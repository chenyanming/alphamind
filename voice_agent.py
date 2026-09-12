"""Local LiveKit call listener for Japanese Phone Call Translator."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys
import unicodedata
from collections.abc import Awaitable
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock
from time import monotonic
from typing import Any, Iterator

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobExecutorType,
    JobProcess,
    cli,
    stt,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice import UserInputTranscribedEvent, room_io
from livekit.plugins import silero

logger = logging.getLogger("japanese-phone-call-translator-voice")

DUPLICATE_TRANSCRIPT_WINDOW_SECONDS = 3.0
_NATIVE_STDERR_LOCK = Lock()
_MARKER_BOUNDARIES = re.compile(r"[\[\](){}<>【】「」『』♪♫.,!?。！？…·・]+")
_DUPLICATE_SEPARATORS = re.compile(r"[\s.,!?。！？…]+")
_NON_SPEECH_LABELS = frozenset(
    {
        "applause",
        "backgroundmusic",
        "backgroundnoise",
        "bgm",
        "blankaudio",
        "clapping",
        "inaudible",
        "laugh",
        "laughing",
        "laughs",
        "laughter",
        "music",
        "musicplaying",
        "noaudio",
        "noise",
        "silence",
        "unintelligible",
        "ノイズ",
        "拍手",
        "無音",
        "笑",
        "笑い",
        "笑い声",
        "聞き取れない",
        "聞き取り不能",
        "音声なし",
        "音楽",
        "雑音",
    }
)


class VoiceConfigurationError(RuntimeError):
    """The local Voice Agent cannot start with its configured transcriber."""


class BufferedLocalTranscriberSTT(stt.STT):
    """Adapt a local WAV transcriber to LiveKit's non-streaming STT contract."""

    def __init__(
        self,
        transcriber: Any,
        *,
        language: str,
        show_native_logs: bool = False,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        if not callable(getattr(transcriber, "transcribe_wav", None)):
            raise TypeError("transcriber must define transcribe_wav(wav, language=...)")
        self._transcriber = transcriber
        self._default_language = _language_code(language)
        self._provider = _required_name(
            getattr(transcriber, "provider", type(transcriber).__name__),
            "transcriber provider",
        )
        self._model = str(getattr(transcriber, "model", "")).strip()
        self._show_native_logs = show_native_logs

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        del conn_options
        selected_language = (
            self._default_language
            if language is NOT_GIVEN
            else _language_code(str(language))
        )
        wav = _wav_bytes(buffer)
        transcribe = self._transcriber.transcribe_wav
        if inspect.iscoroutinefunction(transcribe):
            text = await transcribe(wav, language=selected_language)
        else:
            text = await asyncio.to_thread(
                self._transcribe_without_native_noise,
                transcribe,
                wav,
                selected_language,
            )
        if not isinstance(text, str):
            raise RuntimeError("local transcriber returned a non-text result")
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text.strip(),
                    language=selected_language,
                    confidence=1.0,
                )
            ],
        )

    def _transcribe_without_native_noise(
        self,
        transcribe: Any,
        wav: bytes,
        language: str,
    ) -> Any:
        with _native_stderr(visible=self._show_native_logs):
            return transcribe(wav, language=language)

    async def aclose(self) -> None:
        return None


class _BackgroundTasks:
    def __init__(self, ctx: JobContext) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closing = False
        add_shutdown_callback = getattr(ctx, "add_shutdown_callback", None)
        if callable(add_shutdown_callback):
            add_shutdown_callback(self.drain)

    def create(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any] | None:
        if self._closing:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            return None
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        self._closing = True
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


class JapaneseCallListenerAgent:
    """Own local audio transcription and hand final turns to a specialist."""

    vifu_name = "Japanese Call Listener"
    vifu_capability = "japanese-call-listening"
    vifu_timeout_ms = 30_000
    vifu_handoff_timeout_seconds = 185.0
    vifu_instructions = (
        "Own the local LiveKit audio session, produce final transcripts, and "
        "hand each final turn to the configured specialist Agent. Do not "
        "perform the specialist's domain reasoning."
    )

    def __init__(
        self,
        *,
        transcriber: Any,
        handoff: str,
        language: str = "ja-JP",
        result_topic: str | None = None,
        on_result: Any | None = None,
        on_transcript: Any | None = None,
        debug: bool = False,
    ) -> None:
        self.transcriber = transcriber
        self.handoff = _required_name(handoff, "handoff")
        self.language = _required_name(language, "language")
        self.result_topic = _optional_name(result_topic, "result_topic")
        self.on_result = on_result
        self.on_transcript = on_transcript
        self.debug = debug
        provider = _required_name(
            getattr(transcriber, "provider", type(transcriber).__name__),
            "transcriber provider",
        )
        self.vifu_metadata = {
            "framework": "livekit-agents",
            "role": "japanese-call-listening",
            "transport": "local-console",
            "protocol": "japanese-phone-call.transcript.v1",
            "language": self.language,
            "handoffAgent": self.handoff,
            "transcriptionProvider": provider,
        }
        self._app: Any | None = None
        self._endpoint: str | None = None
        self._stt: BufferedLocalTranscriberSTT | None = None
        self._latest_sequences: dict[str, int] = {}
        self._pending_transcripts: dict[
            str,
            tuple[JobContext, dict[str, Any]],
        ] = {}
        self._turn_workers: dict[str, asyncio.Task[Any]] = {}
        self._recent_transcripts: dict[tuple[str, str], tuple[str, float]] = {}
        self._recent_transcripts_lock = Lock()
        self._server = AgentServer(
            job_executor_type=JobExecutorType.THREAD,
            setup_fnc=self._prewarm,
        )
        self._server.rtc_session()(self.entrypoint)

    @property
    def server(self) -> AgentServer:
        return self._server

    def prepare(self) -> None:
        prepare = getattr(self.transcriber, "prepare", None)
        if callable(prepare):
            prepare()
        try:
            self._stt = BufferedLocalTranscriberSTT(
                self.transcriber,
                language=self.language,
                show_native_logs=self.debug,
            )
        except (TypeError, ValueError) as error:
            raise VoiceConfigurationError(str(error)) from error

    def vifu_bind(
        self,
        app: Any,
        *,
        endpoint: str,
        **_registration: Any,
    ) -> None:
        self._app = app
        self._endpoint = _required_name(endpoint, "endpoint")

    def vifu_run(self) -> Any:
        original_argv = sys.argv
        debug_alias = original_argv[1:] == ["--debug"]
        if len(original_argv) == 1 or debug_alias:
            log_level = "DEBUG" if self.debug or debug_alias else "ERROR"
            sys.argv = [
                original_argv[0],
                "console",
                "--log-level",
                log_level,
            ]
        try:
            with _livekit_console_banners(visible=self.debug or debug_alias):
                return cli.run_app(self._server)
        finally:
            sys.argv = original_argv

    def close(self) -> None:
        close = getattr(self.transcriber, "close", None)
        if callable(close):
            close()
        self._stt = None
        self._app = None
        self._endpoint = None
        self._latest_sequences.clear()
        self._pending_transcripts.clear()
        self._turn_workers.clear()
        with self._recent_transcripts_lock:
            self._recent_transcripts.clear()

    def __call__(self, request: Any) -> dict[str, Any]:
        event = _voice_transcript(request.input, language=self.language)
        with request.trace.stage(
            "validate",
            metadata={
                "framework": "livekit-agents",
                "step": "voice.handoff",
                "provider": self._provider_name(),
                "protocol": "japanese-phone-call.transcript.v1",
                "isFinal": event["isFinal"],
            },
        ):
            output: dict[str, Any] = {
                "schema": "japanese-phone-call.voice-turn.v1",
                "type": "voice_turn",
                "accepted": False,
                "sessionId": event["voiceSessionId"],
                "sequence": event["sequence"],
            }
            if not event["isFinal"]:
                output["ignoredReason"] = "partial"
                return output
            if _is_non_speech_transcript(event["text"]):
                output["ignoredReason"] = "non_speech"
                return output
            if self._is_recent_duplicate(event):
                output["ignoredReason"] = "duplicate"
                return output
            output["accepted"] = True
            utterance: dict[str, Any] = {
                "sequence": event["sequence"],
                "language": event["language"],
                "text": event["text"],
            }
            if event.get("speakerId"):
                utterance["speakerId"] = event["speakerId"]
            output["handoffInput"] = {
                "source": "livekit",
                "language": event["language"],
                "transcriptionProvider": self._provider_name(),
                "transcript": [utterance],
            }
            return output

    def dispatch_transcript(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Run one Voice Agent turn and its configured specialist handoff."""
        if self._app is None or self._endpoint is None:
            raise RuntimeError(
                "Japanese Call Listener Agent is not attached to an application"
            )
        session_id = _session_id(event)
        voice_turn = self._app.invoke(
            self._endpoint,
            event,
            session_id=session_id,
            timeout=35.0,
        )
        output = voice_turn.output
        if not isinstance(output, dict):
            raise RuntimeError("Voice Agent returned an invalid result")
        if not output.get("accepted"):
            return None
        handoff_input = output.get("handoffInput")
        if not isinstance(handoff_input, dict):
            raise RuntimeError("Voice Agent omitted its specialist handoff input")
        specialist = self._app.invoke(
            self.handoff,
            handoff_input,
            session_id=session_id,
            timeout=self.vifu_handoff_timeout_seconds,
        )
        if not isinstance(specialist.output, dict):
            raise RuntimeError("specialist Agent returned an invalid result")
        return specialist.output

    async def entrypoint(self, ctx: JobContext) -> None:
        """Start one local LiveKit transcription session."""
        if self._stt is None:
            self.prepare()
        assert self._stt is not None
        ctx.log_context_fields = {"room": ctx.room.name}
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        participant = await utils.wait_for_participant(ctx.room)
        vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
        session = AgentSession(
            stt=self._stt,
            vad=vad,
            turn_handling={"turn_detection": "vad"},
        )
        background_tasks = _BackgroundTasks(ctx)
        sequence = 0

        @session.on("user_input_transcribed")
        def on_user_transcript(event: UserInputTranscribedEvent) -> None:
            nonlocal sequence
            text = event.transcript.strip()
            if not event.is_final or not text:
                return
            sequence += 1
            transcript = {
                "schema": "japanese-phone-call.transcript.v1",
                "type": "transcript",
                "voiceSessionId": ctx.room.name,
                "participantIdentity": participant.identity,
                "speakerId": event.speaker_id,
                "language": str(event.language or self.language),
                "text": text,
                "isFinal": True,
                "sequence": sequence,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            self._enqueue_final_transcript(ctx, transcript, background_tasks)

        await session.start(
            Agent(
                instructions="Transcribe Japanese speech into final text turns.",
            ),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                text_output=True,
                audio_output=False,
            ),
        )
        logger.info("Local Japanese call transcription started")

    def _enqueue_final_transcript(
        self,
        ctx: JobContext,
        event: dict[str, Any],
        background_tasks: _BackgroundTasks,
    ) -> None:
        session_id = _session_id(event)
        self._latest_sequences[session_id] = _sequence(event)
        self._pending_transcripts[session_id] = (ctx, event)
        worker = self._turn_workers.get(session_id)
        if worker is not None and not worker.done():
            return
        worker = background_tasks.create(
            self._drain_final_transcripts(session_id),
        )
        if worker is not None:
            self._turn_workers[session_id] = worker

    async def _drain_final_transcripts(self, session_id: str) -> None:
        """Process one active turn and retain only the newest waiting turn."""
        try:
            while pending := self._pending_transcripts.pop(session_id, None):
                ctx, event = pending
                await self._handle_final_transcript(ctx, event)
        finally:
            self._turn_workers.pop(session_id, None)

    async def _handle_final_transcript(
        self,
        ctx: JobContext,
        event: dict[str, Any],
    ) -> None:
        session_id = _session_id(event)
        sequence = _sequence(event)
        self._latest_sequences[session_id] = max(
            sequence,
            self._latest_sequences.get(session_id, sequence),
        )
        try:
            if self.on_transcript is not None:
                callback_result = self.on_transcript(event)
                if inspect.isawaitable(callback_result):
                    await callback_result
            result = await asyncio.to_thread(self.dispatch_transcript, event)
            if result is None or self._latest_sequences.get(session_id) != sequence:
                return
            if self.result_topic:
                await ctx.room.local_participant.publish_data(
                    json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                    reliable=True,
                    topic=self.result_topic,
                )
            if self.on_result is not None:
                callback_result = self.on_result(result)
                if inspect.isawaitable(callback_result):
                    await callback_result
        except Exception:
            logger.exception(
                "Voice Agent handoff to %s failed for session %s",
                self.handoff,
                session_id,
            )

    def _prewarm(self, proc: JobProcess) -> None:
        proc.userdata["vad"] = silero.VAD.load()

    def _provider_name(self) -> str:
        return _required_name(
            getattr(self.transcriber, "provider", type(self.transcriber).__name__),
            "transcriber provider",
        )

    def _is_recent_duplicate(self, event: dict[str, Any]) -> bool:
        now = monotonic()
        session_key = (
            event["voiceSessionId"],
            str(event.get("speakerId") or ""),
        )
        transcript_key = _duplicate_transcript_key(event["text"])
        with self._recent_transcripts_lock:
            expired = [
                key
                for key, (_, seen_at) in self._recent_transcripts.items()
                if now - seen_at > DUPLICATE_TRANSCRIPT_WINDOW_SECONDS
            ]
            for key in expired:
                self._recent_transcripts.pop(key, None)
            previous = self._recent_transcripts.get(session_key)
            self._recent_transcripts[session_key] = (transcript_key, now)
        return bool(
            previous
            and previous[0] == transcript_key
            and now - previous[1] <= DUPLICATE_TRANSCRIPT_WINDOW_SECONDS
        )


@contextmanager
def _native_stderr(*, visible: bool) -> Iterator[None]:
    """Hide native model initialization chatter while preserving exceptions."""
    if visible:
        yield
        return
    with _NATIVE_STDERR_LOCK:
        saved_fd: int | None = None
        null_fd: int | None = None
        try:
            sys.stderr.flush()
            stderr_fd = sys.stderr.fileno()
            saved_fd = os.dup(stderr_fd)
            null_fd = os.open(os.devnull, os.O_WRONLY)
        except (AttributeError, OSError):
            if saved_fd is not None:
                os.close(saved_fd)
            if null_fd is not None:
                os.close(null_fd)
            yield
            return
        try:
            os.dup2(null_fd, stderr_fd)
            yield
        finally:
            try:
                sys.stderr.flush()
            finally:
                os.dup2(saved_fd, stderr_fd)
                os.close(saved_fd)
                os.close(null_fd)


@contextmanager
def _livekit_console_banners(*, visible: bool) -> Iterator[None]:
    """Hide only the deprecated LiveKit wrapper banners in product mode."""
    if visible:
        yield
        return
    try:
        from livekit.agents.cli import _legacy

        console = _legacy.AgentsConsole.get_instance()
        original_print = console.print

        def filtered_print(
            child: Any,
            *,
            tag: str = "",
            tag_style: Any | None = None,
        ) -> None:
            if tag in {"Agents", "Deprecated"}:
                return
            original_print(child, tag=tag, tag_style=tag_style)

        console.print = filtered_print
    except (AttributeError, ImportError):
        yield
        return
    try:
        yield
    finally:
        console.print = original_print


def _wav_bytes(buffer: utils.AudioBuffer) -> bytes:
    if isinstance(buffer, list):
        if not buffer:
            return b""
        frame = rtc.combine_audio_frames(buffer)
    else:
        frame = buffer
    resampler = rtc.AudioResampler(
        input_rate=frame.sample_rate,
        output_rate=16_000,
        num_channels=frame.num_channels,
        quality=rtc.AudioResamplerQuality.MEDIUM,
    )
    frames = resampler.push(frame)
    frames.extend(resampler.flush())
    if not frames:
        return b""
    return rtc.combine_audio_frames(frames).to_wav_bytes()


def _voice_transcript(value: object, *, language: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Voice Agent input must be an object")
    event = dict(value)
    schema = event.get("schema", "japanese-phone-call.transcript.v1")
    if schema != "japanese-phone-call.transcript.v1":
        raise ValueError(
            "Voice Agent input schema must be japanese-phone-call.transcript.v1"
        )
    if event.get("type", "transcript") != "transcript":
        raise ValueError("Voice Agent input type must be transcript")
    event["schema"] = schema
    event["type"] = "transcript"
    event["voiceSessionId"] = _session_id(event)
    text = event.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Voice Agent transcript text must not be empty")
    event["text"] = text.strip()
    if not isinstance(event.get("isFinal"), bool):
        raise ValueError("Voice Agent transcript isFinal must be a boolean")
    event["sequence"] = _sequence(event)
    event["language"] = _required_name(event.get("language", language), "language")
    speaker_id = event.get("speakerId")
    if speaker_id is not None:
        event["speakerId"] = _required_name(speaker_id, "speakerId")
    return event


def _is_non_speech_transcript(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    labels = [
        compact
        for part in _MARKER_BOUNDARIES.split(normalized)
        if (compact := _compact_marker_label(part))
    ]
    return not labels or all(label in _NON_SPEECH_LABELS for label in labels)


def _compact_marker_label(value: str) -> str:
    return "".join(value.split()).replace("_", "").replace("-", "")


def _duplicate_transcript_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _DUPLICATE_SEPARATORS.sub("", normalized)


def _session_id(event: dict[str, Any]) -> str:
    return _required_name(event.get("voiceSessionId"), "voiceSessionId")


def _sequence(event: dict[str, Any]) -> int:
    value = event.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Voice Agent transcript sequence must be a positive integer")
    return value


def _language_code(value: str) -> str:
    return _required_name(value, "language").split("-", 1)[0].lower()


def _required_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


def _optional_name(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_name(value, field)


__all__ = [
    "BufferedLocalTranscriberSTT",
    "JapaneseCallListenerAgent",
    "VoiceConfigurationError",
]
