from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from livekit import rtc
from livekit.agents import stt

import main
from voice_agent import BufferedLocalTranscriberSTT, JapaneseCallListenerAgent


def fake_transcriber() -> SimpleNamespace:
    return SimpleNamespace(
        provider="test-local-whisper",
        model="test-whisper.bin",
    )


class RecordingTranscriber:
    provider = "test-local-whisper"
    model = "test-whisper.bin"

    def __init__(self) -> None:
        self.wav = b""
        self.language = ""

    def transcribe_wav(self, wav: bytes, *, language: str) -> str:
        self.wav = wav
        self.language = language
        return " 来週の金曜日です。 "


class RecordingSpeakerIdentifier:
    provider = "test-local-speaker-id"

    def __init__(self, speaker_id: str = "caller-2") -> None:
        self.speaker_id = speaker_id
        self.wav = b""

    def identify_wav(self, wav: bytes) -> str:
        self.wav = wav
        return self.speaker_id


class VoiceAgentContractTests(unittest.TestCase):
    def test_app_voice_agent_owns_realtime_orchestration(self) -> None:
        voice = main.voice_agent

        self.assertEqual(voice.vifu_capability, "japanese-call-listening")
        self.assertEqual(
            voice.vifu_metadata["role"],
            "japanese-call-listening",
        )
        self.assertEqual(voice.handoff, "japanese-reply-agent")
        self.assertEqual(voice.result_topic, main.ASSIST_CARD_TOPIC)

    def test_default_local_runtime_uses_the_quiet_console_log_level(self) -> None:
        voice = JapaneseCallListenerAgent(
            handoff="japanese-reply-agent",
            transcriber=fake_transcriber(),
        )
        captured_argv: list[str] = []

        def capture_run_app(_server: object) -> None:
            captured_argv.extend(sys.argv)

        with (
            patch.object(sys, "argv", ["main.py"]),
            patch("voice_agent.cli.run_app", side_effect=capture_run_app),
        ):
            voice.vifu_run()

        self.assertEqual(
            captured_argv,
            ["main.py", "console", "--log-level", "ERROR"],
        )


class VoiceAgentResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_session_coalesces_waiting_turns_to_the_latest(self) -> None:
        voice = JapaneseCallListenerAgent(
            handoff="japanese-reply-agent",
            transcriber=fake_transcriber(),
        )
        ctx = SimpleNamespace()
        handled_sequences: list[int] = []

        def event(sequence: int) -> dict[str, object]:
            return {
                "voiceSessionId": "voice-1",
                "language": "ja-JP",
                "text": f"turn-{sequence}",
                "isFinal": True,
                "sequence": sequence,
            }

        async def handle(_ctx: object, current: dict[str, object]) -> None:
            handled_sequences.append(int(current["sequence"]))
            if current["sequence"] == 1:
                voice._pending_transcripts["voice-1"] = (ctx, event(2))
                voice._pending_transcripts["voice-1"] = (ctx, event(3))

        voice._pending_transcripts["voice-1"] = (ctx, event(1))
        with patch.object(voice, "_handle_final_transcript", side_effect=handle):
            await voice._drain_final_transcripts("voice-1")

        self.assertEqual(handled_sequences, [1, 3])

    async def test_livekit_audio_is_adapted_to_the_local_transcriber(self) -> None:
        transcriber = RecordingTranscriber()
        adapter = BufferedLocalTranscriberSTT(transcriber, language="ja-JP")
        frame = rtc.AudioFrame.create(
            sample_rate=48_000,
            num_channels=1,
            samples_per_channel=4_800,
        )

        with patch(
            "voice_agent._native_stderr",
            return_value=nullcontext(),
        ) as native_stderr:
            event = await adapter.recognize(frame)

        native_stderr.assert_called_once_with(visible=False)
        self.assertEqual(event.type, stt.SpeechEventType.FINAL_TRANSCRIPT)
        self.assertEqual(event.alternatives[0].text, "来週の金曜日です。")
        self.assertEqual(event.alternatives[0].language, "ja")
        self.assertEqual(transcriber.language, "ja")
        self.assertTrue(transcriber.wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", transcriber.wav[:16])

    async def test_livekit_stt_emits_the_local_speaker_identity(self) -> None:
        transcriber = RecordingTranscriber()
        identifier = RecordingSpeakerIdentifier()
        adapter = BufferedLocalTranscriberSTT(
            transcriber,
            language="ja-JP",
            speaker_identifier=identifier,
        )
        frame = rtc.AudioFrame.create(
            sample_rate=48_000,
            num_channels=1,
            samples_per_channel=4_800,
        )

        with patch("voice_agent._native_stderr", return_value=nullcontext()):
            event = await adapter.recognize(frame)

        self.assertTrue(adapter.capabilities.diarization)
        self.assertEqual(event.alternatives[0].speaker_id, "caller-2")
        self.assertTrue(identifier.wav.startswith(b"RIFF"))

    async def test_final_turn_publishes_the_specialist_result(self) -> None:
        voice = JapaneseCallListenerAgent(
            handoff="japanese-reply-agent",
            result_topic=main.ASSIST_CARD_TOPIC,
            transcriber=fake_transcriber(),
        )
        participant = SimpleNamespace(publish_data=AsyncMock())
        ctx = SimpleNamespace(room=SimpleNamespace(local_participant=participant))
        event = {
            "voiceSessionId": "voice-1",
            "language": "ja-JP",
            "text": "少々お待ちください。",
            "isFinal": True,
            "sequence": 1,
        }
        card = {
            "type": "assist_card",
            "sourceText": "少々お待ちください。",
            "translationZh": "请稍等。",
            "suggestedReplies": [],
            "confirmationRequired": False,
        }

        with patch.object(voice, "dispatch_transcript", return_value=card):
            await voice._handle_final_transcript(ctx, event)

        participant.publish_data.assert_awaited_once()
        self.assertEqual(
            participant.publish_data.await_args.kwargs["topic"],
            main.ASSIST_CARD_TOPIC,
        )

    async def test_result_from_an_interrupted_turn_is_not_published(self) -> None:
        voice = JapaneseCallListenerAgent(
            handoff="japanese-reply-agent",
            result_topic=main.ASSIST_CARD_TOPIC,
            transcriber=fake_transcriber(),
        )
        participant = SimpleNamespace(publish_data=AsyncMock())
        ctx = SimpleNamespace(room=SimpleNamespace(local_participant=participant))
        event = {
            "voiceSessionId": "voice-1",
            "language": "ja-JP",
            "text": "最初の発話です。",
            "isFinal": True,
            "sequence": 1,
        }

        async def finish_after_newer_turn(*_args: object) -> dict[str, object]:
            voice._latest_sequences["voice-1"] = 2
            return {"type": "assist_card"}

        with patch(
            "voice_agent.asyncio.to_thread",
            side_effect=finish_after_newer_turn,
        ):
            await voice._handle_final_transcript(ctx, event)

        participant.publish_data.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
