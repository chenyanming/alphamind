from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from japanese_reply_agent import JapaneseReplyAgent
from reasoning_config import ReasoningConfig, ReasoningConfigurationError
from speaker_identification import DEFAULT_SPEAKER_MODEL, LocalSpeakerIdentifier
from vifu import LocalLlama, LocalWhisper, Vifu
from voice_agent import JapaneseCallListenerAgent, VoiceConfigurationError

ASSIST_CARD_TOPIC = "japanese-phone-call.assist-card.v1"
LOCAL_WHISPER_MODEL = "ggml-small.bin"
LOCAL_REASONING_MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_REASONING_GPU_LAYERS = (
    36
    if sys.platform == "darwin"
    and getattr(LocalLlama, "supports_safe_accelerator_shutdown", False)
    else 0
)
LOCAL_REASONING_TIMEOUT_SECONDS = 30.0
DEMO_TRANSCRIPT = "今、お時間よろしいでしょうか。"


def print_assist_card(card: dict[str, Any]) -> None:
    print()
    print(f"中文: {card.get('translationZh', '')}")
    replies = card.get("suggestedReplies")
    if isinstance(replies, list) and replies:
        print("建议回复:")
        for index, reply in enumerate(replies, start=1):
            if not isinstance(reply, dict):
                continue
            japanese = str(reply.get("japanese") or "").strip()
            chinese = str(reply.get("chinese") or "").strip()
            meaning = f" — {chinese}" if chinese else ""
            print(f"  {index}. {japanese}{meaning}")
    if card.get("confirmationRequired"):
        print(f"需要确认: {card.get('confirmationZh') or '请先确认关键信息。'}")
    print()


def print_transcript_status(event: dict[str, Any]) -> None:
    print()
    role = event.get("speakerRole", "caller")
    if event.get("speakerEnrollment"):
        print("已记住你的声音。现在可以开始接听日语电话。")
        return
    if role == "self":
        print(f"你说: {event.get('text', '')}")
        return
    if role == "unknown":
        print(f"无法确定说话人，已跳过: {event.get('text', '')}")
        return
    print(f"听到对方: {event.get('text', '')}")
    print("正在分析…")


def print_speaker_status(event: dict[str, Any]) -> None:
    if event.get("state") == "enrollment_required":
        print()
        print("请先对着麦克风说一句完整的话，用来识别你的声音。")
        print("例如：これは私の声です。AlphaMindを開始します。")


app = Vifu(
    "AlphaMind",
    workspace=Path(__file__).resolve().parent,
    capture_trace_content=True,
)

transcription_provider = app.provider(
    "local-whisper",
    LocalWhisper(
        model=LOCAL_WHISPER_MODEL,
        language="ja",
    ),
    name="Japanese Local Whisper",
)
reasoning_provider = app.provider(
    "local-qwen",
    LocalLlama(
        model=LOCAL_REASONING_MODEL,
        context_size=8_192,
        default_max_tokens=1_200,
        gpu_layers=LOCAL_REASONING_GPU_LAYERS,
        timeout=LOCAL_REASONING_TIMEOUT_SECONDS,
    ),
    name="Japanese Local Qwen",
)

voice_agent = JapaneseCallListenerAgent(
    language="ja-JP",
    handoff="japanese-reply-agent",
    result_topic=ASSIST_CARD_TOPIC,
    on_result=print_assist_card,
    on_transcript=print_transcript_status,
    on_speaker_status=print_speaker_status,
    speaker_identifier=LocalSpeakerIdentifier(model=DEFAULT_SPEAKER_MODEL),
    transcriber=transcription_provider,
)
reply_agent = JapaneseReplyAgent(ReasoningConfig.local(reasoning_provider))

app.agent(
    "japanese-call-listener",
    voice_agent,
    implementation="livekit-agents",
    providers={"transcription": transcription_provider},
)
app.agent(
    "japanese-reply-agent",
    reply_agent,
    implementation="strands-agents",
    providers={"reasoning": reasoning_provider},
)


def run_demo() -> None:
    """Run one deterministic two-Agent handoff without opening an audio room."""
    app.connect()
    try:
        print_transcript_status({"text": DEMO_TRANSCRIPT})
        card = voice_agent.dispatch_transcript(
            {
                "schema": "japanese-phone-call.transcript.v1",
                "type": "transcript",
                "voiceSessionId": "local-demo",
                "language": "ja-JP",
                "text": DEMO_TRANSCRIPT,
                "isFinal": True,
                "sequence": 1,
            }
        )
        if card is None:
            raise RuntimeError("the Voice Agent did not produce an Assist Card")
        print_assist_card(card)
    finally:
        app.close()


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    debug = arguments == ["--debug"]
    voice_agent.debug = debug
    reply_agent.debug = debug
    try:
        if arguments == ["demo"]:
            run_demo()
            return
        app.run()
    except (
        VoiceConfigurationError,
        ReasoningConfigurationError,
    ) as error:
        raise SystemExit(f"Configuration error: {error}") from None
    finally:
        voice_agent.debug = False
        reply_agent.debug = False


if __name__ == "__main__":
    main()
