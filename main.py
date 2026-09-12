from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from japanese_reply_agent import JapaneseReplyAgent
from reasoning_config import ReasoningConfig, ReasoningConfigurationError
from vifu import LocalLlama, LocalWhisper, Vifu
from voice_agent import JapaneseCallListenerAgent, VoiceConfigurationError

ASSIST_CARD_TOPIC = "japanese-phone-call.assist-card.v1"
LOCAL_WHISPER_MODEL = "ggml-base.bin"
LOCAL_REASONING_MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_REASONING_GPU_LAYERS = (
    36
    if sys.platform == "darwin"
    and getattr(LocalLlama, "supports_safe_accelerator_shutdown", False)
    else 0
)
LOCAL_REASONING_TIMEOUT_SECONDS = 30.0
DEMO_TRANSCRIPT = "明日の午後2時に荷物をお届けしてもよろしいでしょうか。"


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
    print(f"听到日语: {event.get('text', '')}")
    print("正在分析…")


app = Vifu(
    "AlphaMind",
    workspace=Path(__file__).resolve().parent,
    capture_trace_content=True,
)

voice_agent = JapaneseCallListenerAgent(
    language="ja-JP",
    handoff="japanese-reply-agent",
    result_topic=ASSIST_CARD_TOPIC,
    on_result=print_assist_card,
    on_transcript=print_transcript_status,
    transcriber=LocalWhisper(
        model=LOCAL_WHISPER_MODEL,
        language="ja",
    ),
)
configured_reasoning = ReasoningConfig.local(
    LocalLlama(
        model=LOCAL_REASONING_MODEL,
        context_size=8_192,
        default_max_tokens=1_200,
        gpu_layers=LOCAL_REASONING_GPU_LAYERS,
        timeout=LOCAL_REASONING_TIMEOUT_SECONDS,
    )
)
reply_agent = JapaneseReplyAgent(configured_reasoning)

app.agent("japanese-call-listener", voice_agent)
app.agent("japanese-reply-agent", reply_agent)


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
    try:
        if arguments == ["demo"]:
            run_demo()
            return
        app.run()
    except (VoiceConfigurationError, ReasoningConfigurationError) as error:
        raise SystemExit(f"Configuration error: {error}") from None
    finally:
        voice_agent.debug = False


if __name__ == "__main__":
    main()
