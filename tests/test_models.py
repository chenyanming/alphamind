from __future__ import annotations

import unittest

from models import (
    AssistCard,
    CallAssistInput,
)
from pydantic import ValidationError
from result_sink import CallAssistResultSink


class CallAssistModelTests(unittest.TestCase):
    def test_normalizes_one_on_device_whisper_transcript(self) -> None:
        request = CallAssistInput.model_validate(
            {
                "transcript": "来週の金曜日までに見積書を送ってください。",
                "language": "ja-JP",
            }
        )

        self.assertEqual(request.source, "on_device_whisper")
        self.assertEqual(request.transcription_provider, "local-whisper")
        self.assertEqual(request.transcript[0].sequence, 1)
        self.assertEqual(request.transcript[0].language, "ja-JP")
        self.assertEqual(
            request.transcript[0].text,
            "来週の金曜日までに見積書を送ってください。",
        )

    def test_rejects_audio_and_unordered_transcripts_from_runner_input(self) -> None:
        with self.assertRaises(ValidationError):
            CallAssistInput.model_validate(
                {
                    "audio": "base64-audio-does-not-belong-in-the-endpoint",
                    "transcript": "こんにちは。",
                }
            )

        with self.assertRaisesRegex(ValidationError, "strictly increasing"):
            CallAssistInput.model_validate(
                {
                    "transcript": [
                        {"sequence": 2, "text": "二つ目"},
                        {"sequence": 1, "text": "一つ目"},
                    ],
                }
            )

    def test_confirmation_card_requires_confirmation_copy(self) -> None:
        with self.assertRaises(ValidationError):
            AssistCard.model_validate(
                {
                    "voiceSessionId": "voice-1",
                    "sequence": 1,
                    "sourceText": "金曜日までにお願いします。",
                    "translationZh": "请在周五前完成。",
                    "suggestedReplies": [
                        {"japanese": "承知しました。", "chinese": "明白了。"}
                    ],
                    "confirmationRequired": True,
                }
            )

        with self.assertRaises(ValidationError):
            AssistCard.model_validate(
                {
                    "voiceSessionId": "voice-1",
                    "sequence": 1,
                    "sourceText": "   ",
                    "translationZh": "空白",
                    "suggestedReplies": [{"japanese": "はい。", "chinese": "好的。"}],
                    "confirmationRequired": False,
                }
            )

    def test_collects_one_card_as_an_endpoint_result(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        tool_result = sink.publish_from_tool(
            source_text="来週でお願いします。",
            translation_zh="请安排在下周。",
            suggested_replies=[{"japanese": "承知しました。", "chinese": "明白了。"}],
            confirmation_required=False,
        )
        card = sink.finish_agent_turn()
        self.assertEqual(tool_result, {"accepted": True, "sequence": 1})
        self.assertEqual(card.translation_zh, "请安排在下周。")
        self.assertNotIn("voiceSessionId", card.model_dump(by_alias=True))


if __name__ == "__main__":
    unittest.main()
