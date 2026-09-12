from __future__ import annotations

import json
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from japanese_reply_agent import (
    JapaneseReplyAgent,
    SYSTEM_PROMPT,
    _system_prompt,
    create_strands_agent,
    validate_assist_card,
)
from reasoning_config import ReasoningConfig, ReasoningConfigurationError
from result_sink import CallAssistResultSink


class FakeTrace:
    def __init__(self) -> None:
        self.stages: list[tuple[str, object]] = []

    def stage(self, name: str, *, metadata: object = None) -> object:
        self.stages.append((name, metadata))
        return nullcontext()


class CallAssistAgentTests(unittest.TestCase):
    def test_prompt_targets_foreign_residents_and_multiple_relevant_replies(
        self,
    ) -> None:
        self.assertIn("foreign resident in Japan", SYSTEM_PROMPT)
        self.assertIn("two or three", SYSTEM_PROMPT)
        self.assertIn("directly relevant", SYSTEM_PROMPT)
        self.assertNotIn("business-call assistant", SYSTEM_PROMPT)

    def test_local_model_uses_the_prompt_from_python(self) -> None:
        request = SimpleNamespace(instructions="stale dashboard prompt")

        self.assertEqual(
            _system_prompt(request, ReasoningConfig.local(object())),
            SYSTEM_PROMPT,
        )
        self.assertEqual(
            _system_prompt(
                request,
                ReasoningConfig.vifu_profile("managed-reasoning"),
            ),
            "stale dashboard prompt",
        )

    def test_local_reasoning_tool_schema_describes_each_suggested_reply(self) -> None:
        class CapturingProvider:
            model = "test-local-model"

            def __init__(self) -> None:
                self.request: dict[str, object] | None = None
                self.call_count = 0

            def complete(
                self,
                request: dict[str, object],
                *,
                session_id: str,
            ) -> dict[str, object]:
                self.call_count += 1
                if any(
                    message.get("role") == "tool"
                    for message in request.get("messages", [])
                ):
                    return {
                        "choices": [
                            {
                                "message": {"content": ""},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                self.request = request
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_test",
                                        "type": "function",
                                        "function": {
                                            "name": "publish_call_assist",
                                            "arguments": (
                                                '{"translation_simplified_chinese":'
                                                '"请确认金额和日期。",'
                                                '"suggested_replies":[{'
                                                '"japanese_reply":"確認します。",'
                                                '"simplified_chinese_meaning":"我来确认。"},{'
                                                '"japanese_reply":"申しありません。もう一度お願いします。",'
                                                '"simplified_chinese_meaning":"请再说一遍。"}],'
                                                '"confirmation_required":true,'
                                                '"confirmation_simplified_chinese":'
                                                '"请确认三万日元和下周五。"}'
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }

        provider = CapturingProvider()
        sink = CallAssistResultSink()
        request = SimpleNamespace(
            session_id="local-schema-test",
            instructions=None,
        )
        agent = create_strands_agent(
            object(),
            request,
            sink,
            "来週の金曜日までに三万円を振り込んでください。",
            ReasoningConfig.local(provider),
        )

        sink.begin_agent_turn()
        console_output = StringIO()
        with redirect_stdout(console_output):
            agent("Publish the call assistance card.")
        card = sink.finish_agent_turn()

        self.assertEqual(console_output.getvalue(), "")
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(card.suggested_replies[0].japanese, "確認します。")
        self.assertEqual(
            card.suggested_replies[1].japanese,
            "申し訳ありません。もう一度お願いします。",
        )
        assert provider.request is not None
        tool_schema = provider.request["tools"][0]["function"]["parameters"]
        reply_schema = tool_schema["properties"]["suggested_replies"]["items"]
        replies_property = tool_schema["properties"]["suggested_replies"]
        self.assertEqual(replies_property["minItems"], 2)
        self.assertEqual(replies_property["maxItems"], 3)
        self.assertEqual(
            reply_schema["$ref"],
            "#/$defs/GeneratedSuggestedReply",
        )
        reply_definition = tool_schema["$defs"]["GeneratedSuggestedReply"]
        self.assertEqual(
            set(reply_definition["properties"]),
            {"japanese_reply", "simplified_chinese_meaning"},
        )
        self.assertIn(
            "natural Japanese",
            reply_definition["properties"]["japanese_reply"]["description"],
        )
        self.assertIn(
            "Simplified Chinese",
            reply_definition["properties"]["simplified_chinese_meaning"][
                "description"
            ],
        )
        self.assertIn(
            "confirmation_simplified_chinese",
            tool_schema["required"],
        )
        self.assertIn(
            "Simplified Chinese",
            tool_schema["properties"]["confirmation_simplified_chinese"][
                "description"
            ],
        )

    def test_reasoning_provider_never_receives_credentials_from_voice_agent(
        self,
    ) -> None:
        assistant = JapaneseReplyAgent(
            ReasoningConfig.openai_compatible(
                url="https://provider.example/v1/chat/completions",
                model="provider/conversation",
                api_key="configured-for-test",
            )
        )

        self.assertFalse(hasattr(assistant, "use_session_api_key"))

    def test_strands_turn_returns_one_card_from_text_input(self) -> None:
        trace = FakeTrace()
        prompts: list[dict[str, object]] = []

        def fake_create(
            _app: object,
            _request: object,
            result_sink: object,
            source_text: str,
            _reasoning: object,
        ) -> object:
            class FakeStrandsAgent:
                def __call__(self, prompt: str) -> None:
                    prompts.append(json.loads(prompt))
                    result_sink.publish_from_tool(
                        source_text=source_text,
                        translation_zh="请在下周五前发送报价单。",
                        suggested_replies=[
                            {
                                "japanese": "承知しました。金曜日までにお送りします。",
                                "chinese": "明白了，我会在周五前发送。",
                            },
                            {
                                "japanese": "申し訳ありません。期限を変更できますか。",
                                "chinese": "不好意思，可以更改截止日期吗？",
                            },
                        ],
                        confirmation_required=True,
                        confirmation_zh="请确认下周五截止和发送报价单的承诺。",
                    )

            return FakeStrandsAgent()

        assistant = JapaneseReplyAgent(
            ReasoningConfig.vifu_profile("call-assist-reasoning")
        )
        assistant.vifu_bind(object())
        request = SimpleNamespace(
            input={
                "transcript": "来週の金曜日までに見積書を送ってください。",
            },
            trace=trace,
            session_id="conversation-1",
            instructions=None,
        )
        with patch(
            "japanese_reply_agent.create_strands_agent",
            side_effect=fake_create,
        ):
            result = assistant(request)

        self.assertEqual(result["translationZh"], "请在下周五前发送报价单。")
        self.assertTrue(result["confirmationRequired"])
        self.assertEqual(prompts[0]["transcript"][0]["language"], "ja-JP")
        self.assertEqual(
            prompts[0]["constraints"],
            {
                "confirmationRequired": True,
                "replyPerspective": (
                    "Give two or three distinct options from the listener's point of "
                    "view. Include an option that accepts the request and an option "
                    "that asks for a change or clarification when relevant."
                ),
                "replyMustNotCopySource": True,
                "replyCount": "2 or 3",
                "replyOptionsMustBeDistinctAndRelevant": True,
                "preserveJapaneseTermsInReply": ["金曜日"],
                "suggestedReplyShape": "承知しました。金曜日までにお送りします。",
                "suggestedReplyChineseShape": "明白了，我会在周五前发送。",
                "confirmationChineseMustInclude": ["星期五", "下周", "报价单"],
                "confirmationGuidance": (
                    "Ask the user to confirm the stated deadline and commitment. "
                    "Do not ask what an already-stated date, weekday, or action means."
                ),
            },
        )
        self.assertEqual(
            trace.stages[0][1],
            {
                "framework": "strands-agents",
                "providerType": "vifu-profile",
                "profile": "call-assist-reasoning",
                "transcriptionProvider": "local-whisper",
                "attempt": 1,
            },
        )

    def test_retries_a_wrong_weekday_and_echoed_reply_with_same_provider(self) -> None:
        prompts: list[dict[str, object]] = []
        cards = [
            {
                "translation_zh": "请在下周一之前发送报价单。",
                "suggested_replies": [
                    {
                        "japanese": "来週の金曜日までに見積書を送ってください。",
                        "chinese": "请在下周一之前发送报价单。",
                    }
                ],
                "confirmation_required": False,
                "confirmation_zh": "",
            },
            {
                "translation_zh": "请在下周五之前发送报价单。",
                "suggested_replies": [
                    {
                        "japanese": "承知しました。金曜日までにお送りします。",
                        "chinese": "明白了，我会在周五前发送。",
                    },
                    {
                        "japanese": "申し訳ありません。期限を変更できますか。",
                        "chinese": "不好意思，可以更改截止日期吗？",
                    },
                ],
                "confirmation_required": True,
                "confirmation_zh": "请确认下周五截止以及发送报价单的承诺。",
            },
        ]

        def fake_create(
            _app: object,
            _request: object,
            result_sink: object,
            source_text: str,
            _reasoning: object,
        ) -> object:
            values = cards.pop(0)

            class FakeStrandsAgent:
                def __call__(self, prompt: str) -> None:
                    prompts.append(json.loads(prompt))
                    result_sink.publish_from_tool(
                        source_text=source_text,
                        **values,
                    )

            return FakeStrandsAgent()

        assistant = JapaneseReplyAgent(
            ReasoningConfig.vifu_profile("call-assist-reasoning")
        )
        assistant.vifu_bind(object())
        request = SimpleNamespace(
            input={"transcript": "来週の金曜日までに見積書を送ってください。"},
            trace=FakeTrace(),
            session_id="conversation-retry",
            instructions=None,
        )
        with patch(
            "japanese_reply_agent.create_strands_agent",
            side_effect=fake_create,
        ):
            result = assistant(request)

        self.assertEqual(result["translationZh"], "请在下周五之前发送报价单。")
        self.assertTrue(result["confirmationRequired"])
        self.assertNotIn("validationFeedback", prompts[0])
        self.assertIn(
            "The Assist Card must contain at least two response options.",
            prompts[1]["validationFeedback"],
        )
        self.assertEqual(
            prompts[1]["previousRejectedCard"]["translationZh"],
            "请在下周一之前发送报价单。",
        )

    def test_semantic_validation_accepts_a_safe_deadline_card(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="金曜日までにお願いします。",
            translation_zh="请在周五前完成。",
            suggested_replies=[
                {
                    "japanese": "承知しました。金曜日までに対応します。",
                    "chinese": "明白了，我会在周五前处理。",
                },
                {
                    "japanese": "申し訳ありません。期限を変更できますか。",
                    "chinese": "不好意思，可以更改截止日期吗？",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请确认周五截止。",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

    def test_semantic_validation_accepts_delivery_reply_options(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="明日の午後2時に荷物をお届けしてもよろしいでしょうか。",
            translation_zh="明天下午两点给您送货，可以吗？",
            suggested_replies=[
                {
                    "japanese": "はい、午後2時で大丈夫です。",
                    "chinese": "可以，下午两点没问题。",
                },
                {
                    "japanese": "申し訳ありません。別の時間に変更できますか。",
                    "chinese": "不好意思，可以改到其他时间吗？",
                },
                {
                    "japanese": "すみません、もう一度ゆっくりお願いします。",
                    "chinese": "不好意思，请再慢慢说一遍。",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请确认明天下午两点是否方便收货。",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

    def test_semantic_validation_rejects_reversed_delivery_perspective(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="明日の午後2時に荷物をお届けしてもよろしいでしょうか。",
            translation_zh="明天下午两点送货可以吗？",
            suggested_replies=[
                {
                    "japanese": "了解しました。午後2時に荷物をお届けします。",
                    "chinese": "明白了，我会在下午两点送货。",
                },
                {
                    "japanese": "申しありません。別の時間に変更できますか。",
                    "chinese": "不好意思，可以改到其他时间吗？",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请确认明天下午几点可以收货。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("user will deliver" in error for error in errors))
        self.assertTrue(any("申し訳ありません" in error for error in errors))
        self.assertTrue(any("number 2" in error for error in errors))

    def test_semantic_validation_rejects_wrong_week_and_non_japanese_replies(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="来週の金曜日までに見積書を送ってください。",
            translation_zh="请在本周五之前发送预算书。",
            suggested_replies=[
                {"japanese": "金曜日必须翻译为周五", "chinese": "明白。"},
                {"japanese": "金曜日必须翻译为周五", "chinese": "明白。"},
            ],
            confirmation_required=True,
            confirmation_zh="请确认截止日期。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("来週" in error for error in errors))
        self.assertTrue(any("見積書" in error for error in errors))
        self.assertTrue(any("unique" in error for error in errors))
        self.assertTrue(any("hiragana or katakana" in error for error in errors))

    def test_semantic_validation_rejects_non_chinese_translation_and_meaning(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="おはようございます。",
            translation_zh="Good morning.",
            suggested_replies=[
                {
                    "japanese": "はい、お早うございます。",
                    "chinese": "こんにちは、お早うございます。",
                }
            ],
            confirmation_required=False,
            confirmation_zh="用户需要确认吗？",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("translation" in error for error in errors))
        self.assertTrue(any("Chinese meaning" in error for error in errors))
        self.assertIsNone(card.confirmation_zh)

    def test_semantic_validation_rejects_non_chinese_confirmation(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="来週の金曜日までに見積書を送ってください。",
            translation_zh="请在下周五之前发送报价单。",
            suggested_replies=[
                {
                    "japanese": "承知しました。金曜日までにお送りします。",
                    "chinese": "明白了，我会在周五前发送。",
                }
            ],
            confirmation_required=True,
            confirmation_zh="金曜日までに送ってください。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("confirmation" in error for error in errors))

    def test_semantic_validation_rejects_a_near_copy_of_a_long_request(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="来週の金曜日までに見積書を送ってください。",
            translation_zh="请在下周五之前发送报价单。",
            suggested_replies=[
                {
                    "japanese": "金曜日までに見積書を送ってください。",
                    "chinese": "请在周五前发送报价单。",
                }
            ],
            confirmation_required=True,
            confirmation_zh="请确认周五截止。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("too similar" in error for error in errors))

    def test_semantic_validation_preserves_japanese_weekday_in_reply(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="来週の金曜日までに見積書を送ってください。",
            translation_zh="请在下周五之前发送报价单。",
            suggested_replies=[
                {
                    "japanese": "了解しました、週五までに見積書を送ります。",
                    "chinese": "明白了，我会在下周五前发送报价单。",
                }
            ],
            confirmation_required=True,
            confirmation_zh="请确认周五截止。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("金曜日" in error for error in errors))

    def test_semantic_validation_rejects_observed_ungrammatical_reply(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="金曜日までにお願いします。",
            translation_zh="请在周五前完成。",
            suggested_replies=[
                {
                    "japanese": "承知しました。金曜日までに対します。",
                    "chinese": "明白了，我会在周五前处理。",
                }
            ],
            confirmation_required=True,
            confirmation_zh="请确认周五截止。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("対応します" in error for error in errors))

    def test_semantic_validation_rejects_conflicting_week_and_vague_confirmation(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="来週の金曜日までに見積書を送ってください。",
            translation_zh="请在下周五之前发送报价单。",
            suggested_replies=[
                {
                    "japanese": "承知しました。金曜日までにお送りします。",
                    "chinese": "明白了，我会在本周五之前发送。",
                }
            ],
            confirmation_required=True,
            confirmation_zh="下周是哪一天？",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("本周" in error for error in errors))
        self.assertTrue(any("金曜日" in error for error in errors))
        self.assertTrue(any("見積書" in error for error in errors))

    def test_accepts_a_standard_chat_endpoint_message(self) -> None:
        assistant = JapaneseReplyAgent(
            ReasoningConfig.vifu_profile("call-assist-reasoning")
        )
        assistant.vifu_bind(object())
        request = SimpleNamespace(
            input={
                "messages": [{"role": "user", "content": "確認をお願いします。"}],
            },
            trace=FakeTrace(),
            session_id="conversation-2",
            instructions=None,
        )

        def fake_create(
            _app: object,
            _request: object,
            result_sink: object,
            source_text: str,
            _reasoning: object,
        ) -> object:
            class FakeStrandsAgent:
                def __call__(self, _prompt: str) -> None:
                    result_sink.publish_from_tool(
                        source_text=source_text,
                        translation_zh="请确认。",
                        suggested_replies=[
                            {"japanese": "確認します。", "chinese": "我来确认。"},
                            {
                                "japanese": "もう一度説明をお願いします。",
                                "chinese": "请再说明一次。",
                            },
                        ],
                        confirmation_required=True,
                        confirmation_zh="请确认需要核实的信息。",
                    )

            return FakeStrandsAgent()

        with patch(
            "japanese_reply_agent.create_strands_agent",
            side_effect=fake_create,
        ):
            result = assistant(request)

        self.assertEqual(result["sourceText"], "確認をお願いします。")
        self.assertEqual(result["translationZh"], "请确认。")

    def test_unconfigured_reasoning_provider_fails_before_running(self) -> None:
        assistant = JapaneseReplyAgent(ReasoningConfig.unconfigured())

        with self.assertRaisesRegex(
            ReasoningConfigurationError,
            "JAPANESE_CALL_REASONING_URL",
        ):
            assistant.prepare()


if __name__ == "__main__":
    unittest.main()
