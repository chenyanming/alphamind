from __future__ import annotations

import json
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from vifu import LocalProviderError

from japanese_reply_agent import (
    JapaneseReplyAgent,
    SYSTEM_PROMPT,
    _system_prompt,
    analyze_transcript,
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
        self.assertIn("生活在日本的外国人", SYSTEM_PROMPT)
        self.assertIn("含义不同", SYSTEM_PROMPT)
        self.assertIn("部分内容听不清", SYSTEM_PROMPT)
        self.assertIn("不要替用户决定事实", SYSTEM_PROMPT)
        self.assertNotIn("business-call assistant", SYSTEM_PROMPT)
        self.assertNotIn("お届け", SYSTEM_PROMPT)
        self.assertNotIn("配達", SYSTEM_PROMPT)

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
                                            "name": "card",
                                            "arguments": (
                                                '{"translation_zh":"请确认金额和日期。",'
                                                '"replies":[{'
                                                '"japanese":"確認します。",'
                                                '"chinese":"我来确认。"},{'
                                                '"japanese":"申しありません。もう一度お願いします。",'
                                                '"chinese":"请再说一遍。"}],'
                                                '"confirmation_zh":'
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
        reply_schema = tool_schema["properties"]["replies"]["items"]
        replies_property = tool_schema["properties"]["replies"]
        self.assertEqual(replies_property["minItems"], 2)
        self.assertEqual(replies_property["maxItems"], 2)
        self.assertEqual(
            reply_schema["$ref"],
            "#/$defs/GeneratedSuggestedReply",
        )
        reply_definition = tool_schema["$defs"]["GeneratedSuggestedReply"]
        self.assertEqual(
            set(reply_definition["properties"]),
            {"japanese", "chinese"},
        )
        self.assertIn(
            "自然日语",
            reply_definition["properties"]["japanese"]["description"],
        )
        self.assertIn(
            "简体中文",
            reply_definition["properties"]["chinese"]["description"],
        )
        self.assertNotIn("confirm", tool_schema["properties"])
        self.assertIn("confirmation_zh", tool_schema["required"])
        self.assertIn(
            "简体中文",
            tool_schema["properties"]["confirmation_zh"]["description"],
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
        self.assertEqual(
            prompts[0]["latest_ja"],
            "来週の金曜日までに見積書を送ってください。",
        )
        self.assertTrue(prompts[0]["confirm"])
        self.assertEqual(
            prompts[0]["facts"],
            [
                {"ja": "金曜日", "zh": "星期五"},
                {"ja": "来週", "zh": "下周"},
            ],
        )
        self.assertEqual(
            prompts[0]["reply_strategy"],
            "回应来电者；第二项提出相关追问或另一种选择",
        )
        self.assertEqual(
            prompts[0]["output_language"],
            "所有 *_zh 和 replies.chinese 字段只能使用简体中文",
        )
        self.assertEqual(
            prompts[0]["translation_rule"],
            "只翻译 latest_ja，不得添加未提到的信息或回答建议",
        )
        self.assertNotIn("answer_mode", prompts[0])
        self.assertEqual(prompts[0]["utterance_type"], "statement")
        self.assertNotIn("reply_templates", prompts[0])
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
        self.assertNotIn("fix", prompts[0])
        self.assertIn(
            "The Assist Card must contain at least two response options.",
            prompts[1]["fix"],
        )
        self.assertEqual(
            prompts[1]["rejected"]["translationZh"],
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

    def test_yes_no_request_adds_generic_answer_mode(self) -> None:
        prompts: list[dict[str, object]] = []

        class FakeAgent:
            def __call__(self, prompt: str) -> None:
                prompts.append(json.loads(prompt))

        analyze_transcript(
            FakeAgent(),
            [
                {
                    "text": (
                        "登録内容に変更がないか、はい、いえで"
                        "お答えいただけますか。"
                    )
                }
            ],
        )

        self.assertEqual(prompts[0]["answer_mode"], "yes_no")
        self.assertEqual(
            prompts[0]["reply_strategy"],
            "一项肯定回答；一项否定或要求澄清的回答；都使用接听者口吻",
        )
        self.assertEqual(
            prompts[0]["reply_constraints"],
            [
                "一个回答以「はい」开头",
                "另一个回答以「いいえ」开头，或明确请求对方重述需要确认的内容",
                "不能把对方的是非问题原样反问回去",
            ],
        )

    def test_yes_no_request_rejects_two_rephrased_questions(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="変更がないか、はい、いえでお答えいただけますか。",
            translation_zh="是否没有变更，请回答是或否。",
            suggested_replies=[
                {"japanese": "変更がないですね？", "chinese": "没有变更，是吗？"},
                {"japanese": "変更があるんですか？", "chinese": "有变更吗？"},
            ],
            confirmation_required=True,
            confirmation_zh="请确认是否有变更。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("yes/no request" in error for error in errors))

    def test_information_request_adds_listener_role_contract(self) -> None:
        prompts: list[dict[str, object]] = []

        class FakeAgent:
            def __call__(self, prompt: str) -> None:
                prompts.append(json.loads(prompt))

        analyze_transcript(
            FakeAgent(),
            [{"text": "ご予約のお名前と日々を教えてもらいますか"}],
        )

        self.assertEqual(
            prompts[0]["speech_act"],
            "caller_requests_information_from_listener",
        )
        self.assertEqual(prompts[0]["output_mode"], "translation_only")
        self.assertEqual(
            prompts[0]["role_contract"],
            "来电者正在请接听者提供信息；回答必须由接听者说给来电者",
        )
        self.assertTrue(
            any(
                "禁止编造姓名、日期、金额、号码或示例值" in constraint
                for constraint in prompts[0]["information_reply_constraints"]
            )
        )

    def test_coaching_fragment_uses_caller_context_and_listener_role(self) -> None:
        prompts: list[dict[str, object]] = []

        class FakeAgent:
            def __call__(self, prompt: str) -> None:
                prompts.append(json.loads(prompt))

        analyze_transcript(
            FakeAgent(),
            [
                {"sequence": 1, "text": "じゃあ次は出身を一言足してみようか"},
                {"sequence": 2, "text": "「○○出身です」って感じで"},
            ],
        )

        self.assertEqual(
            prompts[0]["previous_ja"],
            ["じゃあ次は出身を一言足してみようか"],
        )
        self.assertEqual(
            prompts[0]["speech_act"],
            "caller_coaches_listener_response",
        )
        self.assertIn(
            "接听者自己的信息",
            prompts[0]["role_contract"],
        )
        self.assertTrue(
            any(
                "不能反问来电者" in constraint
                for constraint in prompts[0]["coaching_reply_constraints"]
            )
        )

    def test_coaching_reply_rejects_reversed_personal_information_question(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="「○○出身です」って感じで",
            translation_zh="用“我来自某地”这样的说法。",
            suggested_replies=[
                {
                    "japanese": "はい、言ってみます。",
                    "chinese": "好的，我试着说。",
                },
                {
                    "japanese": "出身地を具体的に教えていただけますか。",
                    "chinese": "可以具体告诉我您的出生地吗？",
                },
            ],
            confirmation_required=False,
            confirmation_zh="",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("coaching" in error for error in errors))

    def test_information_request_tool_keeps_unknown_values_in_application_code(
        self,
    ) -> None:
        class CapturingProvider:
            model = "test-local-model"

            def __init__(self) -> None:
                self.request: dict[str, object] | None = None

            def complete(
                self,
                request: dict[str, object],
                *,
                session_id: str,
            ) -> dict[str, object]:
                del session_id
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
                                        "id": "call_information_request",
                                        "type": "function",
                                        "function": {
                                            "name": "card",
                                            "arguments": (
                                                '{"translation_zh":'
                                                '"对方请您提供预约姓名和日期。"}'
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
            session_id="information-tool-contract",
            instructions=None,
        )
        source_text = "ご予約のお名前と日々を教えてもらいますか"
        agent = create_strands_agent(
            object(),
            request,
            sink,
            source_text,
            ReasoningConfig.local(provider),
        )

        sink.begin_agent_turn()
        with redirect_stdout(StringIO()):
            agent("翻译信息请求。")
        card = sink.finish_agent_turn()

        assert provider.request is not None
        tool_schema = provider.request["tools"][0]["function"]["parameters"]
        self.assertEqual(tool_schema["required"], ["translation_zh"])
        self.assertEqual(
            [reply.japanese for reply in card.suggested_replies],
            [
                "はい、確認してお伝えします。",
                "申し訳ありません。確認しますので、少々お待ちください。",
            ],
        )
        self.assertEqual(validate_assist_card(source_text, card), [])

    def test_information_request_rejects_role_reversed_translation(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="ご予約のお名前と日々を教えてもらいますか",
            translation_zh="需要对方提供预约的姓名和日期。",
            suggested_replies=[
                {
                    "japanese": "了解しました、名前と日付を記します。",
                    "chinese": "好的，我会记录下来。",
                },
                {
                    "japanese": "申し訳ありません、もう一度お願いします。",
                    "chinese": "不好意思，请再说一遍。",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请确认对方是否准备好了预约信息。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("caller asks the listener" in error for error in errors))

    def test_information_request_rejects_invented_values(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="ご予約のお名前と日々を教えてもらいますか",
            translation_zh="请告诉对方预约姓名和日期。",
            suggested_replies=[
                {
                    "japanese": "お名前は張三で、日付は2023年10月1日です。",
                    "chinese": "姓名是张三，日期是2023年10月1日。",
                },
                {
                    "japanese": "確認しますので、少々お待ちください。",
                    "chinese": "我确认一下，请稍等。",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请确认自己要提供的预约姓名和日期。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("must not invent" in error for error in errors))

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

    def test_semantic_validation_rejects_bad_japanese(self) -> None:
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

        self.assertTrue(any("申し訳ありません" in error for error in errors))

    def test_semantic_validation_accepts_a_summary_for_a_long_noisy_utterance(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text=(
                "失礼いたしました。では、予定だけお伝えします。"
                "ご登録の50所とお車ライフ方法に変更がないかの確認だけ、"
                "はい、いえでお答えいただけますか。"
            ),
            translation_zh=(
                "不好意思，只向您说明安排。对方想确认登记信息是否有变更，"
                "请回答是或否。"
            ),
            suggested_replies=[
                {
                    "japanese": "はい、変更はありません。",
                    "chinese": "是的，没有变更。",
                },
                {
                    "japanese": "すみません、確認内容をもう一度お願いします。",
                    "chinese": "不好意思，请再说一遍要确认的内容。",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请在回答前确认登记信息是否有变更。",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

    def test_plain_question_does_not_require_commitment_confirmation(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="今、お時間よろしいでしょうか？",
            translation_zh="请问现在方便吗？",
            suggested_replies=[
                {"japanese": "はい、大丈夫です。", "chinese": "可以，没问题。"},
                {
                    "japanese": "申し訳ありません。後でもよろしいですか。",
                    "chinese": "不好意思，稍后可以吗？",
                },
            ],
            confirmation_required=False,
            confirmation_zh="",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

    def test_semantic_validation_accepts_an_imperative_translation_of_a_request(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="はい、いえでお答えいただけますか。",
            translation_zh="请回答是或否。",
            suggested_replies=[
                {"japanese": "はい、変更はありません。", "chinese": "是的，没有变更。"},
                {
                    "japanese": "すみません、もう一度お願いします。",
                    "chinese": "不好意思，请再说一遍。",
                },
            ],
            confirmation_required=False,
            confirmation_zh="",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

    def test_semantic_validation_rejects_multiple_values_in_translation(self) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="現在の時間では難しいです。",
            translation_zh="现在这个时间不方便。 / 我稍后再打。",
            suggested_replies=[
                {"japanese": "分かりました。", "chinese": "明白了。"},
                {
                    "japanese": "いつがよろしいですか。",
                    "chinese": "什么时候方便？",
                },
            ],
            confirmation_required=False,
            confirmation_zh="",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("one translation" in error for error in errors))

    def test_semantic_validation_accepts_a_contextual_time_reply(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="明日の午後2時に荷物をお届けしてもよろしいでしょうか。",
            translation_zh="明天下午两点给您送货，可以吗？",
            suggested_replies=[
                {
                    "japanese": "はい、その時間で大丈夫です。",
                    "chinese": "好的，那个时间可以。",
                },
                {
                    "japanese": "申し訳ありません。別の時間に変更できますか。",
                    "chinese": "不好意思，可以改到其他时间吗？",
                },
            ],
            confirmation_required=True,
            confirmation_zh="请确认明天下午两点是否方便收货。",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

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

    def test_semantic_validation_rejects_an_exact_copy_of_a_long_request(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="来週の金曜日までに見積書を送ってください。",
            translation_zh="请在下周五之前发送报价单。",
            suggested_replies=[
                {
                    "japanese": "来週の金曜日までに見積書を送ってください。",
                    "chinese": "请在下周五前发送报价单。",
                }
            ],
            confirmation_required=True,
            confirmation_zh="请确认周五截止。",
        )
        card = sink.finish_agent_turn()

        errors = validate_assist_card(card.source_text, card)

        self.assertTrue(any("must not repeat" in error for error in errors))

    def test_semantic_validation_does_not_block_a_nonidentical_short_reply(
        self,
    ) -> None:
        sink = CallAssistResultSink()
        sink.begin_agent_turn()
        sink.publish_from_tool(
            source_text="現在の時間では難しいです。",
            translation_zh="现在这个时间不方便。",
            suggested_replies=[
                {
                    "japanese": "現在の時間は難しいです、すみません。",
                    "chinese": "现在这个时间确实不方便。",
                },
                {
                    "japanese": "いつがよろしいですか。",
                    "chinese": "什么时候方便？",
                },
            ],
            confirmation_required=False,
            confirmation_zh="",
        )
        card = sink.finish_agent_turn()

        self.assertEqual(validate_assist_card(card.source_text, card), [])

    def test_semantic_validation_rejects_chinese_weekday_in_japanese_reply(
        self,
    ) -> None:
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

        self.assertTrue(any("Chinese weekday" in error for error in errors))

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

    def test_semantic_validation_rejects_conflicting_week_in_reply_meaning(
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

    def test_returns_a_safe_card_when_retries_remain_unusable(self) -> None:
        attempts = 0

        def fake_create(
            _app: object,
            _request: object,
            result_sink: object,
            source_text: str,
            _reasoning: object,
        ) -> object:
            nonlocal attempts
            attempts += 1

            class FakeStrandsAgent:
                def __call__(self, _prompt: str) -> None:
                    result_sink.publish_from_tool(
                        source_text=source_text,
                        translation_zh=source_text,
                        suggested_replies=[
                            {
                                "japanese": "はい、変更はありません。",
                                "chinese": "是的，没有变更。",
                            },
                            {
                                "japanese": "いいえ、変更があります。",
                                "chinese": "不，有变更。",
                            },
                        ],
                        confirmation_required=True,
                        confirmation_zh="请确认登记信息是否有变更。",
                    )

            return FakeStrandsAgent()

        source_text = (
            "失礼いたしました。では、予定だけお伝えします。"
            "ご登録の50所とお車ライフ方法に変更がないかの確認だけ、"
            "はい、いえでお答えいただけますか。"
        )
        assistant = JapaneseReplyAgent(
            ReasoningConfig.vifu_profile("call-assist-reasoning")
        )
        assistant.vifu_bind(object())
        request = SimpleNamespace(
            input={"transcript": source_text},
            trace=FakeTrace(),
            session_id="long-noisy-turn",
            instructions=None,
        )

        with patch(
            "japanese_reply_agent.create_strands_agent",
            side_effect=fake_create,
        ):
            result = assistant(request)

        self.assertEqual(attempts, 2)
        self.assertEqual(
            result["translationZh"],
            "这段话有部分内容没有听清，请先让对方重述关键信息。",
        )
        self.assertEqual(len(result["suggestedReplies"]), 2)
        self.assertTrue(result["confirmationRequired"])
        self.assertIn("请勿直接确认", result["confirmationZh"])

    def test_retries_a_transient_provider_failure_on_the_same_provider(self) -> None:
        attempts = 0

        def fake_create(
            _app: object,
            _request: object,
            result_sink: object,
            source_text: str,
            _reasoning: object,
        ) -> object:
            nonlocal attempts
            attempts += 1
            current_attempt = attempts

            class FakeStrandsAgent:
                def __call__(self, _prompt: str) -> None:
                    if current_attempt == 1:
                        raise LocalProviderError(
                            "OpenAI-compatible Provider request failed"
                        )
                    result_sink.publish_from_tool(
                        source_text=source_text,
                        translation_zh="早上好。",
                        suggested_replies=[
                            {
                                "japanese": "はい、おはようございます。",
                                "chinese": "早上好。",
                            },
                            {
                                "japanese": "本日もよろしくお願いします。",
                                "chinese": "今天也请多关照。",
                            },
                        ],
                        confirmation_required=False,
                        confirmation_zh="",
                    )

            return FakeStrandsAgent()

        assistant = JapaneseReplyAgent(
            ReasoningConfig.vifu_profile("call-assist-reasoning")
        )
        assistant.vifu_bind(object())
        request = SimpleNamespace(
            input={"transcript": "おはようございます。"},
            trace=FakeTrace(),
            session_id="provider-retry",
            instructions=None,
        )

        with patch(
            "japanese_reply_agent.create_strands_agent",
            side_effect=fake_create,
        ):
            result = assistant(request)

        self.assertEqual(attempts, 2)
        self.assertEqual(result["translationZh"], "早上好。")

    def test_does_not_retry_a_provider_authentication_failure(self) -> None:
        attempts = 0

        def fake_create(
            _app: object,
            _request: object,
            _result_sink: object,
            _source_text: str,
            _reasoning: object,
        ) -> object:
            nonlocal attempts
            attempts += 1

            class FakeStrandsAgent:
                def __call__(self, _prompt: str) -> None:
                    raise LocalProviderError(
                        "OpenAI-compatible Provider returned HTTP 401"
                    )

            return FakeStrandsAgent()

        assistant = JapaneseReplyAgent(
            ReasoningConfig.vifu_profile("call-assist-reasoning")
        )
        assistant.vifu_bind(object())
        request = SimpleNamespace(
            input={"transcript": "おはようございます。"},
            trace=FakeTrace(),
            session_id="provider-auth-error",
            instructions=None,
        )

        with (
            patch(
                "japanese_reply_agent.create_strands_agent",
                side_effect=fake_create,
            ),
            self.assertRaisesRegex(LocalProviderError, "HTTP 401"),
        ):
            assistant(request)

        self.assertEqual(attempts, 1)

    def test_unconfigured_reasoning_provider_fails_before_running(self) -> None:
        assistant = JapaneseReplyAgent(ReasoningConfig.unconfigured())

        with self.assertRaisesRegex(
            ReasoningConfigurationError,
            "JAPANESE_CALL_REASONING_URL",
        ):
            assistant.prepare()


if __name__ == "__main__":
    unittest.main()
