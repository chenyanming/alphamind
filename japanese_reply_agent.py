from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from models import CallAssistInput
from native_output import native_stderr
from reasoning_config import ReasoningConfig
from result_sink import CallAssistResultSink
from vifu import AgentRequest

logger = logging.getLogger("alphamind-reasoning")

MAX_REASONING_ATTEMPTS = 2
MAX_REASONING_TOKENS = 240
_WEEKDAY_TRANSLATIONS = {
    "月曜日": ("星期一", "周一", "礼拜一"),
    "火曜日": ("星期二", "周二", "礼拜二"),
    "水曜日": ("星期三", "周三", "礼拜三"),
    "木曜日": ("星期四", "周四", "礼拜四"),
    "金曜日": ("星期五", "周五", "礼拜五"),
    "土曜日": ("星期六", "周六", "礼拜六"),
    "日曜日": ("星期日", "星期天", "周日", "周天", "礼拜日", "礼拜天"),
}
_CONFIRMATION_MARKERS = (
    "今日",
    "明日",
    "明後日",
    "午前",
    "午後",
    "まで",
    "期限",
    "締切",
    "納期",
    "予定",
    "予約",
    "登録",
    "名前",
    "住所",
    "電話番号",
    "変更",
    "確認",
    "約束",
    "円",
    "個",
    "本",
    "ください",
    "お願いします",
)
_SOURCE_TERM_TRANSLATIONS = {
    "明日": ("明天",),
    "午後": ("下午",),
    "来週": ("下周", "下星期", "下礼拜"),
    "今週": ("本周", "这周", "这个星期", "本星期", "这个礼拜"),
}
_UNNATURAL_REPLY_FRAGMENTS = {
    "申しありません": "申し訳ありません",
    "対します": "対応します or 対応いたします",
    "他時": "別の時間",
    "了解了": "分かりました or 承知しました",
}
_CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_JAPANESE_KANA = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")
_JAPANESE_TIME = re.compile(r"(?:午前|午後)?[0-9０-９]+時(?:[0-9０-９]+分)?")
_CHINESE_WEEKDAY_IN_JAPANESE = re.compile(r"週[一二三四五六日天]")
_QUESTION_ENDING = re.compile(r"か[?？。]?$|[?？]$")
_TRANSLATION_SEPARATOR = re.compile(r"\s[/／]\s")
_INFORMATION_REQUEST = re.compile(
    r"(?:教えて(?:いただけ|もらえ|もらい|ください)|お聞かせください)"
)


class GeneratedSuggestedReply(BaseModel):
    """Compact internal fields keep local-model response latency bounded."""

    model_config = ConfigDict(extra="forbid")

    japanese: str = Field(
        min_length=1,
        max_length=60,
        description="用户可以直接对来电者说出的简短、自然日语；不能复述对方原话。",
    )
    chinese: str = Field(
        min_length=1,
        max_length=60,
        description="这句日语回答的简体中文含义；不能包含日语假名。",
    )

SYSTEM_PROMPT = """
你帮助生活在日本的外国人理解并回应日语电话。只分析 `latest_ja`；仅当
最新一句明确指代前文时，才使用 `previous_ja`。不要从示例或前文带入最新
一句没有提到的话题、物品、业务或行动。

语音转写可能包含错字。用简体中文概括能够可靠理解的整体意思；无法辨认的
片段写成“（部分内容听不清）”。任何中文字段都不能复制日文或包含日语假名。
保留能够确定的人名、日期、时间、价格和数量，但不要猜测，不要创造事实、
承诺或动作，也不能颠倒来电者与接听者的角色。`translation_zh` 只能翻译
对方原话，不能加入回答建议，也不能把原句改写成新的追问。
`Xを教えてもらえますか／いただけますか` 表示来电者请接听者提供 X；
翻译、确认提示和建议回答都必须保持这个方向，不能反过来让来电者提供 X。
如果输入没有给出 X 的真实值，绝不能替用户填写示例姓名、日期、金额或号码；
建议用户表示会提供真实信息，或者请对方稍候以便核对。

严格调用一次 `card`。`translation_zh` 是一条简洁的中文解释。`replies` 必须
包含两个简短、礼貌且含义不同的日语回答，并附各自的简体中文含义：第一个
直接回答或回应来电者，第二个提供另一种真实可选的回答、拒绝或澄清方式。
这些是供用户选择的说法，不要替用户决定事实。对方要求回答是或否时，必须
提供肯定与否定/澄清两种接听者回答，不能用“知道了”代替回答。`confirm` 为 true 时，
`confirmation_zh` 说明回答前应确认什么；否则留空。工具调用外不要输出文字。
当输入指定 `output_mode=translation_only` 时，应用会生成信息保护型回答；
工具只要求 `translation_zh`，不要自行生成姓名、日期、号码、回复或确认内容。

以下示例只说明来电者与接听者的视角，不能作为当前电话的上下文：
- `今、お時間よろしいでしょうか。`：中文“请问现在方便吗？”；可选回答
  `はい、大丈夫です。`（可以，没问题。）或
  `申し訳ありません。後でもよろしいですか。`（不好意思，稍后可以吗？）
- `今は難しいです。`：中文“现在不方便。”；可选回答
  `分かりました。ご都合の良い時間を教えてください。`
  （明白了，请告诉我您方便的时间。）或
  `では、また後でお電話します。`（那我稍后再打给您。）
""".strip()


def create_strands_agent(
    app: Any,
    request: AgentRequest,
    result_sink: Any,
    source_text: str,
    reasoning: ReasoningConfig,
) -> Any:
    from strands import Agent, tool
    from strands.hooks import AfterToolsEvent
    from vifu.integrations.strands import (
        local_provider_model,
        openai_compatible_model,
        strands_model,
    )

    confirmation_required = _requires_confirmation(source_text)
    information_request = _caller_requests_listener_information(source_text)

    if information_request:

        @tool(name="card", description="翻译来电者向接听者提出的信息请求。")
        def publish_information_request(
            translation_zh: Annotated[
                str,
                "来电者请接听者提供什么信息；只用简体中文，不能包含日语。",
            ],
        ) -> dict[str, Any]:
            """Publish an information request without inventing the user's values."""
            return result_sink.publish_from_tool(
                source_text=source_text,
                translation_zh=translation_zh,
                suggested_replies=[
                    {
                        "japanese": "はい、確認してお伝えします。",
                        "chinese": "好的，我确认后告诉您。",
                    },
                    {
                        "japanese": "申し訳ありません。確認しますので、少々お待ちください。",
                        "chinese": "不好意思，我确认一下，请稍等。",
                    },
                ],
                confirmation_required=confirmation_required,
                confirmation_zh=(
                    "请先核对自己要向来电者提供的信息是否准确。"
                    if confirmation_required
                    else ""
                ),
            )

        output_tool = publish_information_request
    else:

        @tool(name="card", description="输出一张日语电话辅助卡。")
        def publish_call_assist(
            translation_zh: Annotated[
                str,
                "简洁的简体中文解释；不能复制日文或包含日语假名。",
            ],
            confirmation_zh: Annotated[
                str,
                "回答前要确认的内容，用简体中文；confirm 为 false 时留空。",
            ],
            replies: list[GeneratedSuggestedReply] = Field(
                min_length=2,
                max_length=2,
            ),
        ) -> dict[str, Any]:
            """Produce one validated visual call-assistance card."""
            validated_replies = [
                GeneratedSuggestedReply.model_validate(reply)
                for reply in replies
            ]
            return result_sink.publish_from_tool(
                source_text=source_text,
                translation_zh=translation_zh,
                suggested_replies=[
                    {
                        "japanese": validated.japanese.replace(
                            "申しありません", "申し訳ありません"
                        ),
                        "chinese": validated.chinese,
                    }
                    for validated in validated_replies
                ],
                confirmation_required=confirmation_required,
                confirmation_zh=confirmation_zh,
            )

        output_tool = publish_call_assist

    class EndTurnAfterPublish:
        """Finish after the output tool; a second model acknowledgement adds no value."""

        def register_hooks(self, registry: Any) -> None:
            registry.add_callback(AfterToolsEvent, self.end_turn)

        @staticmethod
        def end_turn(event: Any) -> None:
            event.end_turn = True

    reasoning.validate()
    if reasoning.provider_type == "vifu-profile":
        model = strands_model(
            app,
            request,
            profile=cast(str, reasoning.profile),
            temperature=0.0,
            max_tokens=MAX_REASONING_TOKENS,
        )
    elif reasoning.provider_type == "vifu-local":
        model = local_provider_model(
            request,
            provider=reasoning.provider,
            temperature=0.0,
            max_tokens=MAX_REASONING_TOKENS,
        )
    else:
        model = openai_compatible_model(
            request,
            url=cast(str, reasoning.url),
            model=cast(str, reasoning.model),
            api_key=reasoning.api_key,
            temperature=0.0,
            max_tokens=MAX_REASONING_TOKENS,
        )
    return Agent(
        model=model,
        tools=[output_tool],
        system_prompt=_system_prompt(request, reasoning),
        # The default Strands callback is a developer trace printer. The app
        # records traces and presents the validated Assist Card separately.
        callback_handler=None,
        hooks=[EndTurnAfterPublish()],
    )


def _system_prompt(request: AgentRequest, reasoning: ReasoningConfig) -> str:
    if reasoning.provider_type == "vifu-profile" and request.instructions:
        return request.instructions
    return SYSTEM_PROMPT


def analyze_transcript(
    agent: Any,
    transcript_context: list[dict[str, Any]],
    *,
    validation_feedback: list[str] | None = None,
    rejected_card: dict[str, Any] | None = None,
) -> None:
    source_text = str(transcript_context[-1].get("text") or "")
    confirmation_required = _requires_confirmation(source_text)
    question = _is_question(source_text)
    yes_no = _requests_yes_no_answer(source_text)
    information_request = _caller_requests_listener_information(source_text)
    prompt = {
        "latest_ja": source_text,
        "confirm": confirmation_required,
        "utterance_type": "question" if question else "statement",
        "reply_strategy": (
            "一项肯定回答；一项否定或要求澄清的回答；都使用接听者口吻"
            if yes_no
            else (
                "接听者表示会提供或先确认所需信息；第二项请求澄清或稍候"
                if information_request
                else (
                "直接回答；第二项给出不同的拒绝、另一种选择或询问缺失信息"
                if question
                else "回应来电者；第二项提出相关追问或另一种选择"
                )
            )
        ),
        "output_language": "所有 *_zh 和 replies.chinese 字段只能使用简体中文",
        "translation_rule": "只翻译 latest_ja，不得添加未提到的信息或回答建议",
    }
    if yes_no:
        prompt["answer_mode"] = "yes_no"
        prompt["reply_constraints"] = [
            "一个回答以「はい」开头",
            "另一个回答以「いいえ」开头，或明确请求对方重述需要确认的内容",
            "不能把对方的是非问题原样反问回去",
        ]
    if information_request:
        prompt["output_mode"] = "translation_only"
        prompt["speech_act"] = "caller_requests_information_from_listener"
        prompt["role_contract"] = (
            "来电者正在请接听者提供信息；回答必须由接听者说给来电者"
        )
        prompt["information_reply_constraints"] = [
            "第一个回答直接表示将提供或先确认所需信息",
            "不能要求来电者提供同一信息",
            "输入没有提供用户的真实值；禁止编造姓名、日期、金额、号码或示例值",
            "使用「お伝えします」「確認します」或「少々お待ちください」",
            "确认提示应提醒接听者核对自己将要提供的信息",
        ]
    previous = [
        str(item.get("text") or "")
        for item in transcript_context[-3:-1]
        if item.get("text")
    ]
    if previous:
        prompt["previous_ja"] = previous
    must_preserve = [
        {"ja": japanese, "zh": accepted_chinese[0]}
        for japanese, accepted_chinese in (
            *_WEEKDAY_TRANSLATIONS.items(),
            *_SOURCE_TERM_TRANSLATIONS.items(),
        )
        if japanese in source_text
    ]
    if must_preserve:
        prompt["facts"] = must_preserve
    stated_times = _JAPANESE_TIME.findall(source_text)
    if stated_times:
        prompt["exact_times"] = stated_times
    if validation_feedback:
        prompt["fix"] = validation_feedback
        prompt["rejected"] = rejected_card
    agent(json.dumps(prompt, ensure_ascii=False, separators=(",", ":")))


def validate_assist_card(source_text: str, card: Any) -> list[str]:
    """Return deterministic semantic errors that require a same-Provider retry."""
    errors: list[str] = []
    translation = card.translation_zh
    if not _is_chinese_text(translation):
        errors.append(
            "The Chinese translation must use Simplified Chinese and must not "
            "contain Japanese kana."
        )
    if _TRANSLATION_SEPARATOR.search(translation):
        errors.append("The Chinese translation must contain one translation only.")
    information_request = _caller_requests_listener_information(source_text)
    if information_request and any(
        phrase in translation
        for phrase in (
            "需要对方提供",
            "请对方提供",
            "让对方提供",
            "对方需要提供",
            "向对方询问",
        )
    ):
        errors.append(
            "The caller asks the listener to provide information; the Chinese "
            "translation must not reverse that direction."
        )
    for japanese, accepted_chinese in _WEEKDAY_TRANSLATIONS.items():
        if japanese in source_text and not any(
            candidate in translation for candidate in accepted_chinese
        ):
            errors.append(
                f"{japanese} must be translated as one of: "
                + ", ".join(accepted_chinese)
            )
    for japanese, accepted_chinese in _SOURCE_TERM_TRANSLATIONS.items():
        if japanese in source_text and not any(
            candidate in translation for candidate in accepted_chinese
        ):
            errors.append(
                f"The Chinese translation of {japanese} must include one of: "
                + ", ".join(accepted_chinese)
            )
    confirmation_expected = _requires_confirmation(source_text)
    if confirmation_expected and not card.confirmation_required:
        errors.append(
            "This utterance contains a request, commitment, date, quantity, or "
            "deadline, so confirmation_required must be true."
        )
    normalized_source = "".join(source_text.split())
    normalized_replies = [
        "".join(reply.japanese.split()) for reply in card.suggested_replies
    ]
    if len(normalized_replies) < 2:
        errors.append("The Assist Card must contain at least two response options.")
    if information_request and card.suggested_replies:
        first_reply = card.suggested_replies[0].japanese
        directly_responds = (
            bool(re.match(r"^はい(?:[\s、。,.！!]|$)", first_reply))
            and "教えて" not in first_reply
        ) or any(
            marker in first_reply
            for marker in ("お伝えします", "確認します", "少々お待ち")
        )
        if not directly_responds:
            errors.append(
                "The caller asks the listener for information. The first Japanese "
                "reply must use お伝えします or 確認します, or start with はい "
                "without inventing the requested value."
            )
        source_numbers = set(re.findall(r"[0-9０-９]+", source_text))
        reply_numbers = {
            number
            for reply in card.suggested_replies
            for number in re.findall(r"[0-9０-９]+", reply.japanese)
        }
        invented_numbers = reply_numbers - source_numbers
        if invented_numbers:
            errors.append(
                "The suggested replies must not invent requested values such as "
                "dates or numbers: " + ", ".join(sorted(invented_numbers))
            )
    if _requests_yes_no_answer(source_text):
        has_yes = any(
            re.match(r"^はい(?:[\s、。,.！!]|$)", reply.japanese)
            for reply in card.suggested_replies
        )
        has_no_or_clarification = any(
            re.match(r"^いいえ(?:[\s、。,.！!]|$)", reply.japanese)
            or any(
                marker in reply.japanese
                for marker in ("もう一度", "教えて", "分かりません")
            )
            for reply in card.suggested_replies
        )
        if not has_yes or not has_no_or_clarification:
            errors.append(
                "A yes/no request needs one reply starting with はい and another "
                "starting with いいえ or asking the caller to clarify."
            )
    if any(reply == normalized_source for reply in normalized_replies):
        errors.append(
            "A suggested reply must answer the speaker and must not repeat the "
            "source request."
        )
    if len(set(normalized_replies)) != len(normalized_replies):
        errors.append("Every suggested Japanese reply must be unique.")
    if any(
        not any(
            "\u3040" <= character <= "\u30ff" for character in reply.japanese
        )
        for reply in card.suggested_replies
    ):
        errors.append(
            "Every suggested_replies.japanese value must be natural Japanese "
            "and include hiragana or katakana."
        )
    if any(
        _CHINESE_WEEKDAY_IN_JAPANESE.search(reply.japanese)
        for reply in card.suggested_replies
    ):
        errors.append(
            "A Japanese reply contains Chinese weekday wording; use a natural "
            "Japanese weekday such as 金曜日."
        )
    for fragment, replacements in _UNNATURAL_REPLY_FRAGMENTS.items():
        if any(fragment in reply.japanese for reply in card.suggested_replies):
            errors.append(
                f"The Japanese reply contains the unnatural phrase {fragment}; "
                f"use {replacements}."
            )
    if any(
        not _is_chinese_text(reply.chinese) for reply in card.suggested_replies
    ):
        errors.append(
            "Every suggested reply Chinese meaning must use Simplified Chinese "
            "and must not contain Japanese kana."
        )
    if card.confirmation_required and not _is_chinese_text(
        card.confirmation_zh or ""
    ):
        errors.append(
            "The confirmation must use Simplified Chinese and must not contain "
            "Japanese kana."
        )
    if (
        information_request
        and card.confirmation_required
        and not any(
            marker in (card.confirmation_zh or "")
            for marker in ("自己", "要提供", "需提供", "将提供")
        )
    ):
        errors.append(
            "The caller asks the listener for information. The confirmation must "
            "tell the listener to verify what they will provide."
        )
    for reply in card.suggested_replies:
        for japanese, accepted_chinese in _WEEKDAY_TRANSLATIONS.items():
            if japanese in reply.japanese and not any(
                candidate in reply.chinese for candidate in accepted_chinese
            ):
                errors.append(
                    f"The Chinese reply meaning must translate {japanese} as one of: "
                    + ", ".join(accepted_chinese)
                )
        if "来週" in source_text and any(
            candidate in reply.chinese
            for candidate in _SOURCE_TERM_TRANSLATIONS["今週"]
        ):
            errors.append(
                "The Chinese reply meaning must not change 来週 to 本周, 这周, "
                "or another current-week expression."
            )
        if "今週" in source_text and any(
            candidate in reply.chinese
            for candidate in _SOURCE_TERM_TRANSLATIONS["来週"]
        ):
            errors.append(
                "The Chinese reply meaning must not change 今週 to 下周, 下星期, "
                "or another next-week expression."
            )
    return errors


def _requires_confirmation(source_text: str) -> bool:
    return any(
        marker in source_text
        for marker in (*_CONFIRMATION_MARKERS, *_WEEKDAY_TRANSLATIONS)
    ) or any(character.isdigit() for character in source_text)


def _is_question(source_text: str) -> bool:
    return bool(_QUESTION_ENDING.search(source_text.strip()))


def _requests_yes_no_answer(source_text: str) -> bool:
    return "はい" in source_text and any(
        negative in source_text for negative in ("いいえ", "いえ")
    )


def _caller_requests_listener_information(source_text: str) -> bool:
    return bool(_INFORMATION_REQUEST.search(source_text))


def _is_chinese_text(value: str) -> bool:
    return bool(_CJK_CHARACTER.search(value)) and not _JAPANESE_KANA.search(value)


class JapaneseReplyAgent:
    """Application endpoint that runs Strands with one model configuration."""

    vifu_name = "Japanese Reply Agent"
    vifu_capability = "japanese-call-replies"
    # One corrective retry stays bounded on a cold local model.
    vifu_timeout_ms = 180_000
    vifu_instructions = SYSTEM_PROMPT

    def __init__(self, reasoning: ReasoningConfig | None = None) -> None:
        self._app: Any | None = None
        self.debug = False
        self._reasoning = (
            reasoning if reasoning is not None else ReasoningConfig.from_env()
        )
        self.vifu_metadata = {
            "framework": "strands-agents",
            "role": "japanese-call-replies",
            **self._reasoning.public_metadata(),
        }

    def vifu_bind(self, app: Any, **_registration: Any) -> None:
        self._app = app

    def prepare(self) -> None:
        self._reasoning.validate()
        prepare = getattr(self._reasoning.provider, "prepare", None)
        if callable(prepare):
            with native_stderr(visible=self.debug):
                prepare()

    def close(self) -> None:
        close = getattr(self._reasoning.provider, "close", None)
        if callable(close):
            with native_stderr(visible=self.debug):
                close()

    def __call__(self, request: AgentRequest) -> dict[str, Any]:
        if self._app is None:
            raise RuntimeError(
                "Japanese Reply Agent is not attached to an application"
            )
        value = CallAssistInput.model_validate(_endpoint_input(request.input))
        transcript = [
            utterance.model_dump(by_alias=True, exclude_none=True)
            for utterance in value.transcript
        ]
        source_text = value.transcript[-1].text
        reasoning = self._reasoning
        validation_feedback: list[str] = []
        card = None
        rejected_card: dict[str, Any] | None = None
        for attempt in range(1, MAX_REASONING_ATTEMPTS + 1):
            result_sink = CallAssistResultSink()
            with native_stderr(visible=self.debug):
                agent = create_strands_agent(
                    self._app,
                    request,
                    result_sink,
                    source_text,
                    reasoning,
                )
            result_sink.begin_agent_turn()
            try:
                with request.trace.stage(
                    "decode",
                    metadata={
                        "framework": "strands-agents",
                        **reasoning.public_metadata(),
                        "transcriptionProvider": value.transcription_provider,
                        "attempt": attempt,
                    },
                ):
                    with native_stderr(visible=self.debug):
                        analyze_transcript(
                            agent,
                            transcript,
                            validation_feedback=validation_feedback,
                            rejected_card=rejected_card,
                        )
                card = result_sink.finish_agent_turn()
            except Exception as error:
                result_sink.abort_agent_turn()
                logger.error(
                    "Japanese reply reasoning failed (%s): %s",
                    type(error).__name__,
                    error,
                )
                raise
            validation_feedback = validate_assist_card(source_text, card)
            if not validation_feedback:
                break
            rejected_card = card.model_dump(by_alias=True, exclude_none=True)
            logger.debug(
                "Japanese reply output rejected on attempt %s: %s",
                attempt,
                "; ".join(validation_feedback),
            )
        if card is None:
            raise RuntimeError("Japanese reply reasoning did not produce an Assist Card")
        if validation_feedback:
            logger.warning(
                "Japanese reply reasoning remained uncertain after %s attempts: %s",
                MAX_REASONING_ATTEMPTS,
                "; ".join(validation_feedback),
            )
            return _uncertain_assist_card(
                source_text,
                confirmation_required=_requires_confirmation(source_text),
            )
        return card.model_dump(by_alias=True, exclude_none=True)


def _uncertain_assist_card(
    source_text: str,
    *,
    confirmation_required: bool,
) -> dict[str, Any]:
    """Return a safe clarification card when the local model output is unusable."""
    sink = CallAssistResultSink()
    sink.begin_agent_turn()
    sink.publish_from_tool(
        source_text=source_text,
        translation_zh="这段话有部分内容没有听清，请先让对方重述关键信息。",
        suggested_replies=[
            {
                "japanese": "すみません、もう一度ゆっくりお願いします。",
                "chinese": "不好意思，请再慢慢说一遍。",
            },
            {
                "japanese": "確認する内容を、もう一度教えていただけますか。",
                "chinese": "可以再告诉我需要确认的内容吗？",
            },
        ],
        confirmation_required=confirmation_required,
        confirmation_zh=(
            "部分内容没有听清，请勿直接确认，先让对方重述关键信息。"
            if confirmation_required
            else ""
        ),
    )
    return sink.finish_agent_turn().model_dump(by_alias=True, exclude_none=True)


def _endpoint_input(value: object) -> object:
    if not isinstance(value, dict):
        return value
    transcript = value.get("transcript")
    if transcript is not None:
        return value
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        return value
    latest = messages[-1]
    if not isinstance(latest, dict):
        return value
    content = latest.get("content")
    if isinstance(content, str):
        return {"source": "text", "transcript": content}
    return value
