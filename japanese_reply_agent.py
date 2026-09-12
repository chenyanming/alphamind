from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Annotated, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from models import CallAssistInput
from reasoning_config import ReasoningConfig
from result_sink import CallAssistResultSink
from vifu import AgentRequest

logger = logging.getLogger("alphamind-reasoning")

MAX_REASONING_ATTEMPTS = 3
MAX_REASONING_TOKENS = 500
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
    "配達",
    "お届け",
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
    "見積書": ("报价单", "报价书", "估价单", "估价书"),
}
_UNNATURAL_REPLY_FRAGMENTS = {
    "申しありません": "申し訳ありません",
    "対します": "対応します or 対応いたします",
}
_SINGLE_DIGIT_CHINESE = {
    "0": ("0", "零"),
    "1": ("1", "一"),
    "2": ("2", "二", "两"),
    "3": ("3", "三"),
    "4": ("4", "四"),
    "5": ("5", "五"),
    "6": ("6", "六"),
    "7": ("7", "七"),
    "8": ("8", "八"),
    "9": ("9", "九"),
}
_CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_JAPANESE_KANA = re.compile(r"[\u3040-\u30ff\uff66-\uff9f]")


class GeneratedSuggestedReply(BaseModel):
    """Unambiguous field names for the small model's output tool."""

    model_config = ConfigDict(extra="forbid")

    japanese_reply: str = Field(
        min_length=1,
        max_length=240,
        description=(
            "One short, natural Japanese response option from the listener's "
            "point of view. It must answer the caller's latest statement or "
            "request directly. Never repeat the caller's words as the answer. "
            "Use 申し訳ありません, never 申しありません."
        ),
    )
    simplified_chinese_meaning: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "The Japanese reply's meaning in Simplified Chinese. "
            "Never copy Japanese text into this field."
        ),
    )

SYSTEM_PROMPT = """
You help a foreign resident in Japan understand and answer a Japanese phone
call. The current user interface uses Simplified Chinese. You never produce
audio and never take an action on the user's behalf.

For each final transcript update:
1. Translate the latest Japanese utterance into concise Simplified Chinese.
2. Suggest two or three short, polite Japanese response options with their
   Chinese meanings.
3. Make every option directly relevant to the caller's latest utterance.
   Give distinct choices when useful, such as accept, request a change, or ask
   the caller to repeat or clarify. Do not add an unrelated generic response.
4. If the utterance includes a date, price, quantity, deadline, promise, or
   action item, set confirmation_required.
5. Explain exactly what the user must confirm before a reply.
   Use an empty confirmation_simplified_chinese string when no confirmation
   is required.
6. Preserve uncertainty. Do not invent names, numbers, commitments, or context.
7. Call publish_call_assist exactly once. Do not answer outside the tool call.

Accuracy rules:
- Japanese weekdays map exactly to Chinese: 月曜日=星期一, 火曜日=星期二,
  水曜日=星期三, 木曜日=星期四, 金曜日=星期五, 土曜日=星期六,
  日曜日=星期日.
- Every suggested response must answer the caller's latest utterance.
- Response options must be distinct. Do not rewrite the same answer three ways.
- Requests, deadlines, dates, prices, quantities, promises, and action items
  always require confirmation.

Example:
Source: 明日の午後2時に荷物をお届けしてもよろしいでしょうか。
Translation: 明天下午两点给您送货，可以吗？
Replies:
- はい、午後2時で大丈夫です。 / 可以，下午两点没问题。
- 申し訳ありません。別の時間に変更できますか。 / 不好意思，可以改到其他时间吗？
- すみません、もう一度ゆっくりお願いします。 / 不好意思，请再慢慢说一遍。
Confirmation required: true.
Confirmation: 请确认明天下午2点是否方便收货。

Source: おはようございます。
Translation: 早上好。
Replies:
- おはようございます。 / 早上好。
- お電話ありがとうございます。 / 谢谢您的来电。
Confirmation required: false.
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

    @tool
    def publish_call_assist(
        translation_simplified_chinese: Annotated[
            str,
            "A concise Simplified Chinese translation of the Japanese utterance. "
            "Never answer in English or Japanese.",
        ],
        confirmation_required: Annotated[
            bool,
            "True only when the user must confirm a date, price, quantity, "
            "deadline, promise, request, or action item.",
        ],
        confirmation_simplified_chinese: Annotated[
            str,
            "What the user must confirm, written only in Simplified Chinese; "
            "empty when confirmation_required is false. Preserve every date, "
            "time, and number from the caller. Chinese number words are valid, "
            "so 2, 二, and 两 can express the same stated value.",
        ],
        suggested_replies: list[GeneratedSuggestedReply] = Field(
            min_length=2,
            max_length=3,
        ),
    ) -> dict[str, Any]:
        """Produce one validated visual call-assistance card. Never produces audio."""
        validated_replies = [
            GeneratedSuggestedReply.model_validate(reply)
            for reply in suggested_replies
        ]
        return result_sink.publish_from_tool(
            source_text=source_text,
            translation_zh=translation_simplified_chinese,
            suggested_replies=[
                {
                    "japanese": validated.japanese_reply.replace(
                        "申しありません", "申し訳ありません"
                    ),
                    "chinese": validated.simplified_chinese_meaning,
                }
                for validated in validated_replies
            ],
            confirmation_required=confirmation_required,
            confirmation_zh=confirmation_simplified_chinese,
        )

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
        tools=[publish_call_assist],
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
    preserved_terms = [term for term in _WEEKDAY_TRANSLATIONS if term in source_text]
    confirmation_required = _requires_confirmation(source_text)
    if confirmation_required:
        reply_perspective = (
            "Give two or three distinct options from the listener's point of "
            "view. Include an option that accepts the request and an option "
            "that asks for a change or clarification when relevant."
        )
    else:
        reply_perspective = (
            "Give two or three distinct natural responses to the caller. Every "
            "option must address the latest utterance without invented context."
        )
    confirmation_terms = [
        accepted_chinese[0]
        for japanese, accepted_chinese in (
            *_WEEKDAY_TRANSLATIONS.items(),
            *_SOURCE_TERM_TRANSLATIONS.items(),
        )
        if japanese in source_text
    ]
    confirmation_numbers = [
        {
            "source": number,
            "accepted": list(_number_equivalents(number)),
        }
        for number in re.findall(r"[0-9０-９]+", source_text)
    ]
    constraints: dict[str, Any] = {
        "confirmationRequired": confirmation_required,
        "replyPerspective": reply_perspective,
        "replyMustNotCopySource": True,
        "replyCount": "2 or 3",
        "replyOptionsMustBeDistinctAndRelevant": True,
        "preserveJapaneseTermsInReply": preserved_terms,
    }
    prompt = {
        "task": "Publish assistance for the latest final utterance.",
        "transcript": transcript_context,
        "constraints": constraints,
    }
    if confirmation_required:
        if "お届け" in source_text or "配達" in source_text:
            constraints["scenario"] = (
                "The caller asks whether the caller can deliver an item. The "
                "user receives the delivery and does not deliver the item."
            )
            constraints["suggestedReplyShapes"] = [
                "はい、午後2時で大丈夫です。",
                "申し訳ありません。別の時間に変更できますか。",
                "すみません、もう一度ゆっくりお願いします。",
            ]
            constraints["forbiddenReplyContent"] = [
                "荷物をお届けします",
                "配達します",
            ]
        else:
            weekday = next(
                (term for term in _WEEKDAY_TRANSLATIONS if term in source_text),
                None,
            )
            if "送って" in source_text or "送付" in source_text:
                action = "お送りします。"
            else:
                action = "対応いたします。"
            deadline = f"{weekday}までに" if weekday is not None else ""
            chinese_weekday = (
                _WEEKDAY_TRANSLATIONS[weekday][1] if weekday is not None else ""
            )
            chinese_action = (
                "发送" if "送って" in source_text or "送付" in source_text else "处理"
            )
            constraints["suggestedReplyShape"] = (
                f"承知しました。{deadline}{action}"
            )
            constraints["suggestedReplyChineseShape"] = (
                f"明白了，我会在{chinese_weekday}前{chinese_action}。"
                if chinese_weekday
                else f"明白了，我会{chinese_action}。"
            )
        constraints["confirmationChineseMustInclude"] = confirmation_terms
        if confirmation_numbers:
            constraints["confirmationNumberEquivalents"] = confirmation_numbers
        if "お届け" in source_text or "配達" in source_text:
            constraints["confirmationGuidance"] = (
                "Ask the user to confirm whether the stated delivery time works. "
                "Repeat the exact date and time."
            )
        else:
            constraints["confirmationGuidance"] = (
                "Ask the user to confirm the stated deadline and commitment. Do not "
                "ask what an already-stated date, weekday, or action means."
            )
    if validation_feedback:
        prompt["validationFeedback"] = validation_feedback
        prompt["previousRejectedCard"] = rejected_card
        prompt["retryInstruction"] = (
            "The previous card was rejected. Correct every listed issue and call "
            "publish_call_assist exactly once."
        )
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
    for japanese in _WEEKDAY_TRANSLATIONS:
        if japanese in source_text and not any(
            japanese in reply.japanese for reply in card.suggested_replies
        ):
            errors.append(
                f"At least one Japanese reply must preserve the exact weekday {japanese}."
            )
    if any(reply == normalized_source for reply in normalized_replies):
        errors.append(
            "A suggested reply must answer the speaker and must not repeat the "
            "source request."
        )
    elif len(normalized_source) >= 16 and any(
        SequenceMatcher(None, normalized_source, reply).ratio() >= 0.7
        for reply in normalized_replies
    ):
        errors.append(
            "A suggested reply is too similar to the source request. It must "
            "respond from the user's point of view."
        )
    if len(set(normalized_replies)) != len(normalized_replies):
        errors.append("Every suggested Japanese reply must be unique.")
    if "お届け" in source_text or "配達" in source_text:
        reversed_delivery_phrases = ("荷物をお届けします", "配達します")
        if any(
            phrase in reply.japanese
            for reply in card.suggested_replies
            for phrase in reversed_delivery_phrases
        ):
            errors.append(
                "The caller delivers the item. A suggested reply must not say "
                "that the user will deliver it."
            )
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
    if card.confirmation_required:
        confirmation = card.confirmation_zh or ""
        for number in re.findall(r"[0-9０-９]+", source_text):
            if not any(
                equivalent in confirmation
                for equivalent in _number_equivalents(number)
            ):
                errors.append(
                    f"The confirmation must preserve the stated number {number}."
                )
        for japanese, accepted_chinese in (
            *_WEEKDAY_TRANSLATIONS.items(),
            *_SOURCE_TERM_TRANSLATIONS.items(),
        ):
            if japanese in source_text and not any(
                candidate in confirmation for candidate in accepted_chinese
            ):
                errors.append(
                    f"The confirmation must preserve {japanese} as one of: "
                    + ", ".join(accepted_chinese)
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


def _number_equivalents(value: str) -> tuple[str, ...]:
    normalized = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return _SINGLE_DIGIT_CHINESE.get(normalized, (normalized,))


def _is_chinese_text(value: str) -> bool:
    return bool(_CJK_CHARACTER.search(value)) and not _JAPANESE_KANA.search(value)


class JapaneseReplyAgent:
    """Application endpoint that runs Strands with one model configuration."""

    vifu_name = "Japanese Reply Agent"
    vifu_capability = "japanese-call-replies"
    # Three validated local-model attempts can exceed one minute on a cold CPU.
    vifu_timeout_ms = 180_000
    vifu_instructions = SYSTEM_PROMPT

    def __init__(self, reasoning: ReasoningConfig | None = None) -> None:
        self._app: Any | None = None
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
            prepare()

    def close(self) -> None:
        close = getattr(self._reasoning.provider, "close", None)
        if callable(close):
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
        if card is None or validation_feedback:
            message = (
                "Japanese reply reasoning could not produce a safe Assist Card: "
                + "; ".join(validation_feedback)
            )
            logger.error(message)
            raise RuntimeError(message)
        return card.model_dump(by_alias=True, exclude_none=True)


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
