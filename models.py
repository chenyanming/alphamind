from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptUtterance(BaseModel):
    """One final utterance produced before the managed Agent Run starts."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sequence: int = Field(default=1, ge=1)
    speaker_id: str | None = Field(default=None, alias="speakerId", max_length=128)
    language: str = Field(default="ja-JP", min_length=1, max_length=16)
    text: str = Field(min_length=1, max_length=4_000)

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class CallAssistInput(BaseModel):
    """Text-only input accepted by the Call Assist Endpoint."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: Literal["livekit", "on_device_whisper", "text"] = "on_device_whisper"
    language: str = Field(default="ja-JP", min_length=1, max_length=16)
    transcription_provider: str = Field(
        default="local-whisper",
        alias="transcriptionProvider",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    transcript: list[TranscriptUtterance] = Field(min_length=1, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def normalize_single_transcript(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized: dict[str, Any] = dict(value)
        if (
            "transcriptionProvider" not in normalized
            and "transcription_provider" not in normalized
        ):
            normalized["transcriptionProvider"] = (
                "manual" if normalized.get("source") == "text" else "local-whisper"
            )
        transcript = normalized.get("transcript")
        if isinstance(transcript, str):
            normalized["transcript"] = [
                {
                    "sequence": 1,
                    "language": normalized.get("language", "ja-JP"),
                    "text": transcript,
                }
            ]
        return normalized

    @model_validator(mode="after")
    def transcript_sequences_are_ordered(self) -> CallAssistInput:
        sequences = [utterance.sequence for utterance in self.transcript]
        if any(
            current <= previous for previous, current in zip(sequences, sequences[1:])
        ):
            raise ValueError("transcript sequence values must be strictly increasing")
        return self


class SuggestedReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    japanese: str = Field(
        min_length=1,
        max_length=240,
        description="One short, natural Japanese reply to the speaker.",
    )
    chinese: str = Field(
        min_length=1,
        max_length=300,
        description="The reply's meaning in Simplified Chinese; do not use Japanese.",
    )

    @field_validator("japanese", "chinese", mode="before")
    @classmethod
    def strip_required_reply_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AssistCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["japanese-phone-call.assist-card.v1"] = Field(
        default="japanese-phone-call.assist-card.v1", alias="schema"
    )
    type: Literal["assist_card"] = "assist_card"
    sequence: int = Field(default=1, ge=1)
    source_text: str = Field(alias="sourceText", min_length=1, max_length=4_000)
    translation_zh: str = Field(alias="translationZh", min_length=1, max_length=4_000)
    suggested_replies: list[SuggestedReply] = Field(
        alias="suggestedReplies", min_length=1, max_length=3
    )
    confirmation_required: bool = Field(alias="confirmationRequired")
    confirmation_zh: str | None = Field(
        default=None, alias="confirmationZh", max_length=500
    )
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("source_text", "translation_zh", mode="before")
    @classmethod
    def strip_required_card_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("confirmation_zh", mode="before")
    @classmethod
    def strip_confirmation_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip() or None

    @model_validator(mode="after")
    def confirmation_matches_flag(self) -> AssistCard:
        if self.confirmation_required and not self.confirmation_zh:
            raise ValueError(
                "confirmationZh is required when confirmationRequired is true"
            )
        return self
