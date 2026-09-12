from __future__ import annotations

import threading
from typing import Any

from models import AssistCard


class CallAssistResultSink:
    """Collect exactly one validated card for an Agent invocation."""

    def __init__(self) -> None:
        self._sequence = 0
        self._lock = threading.Lock()
        self._turn_active = False
        self._turn_card: AssistCard | None = None

    def begin_agent_turn(self) -> None:
        with self._lock:
            if self._turn_active:
                raise RuntimeError("a Call Assist Agent turn is already active")
            self._turn_active = True
            self._turn_card = None

    def finish_agent_turn(self) -> AssistCard:
        with self._lock:
            card = self._turn_card
            self._turn_active = False
            self._turn_card = None
        if card is None:
            raise RuntimeError(
                "Call Assist must produce exactly one validated card per turn"
            )
        return card

    def abort_agent_turn(self) -> None:
        with self._lock:
            self._turn_active = False
            self._turn_card = None

    def publish_from_tool(
        self,
        *,
        source_text: str,
        translation_zh: str,
        suggested_replies: list[dict[str, Any]],
        confirmation_required: bool,
        confirmation_zh: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if not self._turn_active:
                raise RuntimeError(
                    "publish_call_assist is only available during an Agent turn"
                )
            if self._turn_card is not None:
                raise RuntimeError(
                    "publish_call_assist may be called only once per Agent turn"
                )
            sequence = self._sequence + 1
            card = AssistCard.model_validate(
                {
                    "sequence": sequence,
                    "sourceText": source_text,
                    "translationZh": translation_zh,
                    "suggestedReplies": suggested_replies,
                    "confirmationRequired": confirmation_required,
                    "confirmationZh": (
                        confirmation_zh if confirmation_required else None
                    ),
                }
            )
            self._sequence = sequence
            self._turn_card = card
        return {"accepted": True, "sequence": sequence}
