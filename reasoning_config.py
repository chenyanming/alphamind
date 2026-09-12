from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit


class ReasoningConfigurationError(RuntimeError):
    """The reply Agent has no single explicit reasoning Provider."""


@dataclass(frozen=True)
class ReasoningConfig:
    provider_type: str
    profile: str | None = None
    url: str | None = None
    model: str | None = None
    api_key: str | None = None
    provider: Any | None = field(default=None, repr=False, compare=False)

    @classmethod
    def unconfigured(cls) -> ReasoningConfig:
        return cls(provider_type="unconfigured")

    @classmethod
    def openai_compatible(
        cls,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
    ) -> ReasoningConfig:
        return cls(
            provider_type="openai-compatible",
            url=url,
            model=model,
            api_key=api_key,
        )

    @classmethod
    def vifu_profile(cls, profile: str) -> ReasoningConfig:
        return cls(provider_type="vifu-profile", profile=profile)

    @classmethod
    def local(cls, provider: Any) -> ReasoningConfig:
        return cls(provider_type="vifu-local", provider=provider)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ReasoningConfig:
        values = os.environ if env is None else env
        profile = values.get("VIFU_REASONING_PROFILE", "").strip()
        url = values.get("JAPANESE_CALL_REASONING_URL", "").strip()
        model = values.get("JAPANESE_CALL_REASONING_MODEL", "").strip()
        api_key = values.get("JAPANESE_CALL_REASONING_API_KEY", "").strip() or None
        direct_configured = bool(url or model or api_key)
        if profile and direct_configured:
            return cls(
                provider_type="ambiguous",
                profile=profile,
                url=url or None,
                model=model or None,
                api_key=api_key,
            )
        if profile:
            return cls.vifu_profile(profile)
        if direct_configured:
            return cls(
                provider_type="openai-compatible",
                url=url or None,
                model=model or None,
                api_key=api_key,
            )
        return cls.unconfigured()

    def validate(self) -> None:
        if self.provider_type == "ambiguous":
            raise ReasoningConfigurationError(
                "configure either VIFU_REASONING_PROFILE or the "
                "JAPANESE_CALL_REASONING_* Provider variables, not both"
            )
        if self.provider_type == "unconfigured":
            raise ReasoningConfigurationError(
                "set JAPANESE_CALL_REASONING_URL and "
                "JAPANESE_CALL_REASONING_MODEL, or "
                "explicitly select VIFU_REASONING_PROFILE"
            )
        if self.provider_type == "vifu-profile":
            if not self.profile:
                raise ReasoningConfigurationError(
                    "VIFU_REASONING_PROFILE must not be empty"
                )
            return
        if self.provider_type == "vifu-local":
            if not callable(getattr(self.provider, "complete", None)):
                raise ReasoningConfigurationError(
                    "the local Vifu reasoning Provider must define complete()"
                )
            return
        if self.provider_type != "openai-compatible":
            raise ReasoningConfigurationError(
                f"unsupported reasoning Provider type: {self.provider_type}"
            )
        if not self.url or not self.model:
            raise ReasoningConfigurationError(
                "JAPANESE_CALL_REASONING_URL and JAPANESE_CALL_REASONING_MODEL "
                "are both required"
            )
        target = urlsplit(self.url)
        loopback = target.hostname in {"127.0.0.1", "localhost", "::1"}
        if target.scheme not in ({"http", "https"} if loopback else {"https"}):
            raise ReasoningConfigurationError(
                "JAPANESE_CALL_REASONING_URL must use HTTPS or loopback HTTP"
            )
        if not loopback and not self.api_key:
            raise ReasoningConfigurationError(
                "JAPANESE_CALL_REASONING_API_KEY is required for a remote "
                "reasoning Provider"
            )

    def public_metadata(self) -> dict[str, str]:
        if self.provider_type == "vifu-profile":
            return {
                "providerType": self.provider_type,
                "profile": self.profile or "",
            }
        if self.provider_type == "vifu-local":
            return {
                "providerType": self.provider_type,
                "provider": str(
                    getattr(self.provider, "provider", type(self.provider).__name__)
                ),
                "model": str(getattr(self.provider, "model", "")),
            }
        metadata = {
            "providerType": self.provider_type,
            "model": self.model or "",
        }
        if self.provider_type == "ambiguous":
            metadata["profile"] = self.profile or ""
        return metadata
