from __future__ import annotations

import unittest

from reasoning_config import ReasoningConfig, ReasoningConfigurationError


class ReasoningConfigTests(unittest.TestCase):
    def test_provider_must_be_configured_explicitly(self) -> None:
        config = ReasoningConfig.from_env({})

        with self.assertRaisesRegex(
            ReasoningConfigurationError,
            "JAPANESE_CALL_REASONING_URL",
        ):
            config.validate()

    def test_generic_llm_aliases_are_not_silent_fallbacks(self) -> None:
        config = ReasoningConfig.from_env(
            {
                "LLM_URL": "http://127.0.0.1:11434/v1/chat/completions",
                "LLM_MODEL": "unexpected-fallback",
            }
        )

        with self.assertRaises(ReasoningConfigurationError):
            config.validate()

    def test_direct_provider_requires_its_own_url_and_model(self) -> None:
        config = ReasoningConfig.from_env(
            {
                "JAPANESE_CALL_REASONING_URL": (
                    "http://127.0.0.1:11434/v1/chat/completions"
                ),
                "JAPANESE_CALL_REASONING_MODEL": "qwen2.5:7b",
            }
        )

        config.validate()
        self.assertEqual(config.provider_type, "openai-compatible")

    def test_remote_direct_provider_requires_its_own_api_key(self) -> None:
        config = ReasoningConfig.from_env(
            {
                "JAPANESE_CALL_REASONING_URL": (
                    "https://provider.example/v1/chat/completions"
                ),
                "JAPANESE_CALL_REASONING_MODEL": "provider/model",
            }
        )

        with self.assertRaisesRegex(
            ReasoningConfigurationError,
            "JAPANESE_CALL_REASONING_API_KEY",
        ):
            config.validate()

    def test_vifu_profile_and_direct_provider_are_mutually_exclusive(self) -> None:
        config = ReasoningConfig.from_env(
            {
                "VIFU_REASONING_PROFILE": "call-assist-reasoning",
                "JAPANESE_CALL_REASONING_URL": (
                    "http://127.0.0.1:11434/v1/chat/completions"
                ),
                "JAPANESE_CALL_REASONING_MODEL": "qwen2.5:7b",
            }
        )

        with self.assertRaisesRegex(ReasoningConfigurationError, "both"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
