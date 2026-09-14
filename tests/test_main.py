from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import main
from reasoning_config import ReasoningConfigurationError
from voice_agent import JapaneseCallListenerAgent, VoiceConfigurationError


class LocalAppTests(unittest.TestCase):
    def setUp(self) -> None:
        key = patch.dict(
            os.environ,
            {main.OPENAI_API_KEY_ENV: "test-provider-key"},
        )
        key.start()
        self.addCleanup(key.stop)

    def test_main_registers_exactly_two_composable_agents(self) -> None:
        registrations = main.app._registrations

        self.assertEqual(
            [agent_id for agent_id, _, _ in registrations],
            ["japanese-call-listener", "japanese-reply-agent"],
        )
        self.assertIsInstance(registrations[0][1], JapaneseCallListenerAgent)
        self.assertIs(registrations[1][1], main.reply_agent)
        self.assertEqual(
            [options["capability"] for _, _, options in registrations],
            ["japanese-call-listening", "japanese-call-replies"],
        )
        self.assertEqual(registrations[0][1].handoff, "japanese-reply-agent")
        self.assertEqual(
            registrations[0][1].transcriber.provider,
            "vifu-local-whisper",
        )
        self.assertEqual(main.reply_agent._reasoning.provider_type, "app-provider")
        self.assertEqual(
            main.reply_agent._reasoning.provider.model,
            main.OPENAI_MODEL,
        )
        self.assertEqual(
            main.reply_agent._reasoning.provider.url,
            main.OPENAI_CHAT_COMPLETIONS_URL,
        )
        self.assertEqual(main.reply_agent._reasoning.provider.timeout, 30.0)
        self.assertEqual(main.reply_agent.vifu_timeout_ms, 180_000)
        self.assertGreater(
            registrations[0][1].vifu_handoff_timeout_seconds,
            main.reply_agent.vifu_timeout_ms / 1_000,
        )
        self.assertEqual(
            registrations[0][1].transcriber.model,
            main.LOCAL_WHISPER_MODEL,
        )
        self.assertEqual(main.LOCAL_WHISPER_MODEL, "ggml-small.bin")
        self.assertEqual(
            registrations[0][2]["metadata"]["role"],
            "japanese-call-listening",
        )
        self.assertEqual(
            registrations[0][2]["metadata"]["handoffAgent"],
            "japanese-reply-agent",
        )
        self.assertEqual(
            registrations[1][2]["metadata"]["role"],
            "japanese-call-replies",
        )
        self.assertEqual(
            registrations[0][2]["metadata"]["implementation"],
            "livekit-agents",
        )
        self.assertEqual(
            registrations[1][2]["metadata"]["implementation"],
            "strands-agents",
        )
        self.assertEqual(
            registrations[0][2]["metadata"]["providerBindings"],
            {
                "transcription": {
                    "providerKey": "local-whisper",
                    "capability": "transcription",
                }
            },
        )
        self.assertEqual(
            registrations[1][2]["metadata"]["providerBindings"],
            {
                "reasoning": {
                    "providerKey": "openai-compatible",
                    "capability": "chat",
                }
            },
        )
        self.assertEqual(
            set(main.app.providers),
            {"local-whisper", "openai-compatible"},
        )
        self.assertEqual(
            main.app.providers["openai-compatible"].descriptor()["settings"],
            {
                "url": main.OPENAI_CHAT_COMPLETIONS_URL,
                "model": main.OPENAI_MODEL,
            },
        )

    def test_main_delegates_the_complete_lifecycle_to_vifu(self) -> None:
        with patch.object(main.app, "run") as run:
            main.main([])

        run.assert_called_once_with()

    def test_debug_mode_controls_native_logs_for_both_agents(self) -> None:
        def assert_debug_enabled() -> None:
            self.assertTrue(main.voice_agent.debug)
            self.assertTrue(main.reply_agent.debug)

        with patch.object(main.app, "run", side_effect=assert_debug_enabled):
            main.main(["--debug"])

        self.assertFalse(main.voice_agent.debug)
        self.assertFalse(main.reply_agent.debug)

    def test_demo_runs_one_two_agent_handoff_and_closes_the_app(self) -> None:
        card = {
            "sourceText": main.DEMO_TRANSCRIPT,
            "translationZh": "请在下周五之前发送报价单。",
            "suggestedReplies": [
                {
                    "japanese": "承知しました。",
                    "chinese": "明白了。",
                },
                {
                    "japanese": "もう一度お願いします。",
                    "chinese": "请再说一遍。",
                },
            ],
            "confirmationRequired": True,
            "confirmationZh": "请确认截止日期。",
        }
        with (
            patch.object(main.app, "connect") as connect,
            patch.object(
                main.voice_agent,
                "dispatch_transcript",
                return_value=card,
            ) as dispatch,
            patch.object(main, "print_transcript_status") as print_status,
            patch.object(main, "print_assist_card") as print_card,
            patch.object(main.app, "close") as close,
        ):
            main.main(["demo"])

        connect.assert_called_once_with()
        event = dispatch.call_args.args[0]
        self.assertEqual(event["text"], main.DEMO_TRANSCRIPT)
        self.assertTrue(event["isFinal"])
        print_status.assert_called_once_with({"text": main.DEMO_TRANSCRIPT})
        print_card.assert_called_once_with(card)
        close.assert_called_once_with()

    def test_demo_closes_the_app_when_the_handoff_fails(self) -> None:
        with (
            patch.object(main.app, "connect"),
            patch.object(
                main.voice_agent,
                "dispatch_transcript",
                side_effect=RuntimeError("handoff failed"),
            ),
            patch.object(main, "print_transcript_status"),
            patch.object(main.app, "close") as close,
            self.assertRaisesRegex(RuntimeError, "handoff failed"),
        ):
            main.main(["demo"])

        close.assert_called_once_with()

    def test_main_reports_provider_configuration_without_a_traceback(self) -> None:
        for error in (
            VoiceConfigurationError("choose an STT Provider"),
            ReasoningConfigurationError("choose a reasoning Provider"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(main.app, "run", side_effect=error),
                self.assertRaisesRegex(SystemExit, "Configuration error"),
            ):
                main.main([])

    def test_main_requires_the_default_openai_key_before_starting_audio(self) -> None:
        with (
            patch.dict(os.environ),
            patch.object(main, "OPENAI_BASE_URL", main.DEFAULT_OPENAI_BASE_URL),
            patch.object(main.app, "run") as run,
        ):
            os.environ.pop(main.OPENAI_API_KEY_ENV, None)
            with self.assertRaisesRegex(
                SystemExit,
                main.OPENAI_API_KEY_ENV,
            ):
                main.main([])

        run.assert_not_called()

    def test_main_allows_a_compatible_provider_without_authentication(self) -> None:
        with (
            patch.dict(os.environ),
            patch.object(main, "OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
            patch.object(main.app, "run") as run,
        ):
            os.environ.pop(main.OPENAI_API_KEY_ENV, None)
            main.main([])

        run.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
