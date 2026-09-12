from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import main
from reasoning_config import ReasoningConfigurationError
from voice_agent import JapaneseCallListenerAgent, VoiceConfigurationError


class LocalAppTests(unittest.TestCase):
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
        self.assertEqual(main.reply_agent._reasoning.provider_type, "vifu-local")
        self.assertEqual(
            main.reply_agent._reasoning.provider.gpu_layers,
            (
                36
                if sys.platform == "darwin"
                and getattr(
                    main.LocalLlama,
                    "supports_safe_accelerator_shutdown",
                    False,
                )
                else 0
            ),
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

    def test_main_delegates_the_complete_lifecycle_to_vifu(self) -> None:
        with patch.object(main.app, "run") as run:
            main.main([])

        run.assert_called_once_with()

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


if __name__ == "__main__":
    unittest.main()
