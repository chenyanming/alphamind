from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vifu import VifuRuntime
from voice_agent import JapaneseCallListenerAgent


class MultiAgentHandoffTests(unittest.TestCase):
    def test_real_runtime_records_listener_and_reply_agent_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = VifuRuntime("japanese-call-two-agent-test", data_dir=directory)
            voice = JapaneseCallListenerAgent(
                handoff="japanese-reply-agent",
                transcriber=SimpleNamespace(
                    provider="test-local-whisper",
                    model="test-whisper.bin",
                ),
            )
            runtime.agent(
                "japanese-call-listener",
                voice,
                capability="japanese-call-listening",
            )
            runtime.agent(
                "japanese-reply-agent",
                lambda request: {
                    "type": "assist_card",
                    "sourceText": request.input["transcript"][0]["text"],
                },
                capability="japanese-call-replies",
            )
            voice._app = runtime
            voice._endpoint = "japanese-call-listener"

            result = voice.dispatch_transcript(
                {
                    "type": "transcript",
                    "voiceSessionId": "voice-runtime-1",
                    "language": "ja-JP",
                    "text": "確認をお願いします。",
                    "isFinal": True,
                    "sequence": 1,
                }
            )
            traces = runtime.pending_traces()
            runtime.close()

        self.assertEqual(result["sourceText"], "確認をお願いします。")
        self.assertEqual(
            {trace["endpoint"] for trace in traces},
            {"japanese-call-listener", "japanese-reply-agent"},
        )

    def test_partial_turn_stops_at_the_voice_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = VifuRuntime("japanese-call-partial-turn-test", data_dir=directory)
            voice = JapaneseCallListenerAgent(
                handoff="japanese-reply-agent",
                transcriber=SimpleNamespace(
                    provider="test-local-whisper",
                    model="test-whisper.bin",
                ),
            )
            runtime.agent("japanese-call-listener", voice)
            runtime.agent(
                "japanese-reply-agent",
                lambda _request: self.fail("partial transcript reached Reply Agent"),
            )
            voice._app = runtime
            voice._endpoint = "japanese-call-listener"

            result = voice.dispatch_transcript(
                {
                    "type": "transcript",
                    "voiceSessionId": "voice-runtime-2",
                    "language": "ja-JP",
                    "text": "確認を",
                    "isFinal": False,
                    "sequence": 1,
                }
            )
            traces = runtime.pending_traces()
            runtime.close()

        self.assertIsNone(result)
        self.assertEqual(
            [trace["endpoint"] for trace in traces],
            ["japanese-call-listener"],
        )

    def test_non_speech_markers_stop_at_the_voice_agent(self) -> None:
        for sequence, text in enumerate(
            ("[音楽]", "(笑)", "[music]", "……"),
            start=1,
        ):
            with (
                self.subTest(text=text),
                tempfile.TemporaryDirectory() as directory,
            ):
                runtime = VifuRuntime("japanese-call-noise-test", data_dir=directory)
                specialist_calls: list[object] = []
                voice = JapaneseCallListenerAgent(
                    handoff="japanese-reply-agent",
                    transcriber=SimpleNamespace(
                        provider="test-local-whisper",
                        model="test-whisper.bin",
                    ),
                )
                runtime.agent("japanese-call-listener", voice)
                runtime.agent(
                    "japanese-reply-agent",
                    lambda request: specialist_calls.append(request) or {},
                )
                voice._app = runtime
                voice._endpoint = "japanese-call-listener"

                result = voice.dispatch_transcript(
                    {
                        "type": "transcript",
                        "voiceSessionId": f"voice-noise-{sequence}",
                        "language": "ja-JP",
                        "text": text,
                        "isFinal": True,
                        "sequence": sequence,
                    }
                )
                traces = runtime.pending_traces()
                runtime.close()

            self.assertIsNone(result)
            self.assertEqual(specialist_calls, [])
            self.assertEqual(
                [trace["endpoint"] for trace in traces],
                ["japanese-call-listener"],
            )

    def test_duplicate_final_turn_is_suppressed_only_during_debounce_window(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = VifuRuntime("japanese-call-duplicate-test", data_dir=directory)
            voice = JapaneseCallListenerAgent(
                handoff="japanese-reply-agent",
                transcriber=SimpleNamespace(
                    provider="test-local-whisper",
                    model="test-whisper.bin",
                ),
            )
            specialist_calls: list[str] = []

            def specialist(request: object) -> dict[str, str]:
                text = request.input["transcript"][0]["text"]
                specialist_calls.append(text)
                return {"type": "assist_card", "sourceText": text}

            runtime.agent("japanese-call-listener", voice)
            runtime.agent("japanese-reply-agent", specialist)
            voice._app = runtime
            voice._endpoint = "japanese-call-listener"

            with patch(
                "voice_agent.monotonic",
                side_effect=(100.0, 101.0, 101.5, 105.0),
                create=True,
            ):
                results = [
                    voice.dispatch_transcript(
                        {
                            "type": "transcript",
                            "voiceSessionId": "voice-duplicate-1",
                            "language": "ja-JP",
                            "text": "確認をお願いします。",
                            "isFinal": True,
                            "sequence": sequence,
                            "speakerId": speaker_id,
                        }
                    )
                    for sequence, speaker_id in (
                        (1, "caller"),
                        (2, "caller"),
                        (3, "agent"),
                        (4, "caller"),
                    )
                ]
            traces = runtime.pending_traces()
            runtime.close()

        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertIsNotNone(results[3])
        self.assertEqual(
            specialist_calls,
            [
                "確認をお願いします。",
                "確認をお願いします。",
                "確認をお願いします。",
            ],
        )
        self.assertEqual(
            [trace["endpoint"] for trace in traces],
            [
                "japanese-call-listener",
                "japanese-reply-agent",
                "japanese-call-listener",
                "japanese-call-listener",
                "japanese-reply-agent",
                "japanese-call-listener",
                "japanese-reply-agent",
            ],
        )


if __name__ == "__main__":
    unittest.main()
