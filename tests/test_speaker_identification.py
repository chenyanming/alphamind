from __future__ import annotations

import unittest
import wave
from io import BytesIO
from unittest.mock import patch

import numpy as np

from speaker_identification import LocalSpeakerIdentifier


def wav_bytes(*, seconds: float = 2.0) -> bytes:
    sample_rate = 16_000
    sample_count = int(sample_rate * seconds)
    samples = (np.sin(np.arange(sample_count) * 0.05) * 8_000).astype(np.int16)
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


class FakeStream:
    def accept_waveform(self, *, sample_rate: int, waveform: np.ndarray) -> None:
        self.sample_rate = sample_rate
        self.waveform = waveform

    def input_finished(self) -> None:
        return None


class FakeExtractor:
    dim = 3

    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = iter(embeddings)

    def create_stream(self) -> FakeStream:
        return FakeStream()

    def is_ready(self, _stream: FakeStream) -> bool:
        return True

    def compute(self, _stream: FakeStream) -> list[float]:
        return next(self._embeddings)


class LocalSpeakerIdentifierTests(unittest.TestCase):
    def test_enrolls_self_then_routes_multiple_other_speakers(self) -> None:
        extractor = FakeExtractor(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.08, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.99, 0.0],
                [0.70, 0.70, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        with patch(
            "speaker_identification._create_extractor",
            return_value=extractor,
        ):
            identifier = LocalSpeakerIdentifier(model="speaker.onnx")
            identifier.prepare(require_model_file=False)

            self.assertEqual(identifier.identify_wav(wav_bytes()), "self-enrollment")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "self")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "caller-1")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "caller-1")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "unknown")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "caller-2")

    def test_short_audio_cannot_enroll_or_be_routed(self) -> None:
        extractor = FakeExtractor([[1.0, 0.0, 0.0]])
        with patch(
            "speaker_identification._create_extractor",
            return_value=extractor,
        ):
            identifier = LocalSpeakerIdentifier(model="speaker.onnx")
            identifier.prepare(require_model_file=False)

            self.assertEqual(
                identifier.identify_wav(wav_bytes(seconds=0.4)),
                "unknown",
            )
            self.assertFalse(identifier.enrolled)

    def test_reset_requires_fresh_self_enrollment(self) -> None:
        extractor = FakeExtractor([[1.0, 0.0], [1.0, 0.0]])
        with patch(
            "speaker_identification._create_extractor",
            return_value=extractor,
        ):
            identifier = LocalSpeakerIdentifier(model="speaker.onnx")
            identifier.prepare(require_model_file=False)
            self.assertEqual(identifier.identify_wav(wav_bytes()), "self-enrollment")

            identifier.reset()

            self.assertFalse(identifier.enrolled)
            self.assertEqual(identifier.identify_wav(wav_bytes()), "self-enrollment")

    def test_known_speakers_use_relative_match_in_the_threshold_gap(self) -> None:
        extractor = FakeExtractor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.58, 0.52, 0.627375487],
                [0.52, 0.58, 0.627375487],
            ]
        )
        with patch(
            "speaker_identification._create_extractor",
            return_value=extractor,
        ):
            identifier = LocalSpeakerIdentifier(model="speaker.onnx")
            identifier.prepare(require_model_file=False)

            self.assertEqual(identifier.identify_wav(wav_bytes()), "self-enrollment")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "caller-1")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "self")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "caller-1")

    def test_first_uncertain_turn_does_not_poison_the_caller_profile(self) -> None:
        extractor = FakeExtractor(
            [
                [1.0, 0.0, 0.0],
                [0.54, 0.841665017, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        with patch(
            "speaker_identification._create_extractor",
            return_value=extractor,
        ):
            identifier = LocalSpeakerIdentifier(model="speaker.onnx")
            identifier.prepare(require_model_file=False)

            self.assertEqual(identifier.identify_wav(wav_bytes()), "self-enrollment")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "unknown")
            self.assertEqual(identifier.identify_wav(wav_bytes()), "caller-1")


if __name__ == "__main__":
    unittest.main()
