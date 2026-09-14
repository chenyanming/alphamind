"""On-device speaker identification for the LiveKit Voice Agent."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np


DEFAULT_SPEAKER_MODEL = (
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
DEFAULT_MATCH_THRESHOLD = 0.60
DEFAULT_NEW_SPEAKER_THRESHOLD = 0.50
DEFAULT_AMBIGUITY_MARGIN = 0.04


class SpeakerIdentificationError(RuntimeError):
    """The local speaker identifier cannot start or classify an utterance."""


@dataclass
class _VoiceProfile:
    embedding: np.ndarray
    samples: int = 1

    def update(self, value: np.ndarray) -> None:
        combined = self.embedding * self.samples + value
        self.samples += 1
        self.embedding = _normalized(combined / self.samples)


class LocalSpeakerIdentifier:
    """Enroll the listener and identify speakers in LiveKit speech turns.

    LiveKit's VAD supplies one completed speech turn at a time. The first long
    enough turn enrolls the nearby listener. Later turns are compared with the
    listener and with stable caller profiles. Ambiguous and very short turns
    stay unknown and never need to be guessed by the Strands Agent.
    """

    provider = "sherpa-onnx-speaker-identification"

    def __init__(
        self,
        *,
        model: str | Path = DEFAULT_SPEAKER_MODEL,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        new_speaker_threshold: float = DEFAULT_NEW_SPEAKER_THRESHOLD,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        enrollment_seconds: float = 1.5,
        identification_seconds: float = 0.6,
        max_callers: int = 4,
        num_threads: int = 2,
    ) -> None:
        if not 0 < new_speaker_threshold < match_threshold <= 1:
            raise ValueError(
                "speaker thresholds must satisfy 0 < new < match <= 1"
            )
        if not 0 <= ambiguity_margin < 1:
            raise ValueError("speaker ambiguity_margin must be between 0 and 1")
        if enrollment_seconds <= 0 or identification_seconds <= 0:
            raise ValueError("speaker audio durations must be positive")
        if max_callers < 1:
            raise ValueError("speaker max_callers must be positive")
        if num_threads < 1:
            raise ValueError("speaker num_threads must be positive")
        self.model = str(model)
        self.match_threshold = match_threshold
        self.new_speaker_threshold = new_speaker_threshold
        self.ambiguity_margin = ambiguity_margin
        self.enrollment_seconds = enrollment_seconds
        self.identification_seconds = identification_seconds
        self.max_callers = max_callers
        self.num_threads = num_threads
        self._extractor: Any | None = None
        self._self: _VoiceProfile | None = None
        self._callers: list[_VoiceProfile] = []
        self._lock = RLock()

    @property
    def enrolled(self) -> bool:
        with self._lock:
            return self._self is not None

    def prepare(self, *, require_model_file: bool = True) -> None:
        with self._lock:
            if self._extractor is not None:
                return
            model_path = _model_path(self.model)
            if require_model_file and not model_path.is_file():
                raise SpeakerIdentificationError(
                    "speaker model was not found at "
                    f"{model_path}. Download {DEFAULT_SPEAKER_MODEL} into "
                    "~/.vifu/models/."
                )
            try:
                self._extractor = _create_extractor(
                    model_path,
                    num_threads=self.num_threads,
                )
            except (ImportError, RuntimeError, ValueError) as error:
                raise SpeakerIdentificationError(
                    f"local speaker model could not start: {error}"
                ) from error

    def reset(self) -> None:
        """Forget session voices while keeping the resident model loaded."""
        with self._lock:
            self._self = None
            self._callers.clear()

    def close(self) -> None:
        with self._lock:
            self.reset()
            self._extractor = None

    def identify_wav(self, wav: bytes) -> str:
        """Return self-enrollment, self, caller-N, or unknown."""
        samples, sample_rate = _wav_samples(wav)
        duration = len(samples) / sample_rate
        with self._lock:
            self.prepare()
            if self._self is None and duration < self.enrollment_seconds:
                return "unknown"
            if self._self is not None and duration < self.identification_seconds:
                return "unknown"
            embedding = self._embedding(samples, sample_rate)
            if embedding is None:
                return "unknown"
            if self._self is None:
                self._self = _VoiceProfile(embedding)
                return "self-enrollment"
            return self._match(embedding)

    def _embedding(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray | None:
        assert self._extractor is not None
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        is_ready = getattr(self._extractor, "is_ready", None)
        if callable(is_ready) and not is_ready(stream):
            return None
        try:
            return _normalized(self._extractor.compute(stream))
        except (RuntimeError, ValueError) as error:
            raise SpeakerIdentificationError(
                f"local speaker embedding failed: {error}"
            ) from error

    def _match(self, embedding: np.ndarray) -> str:
        assert self._self is not None
        self_score = _similarity(self._self.embedding, embedding)
        caller_index = -1
        caller_score = -1.0
        for index, caller in enumerate(self._callers):
            score = _similarity(caller.embedding, embedding)
            if score > caller_score:
                caller_index = index
                caller_score = score

        if (
            self_score >= self.match_threshold
            and self_score >= caller_score + self.ambiguity_margin
        ):
            self._self.update(embedding)
            return "self"
        if (
            caller_index >= 0
            and caller_score >= self.match_threshold
            and caller_score >= self_score + self.ambiguity_margin
        ):
            self._callers[caller_index].update(embedding)
            return f"caller-{caller_index + 1}"

        nearest_score = max(self_score, caller_score)
        if (
            nearest_score <= self.new_speaker_threshold
            and len(self._callers) < self.max_callers
        ):
            self._callers.append(_VoiceProfile(embedding))
            return f"caller-{len(self._callers)}"
        if caller_index >= 0:
            if self_score >= caller_score + self.ambiguity_margin:
                return "self"
            if caller_score >= self_score + self.ambiguity_margin:
                return f"caller-{caller_index + 1}"
        return "unknown"


def _create_extractor(model_path: Path, *, num_threads: int) -> Any:
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=num_threads,
        debug=False,
        provider="cpu",
    )
    if not config.validate():
        raise ValueError(f"invalid speaker model configuration for {model_path}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def _model_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        return Path.home() / ".vifu" / "models" / candidate
    return candidate


def _wav_samples(value: bytes) -> tuple[np.ndarray, int]:
    if not isinstance(value, bytes) or not value:
        raise ValueError("speaker WAV audio must not be empty")
    try:
        with wave.open(BytesIO(value), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValueError("speaker audio must be a valid WAV file") from error
    if sample_width != 2:
        raise ValueError("speaker audio must contain 16-bit PCM samples")
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("speaker audio has an invalid format")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples = np.ascontiguousarray(samples / 32_768.0, dtype=np.float32)
    return samples, sample_rate


def _normalized(value: Any) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if not embedding.size or not math.isfinite(norm) or norm <= 0:
        raise ValueError("speaker model returned an invalid embedding")
    return embedding / norm


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


__all__ = [
    "DEFAULT_SPEAKER_MODEL",
    "LocalSpeakerIdentifier",
    "SpeakerIdentificationError",
]
