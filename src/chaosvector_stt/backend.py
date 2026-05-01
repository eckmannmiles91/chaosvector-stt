"""Inference backends — faster-whisper, OpenVINO, Moonshine, ONNX."""

from __future__ import annotations

import abc
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Backend(abc.ABC):
    """Abstract STT inference backend."""

    @abc.abstractmethod
    def load(self, model_path: str | Path, **kwargs) -> None: ...

    @abc.abstractmethod
    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Transcribe float32 mono 16kHz audio to text."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


class FasterWhisperBackend(Backend):
    """CTranslate2-based Whisper backend (CPU/CUDA)."""

    def __init__(self, device: str = "cpu", compute_type: str = "int8") -> None:
        self._model = None
        self._device = device
        self._compute_type = compute_type

    @property
    def name(self) -> str:
        return "faster-whisper"

    def load(self, model_path: str | Path, **kwargs) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            str(model_path),
            device=self._device,
            compute_type=self._compute_type,
        )
        log.info("faster-whisper loaded: %s (device=%s, compute=%s)",
                 model_path, self._device, self._compute_type)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio, language=language, beam_size=1, vad_filter=True,
        )
        text = " ".join(s.text for s in segments).strip()
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug("faster-whisper: %.0fms → %r", elapsed, text[:80])
        return text


class OpenVinoWhisperBackend(Backend):
    """OpenVINO Whisper backend (CPU/GPU/NPU)."""

    def __init__(self, device: str = "NPU") -> None:
        self._pipeline = None
        self._device = device

    @property
    def name(self) -> str:
        return "openvino-whisper"

    def load(self, model_path: str | Path, **kwargs) -> None:
        import openvino_genai as ov_genai

        log.info("OpenVINO Whisper: compiling for device=%s ...", self._device)
        t0 = time.perf_counter()
        self._pipeline = ov_genai.WhisperPipeline(str(model_path), self._device)
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("OpenVINO Whisper compiled in %.0fms", elapsed)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        t0 = time.perf_counter()
        result = self._pipeline.generate(
            audio, language=f"<|{language}|>", max_new_tokens=128,
        )
        text = str(result).strip()
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug("openvino-whisper: %.0fms → %r", elapsed, text[:80])
        return text


class OnnxWhisperBackend(Backend):
    """ONNX Runtime Whisper backend (CPU/GPU via execution providers)."""

    def __init__(self, providers: list[str] | None = None) -> None:
        self._encoder = None
        self._decoder = None
        self._tokenizer = None
        self._providers = providers or ["CPUExecutionProvider"]

    @property
    def name(self) -> str:
        return "onnx-whisper"

    def load(self, model_path: str | Path, **kwargs) -> None:
        import onnxruntime as ort

        model_dir = Path(model_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4

        self._encoder = ort.InferenceSession(
            str(model_dir / "encoder_model.onnx"),
            sess_options=opts, providers=self._providers,
        )
        self._decoder = ort.InferenceSession(
            str(model_dir / "decoder_model.onnx"),
            sess_options=opts, providers=self._providers,
        )
        log.info("ONNX Whisper loaded from %s (providers=%s)",
                 model_path, self._providers)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        # Placeholder — full encoder-decoder loop requires tokenizer + generation
        raise NotImplementedError("ONNX Whisper transcription loop not yet implemented")


def create_backend(name: str = "faster-whisper", **kwargs) -> Backend:
    """Factory for STT inference backends."""
    if name == "openvino":
        return OpenVinoWhisperBackend(**kwargs)
    if name == "onnx":
        return OnnxWhisperBackend(**kwargs)
    return FasterWhisperBackend(**kwargs)
