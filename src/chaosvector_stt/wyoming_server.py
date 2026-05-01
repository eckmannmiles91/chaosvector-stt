"""Wyoming protocol STT server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
import time
from functools import partial

import numpy as np

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .backend import Backend, create_backend
from .quality import is_silent, post_process

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


class SttHandler(AsyncEventHandler):
    """Handle Wyoming STT events — receive audio, return transcript."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        backend: Backend,
        server_info: Info,
        infer_lock: asyncio.Lock,
        language: str,
    ) -> None:
        super().__init__(reader, writer)
        self.backend = backend
        self.server_info = server_info
        self._lock = infer_lock
        self._language = language
        self._audio_buf = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.server_info.event())
            return True

        if Transcribe.is_type(event.type):
            # Reset buffer for new transcription
            self._audio_buf.clear()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_buf.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            t0 = time.perf_counter()

            # Convert S16LE PCM to float32
            n_samples = len(self._audio_buf) // SAMPLE_WIDTH
            if n_samples == 0:
                await self.write_event(Transcript(text="").event())
                return True

            samples = struct.unpack(f"<{n_samples}h", bytes(self._audio_buf))
            audio = np.array(samples, dtype=np.float32) / 32768.0

            # Skip inference on silence
            if is_silent(audio):
                log.debug("Skipped inference (silence)")
                await self.write_event(Transcript(text="").event())
                self._audio_buf.clear()
                return True

            # Serialize inference (GPU/NPU not concurrent-safe)
            async with self._lock:
                text = await asyncio.to_thread(
                    self.backend.transcribe, audio, self._language
                )

            # Post-process: name corrections, hallucination filter
            text = post_process(text)

            elapsed = (time.perf_counter() - t0) * 1000
            audio_ms = n_samples / SAMPLE_RATE * 1000
            log.info(
                "STT: %.0fms audio → %.0fms inference → %r",
                audio_ms, elapsed, text[:80],
            )

            await self.write_event(Transcript(text=text).event())
            self._audio_buf.clear()
            return True

        return True


def build_info(model_name: str) -> Info:
    """Build Wyoming server info."""
    attr = Attribution(
        name="ChaosVector",
        url="https://github.com/eckmannmiles91/chaosvector-stt",
    )
    return Info(
        asr=[
            AsrProgram(
                name="chaosvector-stt",
                description="ChaosVector STT — pluggable speech-to-text",
                installed=True,
                version="0.1.0",
                attribution=attr,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        installed=True,
                        version="1.0",
                        attribution=attr,
                        languages=["en"],
                    )
                ],
            )
        ]
    )


async def run_server(
    uri: str,
    model_path: str,
    model_name: str,
    backend_name: str = "faster-whisper",
    device: str = "cpu",
    language: str = "en",
    compute_type: str = "int8",
) -> None:
    """Start the Wyoming STT server."""
    kwargs = {}
    if backend_name == "openvino":
        kwargs["device"] = device
    elif backend_name == "faster-whisper":
        kwargs["device"] = device
        kwargs["compute_type"] = compute_type

    backend = create_backend(backend_name, **kwargs)
    backend.load(model_path)
    info = build_info(model_name)
    infer_lock = asyncio.Lock()

    log.info(
        "Starting ChaosVector STT: uri=%s backend=%s device=%s model=%s",
        uri, backend_name, device, model_name,
    )

    server = AsyncServer.from_uri(uri)
    await server.run(
        partial(
            SttHandler,
            backend=backend,
            server_info=info,
            infer_lock=infer_lock,
            language=language,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaosVector STT — Wyoming server")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300", help="Listen URI")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--model-name", default="whisper", help="Model name for info")
    parser.add_argument("--backend", default="faster-whisper",
                        choices=["faster-whisper", "openvino", "onnx"])
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/NPU)")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper compute type")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(
        run_server(
            uri=args.uri,
            model_path=args.model,
            model_name=args.model_name,
            backend_name=args.backend,
            device=args.device,
            language=args.language,
            compute_type=args.compute_type,
        )
    )


if __name__ == "__main__":
    main()
