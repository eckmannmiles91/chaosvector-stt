# ChaosVector STT

Pluggable speech-to-text server with [Wyoming protocol](https://github.com/rhasspy/wyoming) support. Drop-in replacement for wyoming-faster-whisper, wyoming-openvino-whisper, and other Wyoming STT servers.

## Features

- **Pluggable backends**: faster-whisper (CPU/CUDA), OpenVINO (CPU/GPU/NPU), ONNX Runtime
- **Wyoming protocol**: Drop-in replacement for any Wyoming-compatible STT server
- **Built-in quality gates**: Silence detection, hallucination filtering, name corrections
- **Post-STT corrections**: Family name regex fixes baked in (no external config needed)

## Quick Start

```bash
# Install with faster-whisper backend
pip install ".[faster-whisper]"

# Run
chaosvector-stt \
  --model base.en \
  --backend faster-whisper \
  --device cpu \
  --uri tcp://0.0.0.0:10300
```

### OpenVINO backend (Intel NPU)

```bash
pip install ".[openvino]"

chaosvector-stt \
  --model /path/to/distil-whisper-large-v3-int8 \
  --backend openvino \
  --device NPU \
  --uri tcp://0.0.0.0:10300
```

## License

Apache-2.0
