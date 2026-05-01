# Changes from Upstream

ChaosVector STT replaces [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper) and [wyoming-openvino-whisper](custom script) with a unified, pluggable STT server.

## Why we forked
- wyoming-faster-whisper is a single-backend wrapper with no post-processing
- Our custom wyoming-openvino-whisper script on microchaos3 works but isn't maintainable
- Neither has quality gates or name corrections — STT errors propagate to the intent classifier

## What we changed

### Post-STT Quality Gates
- **Upstream:** Raw transcription output, no filtering
- **Ours:** Three-layer defense:
  1. **Silence detection** — RMS energy check before inference. Skips model entirely on quiet audio, saving compute.
  2. **Hallucination blocklist** — Filters known Whisper hallucinations on silence: "thank you", "subscribe", "like and subscribe", "the end", "thanks for watching", etc.
  3. **Family name corrections** — Word-boundary regex fixes: Jenny→Jennie, Clicks→Plex, Kinsley→Kinzleigh, Zoe→Zoey, Lexie→Lexi. These corrections previously lived in the Pi-Fi satellite code — now baked into the STT server.

### Inference Serialization
- **Upstream:** No concurrency protection
- **Ours:** asyncio.Lock around all model inference. Prevents "Infer Request is busy" crashes on GPU/NPU when multiple clients connect simultaneously.

### Pluggable Backend Architecture
- **Upstream:** Single backend per project (faster-whisper OR openvino, not both)
- **Ours:** Factory pattern — FasterWhisperBackend, OpenVinoWhisperBackend, OnnxWhisperBackend. Switch with --backend flag. Same server, any model.

## Benchmarks (microchaos3, same test sentences)

| Model | Avg Latency | Accuracy |
|-------|------------|----------|
| distil-large-v3 (NPU, current) | 320ms | 5/8 exact |
| faster-whisper tiny.en (CPU) | 155ms | 5/8 exact |
| faster-whisper base.en (CPU) | 272ms | 5/8 exact |

All models get family names wrong — our post-STT corrections fix them.
