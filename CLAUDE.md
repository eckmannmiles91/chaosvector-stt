# ChaosVector STT

Wyoming-protocol STT server with pluggable inference backends.

## Architecture
- `src/chaosvector_stt/backend.py` — Pluggable inference: faster-whisper, OpenVINO, ONNX
- `src/chaosvector_stt/quality.py` — Post-processing: silence gate, hallucination filter, name corrections
- `src/chaosvector_stt/wyoming_server.py` — Wyoming TCP server + event handler
- Audio format: 16kHz mono S16LE PCM (Wyoming standard)

## Deployment targets
- **microchaos3** (10.1.1.240) — Panther Lake 50 TOPS NPU, OpenVINO backend
- **microchaos2** (10.1.1.228) — Arrow Lake, CPU fallback (Moonshine/faster-whisper)
- **Pi-Fi speaker** (10.1.1.235) — Consumer, connects via Wyoming TCP

## SSH access
- microchaos2: `ssh root@10.1.1.228`
- microchaos3: `ssh root@10.1.1.240`
- Pi-Fi: `ssh chaos@10.1.1.235`
