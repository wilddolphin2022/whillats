# Whillats

Real-time AI speech processing for WebRTC — Speech-to-Text, Language Model, and Text-to-Speech in a process-isolated architecture.

**Platforms:** macOS (Apple Silicon + Intel, Metal), Linux x86_64 (CPU / NVIDIA CUDA)

## What it does

- **Transcription** — Whisper STT with automatic language detection
- **LLM response** — Gemma-4 (or any GGUF model) via llama-server HTTP API
- **Speech synthesis** — Piper neural TTS with per-language voice routing
- **Multimodal** — Optional video frame input to LLM (YUV → JPEG → base64)
- **Multilingual** — EN / RU / ES / ZH / DE / FR voice + language detection

## Architecture

Three-process design:

```
directcall (WebRTC)  ←─pipes─→  whillats_server (Whisper+Piper)  ←─HTTP─→  llama-server (LLM)
Clang/libc++/-fno-exceptions    GCC/libstdc++                               llama.cpp
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design.

## Quick Start

```bash
# 1. Build whillats_server (Whisper + Piper)
cd src/modules/third_party/whillats
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHILLATS_PIPER=ON
cmake --build build -j$(nproc) --target whillats_server

# 2. Build llama-server
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && cmake -B build -DLLAMA_BUILD_SERVER=ON && cmake --build build -j$(nproc) --target llama-server

# 3. Start llama-server
./llama.cpp/build/bin/llama-server --model /path/to/model.gguf --port 8080

# 4. Run directcall
WHILLATS_SERVER=./build/bin/Release/whillats_server \
PIPER_MODEL=/path/to/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./build/bin/Release/espeak-ng-data \
./out/release/directcall --config config.json
```

In `config.json`: `"llama_model": "http://127.0.0.1:8080"` (URL, not a file path).

See [BUILD.md](BUILD.md) for full build and deployment instructions.

## Automated Deployment

Deploy to a clean Ubuntu server:

```bash
# From webrtcsays.ai repo root:
./deploy-talkingface5.sh root@your-server ~/.ssh/id_key
```

Deploys llama-server + whillats_server + directcall + Piper voices as systemd services.

## Dependencies (auto-fetched by CMake)

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Whisper STT
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — LLM server (separate build)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — Phonemizer for Piper
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Piper TTS inference

## Branch History

| Branch | Description |
|--------|-------------|
| `talkingface5` | llama-server HTTP backend, simplified LlamaDeviceBase |
| `talkingface4` | Embedded llama.cpp, Gemma-4 multimodal, multilingual TTS |
| `talkingface3` | StyleTTS2, Orpheus TTS |
| `talkingface2` | Initial Piper TTS integration |
