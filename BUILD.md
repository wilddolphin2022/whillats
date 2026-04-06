# Whillats Build & Deployment Guide

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design details.

---

## Local Development Build

### Prerequisites

- CMake 3.16+
- GCC 11+ or Clang 14+ (C++17)
- `sudo apt install build-essential cmake ninja-build`
- (Optional) CUDA Toolkit 12.x for GPU acceleration

### Build whillats_server (CPU, Piper TTS)

```bash
cd src/modules/third_party/whillats

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DWHILLATS_PIPER=ON

cmake --build build -j$(nproc) --target whillats_server
```

### Build whillats_server (GPU / CUDA)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DWHILLATS_PIPER=ON \
  -DGGML_CUDA=ON

cmake --build build -j$(nproc) --target whillats_server
```

### StyleTTS2 (alternative to Piper)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DWHILLATS_STYLETTS2=ON

cmake --build build -j$(nproc) --target whillats_server
```

`WHILLATS_PIPER` and `WHILLATS_STYLETTS2` are mutually exclusive.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `WHILLATS_PIPER` | OFF | Piper neural TTS (fast CPU, ONNX) |
| `WHILLATS_STYLETTS2` | ON | StyleTTS2 neural TTS (ONNX) |
| `WHILLATS_OLD_ABI` | OFF | `_GLIBCXX_USE_CXX11_ABI=0` for old ABI compat |
| `GGML_CUDA` | OFF | CUDA GPU backend for Whisper |

### Build Outputs

```
build/bin/Release/whillats_server      # AI server (Whisper + Piper)
build/bin/Release/espeak-ng-data/      # espeak runtime data
build/bin/Release/test_whillats        # Pipeline test
build/bin/Release/test_whillats_server # IPC protocol test
```

---

## Build llama-server (llama.cpp)

As of talkingface5, the LLM is handled by llama-server separately. whillats_server communicates with it over HTTP and does **not** link llama.cpp.

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build -j$(nproc) --target llama-server
```

---

## Build directcall + libdirect.so (WebRTC GN)

The thin whillats client is compiled in-tree by WebRTC's GN build system — no separate libwhillats.so needed.

```bash
cd src   # webrtcsays.ai/src

# Generate build files
gn gen out/release --args='
  target_os="linux"
  is_debug=false
  is_clang=true
  use_sysroot=false
  treat_warnings_as_errors=false
  rtc_include_opus=true
  rtc_include_tests=false
  rtc_build_examples=true
  rtc_build_sdk=false
  rtc_enable_symbol_export=true
  rtc_use_speech_audio_devices=true
  use_custom_libcxx=true
  enable_js_protobuf=false
  rtc_enable_protobuf=false
  enable_libaom=false
'

ninja -C out/release libdirect.so directcall
```

---

## Models

| Model | Size | Source |
|-------|------|--------|
| Whisper small | 466MB | `https://huggingface.co/ggerganov/whisper.cpp` |
| Gemma-4 E4B Q2_K | ~4.5GB | `bartowski/google_gemma-4-E4B-it-GGUF` (gated, HF token) |
| Gemma-4 mmproj BF16 | ~950MB | `ggml-org/gemma-4-E4B-it-GGUF` (gated) |
| Piper EN (lessac low) | 15MB | `rhasspy/piper-voices` |
| Piper RU (ruslan med) | 60MB | `rhasspy/piper-voices` |
| Piper ES (carlfm) | 10MB | `rhasspy/piper-voices` |
| Piper ZH (huayan med) | 60MB | `rhasspy/piper-voices` |

Place in `/opt/models/` (deployment) or `~/models/` (local dev).

---

## Running Locally

### Start llama-server

```bash
./llama.cpp/build/bin/llama-server \
  --model /opt/models/google_gemma-4-E4B-it-Q2_K.gguf \
  --mmproj /opt/models/mmproj-BF16.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --ctx-size 4096 \
  --n-predict 512 \
  --threads 6
```

Wait for `model loaded` and `server is listening on http://127.0.0.1:8080`.

### Start directcall

```bash
cd src

WHILLATS_SERVER=./modules/third_party/whillats/build/bin/Release/whillats_server \
PIPER_MODEL=~/models/piper/en_US-lessac-low.onnx \
PIPER_MODEL_RU=~/models/piper/ru_RU-ruslan-medium.onnx \
PIPER_MODEL_ES=~/models/piper/es_ES-carlfm-x_low.onnx \
PIPER_MODEL_ZH=~/models/piper/zh_CN-huayan-medium.onnx \
ESPEAK_DATA_PATH=./modules/third_party/whillats/build/bin/Release/espeak-ng-data \
./out/release/directcall --config config.talkingface5.json
```

### config.talkingface5.json

```json
{
  "mode": "callee",
  "user_name": "Slim",
  "room_name": "room101",
  "websocket_signaling": true,
  "websocket_port": 3459,
  "whisper": true,
  "llama": true,
  "language": "auto",
  "whisper_model": "/opt/models/ggml-small.bin",
  "llama_model": "http://127.0.0.1:8080",
  "llama_mmproj": "",
  "whisper_threads": 4,
  "llama_threads": 0,
  "tts_threads": 2
}
```

Note: `llama_model` is the llama-server URL, not a file path.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WHILLATS_SERVER` | Yes | Path to `whillats_server` binary |
| `PIPER_MODEL` | Yes | Default (English) Piper `.onnx` model |
| `PIPER_MODEL_RU` | Optional | Russian Piper model |
| `PIPER_MODEL_ES` | Optional | Spanish Piper model |
| `PIPER_MODEL_ZH` | Optional | Chinese Piper model |
| `PIPER_MODEL_DE` | Optional | German Piper model |
| `PIPER_MODEL_FR` | Optional | French Piper model |
| `ESPEAK_DATA_PATH` | Yes | Path to `espeak-ng-data/` directory |

---

## Automated Deployment (clean Ubuntu machine)

Use `deploy-talkingface5.sh` from the webrtcsays.ai repo root:

```bash
cd webrtcsays.ai

# With HF token (for gated Gemma-4 download)
HF_TOKEN=hf_xxx ./deploy-talkingface5.sh root@your-server ~/.ssh/id_key

# Without HF token (if models already on server)
./deploy-talkingface5.sh root@your-server ~/.ssh/id_key
```

### What the script does

| Phase | Action |
|-------|--------|
| 1 | Install system deps (cmake, ninja, depot_tools, pip) |
| 2 | Clone + build llama-server from llama.cpp (CPU-only) |
| 3 | Clone + build whillats_server (talkingface5 branch, Piper TTS) |
| 4 | Clone + build directcall + libdirect.so (WebRTC GN) |
| 5 | Download AI models (Whisper, Gemma-4 Q2_K + mmproj, Piper EN/RU/ES/ZH) |
| 6 | Generate TLS certificates (self-signed, 10 years) |
| 7 | Write runtime scripts and config.talkingface5.json |
| 8 | Install + enable two systemd services |
| 9 | Upload demo.html to www.wilddolphin.us via FTP |

### Deployed Layout

```
/opt/directcall3-dev/
  directcall                        # WebRTC process
  whillats_server.talkingface5      # Whisper + Piper server
  bin/llama-server                  # LLM HTTP server
  lib/libdirect.so                  # WebRTC shared lib
  lib/libonnxruntime.so             # ONNX Runtime for Piper
  config.talkingface5.json          # Runtime config
  run-directcall.sh                 # Launcher (sets LD_LIBRARY_PATH)
  cert.pem / key.pem                # TLS certs
  espeak-ng-data.talkingface5/      # espeak runtime data
  RobotPhoneLogo.jpeg               # Talking face image

/opt/models/
  ggml-small.bin                    # Whisper
  google_gemma-4-E4B-it-Q2_K.gguf  # LLM
  mmproj-BF16.gguf                  # Multimodal projector
  piper/
    en_US-lessac-low.onnx + .json
    ru_RU-ruslan-medium.onnx + .json
    es_ES-carlfm-x_low.onnx + .json
    zh_CN-huayan-medium.onnx + .json
```

### Systemd Services

```
llama-server-talkingface5   # LLM (port 8080, localhost only)
directcall3-talkingface5    # WebRTC + STT/TTS (Requires= llama-server)
```

```bash
# Logs
journalctl -u llama-server-talkingface5 -f
journalctl -u directcall3-talkingface5 -f

# Restart
systemctl restart llama-server-talkingface5
systemctl restart directcall3-talkingface5

# Status
systemctl status llama-server-talkingface5
systemctl status directcall3-talkingface5
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| No response from LLM | llama-server not ready | Check `curl http://127.0.0.1:8080/health` |
| No transcription | Whisper chunk too small / VAD threshold | Check `kVADThreshold`, ensure mic input is reaching server |
| Model speaks in English | Language detection not firing | Ensure `"language": "auto"` in config |
| Echo / AI hears itself | Browser AEC disabled | `echoCancellation: true` in getUserMedia |
| `length_error` crash | C++ ABI mismatch | Ensure client code compiled by WebRTC clang in-tree |
| `libdirect.so` not found | LD_LIBRARY_PATH missing | Use `run-directcall.sh` wrapper |
| llama-server OOM | Model too large for RAM | Use Q2_K quantization; disable mmproj |
| Clean rebuild needed | Stale CMake cache | `rm -rf build && cmake -B build ...` |
