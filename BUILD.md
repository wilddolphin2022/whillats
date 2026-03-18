# Whillats Build & Run Guide

## Architecture

Whillats uses a **client-server split** to isolate AI workloads from the WebRTC process:

- **`libwhillats.so`** — thin client library (no AI deps, safe for libc++ / -fno-exceptions)
- **`whillats_server`** — fat standalone binary (Whisper, Llama, Piper/StyleTTS2, optional CUDA)

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design details.

## Prerequisites

- CMake 3.14+
- GCC 11+ (C++17)
- Build tools: `sudo apt install build-essential cmake`
- (Optional) CUDA Toolkit 12.x for GPU acceleration

## Quick Build

### CPU-only (Piper TTS)

```bash
cd src/modules/third_party/whillats

cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_PIPER=ON \
  -DWHILLATS_OLD_ABI=ON

cmake --build build -j$(nproc)
```

### GPU (CUDA + Piper TTS)

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_PIPER=ON \
  -DWHILLATS_OLD_ABI=ON \
  -DGGML_CUDA=ON

cmake --build build -j$(nproc)
```

Only `whillats_server` links CUDA. `libwhillats.so` stays CPU-only.

### StyleTTS2 (instead of Piper)

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_STYLETTS2=ON \
  -DWHILLATS_OLD_ABI=ON

cmake --build build -j$(nproc)
```

`WHILLATS_PIPER` and `WHILLATS_STYLETTS2` are mutually exclusive.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `WHILLATS_PIPER` | OFF | Piper neural TTS (fast CPU, ONNX) |
| `WHILLATS_STYLETTS2` | ON | StyleTTS2 neural TTS (ONNX) |
| `WHILLATS_OLD_ABI` | OFF | `_GLIBCXX_USE_CXX11_ABI=0` for compat |
| `GGML_CUDA` | OFF | CUDA GPU for Whisper + Llama |

## Build Outputs

```
build/lib/Debug/libwhillats.so            # Thin client (no AI)
build/bin/Debug/whillats_server           # Fat AI server
build/bin/Debug/test_whillats             # Client test (-fno-exceptions)
build/bin/Debug/test_whillats_server      # IPC protocol test
build/bin/debug/espeak-ng-data/           # espeak runtime data
```

## Models

Download before running:

| Model | Size | Download |
|-------|------|----------|
| Whisper base | 142MB | `wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin` |
| Whisper small | 487MB | `wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin` |
| Qwen 1.5B Q4 | 1.0GB | `wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| Piper low | 15MB | `wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx` |
| Piper medium | 60MB | `wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx` |

Place models in `~/webrtcsays.ai/models/`:
```
models/
├── ggml-base.bin
├── ggml-small.bin
├── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
└── piper/
    ├── en_US-lessac-low.onnx
    └── en_US-lessac-low.onnx.json
```

## Running

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WHILLATS_SERVER` | Yes (Linux) | Path to `whillats_server` binary |
| `PIPER_MODEL` | Piper mode | Path to `.onnx` voice model |
| `ESPEAK_DATA_PATH` | Yes | Path to `espeak-ng-data` directory |
| `WHISPER_MODEL` | Auto | Set by API from config |
| `LLAMA_MODEL` | Auto | Set by API from config |
| `LD_LIBRARY_PATH` | Yes | Include `build/lib/debug` |

### test_whillats — Full Pipeline Test

Tests TTS → Whisper → Llama through the server. Built with `-fno-exceptions`.

```bash
cd src/modules/third_party/whillats

LD_LIBRARY_PATH=./build/lib/debug:./build/bin \
PIPER_MODEL=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./build/bin/Debug/whillats_server \
./build/bin/Debug/test_whillats \
  --whisper_model=$HOME/webrtcsays.ai/models/ggml-base.bin \
  --llama_model=$HOME/webrtcsays.ai/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --llama
```

Expected output:
```
[test] TTS: 49152 samples
[test] Saved synthesized_audio.wav
[test] Saved synthesized_audio_long.wav
[test] Whisper: Hello, this is a test of text to speak.
[test] Whisper PASSED
[test] Llama: I don't have a name.
[test] Llama PASSED
```

### test_whillats_server — IPC Protocol Test

Tests the IPC layer directly (independent of libwhillats.so).

```bash
LD_LIBRARY_PATH=./build/lib/debug:./build/bin \
ESPEAK_DATA_PATH=./build/bin/debug/espeak-ng-data \
./build/bin/Debug/test_whillats_server \
  --server=./build/bin/Debug/whillats_server \
  --piper_model=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
  --espeak_data=./build/bin/debug/espeak-ng-data \
  --llama_model=$HOME/webrtcsays.ai/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --whisper_model=$HOME/webrtcsays.ai/models/ggml-base.bin \
  --all
```

### directcall — WebRTC Live

```bash
cd ~/webrtcsays.ai/src

LD_LIBRARY_PATH=./modules/third_party/whillats/build/lib/debug:./modules/third_party/whillats/build/bin \
PIPER_MODEL=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./modules/third_party/whillats/build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./modules/third_party/whillats/build/bin/Debug/whillats_server \
./out/debug/directcall --config ../config.talking-face.json
```

### StyleTTS2 Mode

```bash
cmake -B build -DWHILLATS_STYLETTS2=ON -DWHILLATS_OLD_ABI=ON
cmake --build build -j$(nproc)

STYLETTS2_MODEL_DIR=$HOME/webrtcsays.ai/models/styletts2 \
ESPEAK_DATA_PATH=./build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./build/bin/Debug/whillats_server \
./build/bin/Debug/test_whillats
```

## Performance

Tested on AMD EPYC 7H12 (8 cores) + NVIDIA A100 40GB:

| Component | CPU | GPU (A100) |
|-----------|-----|------------|
| Piper TTS (low, 16kHz) | ~0.5s/sentence | ~0.5s (CPU-only ONNX) |
| Whisper base (30s audio) | ~15s | ~4s |
| Llama 1.5B Q4 (per sentence) | ~1.0s | ~0.5s |
| Model preload (all 3) | ~15s | ~8s |

Models preload in background on server startup. First query is fast.

## Dependencies (auto-fetched)

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (for Piper/StyleTTS2)
- [Piper](https://github.com/OHF-Voice/piper1-gpl) (when `WHILLATS_PIPER=ON`)

## Troubleshooting

- **`length_error` crash**: Ensure `directcall` uses `libwhillats.so` (thin) and `WHILLATS_SERVER` points to the fat server binary. Never load AI code in the WebRTC process.
- **Whisper slow**: Use `ggml-base.bin` (142MB) instead of `ggml-small.bin` (487MB). Enable GPU with `-DGGML_CUDA=ON`.
- **Llama timeout**: Models preload on server startup. If still slow, reduce model size or enable GPU.
- **No audio**: Check `PIPER_MODEL` and `ESPEAK_DATA_PATH` are set correctly.
- **Clean rebuild**: `rm -rf build && cmake -B build ...`
