# Whillats Architecture

## Overview

Whillats is a C++ library providing real-time AI speech processing for WebRTC applications. It integrates Speech-to-Text (Whisper), Language Model (Llama), and Text-to-Speech (Piper/StyleTTS2/Orpheus) into a client-server architecture that isolates AI workloads from the WebRTC process.

## Architecture

```
┌──────────────────────────────────┐     pipes      ┌──────────────────────────────────┐
│          directcall              │  ◄──────────►  │         whillats_server           │
│     (WebRTC, libc++,             │                │    (AI backends, libstdc++,       │
│      -fno-exceptions)            │                │     optional CUDA)               │
│                                  │                │                                  │
│  ┌────────────────────────────┐  │                │  ┌────────────────────────────┐  │
│  │    libwhillats.so (thin)   │  │                │  │   Whisper (whisper.cpp)     │  │
│  │  - WhillatsTTS API         │──┼── IPC ────────►│  │   Llama (llama.cpp)        │  │
│  │  - WhillatsTranscriber API │  │                │  │   Piper TTS (ONNX)         │  │
│  │  - WhillatsLlama API       │  │                │  │   StyleTTS2 (ONNX)         │  │
│  │  - TalkingFace (in-proc)   │  │                │  │   Orpheus TTS (llama+ONNX) │  │
│  │  - IPC client stubs        │  │                │  │   espeak-ng (fallback)     │  │
│  └────────────────────────────┘  │                │  └────────────────────────────┘  │
└──────────────────────────────────┘                └──────────────────────────────────┘
```

### Why Client-Server?

WebRTC (directcall) is compiled with Clang/libc++ and `-fno-exceptions`. The AI libraries (llama.cpp, whisper.cpp, ONNX Runtime) use GCC/libstdc++ with exceptions. Mixing these in a single process causes `std::string` ABI crashes (`length_error` in `-fno-exceptions` mode). The server process runs in pure libstdc++ with exceptions enabled, completely isolated from WebRTC.

### Components

**libwhillats.so** — Thin client library (CPU-only, no AI dependencies)
- Provides `WhillatsTTS`, `WhillatsTranscriber`, `WhillatsLlama` API classes
- Internally routes all calls through IPC to `whillats_server`
- Contains `TalkingFace` (video lip-sync, runs in-process)
- Zero dependency on whisper, llama, ggml, onnxruntime, espeak
- Safe to load into any process regardless of C++ runtime

**whillats_server** — Fat standalone binary (all AI backends)
- Links whisper.cpp, llama.cpp, ggml, espeak-ng, onnxruntime
- Can be built with or without CUDA (`-DGGML_CUDA=ON`)
- Started automatically by libwhillats.so as a subprocess
- Communicates via pipes with binary IPC protocol

## IPC Protocol

Length-prefixed binary messages over Unix pipes:

```
┌──────────┬──────────────┬────────────────┐
│ type (1B)│ length (4B)  │ payload (var)  │
└──────────┴──────────────┴────────────────┘
```

Message types:
| Type | Code | Direction | Description |
|------|------|-----------|-------------|
| MSG_CONFIG | 0x30 | client→server | Model paths, thread counts, language |
| MSG_WHISPER_START | 0x01 | client→server | Start Whisper model |
| MSG_WHISPER_STOP | 0x02 | client→server | Stop Whisper |
| MSG_WHISPER_AUDIO | 0x03 | client→server | Audio chunk (or flush if len=0) |
| MSG_WHISPER_RESULT | 0x04 | server→client | Transcription text |
| MSG_WHISPER_LANGUAGE | 0x05 | server→client | Detected language |
| MSG_LLAMA_START | 0x10 | client→server | Start Llama model |
| MSG_LLAMA_STOP | 0x11 | client→server | Stop Llama |
| MSG_LLAMA_ASK | 0x12 | client→server | Prompt text |
| MSG_LLAMA_RESPONSE | 0x13 | server→client | Response text (per sentence) |
| MSG_LLAMA_VIDEO_FRAME | 0x14 | client→server | YUV frame for multimodal |
| MSG_TTS_START | 0x20 | client→server | Start TTS engine |
| MSG_TTS_STOP | 0x21 | client→server | Stop TTS |
| MSG_TTS_SPEAK | 0x22 | client→server | Text + language to synthesize |
| MSG_TTS_AUDIO | 0x23 | server→client | PCM audio samples |
| MSG_TTS_DONE | 0x24 | server→client | Synthesis complete signal |
| MSG_SHUTDOWN | 0xFF | client→server | Shutdown server |

## Build

### Prerequisites

- CMake 3.16+
- GCC 11+ (for libstdc++ C++17)
- espeak-ng (built as dependency)
- ONNX Runtime 1.17+ (for Piper/StyleTTS2, fetched automatically)
- CUDA Toolkit 12.x (optional, for GPU acceleration)

### CPU Build (default)

```bash
cd src/modules/third_party/whillats
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_PIPER=ON \
  -DWHILLATS_OLD_ABI=ON
cmake --build build -j$(nproc)
```

### GPU Build (CUDA)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_PIPER=ON \
  -DWHILLATS_OLD_ABI=ON \
  -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Only `whillats_server` links CUDA. `libwhillats.so` remains CPU-only.

### Build Outputs

```
build/lib/Debug/libwhillats.so       # Thin client library
build/bin/Debug/whillats_server      # Fat AI server (CPU or GPU)
build/bin/Debug/test_whillats        # Client test (uses server via IPC)
build/bin/Debug/test_whillats_server # IPC protocol test
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `WHILLATS_PIPER` | OFF | Enable Piper neural TTS (fast CPU) |
| `WHILLATS_STYLETTS2` | ON | Enable StyleTTS2 neural TTS |
| `WHILLATS_OLD_ABI` | OFF | Use `_GLIBCXX_USE_CXX11_ABI=0` for ABI compat |
| `GGML_CUDA` | OFF | Enable CUDA GPU acceleration for Whisper+Llama |

`WHILLATS_PIPER` and `WHILLATS_STYLETTS2` are mutually exclusive.

## TTS Engines

### Piper (`-DWHILLATS_PIPER=ON`)

Fast CPU-optimized neural TTS. Best for real-time on machines without GPU.

- Model: ONNX format, ~15-60MB
- Sample rate: 16000Hz (low quality) or 22050Hz (medium)
- Languages: Single-language per model (~60 languages available)
- Runs in a forked subprocess (isolates ONNX Runtime)

Models:
- `en_US-lessac-low.onnx` — English, 16kHz, ~15MB, fastest
- `en_US-lessac-medium.onnx` — English, 22050Hz, ~60MB, better quality

### StyleTTS2 (`-DWHILLATS_STYLETTS2=ON`)

High-quality neural TTS with style transfer. Requires more CPU/GPU.

- Models: Multiple ONNX files in a directory
- Sample rate: 24000Hz (resampled to 16000Hz)
- English only

### Orpheus

Llama-based TTS with SNAC audio codec. Experimental.

- Model: GGUF (llama.cpp) + SNAC ONNX decoder
- Requires significant compute (CPU or GPU)

### espeak-ng (fallback)

Rule-based TTS. Always available, supports 100+ languages.

- No neural model needed
- Low quality but instant synthesis
- Used as phonemizer for Piper and StyleTTS2

## Whisper Models

| Model | Size | Speed (CPU) | Quality |
|-------|------|-------------|---------|
| `ggml-base.bin` | 142MB | ~3x real-time | Good for commands |
| `ggml-small.bin` | 487MB | ~0.5x real-time | Better accuracy |

## Llama Models

| Model | Size | Speed (CPU) | Notes |
|-------|------|-------------|-------|
| `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | 1.0GB | ~500ms/sentence | Fast, good for chat |

## Running

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `WHILLATS_SERVER` | Yes (Linux) | Path to `whillats_server` binary |
| `PIPER_MODEL` | Yes (Piper) | Path to Piper ONNX model |
| `ESPEAK_DATA_PATH` | Yes | Path to espeak-ng data directory |
| `WHISPER_MODEL` | Auto | Path to Whisper GGML model (set by API) |
| `LLAMA_MODEL` | Auto | Path to Llama GGUF model (set by API) |
| `LLAMA_MMPROJ` | Optional | Path to multimodal projector |
| `STYLETTS2_MODEL_DIR` | StyleTTS2 | Path to StyleTTS2 model directory |
| `STYLETTS2_USE_CUDA` | Optional | Enable CUDA for StyleTTS2 |
| `ORPHEUS_MODEL` | Orpheus | Path to Orpheus GGUF model |
| `SNAC_MODEL` | Orpheus | Path to SNAC ONNX decoder |

### directcall (WebRTC)

```bash
cd ~/webrtcsays.ai/src

LD_LIBRARY_PATH=./modules/third_party/whillats/build/lib/debug:./modules/third_party/whillats/build/bin \
PIPER_MODEL=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./modules/third_party/whillats/build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./modules/third_party/whillats/build/bin/Debug/whillats_server \
./out/debug/directcall --config ../config.talking-face.json
```

### test_whillats (all components via server)

```bash
cd ~/webrtcsays.ai/src/modules/third_party/whillats

LD_LIBRARY_PATH=./build/lib/debug:./build/bin \
PIPER_MODEL=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./build/bin/Debug/whillats_server \
./build/bin/Debug/test_whillats \
  --whisper_model=$HOME/webrtcsays.ai/models/ggml-base.bin \
  --llama_model=$HOME/webrtcsays.ai/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --llama
```

### test_whillats_server (IPC protocol test)

```bash
cd ~/webrtcsays.ai/src/modules/third_party/whillats

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

### Orpheus TTS test

```bash
ORPHEUS_MODEL=$HOME/webrtcsays.ai/models/orpheus/orpheus-finetuned-3b-q4_k_m.gguf \
SNAC_MODEL=$HOME/webrtcsays.ai/models/orpheus/snac24_int2wav_static.onnx \
./build/bin/Debug/test_whillats --orpheus
```

### StyleTTS2 test

```bash
cmake -B build -DWHILLATS_STYLETTS2=ON -DWHILLATS_OLD_ABI=ON
cmake --build build -j$(nproc)

STYLETTS2_MODEL_DIR=$HOME/webrtcsays.ai/models/styletts2 \
ESPEAK_DATA_PATH=./build/bin/debug/espeak-ng-data \
./build/bin/Debug/test_whillats
```

## Source Files

### Client Library (libwhillats.so)

| File | Description |
|------|-------------|
| `whillats.h` | Public API: WhillatsTTS, WhillatsTranscriber, WhillatsLlama |
| `whillats.cc` | API implementation — routes to server via IPC |
| `whillats_client.h/cc` | IPC client: fork/exec server, pipe communication |
| `whillats_ipc.h` | Binary protocol: message types, read/write helpers |
| `whillats_utils.h/cc` | Audio resampling, YUV conversion utilities |
| `talking_face.h/cc` | Video lip-sync animation (in-process) |
| `whillats_export.h` | DLL export macros |

### Server (whillats_server)

| File | Description |
|------|-------------|
| `whillats_server.cc` | Main loop: reads IPC commands, dispatches to backends |
| `whisper_transcription.h/cc` | Whisper speech-to-text (whisper.cpp) |
| `llama_device_base.h/cc` | Llama text generation + multimodal (llama.cpp + mtmd) |
| `piper_tts.h/cc` | Piper neural TTS controller |
| `piper_subprocess.h/cc` | Piper ONNX isolation via fork |
| `styletts2_tts.h/cc` | StyleTTS2 neural TTS (ONNX) |
| `orpheus_tts.h/cc` | Orpheus llama-based TTS + SNAC decoder |
| `espeak_tts.h/cc` | espeak-ng rule-based TTS fallback |
| `whisper_helpers.h` | Logging macros, time utilities |

### Tests

| File | Description |
|------|-------------|
| `test/test_whillats.cc` | End-to-end test via thin client (uses server) |
| `test/test_whillats_server.cc` | Direct IPC protocol test |
| `test/test_utils.h/cc` | WAV file writer, command-line parser |

## Audio Pipeline

```
Browser Mic → WebRTC → directcall → WhillatsTranscriber → [IPC] → Whisper
                                                                      │
                                                            transcribed text
                                                                      │
                                                                      ▼
                                                              WhillatsLlama → [IPC] → Llama
                                                                                        │
                                                                              response text
                                                                                        │
                                                                                        ▼
                                                              WhillatsTTS → [IPC] → Piper/StyleTTS2
                                                                                        │
                                                                                  PCM audio
                                                                                        │
                        directcall ← SetTTSBuffer ← ttsAudioCallback ← [IPC] ◄─────────┘
                            │
                            ▼
                    WebRTC → Browser Speaker
```

## Performance (CPU: AMD EPYC 7H12, 8 cores @ 2.6GHz)

| Component | Latency | Notes |
|-----------|---------|-------|
| Piper TTS | ~0.5s per sentence | 16kHz, low model |
| Whisper base | ~15s per 30s audio | Real-time factor ~0.5x |
| Whisper small | ~35s per 30s audio | Better accuracy |
| Llama 1.5B Q4 | ~0.5s per sentence | 100 token max |
| Full pipeline | ~20-40s end-to-end | Whisper dominates |

With A100 GPU (`-DGGML_CUDA=ON`), expect 5-10x speedup for Whisper and Llama.
