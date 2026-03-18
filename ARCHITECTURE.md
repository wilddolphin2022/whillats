# Whillats Architecture

## Overview

Whillats is a C++ library providing real-time AI speech processing for WebRTC applications. It integrates Speech-to-Text (Whisper), Language Model (Llama), and Text-to-Speech (Piper/StyleTTS2/Orpheus) into a client-server architecture that isolates AI workloads from the WebRTC process.

## Architecture

```
┌──────────────────────────────────┐     pipes      ┌──────────────────────────────────┐
│          directcall              │  ◄──────────►  │         whillats_server           │
│     (WebRTC clang, libc++,       │                │    (GCC, libstdc++,               │
│      -fno-exceptions)            │                │     optional CUDA/A100)           │
│                                  │                │                                  │
│  ┌────────────────────────────┐  │                │  ┌────────────────────────────┐  │
│  │  Whillats client (in-tree) │  │                │  │   Whisper (whisper.cpp)     │  │
│  │  Compiled by WebRTC clang  │  │                │  │   Llama (llama.cpp)        │  │
│  │  - WhillatsTTS API         │──┼── IPC ────────►│  │   Piper TTS (ONNX)         │  │
│  │  - WhillatsTranscriber API │  │                │  │   StyleTTS2 (ONNX)         │  │
│  │  - WhillatsLlama API       │  │                │  │   Orpheus TTS (llama+ONNX) │  │
│  │  - TalkingFace (in-proc)   │  │                │  │   espeak-ng (fallback)     │  │
│  │  - IPC client stubs        │  │                │  └────────────────────────────┘  │
│  └────────────────────────────┘  │                │                                  │
└──────────────────────────────────┘                └──────────────────────────────────┘
```

### Why Client-Server?

WebRTC (directcall) is compiled with Clang/libc++ and `-fno-exceptions`. The AI libraries (llama.cpp, whisper.cpp, ONNX Runtime) use GCC/libstdc++ with exceptions. Mixing these in a single process causes `std::string` ABI crashes. The server process runs in pure libstdc++ with exceptions enabled, completely isolated from WebRTC.

### In-Tree Build (final architecture)

The thin whillats client is compiled **directly by WebRTC's GN build system** using the same Clang compiler, libc++, sysroot, and `-fno-exceptions` flags as directcall. This guarantees zero ABI mismatch — every `std::string`, `std::vector`, and C++ object in the client uses the exact same runtime as the rest of WebRTC.

No external `libwhillats.so` is needed. The client sources are listed in `modules/audio_device/BUILD.gn`.

### Components

**Whillats client (in-tree, compiled by WebRTC clang)**
- `whillats.cc` — API classes: WhillatsTTS, WhillatsTranscriber, WhillatsLlama
- `whillats_client.cc` — IPC stubs: fork/exec server, pipe communication (pure C malloc in reader thread)
- `talking_face.cc` — Video lip-sync animation (in-process, 10ms frame-by-frame)
- `whillats_utils.cc` — Audio resampling, YUV conversion
- `stb_image_impl.c` — stb_image compiled as C (avoids C++ warning noise)

**whillats_server** — Fat standalone binary (GCC/libstdc++, all AI backends)
- Links whisper.cpp, llama.cpp, ggml, espeak-ng, onnxruntime
- Can be built with CUDA (`-DGGML_CUDA=ON`) for GPU acceleration
- Started automatically by client as subprocess via fork/exec
- Communicates via pipes with binary IPC protocol
- Preloads Whisper + Llama models on startup (background threads)
- Redirects stdout→stderr to prevent library output corrupting IPC pipe

## IPC Protocol

Length-prefixed binary messages over Unix pipes:

```
┌──────────┬──────────────┬────────────────┐
│ type (1B)│ length (4B)  │ payload (var)  │
└──────────┴──────────────┴────────────────┘
```

Key design decisions:
- Header + payload written atomically (single `write()` call via malloc'd buffer)
- Server writes protected by mutex (Llama + TTS callbacks run on different threads)
- Reader thread uses ONLY C malloc/free — no C++ allocations (prevents `length_error` in `-fno-exceptions`)
- Piper child closes all inherited fds (prevents IPC pipe corruption)
- Server redirects stdout→stderr (`dup2`) before any library code runs

| Type | Code | Direction | Description |
|------|------|-----------|-------------|
| MSG_CONFIG | 0x30 | client→server | Model paths, thread counts, language |
| MSG_WHISPER_START | 0x01 | client→server | Confirm Whisper ready (preloaded) |
| MSG_WHISPER_AUDIO | 0x03 | client→server | Audio chunk (len=0 for flush) |
| MSG_WHISPER_RESULT | 0x04 | server→client | Transcription text |
| MSG_WHISPER_LANGUAGE | 0x05 | server→client | Detected language |
| MSG_LLAMA_START | 0x10 | client→server | Confirm Llama ready (preloaded) |
| MSG_LLAMA_ASK | 0x12 | client→server | Prompt text |
| MSG_LLAMA_RESPONSE | 0x13 | server→client | Response text (per sentence) |
| MSG_LLAMA_VIDEO_FRAME | 0x14 | client→server | YUV frame for multimodal |
| MSG_TTS_START | 0x20 | client→server | Start TTS engine |
| MSG_TTS_SPEAK | 0x22 | client→server | Text + language to synthesize |
| MSG_TTS_AUDIO | 0x23 | server→client | PCM int16 audio samples |
| MSG_TTS_DONE | 0x24 | server→client | Synthesis complete signal |
| MSG_SHUTDOWN | 0xFF | client→server | Shutdown server |

## Build

### Server Build (CMake — CPU or GPU)

```bash
cd src/modules/third_party/whillats

# CPU-only with Piper
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_PIPER=ON -DWHILLATS_OLD_ABI=ON
cmake --build build -j$(nproc)

# GPU (CUDA) with Piper
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DWHILLATS_PIPER=ON -DWHILLATS_OLD_ABI=ON -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

### directcall Build (GN — automatically includes thin client)

```bash
cd src
gn gen out/debug
ninja -C out/debug directcall
```

The GN build compiles whillats client sources in-tree. No `libwhillats.so` linking needed.

### Build Outputs

```
out/debug/directcall                              # WebRTC app (includes thin whillats)
modules/third_party/whillats/build/bin/Debug/
  whillats_server                                  # Fat AI server (CPU or GPU)
  test_whillats                                    # Client test (-fno-exceptions)
  test_whillats_server                             # IPC protocol test
  espeak-ng-data/                                  # espeak runtime data
```

## Running

### directcall (WebRTC Live)

```bash
cd ~/webrtcsays.ai/src

PIPER_MODEL=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./modules/third_party/whillats/build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./modules/third_party/whillats/build/bin/Debug/whillats_server \
./out/debug/directcall --config ../config.talking-face.json
```

### test_whillats (Full Pipeline via Server)

```bash
cd src/modules/third_party/whillats

PIPER_MODEL=$HOME/webrtcsays.ai/models/piper/en_US-lessac-low.onnx \
ESPEAK_DATA_PATH=./build/bin/debug/espeak-ng-data \
WHILLATS_SERVER=./build/bin/Debug/whillats_server \
./build/bin/Debug/test_whillats \
  --whisper_model=$HOME/webrtcsays.ai/models/ggml-base.bin \
  --llama_model=$HOME/webrtcsays.ai/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --llama
```

## Audio Pipeline

```
Browser Mic → WebRTC → directcall → WhillatsTranscriber → [IPC] → Whisper (GPU)
                                                                      │
                                                            transcribed text
                                                                      │
                                     RecThreadProcess ← [IPC] ← WhillatsLlama → Llama (GPU)
                                           │                              │
                                      speakText()                   response text
                                           │                              │
                                     WhillatsTTS → [IPC] ──────► Piper TTS (CPU)
                                                                      │
                                                                 PCM audio
                                                                      │
                RecThreadProcess ← SetTTSBuffer ← ttsAudioCallback ← [IPC]
                     │
              feedAudio(10ms) → TalkingFace (lip-sync)
                     │
              WebRTC → Browser (audio + animated video)
```

## Performance (AMD EPYC 7H12 + NVIDIA A100 40GB)

| Component | CPU | GPU (A100) |
|-----------|-----|------------|
| Piper TTS (low, 16kHz) | ~0.5s/sentence | ~0.5s (CPU ONNX) |
| Whisper base (30s audio) | ~15s | ~4s |
| Llama 1.5B Q4 (per sentence) | ~1.0s | ~0.1s |
| Model preload (all 3) | ~15s | ~8s |

Models preload in background on server startup. First query is fast.

## Source Files

### Client (compiled by WebRTC GN build)

| File | Description |
|------|-------------|
| `whillats.h` | Public API: WhillatsTTS, WhillatsTranscriber, WhillatsLlama |
| `whillats.cc` | API implementation — routes to server via IPC |
| `whillats_client.h/cc` | IPC client: fork/exec server, pipe I/O (C malloc only) |
| `whillats_ipc.h` | Binary protocol: message types, read/write (C malloc) |
| `whillats_utils.h/cc` | Audio resampling, YUV conversion |
| `talking_face.h/cc` | Video lip-sync (10ms frame-by-frame audio energy) |
| `stb_image_impl.c` | stb_image compiled as C |

### Server (compiled by CMake/GCC)

| File | Description |
|------|-------------|
| `whillats_server.cc` | Main loop: IPC commands → AI backends, model preload |
| `whisper_transcription.h/cc` | Whisper STT (whisper.cpp) |
| `llama_device_base.h/cc` | Llama text generation + multimodal (llama.cpp) |
| `piper_tts.h/cc` | Piper neural TTS controller |
| `piper_subprocess.h/cc` | Piper ONNX isolation via fork (closes inherited fds) |
| `styletts2_tts.h/cc` | StyleTTS2 neural TTS (ONNX) |
| `orpheus_tts.h/cc` | Orpheus llama-based TTS + SNAC decoder |
| `espeak_tts.h/cc` | espeak-ng rule-based TTS fallback |
| `whisper_helpers.h` | Logging (stderr only), time utilities |
