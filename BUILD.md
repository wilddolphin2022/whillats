# Build Instructions

This project uses CMake for building the `whillats` library and its dependencies, including the Agora example.

## Prerequisites

*   **CMake:** Version 3.14 or higher.
*   **C/C++ Compiler:** A modern compiler supporting C++17 (e.g., GCC, Clang).
*   **Build Tools:** Standard build utilities (e.g., `make` or `ninja`, `binutils`).
    *   On Debian/Ubuntu: `sudo apt update && sudo apt install build-essential cmake`
    *   On Fedora/RHEL: `sudo dnf groupinstall "Development Tools" && sudo dnf install cmake`
*   **(Optional) NVIDIA CUDA Toolkit:** If building with GPU support for Whisper/Llama on Linux. Ensure `nvcc` is in your PATH.
*   **Internet Connection:** Required during the CMake configuration phase to download dependencies (Whisper, Llama, eSpeak-NG, PCAudioLib, Agora SDK).

## Build Steps

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> whillats
    cd whillats
    ```

2.  **Configure using CMake:**
    This step downloads all dependencies using FetchContent and prepares the build system.
    ```bash
    cmake -B build
    ```
    *   If you have CUDA installed and want to enable GPU support (Linux only):
        ```bash
        # CMake should detect CUDA automatically if nvcc is in PATH.
        # The build is configured to enable GGML_CUDA=ON automatically on Linux.
        cmake -B build 
        ```

3.  **Build:**
    Compile the library, dependencies, and the example executable.
    ```bash
    cmake --build build
    ```
    *   You can use parallel builds: `cmake --build build -j $(nproc)`

4.  **Run Example (Optional):**
    The Agora example executable will be located in `build/bin`. When run it will show options including channel id and token that could be obtained from Agora. Example listens for PCM audio and invokes Llama to make conversation.
    ```bash
    ./build/bin/transceiver_yuv_pcm <args...>
    ```

## StyleTTS2 Neural TTS (Optional)

To build with high-quality neural text-to-speech using [StyleTTS2](https://github.com/DDATT/StyleTTS2-onnx-cpp):

```bash
# macOS
cmake -B build -DWHILLATS_STYLETTS2=ON -DGGML_METAL=ON
cmake --build build --config Release

# Linux
cmake -B build -DWHILLATS_STYLETTS2=ON
cmake --build build --config Release

# Linux with CUDA GPU acceleration
cmake -B build -DWHILLATS_STYLETTS2=ON -DGGML_CUDA=ON
cmake --build build --config Release
```

Or use Make targets:
```bash
make styletts2          # macOS debug
make styletts2-release  # macOS release
make styletts2-linux    # Linux debug
make styletts2-linux-cuda  # Linux with CUDA
```

### StyleTTS2 Model Setup

1.  Download ONNX models from [HuggingFace](https://huggingface.co/DDATT/StyleTTS2-ONNX-Cpp/tree/main)
2.  Place them in a `trained_models/` directory:
    ```
    trained_models/
    ├── plbert_simp.onnx
    ├── bert_encoder.onnx
    ├── final_simp.onnx
    ├── ref_s.bin        (voice style embedding)
    └── ref_p.bin        (predictor embedding)
    ```
3.  Run with environment variables:
    ```bash
    STYLETTS2_MODEL_DIR=./trained_models \
    ESPEAK_DATA_PATH=./build/bin/Release/espeak-ng-data \
    ./build/bin/Release/test_whillats --tts
    ```

ONNX Runtime is automatically downloaded during CMake configuration. To use a custom install, pass `-DONNXRUNTIME_DIR=/path/to/onnxruntime`.

## Dependencies

The following dependencies are automatically downloaded and built via CMake's `FetchContent`:

*   [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
*   [llama.cpp](https://github.com/ggerganov/llama.cpp)
*   [espeak-ng](https://github.com/espeak-ng/espeak-ng)
*   [pcaudiolib](https://github.com/espeak-ng/pcaudiolib) (Dependency for espeak-ng)
*   [Agora RTC SDK for Linux](https://www.agora.io/en/) (Gateway SDK version downloaded from URL)
*   [ONNX Runtime](https://github.com/microsoft/onnxruntime) (When StyleTTS2 is enabled)

## Caveats

*   **Clean Builds:** If you encounter persistent build errors, especially after changing CMake options or dependency versions, try removing the build directory (`rm -rf build`) and re-running the CMake configuration step.
*   **System Libraries:** The build is configured to use the versions of Whisper, Llama, and eSpeak-NG downloaded by FetchContent. If you have system-wide installations of these libraries (e.g., in `/usr/local/lib`), they might interfere if CMake is not configured correctly. The current setup attempts to prioritize the FetchContent builds.
*   **eSpeak-NG Data Path:** The `espeak-ng-data` directory required by eSpeak-NG at runtime is copied to `build/bin/espeak-ng-data`. The C++ code using the espeak API must be initialized with this path (currently passed via the `ESPEAK_DATA_PATH` compile definition).
*   **Agora SDK Version:** The build fetches a specific version of the Agora RTC SDK via URL. If this URL becomes invalid, the build will fail during the CMake configuration phase.
*   **Agora Example Code:** The Agora example code (`example/agora-whillats-bot`) is based on examples provided by [Agora.io](https://www.agora.io/en/). 