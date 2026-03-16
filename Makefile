.PHONY: build clean debug release test test_release example example_release deps-ios ios ios-debug ios-clean styletts2 styletts2-release styletts2-linux styletts2-linux-cuda test-styletts2

# --- Platform detection: Metal on macOS, CUDA on Linux ---
# Pass NO_CUDA=1 to disable CUDA on Linux (e.g. make debug NO_CUDA=1)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    GPU_FLAGS = -DGGML_METAL=ON -DGGML_CUDA=OFF
else
    ifdef NO_CUDA
        GPU_FLAGS = -DGGML_METAL=OFF -DGGML_CUDA=OFF
    else
        GPU_FLAGS = -DGGML_METAL=OFF -DGGML_CUDA=ON
    endif
endif

# --- Standard Linux/macOS Build ---
build:
	cmake -B build $(GPU_FLAGS)

debug: build
	cmake --build build --config debug

release:
	cmake -B build -DCMAKE_BUILD_TYPE=release $(GPU_FLAGS)
	cmake --build build --config release

# Default test target (uses debug build)
test: debug
	@echo "Running debug test..."
	@if [ -f ./build/bin/test_whillats ]; then \
	    ./build/bin/test_whillats; \
	else \
	    echo "Test executable not found (skipped on iOS?)"; \
	fi

# Release test target
test_release: release
	@echo "Running release test..."
	@if [ -f ./build/bin/test_whillats ]; then \
	    ./build/bin/test_whillats; \
	else \
	    echo "Test executable not found (skipped on iOS?)"; \
	fi

# Note: Example targets will only work if built on Linux
# Default example target (uses debug build)
example: debug
	@echo "Running debug example (Linux Only)..."
	@if [ -f ./build/bin/transceiver_yuv_pcm ]; then \
	    ./build/bin/transceiver_yuv_pcm; \
	else \
	    echo "Example executable not found (skipped on non-Linux?)"; \
	fi

# Release example target
example_release: release
	@echo "Running release example (Linux Only)..."
	@if [ -f ./build/bin/transceiver_yuv_pcm ]; then \
	    ./build/bin/transceiver_yuv_pcm; \
	else \
	    echo "Example executable not found (skipped on non-Linux?)"; \
	fi

# --- StyleTTS2 Build ---
# Builds whillats with StyleTTS2 neural TTS (ONNX Runtime auto-downloaded)

styletts2:
	cmake -B build -DWHILLATS_STYLETTS2=ON $(GPU_FLAGS)
	cmake --build build --config Debug

styletts2-release:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHILLATS_STYLETTS2=ON $(GPU_FLAGS)
	cmake --build build --config Release

styletts2-linux:
	cmake -B build -DWHILLATS_STYLETTS2=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF -DWHILLATS_OLD_ABI=ON
	cmake --build build --config Debug

styletts2-linux-cuda:
	cmake -B build -DWHILLATS_STYLETTS2=ON -DGGML_METAL=OFF -DGGML_CUDA=ON -DWHILLATS_OLD_ABI=ON
	cmake --build build --config Debug

test-styletts2: styletts2
	@echo "Running StyleTTS2 test..."
	@if [ -f ./build/bin/test_whillats ]; then \
	    STYLETTS2_MODEL_DIR=./trained_models \
	    ESPEAK_DATA_PATH=./build/bin/Debug/espeak-ng-data \
	    ./build/bin/Debug/test_whillats --tts; \
	else \
	    echo "Test executable not found. Build with 'make styletts2' first."; \
	fi

# --- iOS Framework Build ---
# Prerequisite: Requires an iOS CMake toolchain file (e.g., cmake/ios.toolchain.cmake)
# Assumes Xcode and command-line tools are installed.

# Use = instead of ?= to force assignment
IOS_TOOLCHAIN_FILE = ${PWD}/cmake/ios.toolchain.cmake # Default path, can be overridden
IOS_DEPLOYMENT_TARGET ?= 16.4 # Default target, can be overridden
IOS_PLATFORM ?= OS64COMBINED # Platform for toolchain (e.g., OS64, SIMULATOR64, OS64COMBINED)

# Export the variable to make it available in the sub-shell environment
export IOS_TOOLCHAIN_FILE

deps-ios:
	@echo "Running initial CMake configure to fetch sources..."
	@# This ensures FetchContent downloads sources (like build scripts)
	@# Linking step will be skipped due to if(EXISTS...) check in CMakeLists.txt
	cmake -S . -B build-ios -G Xcode \
	      -DCMAKE_SYSTEM_NAME=iOS \
	      -DCMAKE_OSX_DEPLOYMENT_TARGET=$(IOS_DEPLOYMENT_TARGET) \
	      -DCMAKE_TOOLCHAIN_FILE=$(IOS_TOOLCHAIN_FILE) \
	      -DPLATFORM=$(IOS_PLATFORM) || echo "Initial CMake configure complete (ignore potential plist/link errors)."
	@# Patch upstream mtmd-audio.cpp if needed (fresh clone case)
	@if [ -f third_party/llama.cpp/tools/mtmd/mtmd-audio.cpp ]; then \
	  sed -i '' 's/std::vector data(\(filters.n_mel \* filters.n_fft, 0.0f\));/std::vector<float> data(\1);/' third_party/llama.cpp/tools/mtmd/mtmd-audio.cpp; \
	fi
	@echo "Skipping external XCFramework scripts; dependencies will be built via CMake targets."

# iOS build using Xcode generator. Produces whillats.framework under build-ios/bin/release (normalized lowercase)
IOS_ARCHS?=arm64
ios: deps-ios
	@echo "--- Current directory for make: $(shell pwd) ---"
	@echo "Configuring iOS build (SDK target $(IOS_DEPLOYMENT_TARGET), arch $(IOS_ARCHS))..."
	rm -rf build-ios
	cmake -S . -B build-ios -G Xcode \
	      -DCMAKE_SYSTEM_NAME=iOS \
	      -DCMAKE_OSX_ARCHITECTURES=$(IOS_ARCHS) \
	      -DCMAKE_OSX_DEPLOYMENT_TARGET=$(IOS_DEPLOYMENT_TARGET) \
	      -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	      -U CMAKE_TOOLCHAIN_FILE \
	      -DGGML_METAL=ON -DGGML_OPENMP=OFF -DLLAMA_OPENMP=OFF -DWHISPER_OPENMP=OFF
	@# Patch upstream mtmd-audio.cpp if needed (fresh clone case)
	@if [ -f third_party/llama.cpp/tools/mtmd/mtmd-audio.cpp ]; then \
	  sed -i '' 's/std::vector data(\(filters.n_mel \* filters.n_fft, 0.0f\));/std::vector<float> data(\1);/' third_party/llama.cpp/tools/mtmd/mtmd-audio.cpp; \
	fi
	@echo "Building iOS release configuration..."
	cmake --build build-ios --config Release --target whillats --parallel 4
	@# Normalize directories to lowercase for GN lookups
	@mkdir -p build-ios/bin/release build-ios/bin/debug
	@if [ -d build-ios/lib/Release ]; then \
	  rsync -a --delete build-ios/lib/Release/ build-ios/bin/release/; \
	fi
	@if [ -d build-ios/lib/Debug ]; then \
	  rsync -a --delete build-ios/lib/Debug/ build-ios/bin/debug/; \
	fi
	@echo "iOS build complete. Output (framework expected): build-ios/bin/release"

ios-debug: deps-ios
	@echo "--- Current directory for make: $(shell pwd) ---"
	@echo "Configuring iOS build (SDK target $(IOS_DEPLOYMENT_TARGET), arch $(IOS_ARCHS))..."
	rm -rf build-ios
	cmake -S . -B build-ios -G Xcode \
	      -DCMAKE_SYSTEM_NAME=iOS \
	      -DCMAKE_OSX_ARCHITECTURES=$(IOS_ARCHS) \
	      -DCMAKE_OSX_DEPLOYMENT_TARGET=$(IOS_DEPLOYMENT_TARGET) \
	      -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	      -U CMAKE_TOOLCHAIN_FILE \
	      -DGGML_METAL=ON -DGGML_OPENMP=OFF -DLLAMA_OPENMP=OFF -DWHISPER_OPENMP=OFF
	@# Patch upstream mtmd-audio.cpp if needed (fresh clone case)
	@if [ -f third_party/llama.cpp/tools/mtmd/mtmd-audio.cpp ]; then \
	  sed -i '' 's/std::vector data(\(filters.n_mel \* filters.n_fft, 0.0f\));/std::vector<float> data(\1);/' third_party/llama.cpp/tools/mtmd/mtmd-audio.cpp; \
	fi
	@echo "Building iOS debug configuration..."
	cmake --build build-ios --config Debug --target whillats --parallel 4
	@# Normalize directories to lowercase for GN lookups
	@mkdir -p build-ios/bin/release build-ios/bin/debug
	@if [ -d build-ios/lib/Release ]; then \
	  rsync -a --delete build-ios/lib/Release/ build-ios/bin/release/; \
	fi
	@if [ -d build-ios/lib/Debug ]; then \
	  rsync -a --delete build-ios/lib/Debug/ build-ios/bin/debug/; \
	fi
	@echo "iOS debug build complete. Output (framework expected): build-ios/bin/debug"

clean:
	rm -rf build build-ios

ios-clean:
	rm -rf build-ios
