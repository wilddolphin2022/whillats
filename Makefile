.PHONY: build clean debug release test test_release example example_release deps-ios ios

# --- Standard Linux/macOS Build ---
build:
	cmake -B build -DGGML_METAL=ON

debug: build
	cmake --build build --config debug

release:
	cmake -B build -DCMAKE_BUILD_TYPE=release -DGGML_METAL=OFF
	cmake --build build --config release

# Default test target (uses debug build)
test: debug
	@echo "Running debug est..."
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
	@echo "Building iOS dependencies (XCFrameworks)..."
	@(cd third_party/llama.cpp && ./build-xcframework.sh)
	@(cd third_party/whisper.cpp && ./build-xcframework.sh)

ios: deps-ios
	@echo "--- Current directory for make: $(shell pwd) ---"
	@echo "Re-configuring iOS build with dependencies present..."
	@# Re-run CMake configure. FetchContent won't re-download.
	@# The if(EXISTS...) check should now find frameworks and enable linking.
	cmake -S . -B build-ios -G Xcode \
	      -DCMAKE_SYSTEM_NAME=iOS \
	      -DCMAKE_OSX_DEPLOYMENT_TARGET=$(IOS_DEPLOYMENT_TARGET) \
	      -DCMAKE_TOOLCHAIN_FILE=$(IOS_TOOLCHAIN_FILE) \
	      -DPLATFORM=$(IOS_PLATFORM)
	@echo "Building iOS framework (release)..."
	cmake --build build-ios --config release

clean:
	rm -rf build build-ios
