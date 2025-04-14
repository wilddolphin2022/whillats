.PHONY: build clean debug release test test_release example example_release

build:
	cmake -B build

debug: build
	cmake --build build --config Debug

release: build
	cmake --build build --config Release

# Default test target (uses Debug build)
test: debug
	@echo "Running Debug Test..."
	@./build/bin/test_whillats

# Release test target
test_release: release
	@echo "Running Release Test..."
	@./build/bin/test_whillats

# Note: Example targets will only work if built on Linux
# Default example target (uses Debug build)
example: debug
	@echo "Running Debug Example (Linux Only)..."
	@if [ -f ./build/bin/transceiver_yuv_pcm ]; then \
	    ./build/bin/transceiver_yuv_pcm; \
	else \
	    echo "Example executable not found (skipped on non-Linux?)"; \
	fi

# Release example target
example_release: release
	@echo "Running Release Example (Linux Only)..."
	@if [ -f ./build/bin/transceiver_yuv_pcm ]; then \
	    ./build/bin/transceiver_yuv_pcm; \
	else \
	    echo "Example executable not found (skipped on non-Linux?)"; \
	fi

clean:
	rm -rf build