.PHONY: build clean debug release

build:
	cmake -B build

debug: build
	cmake --build build --config Debug

release: build
	cmake --build build --config Release

clean:
	rm -rf build