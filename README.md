# Whillats

A C++ library that combines text-to-speech, speech recognition, and large language model capabilities.

## Features

- Text-to-speech synthesis using eSpeak-NG
- Speech recognition using Whisper
- Language model processing using LLaMA

## Prerequisites

- CMake 3.14 or higher
- C++14 compatible compiler
- macOS or Linux operating system

## Dependencies

The following dependencies are included as submodules:
- whisper.cpp
- llama.cpp
- espeak-ng
- pcaudiolib

## Building

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/wilddolphin2022/whillats.git
cd whillats

chmod +x third_party.sh
./third_party.sh

cmake -B build
cmake --build build --config Release
cmake --build build --config Debug
