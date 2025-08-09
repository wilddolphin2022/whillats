# Whillats Library

This project provides the Whillats library, integrating speech-to-text (Whisper.cpp), Language model processing (Llama.cpp) and text-to-speech (eSpeak-NG)

It includes an example demonstrating integration with the Agora RTC SDK (Linux only due to Agora constraints). 

Platforms tested - Mac x86, Mac Silicon, iOS arm64 and Linux x64. 
Macs use Metal so Silicon with M4 chip will be fast. 
Linux use NVIDIA CUDA drivers for HW acceleration.

## Building

Please see [BUILD.md](BUILD.md) for instructions on how to build the library and the example.

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

cmake -B build
cmake --build build --config Release
cmake --build build --config Debug
