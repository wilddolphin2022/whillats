#pragma once

#include <string>
#include <vector>
#include "whisper.h"

class WhisperTranscription {
public:
    WhisperTranscription();
    ~WhisperTranscription();

    bool initialize(const std::string& model_path);
    std::string transcribe(const uint16_t* audio_data, size_t num_samples);

    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int FRAME_SIZE = 160;  // 10ms at 16kHz

private:
    struct whisper_context* ctx_;
    std::vector<float> convert_audio(const uint16_t* audio_data, size_t num_samples);
};