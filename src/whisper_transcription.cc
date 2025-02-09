#include "whisper_transcription.h"

WhisperTranscription::WhisperTranscription() : ctx_(nullptr) {}

WhisperTranscription::~WhisperTranscription() {
    if (ctx_) {
        whisper_free(ctx_);
    }
}

bool WhisperTranscription::initialize(const std::string& model_path) {
    ctx_ = whisper_init_from_file(model_path.c_str());
    return ctx_ != nullptr;
}

std::string WhisperTranscription::transcribe(const uint16_t* audio_data, size_t num_samples) {
    if (!ctx_ || !audio_data || num_samples == 0) {
        return "";
    }

    std::vector<float> float_audio = convert_audio(audio_data, num_samples);
    
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    if (whisper_full(ctx_, params, float_audio.data(), float_audio.size()) != 0) {
        return "";
    }

    const int n_segments = whisper_full_n_segments(ctx_);
    std::string result;
    for (int i = 0; i < n_segments; ++i) {
        result += whisper_full_get_segment_text(ctx_, i);
    }

    return result;
}

std::vector<float> WhisperTranscription::convert_audio(const uint16_t* audio_data, size_t num_samples) {
    std::vector<float> float_audio(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        // Convert from int16 range [-32768, 32767] to float [-1, 1]
        float_audio[i] = static_cast<float>(static_cast<int16_t>(audio_data[i])) / 32768.0f;
    }
    return float_audio;
}