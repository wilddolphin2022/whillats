#pragma once

class WhillatsSetResponseCallback {
    public:
    explicit WhillatsSetResponseCallback(
        std::function<void(bool, const std::string&)> on_complete)
        : on_complete_(on_complete) {}
    void OnResponseComplete(bool success, const std::string& response) {
        on_complete_(success, response);
    }

    private:
    std::function<void(bool, const std::string&)> on_complete_;
};

class WhillatsSetAudioCallback {
    public:
    explicit WhillatsSetAudioCallback(
        std::function<void(bool, const std::vector<uint16_t>&)> on_complete)
        : on_complete_(on_complete) {}
    void OnBufferComplete(bool success, const std::vector<uint16_t>& buffer) {
        on_complete_(success, buffer);
    }

    private:
    std::function<void(bool, const std::vector<uint16_t>& buffer)> on_complete_;
};

#include "silence_finder.h"
#include "espeak_tts.h"
#include "whisper_transcription.h"
#include "llama_device_base.h"
