/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Orpheus TTS: uses llama.cpp for audio-token generation
 *  and SNAC ONNX decoder for waveform reconstruction.
 */

#ifndef ORPHEUS_TTS_H
#define ORPHEUS_TTS_H

#include "whillats.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace Ort { class Env; class Session; class RunOptions; }

class OrpheusTTS {
public:
    OrpheusTTS(WhillatsSetAudioCallback callback);
    ~OrpheusTTS();

    bool start(const std::string& orpheus_model_path,
               const std::string& snac_onnx_path);
    void stop();

    void queueText(const char* text, const char* voice = "tara");

    static const int getSampleRate();

private:
    bool runProcessingThread();

    // Token decoding: Orpheus custom token -> codebook id
    int tokenToId(const std::string& token_text, int index);

    // Convert 28 audio tokens (4 frames) into PCM via SNAC
    std::vector<int16_t> convertToAudio(const std::vector<int>& tokens);

    WhillatsSetAudioCallback _callback;

    // Llama context for Orpheus model
    struct llama_model* _model = nullptr;
    struct llama_context* _ctx = nullptr;
    struct llama_sampler* _sampler = nullptr;

    // SNAC decoder (ONNX) - opaque to avoid header dependency
    struct SnacDecoder;
    std::unique_ptr<SnacDecoder> _snac;

    // Processing
    bool _running{false};
    std::thread _processingThread;
    std::queue<std::pair<std::string, std::string>> _textQueue; // text, voice
    std::mutex _queueMutex;
    std::condition_variable _queueCondition;
    std::atomic<bool> _initialized{false};

    static constexpr int SNAC_SAMPLE_RATE = 24000;
    static constexpr int OUTPUT_SAMPLE_RATE = 16000;
    static constexpr int TOKENS_PER_FRAME = 7;
    static constexpr int FRAMES_PER_CHUNK = 12;
    static constexpr int TOKENS_PER_CHUNK = TOKENS_PER_FRAME * FRAMES_PER_CHUNK;
};

#endif // ORPHEUS_TTS_H
