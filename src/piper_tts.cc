/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Piper TTS implementation using libpiper C API.
 *  Designed for CPU-only deployment with real-time performance.
 */

#include "piper_tts.h"
#include "whillats_utils.h"
#include "whisper_helpers.h"
#include <piper.h>
#include <cmath>
#include <algorithm>

PiperTTS::PiperTTS(WhillatsSetAudioCallback callback)
    : _callback(callback) {}

PiperTTS::~PiperTTS() {
    stop();
}

bool PiperTTS::start(const std::string& model_path,
                     const std::string& espeak_data_path,
                     const std::string& config_path) {
    if (_initialized) return true;

    const char* cfg = config_path.empty() ? nullptr : config_path.c_str();
    _synth = piper_create(model_path.c_str(), cfg, espeak_data_path.c_str());
    if (!_synth) {
        LOG_E("PiperTTS: Failed to create synthesizer from " << model_path);
        return false;
    }
    LOG_I("PiperTTS: Loaded model: " << model_path);

    _initialized = true;
    _running = true;
    _processingThread = std::thread([this]() {
        while (_running) {
            if (!runProcessingThread()) break;
        }
    });
    LOG_I("PiperTTS: Started");
    return true;
}

void PiperTTS::stop() {
    if (_running) {
        _running = false;
        _queueCondition.notify_all();
        if (_processingThread.joinable())
            _processingThread.join();
    }
    if (_synth) {
        piper_free(_synth);
        _synth = nullptr;
    }
    _initialized = false;
    LOG_I("PiperTTS: Stopped");
}

void PiperTTS::queueText(const char* text, const char* language) {
    if (!_initialized || !text) return;
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _textQueue.push({std::string(text), std::string(language ? language : "en")});
    }
    _queueCondition.notify_one();
}

bool PiperTTS::runProcessingThread() {
    std::string text;
    std::string language;

    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        _queueCondition.wait_for(lock, std::chrono::milliseconds(100),
            [this] { return !_textQueue.empty() || !_running; });
        if (!_running) return false;
        if (_textQueue.empty()) return true;
        text = _textQueue.front().first;
        language = _textQueue.front().second;
        _textQueue.pop();
    }

    if (text.empty()) return true;

    LOG_I("PiperTTS: Synthesizing (" << text.size() << " chars): "
          << text.substr(0, 60) << (text.size() > 60 ? "..." : ""));

    int rc = piper_synthesize_start(_synth, text.c_str(), nullptr);
    if (rc != PIPER_OK) {
        LOG_E("PiperTTS: synthesize_start failed: " << rc);
        _callback.OnSynthesisComplete();
        return true;
    }

    std::vector<int16_t> all_audio;
    piper_audio_chunk chunk;

    while (_running) {
        rc = piper_synthesize_next(_synth, &chunk);
        if (rc == PIPER_DONE) break;
        if (rc != PIPER_OK) {
            LOG_E("PiperTTS: synthesize_next failed: " << rc);
            break;
        }

        if (chunk.samples && chunk.num_samples > 0) {
            // Convert float samples to int16
            for (size_t i = 0; i < chunk.num_samples; ++i) {
                float v = chunk.samples[i] * 32767.0f;
                v = std::max(-32768.0f, std::min(32767.0f, v));
                all_audio.push_back(static_cast<int16_t>(v));
            }

            int src_rate = chunk.sample_rate;
            LOG_V("PiperTTS: Chunk " << chunk.num_samples << " samples at " << src_rate << "Hz"
                  << (chunk.is_last ? " (last)" : ""));
        }

        if (chunk.is_last) break;
    }

    if (!all_audio.empty()) {
        int src_rate = chunk.sample_rate > 0 ? chunk.sample_rate : 22050;
        _outputSampleRate = src_rate;

        LOG_I("PiperTTS: Generated " << all_audio.size() << " samples at " << src_rate << "Hz");
        std::vector<uint16_t> u16(all_audio.begin(), all_audio.end());
        _callback.OnBufferComplete(true, u16);
    }

    _callback.OnSynthesisComplete();
    return true;
}

const int PiperTTS::getSampleRate() {
    return _outputSampleRate;
}
