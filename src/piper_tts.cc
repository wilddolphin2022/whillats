/*
 *  Piper TTS using subprocess to avoid libc++/libstdc++ ONNX crash.
 *  All ONNX calls happen in a forked child process.
 */

#include "piper_tts.h"
#include "piper_subprocess.h"
#include "whisper_helpers.h"
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

    _subprocess = std::make_unique<PiperSubprocess>();
    if (!_subprocess->start(model_path, espeak_data_path)) {
        LOG_E("PiperTTS: Failed to start subprocess");
        _subprocess.reset();
        return false;
    }
    LOG_I("PiperTTS: Subprocess started for model: " << model_path);

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
    if (_subprocess) {
        _subprocess->stop();
        _subprocess.reset();
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

    if (text.empty() || !_subprocess) return true;

    LOG_I("PiperTTS: Synthesizing (" << text.size() << " chars): "
          << text.substr(0, 60) << (text.size() > 60 ? "..." : ""));

    auto audio = _subprocess->synthesize(text);
    _outputSampleRate = _subprocess->getSampleRate();

    if (!audio.empty()) {
        LOG_I("PiperTTS: Generated " << audio.size() << " samples at " << _outputSampleRate << "Hz");
        std::vector<uint16_t> u16(audio.begin(), audio.end());
        _callback.OnBufferComplete(true, u16);
    } else {
        LOG_W("PiperTTS: No audio generated");
    }

    _callback.OnSynthesisComplete();
    return true;
}

const int PiperTTS::getSampleRate() {
    return _outputSampleRate;
}
