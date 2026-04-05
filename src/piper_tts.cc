/*
 *  Piper TTS using subprocess to avoid libc++/libstdc++ ONNX crash.
 *  All ONNX calls happen in a forked child process.
 *  Multiple language models are supported: each language gets its own
 *  subprocess. Synthesis is routed by language tag; falls back to the
 *  default model if no language-specific one is registered.
 */

#include "piper_tts.h"
#include "piper_subprocess.h"
#include "whisper_helpers.h"
#include <cmath>
#include <algorithm>
#include <unistd.h>

static constexpr int TARGET_SAMPLE_RATE = 16000;

// Linear-interpolation resampler. Good enough for TTS→Whisper pipeline.
static std::vector<int16_t> resample_to_16k(const std::vector<int16_t>& in, int src_rate) {
    if (src_rate == TARGET_SAMPLE_RATE || in.empty()) return in;
    double ratio = static_cast<double>(src_rate) / TARGET_SAMPLE_RATE;
    size_t out_len = static_cast<size_t>(in.size() / ratio);
    std::vector<int16_t> out(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        double pos = i * ratio;
        size_t lo  = static_cast<size_t>(pos);
        size_t hi  = lo + 1 < in.size() ? lo + 1 : lo;
        double frac = pos - lo;
        out[i] = static_cast<int16_t>(in[lo] + frac * (in[hi] - in[lo]));
    }
    LOG_I("PiperTTS: Resampled " << in.size() << " samples @" << src_rate
          << "Hz → " << out.size() << " samples @" << TARGET_SAMPLE_RATE << "Hz");
    return out;
}

PiperTTS::PiperTTS(WhillatsSetAudioCallback callback)
    : _callback(callback) {}

PiperTTS::~PiperTTS() {
    stop();
}

bool PiperTTS::start(const std::string& model_path,
                     const std::string& espeak_data_path,
                     const std::string& /*config_path*/) {
    if (_initialized) return true;

    _espeakDataPath = espeak_data_path;

    auto sub = std::make_unique<PiperSubprocess>();
    if (!sub->start(model_path, espeak_data_path)) {
        LOG_E("PiperTTS: Failed to start default subprocess");
        return false;
    }
    LOG_I("PiperTTS: Default subprocess started for model: " << model_path);
    _outputSampleRate = sub->getSampleRate();
    _subprocesses[_defaultLang] = std::move(sub);

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

bool PiperTTS::addLangModel(const std::string& lang, const std::string& model_path) {
    if (!_initialized) {
        LOG_E("PiperTTS::addLangModel called before start()");
        return false;
    }
    if (model_path.empty()) return false;

    // Fail fast if the model file doesn't exist rather than crashing the child
    if (access(model_path.c_str(), R_OK) != 0) {
        LOG_E("PiperTTS: model file not readable for lang=" << lang << ": " << model_path);
        return false;
    }

    auto sub = std::make_unique<PiperSubprocess>();
    if (!sub->start(model_path, _espeakDataPath)) {
        LOG_E("PiperTTS: Failed to start subprocess for lang=" << lang);
        return false;
    }
    LOG_I("PiperTTS: Subprocess started for lang=" << lang << " model=" << model_path);
    std::lock_guard<std::mutex> lock(_queueMutex);
    _subprocesses[lang] = std::move(sub);
    return true;
}

PiperSubprocess* PiperTTS::subprocessForLang(const std::string& lang) {
    auto it = _subprocesses.find(lang);
    if (it != _subprocesses.end()) return it->second.get();
    // Try stripping region suffix: "en-US" → "en"
    auto dash = lang.find('-');
    if (dash != std::string::npos) {
        it = _subprocesses.find(lang.substr(0, dash));
        if (it != _subprocesses.end()) return it->second.get();
    }
    // Fall back to default
    it = _subprocesses.find(_defaultLang);
    if (it != _subprocesses.end()) return it->second.get();
    return nullptr;
}

void PiperTTS::stop() {
    if (_running) {
        _running = false;
        _queueCondition.notify_all();
        if (_processingThread.joinable())
            _processingThread.join();
    }
    _subprocesses.clear();
    _initialized = false;
    LOG_I("PiperTTS: Stopped");
}

void PiperTTS::queueText(const char* text, const char* language) {
    if (!_initialized || !text) return;
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _textQueue.push({std::string(text), std::string(language ? language : _defaultLang)});
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
        text     = _textQueue.front().first;
        language = _textQueue.front().second;
        _textQueue.pop();
    }

    if (text.empty()) return true;

    PiperSubprocess* sub = subprocessForLang(language);
    if (!sub) {
        LOG_E("PiperTTS: No subprocess available for lang=" << language);
        _callback.OnSynthesisComplete();
        return true;
    }

    LOG_I("PiperTTS: Synthesizing [" << language << "] (" << text.size() << " chars): "
          << text.substr(0, 60) << (text.size() > 60 ? "..." : ""));

    auto audio = sub->synthesize(text);
    int native_rate = sub->getSampleRate();

    if (!audio.empty()) {
        LOG_I("PiperTTS: Generated " << audio.size() << " samples at " << native_rate << "Hz");
        if (native_rate != TARGET_SAMPLE_RATE)
            audio = resample_to_16k(audio, native_rate);
        _outputSampleRate = TARGET_SAMPLE_RATE;
        std::vector<uint16_t> u16(audio.begin(), audio.end());
        _callback.OnBufferComplete(true, u16);
    } else {
        LOG_W("PiperTTS: No audio generated for lang=" << language);
    }

    _callback.OnSynthesisComplete();
    return true;
}

const int PiperTTS::getSampleRate() {
    return _outputSampleRate;
}
