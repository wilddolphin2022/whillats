/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Piper TTS: fast CPU-based neural TTS using libpiper.
 *  Models are small ONNX files (~15-60MB), real-time on 8 cores.
 */

#ifndef PIPER_TTS_H
#define PIPER_TTS_H

#include "whillats.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

class PiperSubprocess;

class PiperTTS {
public:
    PiperTTS(WhillatsSetAudioCallback callback);
    ~PiperTTS();

    // Start with the default (fallback) model.
    bool start(const std::string& model_path,
               const std::string& espeak_data_path,
               const std::string& config_path = "");
    void stop();

    // Register an additional language-specific model. Call after start().
    bool addLangModel(const std::string& lang, const std::string& model_path);

    void queueText(const char* text, const char* language = "en");

    const int getSampleRate();

private:
    bool runProcessingThread();
    PiperSubprocess* subprocessForLang(const std::string& lang);

    WhillatsSetAudioCallback _callback;
    std::string _espeakDataPath;

    // "en" (or whatever default) is always in the map after start().
    // Other languages are added via addLangModel().
    std::map<std::string, std::unique_ptr<PiperSubprocess>> _subprocesses;

    bool _running{false};
    std::thread _processingThread;
    std::queue<std::pair<std::string, std::string>> _textQueue;
    std::mutex _queueMutex;
    std::condition_variable _queueCondition;
    std::atomic<bool> _initialized{false};
    int _outputSampleRate = 22050;
    std::string _defaultLang = "en";
};

#endif // PIPER_TTS_H
