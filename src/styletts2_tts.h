/*
 *  (c) 2025, wilddolphin2025 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef STYLETTS2_TTS_H
#define STYLETTS2_TTS_H

#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "whillats.h"
#include "whisper_helpers.h"

#include <onnxruntime_cxx_api.h>
#include <espeak-ng/speak_lib.h>

#ifdef toupper
#undef toupper
#endif
#ifdef tolower
#undef tolower
#endif

class StyleTTS2TTS {
public:
    StyleTTS2TTS(WhillatsSetAudioCallback callback,
                 const std::string& modelDir,
                 const std::string& espeakDataDir,
                 bool useCuda = false);
    ~StyleTTS2TTS();

    bool start();
    void stop();
    void queueText(const std::string& text, const std::string& language);
    void setThreadCount(int n) { _nThreads = n; }
    void loadStyle(const std::string& styleFile, const std::string& predictorFile);

    static const int getSampleRate();

private:
    void initPhonemizer(const std::string& voice, const std::string& espeakData);
    std::string phonemize(const std::string& text);
    std::vector<int64_t> textToSequence(const std::string& text);

    std::vector<int16_t> synthesize(const std::string& text, float speed = 1.0f);

    bool runProcessingThread();

    static std::vector<char32_t> utf8ToCodepoints(const std::string& str);
    static std::vector<float> loadBinaryFile(const std::string& filename);

    static const int SAMPLE_RATE = 24000;
    static constexpr float MAX_WAV_VALUE = 32767.0f;

    WhillatsSetAudioCallback _callback;

    std::string _modelDir;
    std::string _espeakDataDir;
    bool _useCuda;

    Ort::Env _env{nullptr};
    Ort::SessionOptions _sessionOptions;
    std::unique_ptr<Ort::Session> _plBert;
    std::unique_ptr<Ort::Session> _bertEncoder;
    std::unique_ptr<Ort::Session> _model;

    std::vector<float> _styleEmbedding;
    std::vector<float> _predictorEmbedding;

    std::unordered_map<char32_t, int64_t> _symbolToId;

    bool _running{false};
    std::thread _processingThread;
    std::queue<std::pair<std::string, std::string>> _textQueue;
    std::mutex _queueMutex;
    std::condition_variable _queueCondition;

    bool _initialized{false};
    bool _phonemizerReady{false};
    int _nThreads = 0; // 0 = auto
};

#endif // STYLETTS2_TTS_H
