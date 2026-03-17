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

#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>

// #include "silence_finder.h" // removed: not needed for iOS build
#include "whillats.h"
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(WHILLATS_USE_SERVER)
#include "whillats_client.h"
#include "whillats_ipc.h"
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
class WhisperTranscriber {
 public:
  WhisperTranscriber(const char*, WhillatsSetResponseCallback, WhillatsSetLanguageCallback = {nullptr, nullptr}) {}
  ~WhisperTranscriber() = default;
  bool start() { return false; }
  void stop() {}
  void processAudioBuffer(uint8_t*, size_t) {}
};
#else
#include "whisper_transcription.h"
#endif
#include "llama_device_base.h"
#include "whillats_utils.h"

// Runtime check: are we inside whillats_server? If so, use in-process path.
#if defined(WHILLATS_USE_SERVER)
static bool isServerProcess() {
    static int cached = -1;
    if (cached < 0) cached = (getenv("WHILLATS_IS_SERVER") != nullptr) ? 1 : 0;
    return cached != 0;
}
#endif

// Shared server connection (lazy-initialized, one per process)
#if defined(WHILLATS_USE_SERVER)
static std::shared_ptr<WhillatsServerConnection> s_serverConn;
static std::mutex s_serverMutex;

static std::shared_ptr<WhillatsServerConnection> getOrCreateServer() {
    std::lock_guard<std::mutex> lock(s_serverMutex);
    if (s_serverConn && s_serverConn->isRunning())
        return s_serverConn;

    s_serverConn = std::make_shared<WhillatsServerConnection>();

    // Find whillats_server binary next to libwhillats.so
    std::string server_path;
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        std::string dir(exe_path);
        size_t slash = dir.find_last_of('/');
        if (slash != std::string::npos) dir = dir.substr(0, slash);
        // Try same dir, then ../bin/Debug
        std::string candidates[] = {
            dir + "/whillats_server",
            dir + "/../bin/Debug/whillats_server",
            dir + "/../../modules/third_party/whillats/build/bin/Debug/whillats_server",
        };
        for (auto& c : candidates) {
            if (access(c.c_str(), X_OK) == 0) { server_path = c; break; }
        }
    }
    // Fallback: env var
    if (server_path.empty()) {
        const char* env = getenv("WHILLATS_SERVER");
        if (env && access(env, X_OK) == 0) server_path = env;
    }
    if (server_path.empty()) {
        fprintf(stderr, "[whillats] ERROR: Cannot find whillats_server binary. "
                "Set WHILLATS_SERVER env var.\n");
        return nullptr;
    }

    whillats_ipc::ConfigMsg cfg{};
    const char* e;
    if ((e = getenv("WHISPER_MODEL")))     strncpy(cfg.whisper_model, e, sizeof(cfg.whisper_model)-1);
    if ((e = getenv("LLAMA_MODEL")))       strncpy(cfg.llama_model, e, sizeof(cfg.llama_model)-1);
    if ((e = getenv("LLAMA_MMPROJ")))      strncpy(cfg.llama_mmproj, e, sizeof(cfg.llama_mmproj)-1);
    if ((e = getenv("PIPER_MODEL")))       strncpy(cfg.piper_model, e, sizeof(cfg.piper_model)-1);
    if ((e = getenv("ESPEAK_DATA_PATH")))  strncpy(cfg.espeak_data, e, sizeof(cfg.espeak_data)-1);
    strncpy(cfg.language, "en", sizeof(cfg.language)-1);
    cfg.whisper_threads = 4;
    cfg.llama_threads = 6;
    cfg.tts_threads = 2;

    if (!s_serverConn->start(server_path, cfg)) {
        fprintf(stderr, "[whillats] ERROR: Failed to start whillats_server at %s\n", server_path.c_str());
        s_serverConn.reset();
        return nullptr;
    }
    fprintf(stderr, "[whillats] Server started: %s\n", server_path.c_str());
    return s_serverConn;
}
#endif // WHILLATS_USE_SERVER

#if defined(WHILLATS_PIPER)
#include "piper_tts.h"

static int s_piperSampleRate = 16000;

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback)
{
#if defined(WHILLATS_USE_SERVER)
    _useServer = !isServerProcess();
#endif
    if (!_useServer) _piper = std::make_unique<PiperTTS>(callback);
}

WhillatsTTS::~WhillatsTTS() {}

void WhillatsTTS::queueText(const char* text) {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _ttsClient) { _ttsClient->queueText(text, "en"); return; }
#endif
    if (_piper) _piper->queueText(text, "en");
}

void WhillatsTTS::queueText(const char* text, const char* language) {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _ttsClient) { _ttsClient->queueText(text, language); return; }
#endif
    if (_piper) _piper->queueText(text, language);
}

bool WhillatsTTS::start() {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer) {
        _conn = getOrCreateServer();
        if (!_conn) return false;
        _conn->setTtsCallback(_callback);
        _ttsClient = std::make_unique<WhillatsTTSClient>(*_conn);
        return _ttsClient->start();
    }
#endif
    if (_piper) {
        const char* model = getenv("PIPER_MODEL");
        const char* espeak = getenv("ESPEAK_DATA_PATH");
        if (!model || !model[0]) {
            fprintf(stderr, "[whillats] PiperTTS: PIPER_MODEL env var not set\n");
            return false;
        }
        bool ok = _piper->start(model, espeak ? espeak : "");
        if (ok) s_piperSampleRate = _piper->getSampleRate();
        return ok;
    }
    return false;
}

bool WhillatsTTS::start(bool) { return start(); }

void WhillatsTTS::stop() {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _ttsClient) { _ttsClient->stop(); return; }
#endif
    if (_piper) _piper->stop();
}

void WhillatsTTS::setThreadCount(int) {}

int WhillatsTTS::getSampleRate() { return s_piperSampleRate; }

void WhillatsTTS::enableSpeakerphone() {}
void WhillatsTTS::disableSpeakerphone() {}

#elif defined(WHILLATS_STYLETTS2)
#include "styletts2_tts.h"

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback)
{
    const char* modelDir = getenv("STYLETTS2_MODEL_DIR");
    const char* espeakData = getenv("ESPEAK_DATA_PATH");
    bool useCuda = getenv("STYLETTS2_USE_CUDA") != nullptr;

    if (modelDir && espeakData) {
        _styletts2 = std::make_unique<StyleTTS2TTS>(callback,
            std::string(modelDir), std::string(espeakData), useCuda);
    } else {
        LOG_W("StyleTTS2: STYLETTS2_MODEL_DIR or ESPEAK_DATA_PATH not set");
    }
}

WhillatsTTS::~WhillatsTTS() {}

void WhillatsTTS::queueText(const char* text) {
    if (_styletts2) _styletts2->queueText(std::string(text), "en");
}

void WhillatsTTS::queueText(const char* text, const char* language) {
    if (_styletts2) _styletts2->queueText(std::string(text),
        std::string(language ? language : "en"));
}

bool WhillatsTTS::start() {
    if (_styletts2) return _styletts2->start();
    return false;
}

bool WhillatsTTS::start(bool) { return start(); }

void WhillatsTTS::stop() { if (_styletts2) _styletts2->stop(); }

void WhillatsTTS::setThreadCount(int n) {
    if (_styletts2) _styletts2->setThreadCount(n);
}

int WhillatsTTS::getSampleRate() { return StyleTTS2TTS::getSampleRate(); }

void WhillatsTTS::enableSpeakerphone() {}
void WhillatsTTS::disableSpeakerphone() {}

#elif !defined(__APPLE__)
#include "espeak_tts.h"

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback)
{
    _espeak_tts = std::make_unique<ESpeakTTS>(callback);
}

WhillatsTTS::~WhillatsTTS() {}

void WhillatsTTS::queueText(const char* text) {
    _espeak_tts->queueText(std::string(text), "en");
}

void WhillatsTTS::queueText(const char* text, const char* language) {
    _espeak_tts->queueText(std::string(text), std::string(language ? language : "en"));
}

bool WhillatsTTS::start() {
    return _espeak_tts->start();
}

bool WhillatsTTS::start(bool /*withAudio*/) {
    return start();
}

void WhillatsTTS::stop() {
    _espeak_tts->stop();
}

int WhillatsTTS::getSampleRate() {
    return ESpeakTTS::getSampleRate();
}

void WhillatsTTS::enableSpeakerphone() {}

void WhillatsTTS::disableSpeakerphone() {}

#elif defined(__APPLE__) && TARGET_OS_IPHONE
#import "whillats_synth.h"
// Delegate to Objective-C AVFoundation wrapper
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback),
      _wrapper(std::make_unique<WhillatsSpeechSynthesizerWrapper>()) {}

WhillatsTTS::~WhillatsTTS() {
    stop();
}

bool WhillatsTTS::start(bool enableProcessor) {
    _wrapper->initialize(&_callback, enableProcessor);
    return true;
}

void WhillatsTTS::stop() {
    _wrapper->stop();
}

int WhillatsTTS::getSampleRate() {
    // AVAudioEngine is configured for 16kHz
    return 16000;
}

void WhillatsTTS::queueText(const char* text, const char* language) {
    _wrapper->synthesize(std::string(text), std::string(language));
}

void WhillatsTTS::enableSpeakerphone() {
    _wrapper->enableSpeakerphone();
}

void WhillatsTTS::disableSpeakerphone() {
    _wrapper->disableSpeakerphone();
}

#elif defined(__APPLE__) && TARGET_OS_OSX

#include "synthesis.h"
// Delegate to Synthesis class for process-based synthesis
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback),
      _synth(std::make_unique<Synthesis>(callback)) {}

WhillatsTTS::~WhillatsTTS() = default;

bool WhillatsTTS::start(bool) {
    return _synth->start();
}

void WhillatsTTS::stop() {
    _synth->stop();
}

int WhillatsTTS::getSampleRate() {
    return Synthesis::getSampleRate();
}

void WhillatsTTS::queueText(const char* text, const char* language) {
    _synth->queueText(std::string(text), std::string(language));
}

void WhillatsTTS::enableSpeakerphone() {
    // No-op
}

void WhillatsTTS::disableSpeakerphone() {
    // No-op
}
#endif // TTS_PLATFORMS

#if defined(__APPLE__) && !defined(WHILLATS_STYLETTS2)
// Provide a default no-arg start() on Apple that forwards to start(bool)
bool WhillatsTTS::start() {
    return start(true);
}
#endif

void WhillatsSetResponseCallback::OnResponseComplete(bool success, const char* response) {
    if (callback_) {
        callback_(success, response, user_data_);
    }
}

WhillatsTranscriber::WhillatsTranscriber(const char* model_path, 
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback language_callback) : 
    _callback(callback),
    _language_callback(language_callback) {
#if defined(WHILLATS_USE_SERVER)
    _useServer = !isServerProcess();
    if (_useServer) {
        if (model_path && model_path[0]) setenv("WHISPER_MODEL", model_path, 1);
    } else
#endif
    {
        _whisper_transcriber = std::make_unique<WhisperTranscriber>(model_path, callback, language_callback);
    }
}

WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback)
    : WhillatsTranscriber(model_path, callback, {nullptr, nullptr}) {}

WhillatsTranscriber::~WhillatsTranscriber() {}

void WhillatsTranscriber::processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize) {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _whisperClient) { _whisperClient->processAudioBuffer(playoutBuffer, playoutBufferSize); return; }
#endif
    if (_whisper_transcriber) _whisper_transcriber->processAudioBuffer(playoutBuffer, playoutBufferSize);
}

bool WhillatsTranscriber::start() {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer) {
        _conn = getOrCreateServer();
        if (!_conn) return false;
        _conn->setWhisperCallback(_callback);
        _conn->setLanguageCallback(_language_callback);
        _whisperClient = std::make_unique<WhillatsTranscriberClient>(*_conn);
        return _whisperClient->start();
    }
#endif
    return _whisper_transcriber ? _whisper_transcriber->start() : false;
}

void WhillatsTranscriber::stop() {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _whisperClient) { _whisperClient->stop(); return; }
#endif
    if (_whisper_transcriber) _whisper_transcriber->stop();
} 

std::string WhillatsTranscriber::getLanguage() const { return _language; }

void WhillatsTranscriber::setLanguage(const char* language) {
    if (language) _language = language;
}

void WhillatsTranscriber::setThreadCount(int n) {
    _threadCount = n;
    if (_whisper_transcriber) _whisper_transcriber->setThreadCount(n);
}

WhillatsLlama::WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback) 
    : _callback(callback) {
#if defined(WHILLATS_USE_SERVER)
    _useServer = !isServerProcess();
    if (_useServer) {
        if (model_path && model_path[0]) setenv("LLAMA_MODEL", model_path, 1);
    } else
#endif
    {
        _llama_device = std::make_unique<LlamaDeviceBase>(model_path, "", _callback);
    }
}

WhillatsLlama::WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback)
    : _callback(callback) {
#if defined(WHILLATS_USE_SERVER)
    _useServer = !isServerProcess();
    if (_useServer) {
        if (model_path && model_path[0]) setenv("LLAMA_MODEL", model_path, 1);
        if (mmproj_path && mmproj_path[0]) setenv("LLAMA_MMPROJ", mmproj_path, 1);
    } else
#endif
    {
        _llama_device = std::make_unique<LlamaDeviceBase>(model_path, mmproj_path, _callback);
    }
}

WhillatsLlama::~WhillatsLlama() {}

bool WhillatsLlama::start() {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer) {
        _conn = getOrCreateServer();
        if (!_conn) return false;
        _conn->setLlamaCallback(_callback);
        _llamaClient = std::make_unique<WhillatsLlamaClient>(*_conn);
        return _llamaClient->start();
    }
#endif
    return _llama_device ? _llama_device->start() : false;
}

bool WhillatsLlama::isRunning() const {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer) return _llamaClient != nullptr;
#endif
    return _llama_device && _llama_device->isRunning();
}

void WhillatsLlama::setThreadCount(int n) {
    if (_llama_device) _llama_device->setThreadCount(n);
}

void WhillatsLlama::stop() {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _llamaClient) { _llamaClient->stop(); return; }
#endif
    if (_llama_device) _llama_device->stop();
} 

void WhillatsLlama::askLlama(const char* prompt) {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _llamaClient) { _llamaClient->askLlama(prompt); return; }
#endif
    if (_llama_device) _llama_device->askLlama(prompt);
}

void WhillatsLlama::askWithImageFile(const char *prompt, const char *image_file, int width, int height) {
    if (!_llama_device) return;
    YUVData yuv;
    load_yuv(yuv, image_file, width, height);
    _llama_device->askWithImage(prompt, yuv);
}

void WhillatsLlama::askWithYUVRaw(
    const char* prompt, const uint8_t* y_plane, const uint8_t* u_plane,
    const uint8_t* v_plane, int width, int height, size_t y_size, size_t uv_size) {
  if (!prompt || !y_plane || !u_plane || !v_plane) return;
  YUVData data;
  data.width = width; data.height = height;
  data.y_size = y_size; data.uv_size = uv_size;
  data.y = std::make_unique<uint8_t[]>(y_size);
  std::memcpy(data.y.get(), y_plane, y_size);
  data.u = std::make_unique<uint8_t[]>(uv_size);
  std::memcpy(data.u.get(), u_plane, uv_size);
  data.v = std::make_unique<uint8_t[]>(uv_size);
  std::memcpy(data.v.get(), v_plane, uv_size);
  if (_llama_device) _llama_device->askWithImage(prompt, data);
}

void WhillatsLlama::receiveVideoFrame(const YUVData& yuv) {
#if defined(WHILLATS_USE_SERVER)
    if (_useServer && _llamaClient) { _llamaClient->receiveVideoFrame(yuv); return; }
#endif
    if (_llama_device) _llama_device->receiveVideoFrame(yuv);
}

bool WHILLATS_API save_yuv_as_bmp(const YUVData& yuv, const char* path) {
    clip_image_u8* img_clip = yuv_to_clip(yuv);
    save_clip_as_bmp(*img_clip, path); 
    free_clip(img_clip);
    return true;
}
