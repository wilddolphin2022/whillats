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
#include "whillats_client.h"
#include "whillats_ipc.h"
#include "whillats_utils.h"

// Thin client library: always uses whillats_server subprocess

// Shared server connection (lazy-initialized, one per process)
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

// ================================================================
// TTS — server-backed on Linux, platform-native on Apple
// ================================================================
#if defined(__APPLE__) && TARGET_OS_IPHONE
#import "whillats_synth.h"
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback), _wrapper(std::make_unique<WhillatsSpeechSynthesizerWrapper>()) {}
WhillatsTTS::~WhillatsTTS() { stop(); }
bool WhillatsTTS::start(bool ep) { _wrapper->initialize(&_callback, ep); return true; }
bool WhillatsTTS::start() { return start(true); }
void WhillatsTTS::stop() { _wrapper->stop(); }
int WhillatsTTS::getSampleRate() { return 16000; }
void WhillatsTTS::queueText(const char* t) { _wrapper->synthesize(std::string(t), "en"); }
void WhillatsTTS::queueText(const char* t, const char* l) { _wrapper->synthesize(std::string(t), std::string(l)); }
void WhillatsTTS::enableSpeakerphone() { _wrapper->enableSpeakerphone(); }
void WhillatsTTS::disableSpeakerphone() { _wrapper->disableSpeakerphone(); }
void WhillatsTTS::setThreadCount(int) {}

#elif defined(__APPLE__) && TARGET_OS_OSX
#include "synthesis.h"
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback), _synth(std::make_unique<Synthesis>(callback)) {}
WhillatsTTS::~WhillatsTTS() = default;
bool WhillatsTTS::start(bool) { return _synth->start(); }
bool WhillatsTTS::start() { return start(true); }
void WhillatsTTS::stop() { _synth->stop(); }
int WhillatsTTS::getSampleRate() { return Synthesis::getSampleRate(); }
void WhillatsTTS::queueText(const char* t) { _synth->queueText(std::string(t), "en"); }
void WhillatsTTS::queueText(const char* t, const char* l) { _synth->queueText(std::string(t), std::string(l)); }
void WhillatsTTS::enableSpeakerphone() {}
void WhillatsTTS::disableSpeakerphone() {}
void WhillatsTTS::setThreadCount(int) {}

#else
// Linux: always use whillats_server for TTS
WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback) : _callback(callback) {}
WhillatsTTS::~WhillatsTTS() {}
bool WhillatsTTS::start() {
    _conn = getOrCreateServer();
    if (!_conn) return false;
    _conn->setTtsCallback(_callback.callback_, _callback.user_data_);
    _ttsClient = std::make_unique<WhillatsTTSClient>(*_conn);
    return _ttsClient->start();
}
bool WhillatsTTS::start(bool) { return start(); }
void WhillatsTTS::stop() { if (_ttsClient) _ttsClient->stop(); }
void WhillatsTTS::queueText(const char* t) { if (_ttsClient) _ttsClient->queueText(t, "en"); }
void WhillatsTTS::queueText(const char* t, const char* l) { if (_ttsClient) _ttsClient->queueText(t, l); }
int WhillatsTTS::getSampleRate() { return 16000; }
void WhillatsTTS::setThreadCount(int) {}
void WhillatsTTS::enableSpeakerphone() {}
void WhillatsTTS::disableSpeakerphone() {}
#endif

// ================================================================
// Callback — out-of-line definition
// ================================================================
void WhillatsSetResponseCallback::OnResponseComplete(bool success, const char* response) {
    if (callback_) callback_(success, response, user_data_);
}

// ================================================================
// Whisper Transcriber — server-backed on Linux
// ================================================================
WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback language_callback)
    : _callback(callback), _language_callback(language_callback) {
    if (model_path && model_path[0]) setenv("WHISPER_MODEL", model_path, 1);
}

WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback)
    : WhillatsTranscriber(model_path, callback, {nullptr, nullptr}) {}

WhillatsTranscriber::~WhillatsTranscriber() {}

bool WhillatsTranscriber::start() {
    _conn = getOrCreateServer();
    if (!_conn) return false;
    _conn->setWhisperCallback(_callback.callback_, _callback.user_data_);
    _conn->setLanguageCallback(_language_callback.callback_, _language_callback.user_data_);
    _whisperClient = std::make_unique<WhillatsTranscriberClient>(*_conn);
    return _whisperClient->start();
}

void WhillatsTranscriber::stop() { if (_whisperClient) _whisperClient->stop(); }

void WhillatsTranscriber::processAudioBuffer(uint8_t* buf, const size_t size) {
    if (_whisperClient) _whisperClient->processAudioBuffer(buf, size);
}

std::string WhillatsTranscriber::getLanguage() const { return _language; }
void WhillatsTranscriber::setLanguage(const char* lang) { if (lang) _language = lang; }
void WhillatsTranscriber::setThreadCount(int n) { _threadCount = n; }

// ================================================================
// Llama — server-backed on Linux
// ================================================================
WhillatsLlama::WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback)
    : _callback(callback) {
    if (model_path && model_path[0]) setenv("LLAMA_MODEL", model_path, 1);
}

WhillatsLlama::WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback)
    : _callback(callback) {
    if (model_path && model_path[0]) setenv("LLAMA_MODEL", model_path, 1);
    if (mmproj_path && mmproj_path[0]) setenv("LLAMA_MMPROJ", mmproj_path, 1);
}

WhillatsLlama::~WhillatsLlama() {}

bool WhillatsLlama::start() {
    _conn = getOrCreateServer();
    if (!_conn) return false;
    _conn->setLlamaCallback(_callback.callback_, _callback.user_data_);
    _llamaClient = std::make_unique<WhillatsLlamaClient>(*_conn);
    return _llamaClient->start();
}

bool WhillatsLlama::isRunning() const { return _llamaClient != nullptr; }
void WhillatsLlama::setThreadCount(int) {}
void WhillatsLlama::stop() { if (_llamaClient) _llamaClient->stop(); }
void WhillatsLlama::askLlama(const char* prompt) { if (_llamaClient) _llamaClient->askLlama(prompt); }
void WhillatsLlama::askWithImageFile(const char*, const char*, int, int) {}
void WhillatsLlama::askWithYUVRaw(const char*, const uint8_t*, const uint8_t*,
    const uint8_t*, int, int, size_t, size_t) {}

void WhillatsLlama::receiveVideoFrame(const YUVData& yuv) {
    if (_llamaClient) _llamaClient->receiveVideoFrame(yuv);
}

bool WHILLATS_API save_yuv_as_bmp(const YUVData& yuv, const char* path) {
    (void)yuv; (void)path;
    return false;
}
