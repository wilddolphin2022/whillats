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
#if defined(__APPLE__) && TARGET_OS_IPHONE
// iOS stub for WhisperTranscriber to satisfy symbols when whisper is excluded
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

#if defined(WHILLATS_STYLETTS2)
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

bool WhillatsTTS::start(bool /*withAudio*/) {
    return start();
}

void WhillatsTTS::stop() {
    if (_styletts2) _styletts2->stop();
}

void WhillatsTTS::setThreadCount(int n) {
    if (_styletts2) _styletts2->setThreadCount(n);
}

int WhillatsTTS::getSampleRate() {
    return StyleTTS2TTS::getSampleRate();
}

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

WhillatsTranscriber::WhillatsTranscriber(const char* model_path, 
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback language_callback) : 
    _callback(callback),
    _language_callback(language_callback),
    _whisper_transcriber(std::make_unique<WhisperTranscriber>(model_path, callback, language_callback)) {}

// Convenience overload that defaults language callback to null
WhillatsTranscriber::WhillatsTranscriber(const char* model_path,
    WhillatsSetResponseCallback callback)
    : WhillatsTranscriber(model_path, callback, {nullptr, nullptr}) {}

WhillatsTranscriber::~WhillatsTranscriber() {}

void WhillatsTranscriber::processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize) {
    _whisper_transcriber->processAudioBuffer(playoutBuffer, playoutBufferSize);
}

bool WhillatsTranscriber::start() {
    return _whisper_transcriber->start();
}

void WhillatsTranscriber::stop() {
    _whisper_transcriber->stop();
} 

std::string WhillatsTranscriber::getLanguage() const {
    return _language;
}

void WhillatsTranscriber::setLanguage(const char* language) {
    if (language) {
        _language = language;
    }
}

void WhillatsTranscriber::setThreadCount(int n) {
    _threadCount = n;
    if (_whisper_transcriber) _whisper_transcriber->setThreadCount(n);
}

WhillatsLlama::WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback) 
    : _callback(callback),
      _llama_device(std::make_unique<LlamaDeviceBase>(model_path, "", _callback)) {}

WhillatsLlama::WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback)
    : _callback(callback),
      _llama_device(std::make_unique<LlamaDeviceBase>(model_path, mmproj_path, _callback)) {}

WhillatsLlama::~WhillatsLlama() {}

bool WhillatsLlama::start() {
    return _llama_device->start();
}

bool WhillatsLlama::isRunning() const {
    return _llama_device && _llama_device->isRunning();
}

void WhillatsLlama::setThreadCount(int n) {
    if (_llama_device) _llama_device->setThreadCount(n);
}

void WhillatsLlama::stop() {
    _llama_device->stop();
} 

void WhillatsLlama::askLlama(const char* prompt) {
    _llama_device->askLlama(prompt);
}

void WhillatsLlama::askWithImageFile(const char *prompt, const char *image_file, int width, int height) {
    YUVData yuv;
    load_yuv(yuv, image_file, width, height);
    _llama_device->askWithImage(prompt, yuv);
}

// Member method to send raw YUV planes to the llama device
void WhillatsLlama::askWithYUVRaw(
    const char* prompt,
    const uint8_t* y_plane,
    const uint8_t* u_plane,
    const uint8_t* v_plane,
    int width,
    int height,
    size_t y_size,
    size_t uv_size) {
  if (!prompt || !y_plane || !u_plane || !v_plane) return;
  // Deep-copy the planes into YUVData
  YUVData data;
  data.width = width;
  data.height = height;
  data.y_size = y_size;
  data.uv_size = uv_size;
  data.y = std::make_unique<uint8_t[]>(y_size);
  std::memcpy(data.y.get(), y_plane, y_size);
  data.u = std::make_unique<uint8_t[]>(uv_size);
  std::memcpy(data.u.get(), u_plane, uv_size);
  data.v = std::make_unique<uint8_t[]>(uv_size);
  std::memcpy(data.v.get(), v_plane, uv_size);

  // Forward to the underlying device
  _llama_device->askWithImage(prompt, data);
}

void WhillatsLlama::receiveVideoFrame(const YUVData& yuv) {
    _llama_device->receiveVideoFrame(yuv);
}

bool WHILLATS_API save_yuv_as_bmp(const YUVData& yuv, const char* path) {
    clip_image_u8* img_clip = yuv_to_clip(yuv);
    save_clip_as_bmp(*img_clip, path); 
    free_clip(img_clip);
    return true;
}
