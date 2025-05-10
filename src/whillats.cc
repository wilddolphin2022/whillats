/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
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

#include "whillats.h"
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "whillats_utils.h"

#if TTS_PLATFORMS
#include "espeak_tts.h"

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback),
      _espeak_tts(std::make_unique<ESpeakTTS>(callback)) {}

WhillatsTTS::~WhillatsTTS() {}

void WhillatsTTS::queueText(const char* text, const char* language) {
    _espeak_tts->queueText(std::string(text), std::string(language));
}

bool WhillatsTTS::start(bool) {
    return _espeak_tts->start();
}

void WhillatsTTS::stop() {
    _espeak_tts->stop();
}

int WhillatsTTS::getSampleRate() {
    return ESpeakTTS::getSampleRate();
}

void WhillatsTTS::enableSpeakerphone() {}

void WhillatsTTS::disableSpeakerphone() {}

#elif TARGET_OS_IOS
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

#elif TARGET_OS_OSX

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

WhillatsTranscriber::WhillatsTranscriber(const char* model_path, 
    WhillatsSetResponseCallback callback,
    WhillatsSetLanguageCallback language_callback) : 
    _callback(callback),
    _language_callback(language_callback),
    _whisper_transcriber(std::make_unique<WhisperTranscriber>(model_path, callback, language_callback)) {}

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

void WhillatsTranscriber::setLanguage(const std::string& language) { 
    _whisper_transcriber->setLanguage(language); 
}

void WhillatsTranscriber::setDetectLanguage(bool detectLanguage) { 
    _whisper_transcriber->setDetectLanguage(detectLanguage);
}

std::string WhillatsTranscriber::getLanguage() { 
    return _whisper_transcriber->getLanguage(); 
}

void WhillatsTranscriber::setVADThreshold(float threshold) { 
    _whisper_transcriber->setVADThreshold(threshold);
}

float WhillatsTranscriber::getVADThreshold() { 
    return _whisper_transcriber->getVADThreshold();
}

WhillatsLlama::WhillatsLlama(
    const char* model_path, 
    const char* mmproj_path,
    WhillatsSetResponseCallback callback) 
    : _callback(callback),
      _llama_device(std::make_unique<LlamaDeviceBase>(model_path, mmproj_path, callback)) {}

WhillatsLlama::~WhillatsLlama() {}

bool WhillatsLlama::start() {
    return _llama_device->start();
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