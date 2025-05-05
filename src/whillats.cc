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

#include "whillats.h"
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "whillats_utils.h"

#if TTS_PLATFORMS
#include "espeak_tts.h"

// Temporary fix for non-iOS builds
// #ifndef TARGET_OS_OSX
// #pragma message("PLATFORM_DARWIN is not defined")
// ESpeakTTS::ESpeakTTS(WhillatsSetAudioCallback callback) : _callback(callback) {}
// ESpeakTTS::~ESpeakTTS() {}
// const int ESpeakTTS::getSampleRate() { return 16000; }
// bool ESpeakTTS::start() { return false; }
// void ESpeakTTS::stop() { }
// void ESpeakTTS::queueText(const std::string& text, const std::string& language) { }
// #endif

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback),
      _espeak_tts(std::make_unique<ESpeakTTS>(callback)) {}

WhillatsTTS::~WhillatsTTS() {}

void WhillatsTTS::queueText(const char* text, const char* language) {
    _espeak_tts->queueText(std::string(text), std::string(language));
}

bool WhillatsTTS::start() {
    return _espeak_tts->start();
}

void WhillatsTTS::stop() {
    _espeak_tts->stop();
}

int WhillatsTTS::getSampleRate() {
    return ESpeakTTS::getSampleRate();
}

void WhillatsTTS::enableSpeakerphone() {
}

void WhillatsTTS::disableSpeakerphone() {
}

#else // !TTS_PLATFORMS

#include "whillats_synth.h"

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback) : 
    _callback(callback),
    _speech_synthesizer(std::make_unique<WhillatsSpeechSynthesizerWrapper>()) { }

WhillatsTTS::~WhillatsTTS() { 
    _speech_synthesizer.reset();
}

bool WhillatsTTS::start() { 
    _speech_synthesizer->initialize(&_callback);

    return true; 
}
void WhillatsTTS::stop() { 
    _speech_synthesizer->stop();
}
int WhillatsTTS::getSampleRate() { 
    return 16000; 
}

void WhillatsTTS::queueText(const char* text, const char* language) { 
    _speech_synthesizer->synthesize(text, language);
}

void WhillatsTTS::enableSpeakerphone() {
#if TARGET_OS_IOS
    _speech_synthesizer->enableSpeakerphone();
#endif
}

void WhillatsTTS::disableSpeakerphone() {
#if TARGET_OS_IOS
    _speech_synthesizer->disableSpeakerphone();
#endif
}   

#endif // !TTS_PLATFORMS

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

void WhillatsLlama::askWithImage(const char *prompt, const YUVData& yuv) {
    _llama_device->askWithImage(prompt, yuv);
}

void WhillatsLlama::askWithImageFile(const char *prompt, const char *image_file, int width, int height) {
    YUVData yuv;
    load_yuv(yuv, image_file, width, height);
    _llama_device->askWithImage(prompt, yuv);
}