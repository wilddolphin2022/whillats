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

#include "silence_finder.h"
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "espeak_tts.h"

WhillatsTTS::WhillatsTTS(WhillatsSetAudioCallback callback)
    : _callback(callback),
      _espeak_tts(std::make_unique<ESpeakTTS>(callback)) {}

WhillatsTTS::~WhillatsTTS() {}

void WhillatsTTS::queueText(const std::string& text) {
    _espeak_tts->queueText(text);
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

WhillatsTranscriber::WhillatsTranscriber(const std::string& model_path, WhillatsSetResponseCallback callback) 
    : _callback(callback),
      _whisper_transcriber(std::make_unique<WhisperTranscriber>(model_path, callback)) {}

WhillatsTranscriber::~WhillatsTranscriber() {}

void WhillatsTranscriber::processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize) {
    _whisper_transcriber->ProcessAudioBuffer(playoutBuffer, playoutBufferSize);
}

bool WhillatsTranscriber::start() {
    return _whisper_transcriber->start();
}

void WhillatsTranscriber::stop() {
    _whisper_transcriber->stop();
} 

WhillatsLlama::WhillatsLlama(const std::string& model_path, WhillatsSetResponseCallback callback) 
    : _callback(callback),
      _llama_device(std::make_unique<LlamaDeviceBase>(model_path, callback)) {}

WhillatsLlama::~WhillatsLlama() {}

bool WhillatsLlama::start() {
    return _llama_device->start();
}

void WhillatsLlama::stop() {
    _llama_device->stop();
} 

void WhillatsLlama::askLlama(const std::string& prompt) {
    _llama_device->askLlama(prompt);
}