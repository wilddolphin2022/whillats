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

#ifndef WHILLATS_H
#define WHILLATS_H

#include "whillats_export.h"

#pragma once


class WHILLATS_API WhillatsSetResponseCallback {
    public:
    explicit WhillatsSetResponseCallback(
        std::function<void(bool, const std::string&)> on_complete)
        : on_complete_(on_complete) {}
    void OnResponseComplete(bool success, const std::string& response) {
        on_complete_(success, response);
    }

    private:
    std::function<void(bool, const std::string&)> on_complete_;
};

class WHILLATS_API WhillatsSetAudioCallback {
    public:
    explicit WhillatsSetAudioCallback(
        std::function<void(bool, const std::vector<uint16_t>&)> on_complete)
        : on_complete_(on_complete) {}
    void OnBufferComplete(bool success, const std::vector<uint16_t>& buffer) {
        on_complete_(success, buffer);
    }

    private:
    std::function<void(bool, const std::vector<uint16_t>& buffer)> on_complete_;
};

class ESpeakTTS;
class WhisperTranscriber;
class LlamaDeviceBase;

class WHILLATS_API WhillatsTTS {
  public:
    WhillatsTTS(WhillatsSetAudioCallback callback);
    ~WhillatsTTS();

    bool start();
    void stop();
    void queueText(const std::string& text);

    static int getSampleRate();

  private:
    WhillatsSetAudioCallback _callback;
    std::unique_ptr<ESpeakTTS> _espeak_tts; 
};

class WHILLATS_API WhillatsTranscriber {
  public:
    WhillatsTranscriber(const std::string& model_path, WhillatsSetResponseCallback callback);
    ~WhillatsTranscriber();

    bool start();
    void stop();
    void processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize);

  private:
    WhillatsSetResponseCallback _callback; 
    std::unique_ptr<WhisperTranscriber> _whisper_transcriber; 
};

class WHILLATS_API WhillatsLlama {
  public:
    WhillatsLlama(const std::string& model_path, WhillatsSetResponseCallback callback);
    ~WhillatsLlama();

    bool start();
    void stop();
    void askLlama(const std::string& prompt);
  private:
    WhillatsSetResponseCallback _callback;
    std::unique_ptr<LlamaDeviceBase> _llama_device;
};

#endif // WHILLATS_H