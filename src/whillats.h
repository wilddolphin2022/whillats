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
#include <cstdint>

// Change to C-style function pointer callbacks
typedef void (*ResponseCallback)(bool success, const char* response, void* user_data);
typedef void (*AudioCallback)(bool success, const uint16_t* buffer, size_t buffer_size, void* user_data);

class WHILLATS_API WhillatsSetResponseCallback {
public:
    WhillatsSetResponseCallback(ResponseCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}
    
    void OnResponseComplete(bool success, const char* response) {
        if (callback_) {
            callback_(success, response, user_data_);
        }
    }

private:
    ResponseCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsSetAudioCallback {
public:
    WhillatsSetAudioCallback(AudioCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}
    
    void OnBufferComplete(bool success, const std::vector<uint16_t>& buffer) {
        if (callback_) {
            callback_(success, buffer.data(), buffer.size(), user_data_);
        }
    }

private:
    AudioCallback callback_;
    void* user_data_;
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
    void queueText(const char* text);

    static int getSampleRate();

  private:
    WhillatsSetAudioCallback _callback;
    std::unique_ptr<ESpeakTTS> _espeak_tts; 
};

class WHILLATS_API WhillatsTranscriber {
  public:
    WhillatsTranscriber(const char* model_path, WhillatsSetResponseCallback callback);
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
    WhillatsLlama(const char*model_path, WhillatsSetResponseCallback callback);
    ~WhillatsLlama();

    bool start();
    void stop();
    void askLlama(const char* prompt);
  private:
    WhillatsSetResponseCallback _callback;
    std::unique_ptr<LlamaDeviceBase> _llama_device;
};

#endif // WHILLATS_H