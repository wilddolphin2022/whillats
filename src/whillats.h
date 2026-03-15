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
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Callback function pointer types
typedef void (*ResponseCallback)(bool success, const char* response, void* user_data);
typedef void (*AudioCallback)(bool success, const uint16_t* buffer, size_t buffer_size, void* user_data);
typedef void (*LanguageCallback)(bool success, const char* language, void* user_data);

struct clip_image_u8 {
    int width;
    int height;
    uint8_t* data; // RGB, interleaved [R,G,B,R,G,B,...]
};

// Simple YUV frame container used by llama video ingest
struct YUVData {
    int width = 0;
    int height = 0;
    size_t y_size = 0;
    size_t uv_size = 0;
    std::unique_ptr<uint8_t[]> y;
    std::unique_ptr<uint8_t[]> u;
    std::unique_ptr<uint8_t[]> v;
};

class WhisperTranscriber;
class LlamaDeviceBase;
#if defined(WHILLATS_STYLETTS2)
class StyleTTS2TTS;
#else
class ESpeakTTS;
#endif
class WhillatsSpeechSynthesizerWrapper;
class Synthesis;

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

    void OnSynthesisComplete() {
        if (callback_) {
            // Call with success = false, empty buffer to signal completion
            callback_(false, nullptr, 0, user_data_);
        }
    }
private:
    AudioCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsSetLanguageCallback {
public:
    WhillatsSetLanguageCallback(LanguageCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    void OnLanguageChanged(bool success, const char* language) {
        if (callback_) {
            callback_(success, language, user_data_);
        }
    }

    // Backward compatibility shim
    void OnLanguageDetected(bool success, const std::string& language) {
        OnLanguageChanged(success, language.c_str());
    }

private:
    LanguageCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsTTS {
  public:
    WhillatsTTS(WhillatsSetAudioCallback callback);
    ~WhillatsTTS();

    bool start();
    bool start(bool withAudio);
    void stop();
    void queueText(const char* text);
    void queueText(const char* text, const char* language);
    void enableSpeakerphone();
    void disableSpeakerphone();

    static int getSampleRate();

  private:
    WhillatsSetAudioCallback _callback;
#if defined(WHILLATS_STYLETTS2)
    std::unique_ptr<StyleTTS2TTS> _styletts2;
#else
#if !defined(__APPLE__)
    std::unique_ptr<ESpeakTTS> _espeak_tts;
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
    std::unique_ptr<WhillatsSpeechSynthesizerWrapper> _wrapper;
#elif defined(__APPLE__) && TARGET_OS_OSX
    std::unique_ptr<Synthesis> _synth;
#endif
#endif
};

class WHILLATS_API WhillatsTranscriber {
  public:
    WhillatsTranscriber(const char* model_path, WhillatsSetResponseCallback callback);
    WhillatsTranscriber(const char* model_path, WhillatsSetResponseCallback callback, WhillatsSetLanguageCallback languageCallback);
    ~WhillatsTranscriber();

    bool start();
    void stop();

    void processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize);

    // Language control used by factory
    std::string getLanguage() const;
    void setLanguage(const char* language);

  private:
    WhillatsSetResponseCallback _callback; 
    WhillatsSetLanguageCallback _language_callback;
    std::unique_ptr<WhisperTranscriber> _whisper_transcriber; 
    std::string _language = "en";
};

class WHILLATS_API WhillatsLlama {
  public:
    WhillatsLlama(const char* model_path, WhillatsSetResponseCallback callback);
    WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback);
    ~WhillatsLlama();

    bool start();
    void stop();
    void askLlama(const char* prompt);
    void askWithImageFile(const char* prompt, const char* image_file, int width, int height);
    void askWithYUVRaw(
        const char* prompt,
        const uint8_t* y_plane,
        const uint8_t* u_plane,
        const uint8_t* v_plane,
        int width,
        int height,
        size_t y_size,
        size_t uv_size);

    // Accept a video frame for multimodal prompts (no-op on iOS)
    void receiveVideoFrame(const YUVData& yuv);

  private:
    WhillatsSetResponseCallback _callback;
    std::unique_ptr<LlamaDeviceBase> _llama_device;
};

// Helper functions
bool WHILLATS_API save_yuv_as_bmp(const YUVData& yuv, const char* path);

#endif // WHILLATS_H
