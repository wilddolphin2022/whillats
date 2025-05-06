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

// Include TargetConditionals for TARGET_OS_IOS macro
#if defined(__APPLE__)
    #include <TargetConditionals.h>
    // Undefine toupper/tolower macros from espeak compat to avoid conflicts in STL headers
#ifdef toupper
#undef toupper
#endif
#ifdef tolower
#undef tolower
#endif
    // Exclude TTS (espeak-ng) for iOS builds
    #if  TARGET_OS_IOS || TARGET_OS_OSX
        #define TTS_PLATFORMS 0 // Building for iOS
    #else
        #define TTS_PLATFORMS 1 // Building for macOS or other non-iOS platforms
    #endif
#else
    #define TTS_PLATFORMS 1 // Building for other platforms
#endif

#if defined(_MSC_VER)
    #define WHILLATS_EXPORT __declspec(dllexport)
    #define WHILLATS_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    #define WHILLATS_EXPORT __attribute__((visibility("default")))
    #define WHILLATS_IMPORT __attribute__((visibility("default")))
#else
    #define WHILLATS_EXPORT
    #define WHILLATS_IMPORT
#endif

#ifdef WHILLATS_BUILDING_DLL
    #define WHILLATS_API WHILLATS_EXPORT
#else
    #define WHILLATS_API WHILLATS_IMPORT
#endif

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>

// Change to C-style function pointer callbacks
typedef void (*ResponseCallback)(bool success, const char* response, void* user_data);
typedef void (*AudioCallback)(bool success, const uint16_t* buffer, size_t buffer_size, void* user_data);

struct clip_image_u8 {
    int width;
    int height;
    uint8_t* data; // RGB, interleaved [R,G,B,R,G,B,...]
};

struct YUVData {
    std::unique_ptr<uint8_t[]> y;
    std::unique_ptr<uint8_t[]> u;
    std::unique_ptr<uint8_t[]> v;
    int width;
    int height;
    size_t y_size;
    size_t uv_size;
};

class WhisperTranscriber;
class LlamaDeviceBase;
class ESpeakTTS;
class WhillatsSpeechSynthesizerWrapper;

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
    WhillatsSetLanguageCallback(ResponseCallback callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}
    
    void OnLanguageDetected(bool success, const std::string& language) {
        if (callback_) {
            callback_(success, language.c_str(), user_data_);
        }
    }

private:
    ResponseCallback callback_;
    void* user_data_;
};

class WHILLATS_API WhillatsTTS {
  public:
    WhillatsTTS(WhillatsSetAudioCallback callback);
    ~WhillatsTTS();

    bool start();
    void stop();
    void queueText(const char* text, const char* language);
    void enableSpeakerphone();
    void disableSpeakerphone();

    static int getSampleRate();

  private:
    WhillatsSetAudioCallback _callback;
#if TTS_PLATFORMS
    std::unique_ptr<ESpeakTTS> _espeak_tts;
#else    
    std::unique_ptr<WhillatsSpeechSynthesizerWrapper> _speech_synthesizer;
public:
    void setNotificationName(const char* name);
#endif
};

class WHILLATS_API WhillatsTranscriber {
  public:
    WhillatsTranscriber(const char* model_path, 
        WhillatsSetResponseCallback callback,
        WhillatsSetLanguageCallback language_callback);

    ~WhillatsTranscriber();

    bool start();
    void stop();

    void processAudioBuffer(uint8_t* playoutBuffer, const size_t playoutBufferSize);

    void setLanguage(const std::string& language);
    void setDetectLanguage(bool detectLanguage);
    std::string getLanguage();
    void setVADThreshold(float threshold);
    float getVADThreshold();

  private:
    WhillatsSetResponseCallback _callback; 
    WhillatsSetLanguageCallback _language_callback;
    std::unique_ptr<WhisperTranscriber> _whisper_transcriber; 
};

class WHILLATS_API WhillatsLlama {
  public:
    WhillatsLlama(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback);
    ~WhillatsLlama();

    bool start();
    void stop();
    void askLlama(const char* prompt);
    void askWithImage(const char *prompt, const YUVData& yuv);
    void askWithImageFile(const char *prompt, const char *image_file, int width, int height);
  private:
    WhillatsSetResponseCallback _callback;
    std::unique_ptr<LlamaDeviceBase> _llama_device;
};

#endif // WHILLATS_H
