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

#ifndef WHISPER_TRANSCRIPTION_H
#define WHISPER_TRANSCRIPTION_H

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <complex>
#include <atomic>
#include <memory>

#include <whisper.h>
#include "whisper_helpers.h"
#include "whillats.h"

class WhisperTranscriber {
public:
    WhisperTranscriber(const char* modelPath, 
        WhillatsSetResponseCallback callback,
        WhillatsSetLanguageCallback languageCallback);
    ~WhisperTranscriber();

    bool start();
    void stop();

    void processAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize);

    void setLanguage(const std::string& language) { _language = language; }
    void setDetectLanguage(bool detectLanguage) { _detectLanguage = detectLanguage; }
    std::string getLanguage() { return _language; }

    void setVADThreshold(float threshold) { kVADThreshold = threshold; }
    float getVADThreshold() { return kVADThreshold; }
    void setThreadCount(int n) { _nThreads = n; }

private:
    bool InitializeWhisperModel(const std::string& modelPath);
    bool TranscribeAudioNonBlocking(const std::vector<float>& samples);
    void ProcessTokens(const std::vector<whisper_token>& tokens);
    bool RunProcessingThread();
    void ProcessRemainingAudio();
    bool ValidateWhisperModel(const std::string& modelPath);
    whisper_context* TryAlternativeInitMethods(const std::string& modelPath);
    bool responseValidate(std::string& text);
    void fft_forward(std::vector<std::complex<float>>& data, int n);
    bool vad_simple(const std::vector<float>& pcmf32, int sample_rate, int last_ms,
                    float vad_thold, float freq_thold, bool verbose);

    std::unique_ptr<AudioRingBuffer<float>> _audioBuffer;
    whisper_context* _ctx;
    whisper_state* _state;
    std::mutex _state_mutex;
    WhillatsSetResponseCallback _responseCallback;
    WhillatsSetLanguageCallback _languageCallback;
    std::string _fullTranscription; // Accumulate text for current segment
    bool _segmentComplete;          // Flag to reset transcription
    std::string _model_path;
    std::string _language = "auto";
    bool _detectLanguage = false;

    std::vector<whisper_token> _pastTokens;
    int _nPast = 0;
    const int _maxContext = 224;

    std::thread _processingThread;
    bool _running;

    // Guards _processingThread and _running to prevent races in start/stop
    mutable std::mutex _threadMutex;

    struct {
        float noise_level = 0.001f;
    } noise_profile;

    float kVADThreshold = 0.75;
    int _nThreads = 0; // 0 = auto

    static const size_t kMinPhraseSamples = 32000;  // 2s at 16kHz — VAD trigger threshold
    static const size_t kMaxPhraseSamples = 160000; // 10s at 16kHz — max context for language detection

    static const size_t kRingBufferSizeIncrement = 60 * WHISPER_SAMPLE_RATE; 


    static const bool kDebug = false;
};

#endif