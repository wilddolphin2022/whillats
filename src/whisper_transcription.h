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
    WhisperTranscriber(const char* modelPath, WhillatsSetResponseCallback callback);
    ~WhisperTranscriber();

    void ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize);
    bool start();
    void stop();

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
    std::string _fullTranscription; // Accumulate text for current segment
    bool _segmentComplete;          // Flag to reset transcription
    std::string _model_path;

    std::vector<whisper_token> _pastTokens;
    int _nPast = 0;
    const int _maxContext = 224;

    std::thread _processingThread;
    bool _running;

    struct {
        float noise_level = 0.001f;
    } noise_profile;

    static const size_t kMinPhraseSamples = 32000;  // 200ms at 16kHz
    static const size_t kMaxPhraseSamples = 64000; // 1s at 16kHz

    static const size_t kRingBufferSizeIncrement = 60 * WHISPER_SAMPLE_RATE; 

    static const bool kDebug = false;
};

#endif