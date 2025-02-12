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

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>
#include <complex>
#include <cmath>

// Add M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "whillats.h"
#include "silence_finder.h"
#include "whisper_helpers.h"

struct whisper_context;

class WhisperTranscriber {
 private:
  std::string _model_path;
  whisper_context* _whisperContext;

  std::thread _processingThread;
  std::atomic<bool> _running;
  std::atomic<bool> _processingActive;

  // Constants for audio processing
  static constexpr int kSampleRate = 16000;       // 16 kHz
  static constexpr int kChannels = 1;             // Mono
  static constexpr int kBufferDurationMs = 10;    // 10ms buffer
  static constexpr int kTargetDurationSeconds = 3; // 3-second segments for Whisper
  static constexpr int kRingBufferSizeIncrement = kSampleRate * kTargetDurationSeconds * 10; // in samples

  static constexpr size_t kTargetSamples = kSampleRate * 12;  // 12 seconds (in samples)
  static constexpr size_t kSilenceSamples = 16000; // 1 second of silence at 16kHz

  // Replace vector of chunks with ring buffer
  std::unique_ptr<AudioRingBuffer<float>> _audioBuffer;
  std::mutex _audioMutex;

  // State to keep track if we're in a voice segment
  bool _inVoiceSegment = false;
  size_t _samplesSinceVoiceStart = 0;
  size_t _silentSamplesCount = 0; // New: Count of silent samples
  void handleOverflow();

  std::vector<int16_t> _processingBuffer;
  
  // Updated constants
  static constexpr float voiceStartThreshold = 0.12f;  // Higher threshold to start voice
  static constexpr float voiceEndThreshold = 0.08f;    // Lower threshold to end voice
  static constexpr size_t kMinVoiceFrames = 3;
  static constexpr size_t kMinSilenceFrames = 50;

  std::chrono::steady_clock::time_point _lastTranscriptionStart;
  std::chrono::steady_clock::time_point _lastTranscriptionEnd;

  WhillatsSetResponseCallback _responseCallback;

  static constexpr size_t kPrerollBufferSize = 1600;  // 100ms at 16kHz (in samples)
  std::vector<float> _prerollBuffer;              // Changed from uint8_t to float

  // Add FFT helper function declaration
  static void fft_forward(std::vector<std::complex<float>>& data, int n);

  // Existing VAD helper function declaration
  static bool vad_simple(const std::vector<float>& pcmf32,
                        int sample_rate,
                        int last_ms,
                        float vad_thold,
                        float freq_thold,
                        bool verbose);

  bool InitializeWhisperModel(const std::string& modelPath);
  whisper_context* TryAlternativeInitMethods(const std::string& modelPath);
  bool ValidateWhisperModel(const std::string& modelPath);
  bool TranscribeAudioNonBlocking(const std::vector<float>& pcmf32);
  bool RunProcessingThread();

 public:

  WhisperTranscriber(
      const char* model_path,
      WhillatsSetResponseCallback callback);
  
  ~WhisperTranscriber();

  void ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize);

  bool start();
  void stop();
};