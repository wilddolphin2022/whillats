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

#include "whillats.h"
#include "silence_finder.h"

struct whisper_context;
class AudioRingBuffer;

class WhisperTranscriber {
 private:
  std::string _model_path;
  whisper_context* _whisperContext;
  std::unique_ptr<AudioRingBuffer> _audioBuffer; 

  std::thread _processingThread;
  std::atomic<bool> _running;
  std::atomic<bool> _processingActive;

  // Constants for audio processing
  static constexpr int kSampleRate = 16000;       // 16 kHz
  static constexpr int kChannels = 1;             // Mono
  static constexpr int kBufferDurationMs = 10;    // 10ms buffer
  static constexpr int kTargetDurationSeconds = 3; // 3-second segments for Whisper
  static constexpr int kRingBufferSizeIncrement = kSampleRate * kTargetDurationSeconds * 2 * 10; // 10-seconds increment for ring buffer size

  static constexpr size_t kTargetSamples = kSampleRate * 12 * 2; // 12 seconds of audio
  static constexpr size_t kSilenceSamples = 16000; // 1 second of silence at 16kHz

  // Accumulated buffer for Whisper processing
  std::vector<uint8_t> _accumulatedByteBuffer;
  std::atomic<size_t> _overflowCount;
  std::atomic<size_t> _ringBufferSize; // 10 segments buffer size

  bool TranscribeAudioNonBlocking(const std::vector<float>& pcmf32);
  bool RunProcessingThread();
  bool ValidateWhisperModel(const std::string& modelPath);
  bool InitializeWhisperModel(const std::string& modelPath);
  whisper_context* TryAlternativeInitMethods(const std::string& modelPath);

  // State to keep track if we're in a voice segment
  bool _inVoiceSegment = false;
  size_t _samplesSinceVoiceStart = 0;
  size_t _silentSamplesCount = 0; // New: Count of silent samples
  void handleOverflow();

  std::vector<int16_t> _processingBuffer;
  
    // Add new members for voice detection state
    struct VoiceDetectionState {
      float lastThreshold = 0.0f;
      size_t consecutiveVoiceFrames = 0;
      size_t consecutiveSilenceFrames = 0;
  } _voiceState;

  // Updated constants
  static constexpr float voiceStartThreshold = 0.12f;  // Higher threshold to start voice
  static constexpr float voiceEndThreshold = 0.08f;    // Lower threshold to end voice
  static constexpr size_t kMinVoiceFrames = 3;
  static constexpr size_t kMinSilenceFrames = 5;

  std::chrono::steady_clock::time_point _lastTranscriptionStart;
  std::chrono::steady_clock::time_point _lastTranscriptionEnd;

  WhillatsSetResponseCallback _responseCallback;
 public:

  WhisperTranscriber(
      const char* model_path,
      WhillatsSetResponseCallback callback);
  
  ~WhisperTranscriber();

  void ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize);

  bool start();
  void stop();
};