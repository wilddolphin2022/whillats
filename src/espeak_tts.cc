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

#include <thread>

#include "whillats.h"
#include "espeak_tts.h"

static constexpr int kSampleRate = 16000;       // 16 kHz
static constexpr int kChannels = 1;             // Mono
static constexpr int kBufferDurationMs = 10;    // 10ms buffer
static constexpr int kTargetDurationSeconds = 3; // 3-second segments for Whisper
static constexpr int kRingBufferSizeIncrement = kSampleRate * kTargetDurationSeconds * 2; // 3-seconds of 16-bit samples

ESpeakTTS::ESpeakTTS(WhillatsSetAudioCallback callback)
    : _callback(callback),
      last_read_time_(std::chrono::steady_clock::now()),
      ring_buffer_(new AudioRingBuffer(kRingBufferSizeIncrement)) {  // Size in bytes for 16-bit samples 
    espeak_AUDIO_OUTPUT output = AUDIO_OUTPUT_SYNCHRONOUS;
    int Buflength = 500;
    const char* path = NULL;
    int Options = 0;
    char Voice[] = {"English"};

    if (espeak_Initialize(output, Buflength, path, Options) == EE_INTERNAL_ERROR) {
        LOG_E("ESpeakTTS initialization failed!");
        return;
    }

    espeak_SetVoiceByName(Voice);
    const char* langNativeString = "en";
    espeak_VOICE voice;
    memset(&voice, 0, sizeof(espeak_VOICE));
    voice.languages = langNativeString;
    voice.name = "US";
    voice.variant = 1;
    voice.gender = 1;
    espeak_SetVoiceByProperties(&voice);

    espeak_SetParameter(espeakRATE, 180, 0);
    espeak_SetParameter(espeakVOLUME, 75, 0);
    espeak_SetParameter(espeakPITCH, 150, 0);
    espeak_SetParameter(espeakRANGE, 100, 0);
    espeak_SetParameter((espeak_PARAMETER)11, 0, 0);

    espeak_SetSynthCallback(&ESpeakTTS::internalSynthCallback);
}

void ESpeakTTS::synthesize(const char* text) {
    if (!text) return;
    
    // Clear output buffer
    _buffer.clear();

    // Pre-allocate buffer with estimated size
    size_t estimated_samples = strlen(text) * 200;  // More generous estimate
    _buffer.reserve(estimated_samples);

    unsigned int position = 0, end_position = 0, flags = espeakCHARS_AUTO;

    // Process entire text at once
    espeak_ERROR result = espeak_Synth(text, strlen(text) + 1,
                                     position, POS_CHARACTER, 
                                     end_position, flags, NULL, 
                                     reinterpret_cast<void*>(this));
    
    if (result != EE_OK) {
        LOG_E("Synthesis failed with error: " << result);
        return;
    }

    result = espeak_Synchronize();
    if (result != EE_OK) {
        LOG_E("Synchronization failed with error: " << result);
        return;
    }

    // Read all available samples from the ring buffer
    size_t bytes_available = ring_buffer_->availableToRead();
    if (bytes_available > 0) {
        // Convert bytes to samples (2 bytes per sample)
        size_t samples_available = bytes_available / sizeof(uint16_t);
        _buffer.resize(samples_available);
        
        // Read the data
        if (!ring_buffer_->read(reinterpret_cast<uint8_t*>(_buffer.data()), bytes_available)) {
            LOG_E("Failed to read from ring buffer");
            _buffer.clear();
            return;
        }
        LOG_V("Read " << samples_available << " samples from ring buffer");
    }
}

int ESpeakTTS::internalSynthCallback(short* wav, int numsamples, espeak_EVENT* events) {
    if (wav == nullptr || numsamples <= 0) {
        return 1;  // Signal error
    }

    ESpeakTTS* context = static_cast<ESpeakTTS*>(events->user_data);
    if (!context) return 1;  // Signal error

    // Calculate size in bytes
    size_t bytes_to_write = numsamples * sizeof(short);
    
    // Write samples to ring buffer
    if (!context->ring_buffer_->write(reinterpret_cast<uint8_t*>(wav), bytes_to_write)) {
        LOG_E("Failed to write to ring buffer");
        return 1;  // Signal error
    }
    
    return 0;  // Success
}

const int ESpeakTTS::getSampleRate() {
    return kSampleRate;
}

bool ESpeakTTS::start() {
    if (!_running) {
        _running = true;
        _processingThread = std::thread([this] {
            while (_running && RunProcessingThread()) {
            }
        });
        return true;
    }
    return false;
}

void ESpeakTTS::stop() {
    if (_running) {
        _running = false;
        _queueCondition.notify_all();
        
        if (_processingThread.joinable()) {
            _processingThread.join();
        }
    }
}

void ESpeakTTS::queueText(const std::string& text) {
    if (!text.empty()) {
        {
            std::lock_guard<std::mutex> lock(_queueMutex);
            _textQueue.push(text);
        }
        _queueCondition.notify_one();
    }
}

bool ESpeakTTS::RunProcessingThread() {
    std::string textToSynth;
    bool shouldSynth = false;

    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (_queueCondition.wait_for(lock, std::chrono::milliseconds(100), 
            [this] { return !_textQueue.empty() || !_running; })) {
            
            if (!_running) return false;
            
            if (!_textQueue.empty()) {
                textToSynth = _textQueue.front();
                _textQueue.pop();
                shouldSynth = true;
            }
        }
    }

    if (shouldSynth) {
        std::vector<uint16_t> buffer;
        synthesize(textToSynth.c_str());
        
        if (!_buffer.empty()) {
            _callback.OnBufferComplete(true, _buffer);
        }
    }

    return true;
}

ESpeakTTS::~ESpeakTTS() {
    stop();
    espeak_Terminate();
}