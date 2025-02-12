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
      _audioBuffer(new AudioRingBuffer<uint16_t>(kRingBufferSizeIncrement)) {   
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
    
    // Clear output buffer and ring buffer
    _buffer.clear();
    _audioBuffer->clear();  // Clear ring buffer before new synthesis

    // Process entire text at once
    espeak_ERROR result = espeak_Synth(text, strlen(text) + 1,
                                     0, POS_CHARACTER, 
                                     0, espeakCHARS_AUTO, NULL, 
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

    // Read all available samples from the ring buffer in order
    const size_t chunk_size = kSampleRate / 10;  // 100ms chunks
    std::vector<uint16_t> temp_buffer(chunk_size);

    while (true) {
        size_t samples_available = _audioBuffer->availableToRead();
        if (samples_available == 0) {
            break;  // No more data to read
        }

        // Read in small chunks to maintain order
        size_t samples_to_read = std::min(samples_available, chunk_size);
        if (!_audioBuffer->read(temp_buffer.data(), samples_to_read)) {
            LOG_E("Failed to read from ring buffer");
            break;
        }

        // Append to main buffer
        _buffer.insert(_buffer.end(), temp_buffer.begin(), temp_buffer.begin() + samples_to_read);
    }

    LOG_I("Total synthesized samples: " << _buffer.size());
}

int ESpeakTTS::internalSynthCallback(short* wav, int numsamples, espeak_EVENT* events) {
    if (!events || !events->user_data) {
        LOG_W("Invalid event data in callback");
        return 1;
    }

    ESpeakTTS* context = static_cast<ESpeakTTS*>(events->user_data);

    // Handle end of synthesis
    if (wav == nullptr || numsamples <= 0) {
        LOG_V("End of synthesis marker received");
        return 0;  // Success - this is normal end of synthesis
    }

    // Cast short* to uint16_t* since they're the same size and represent the same data
    const uint16_t* samples = reinterpret_cast<const uint16_t*>(wav);
    
    // Write samples directly to ring buffer
    if (!context->_audioBuffer->write(samples, numsamples)) {
        LOG_E("Failed to write to ring buffer");
        return 1;  // Signal error
    }
    
    LOG_V("Successfully wrote " << numsamples << " samples to ring buffer");
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
        // Clear any previous data
        _buffer.clear();
        
        // Synthesize the text
        synthesize(textToSynth.c_str());
        
        // Only send callback if we have data
        if (!_buffer.empty()) {
            LOG_V("Sending " << _buffer.size() << " samples to callback");
            _callback.OnBufferComplete(true, _buffer);
        } else {
            LOG_W("No audio data generated for text: " << textToSynth);
        }
    }

    return true;
}

ESpeakTTS::~ESpeakTTS() {
    stop();
    espeak_Terminate();
}