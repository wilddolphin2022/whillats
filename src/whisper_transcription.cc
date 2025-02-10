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

#include <iostream>
#include <regex>

#include <whisper.h>
#include "whisper_transcription.h"
#include "whisper_helpers.h"

WhisperTranscriber::WhisperTranscriber(
    const char* model_path,
    WhillatsSetResponseCallback callback) 
    : _model_path(model_path),
      _responseCallback(callback),
      _whisperContext(nullptr),
      _audioBuffer(new AudioRingBuffer(kRingBufferSizeIncrement)),
      _running(false),
      _processingActive(false),
      _overflowCount(0),
      _ringBufferSize(kRingBufferSizeIncrement)
{
    // Reserve space for audio buffer
    _accumulatedByteBuffer.reserve(kSampleRate * kTargetDurationSeconds * 2); // 16-bit samples
 
    // Initialize Whisper context
    if (!InitializeWhisperModel(_model_path) || !_whisperContext) {
        LOG_E("Failed to initialize Whisper model");
        _whisperContext = TryAlternativeInitMethods(_model_path);
        if (!_whisperContext) {
            LOG_E("Failed to initialize Whisper model alternative ways");
        }
    }
}

WhisperTranscriber::~WhisperTranscriber() {
    stop();
    if (_whisperContext) {
        whisper_free(_whisperContext);
    }
}

bool WhisperTranscriber::TranscribeAudioNonBlocking(const std::vector<float>& pcmf32) {
    // Prevent multiple simultaneous processing attempts
    if (_processingActive.exchange(true)) {
        LOG_E("Whisper transcription already in progress");
        return false;
    }

    // Validate context
    if (!_whisperContext) {
        LOG_E("Whisper context is null during transcription");
        _responseCallback.OnResponseComplete(false, "Whisper context is null");
        _processingActive = false;
        return false;
    }

    // Validate input
    if (pcmf32.empty()) {
        LOG_E("Empty audio buffer for transcription");
        _responseCallback.OnResponseComplete(false, "Empty audio buffer");
        _processingActive = false;
        return false;
    }

    // Input size validation
    if (pcmf32.size() < kSampleRate || pcmf32.size() > kTargetSamples) {
        LOG_E("Unexpected audio input size: " << pcmf32.size());
        _processingActive = false;
        return false;
    }

    // Input data validation without exceptions
    bool validInput = true;
    float sum = 0.0f, squaredSum = 0.0f;
    float minVal = pcmf32[0], maxVal = pcmf32[0];

    for (size_t i = 0; i < pcmf32.size(); ++i) {
        float sample = pcmf32[i];
        
        // Check for NaN or infinite values
        if (!(sample == sample) || std::abs(sample) > 1.0f) {
            LOG_E("Invalid sample at index " << i << ": " << sample);
            validInput = false;
            break;
        }

        sum += sample;
        squaredSum += sample * sample;
        minVal = std::min(minVal, sample);
        maxVal = std::max(maxVal, sample);
    }

    if (!validInput) {
        _processingActive = false;
        return false;
    }

    // Compute statistics
    float mean = sum / pcmf32.size();
    float variance = (squaredSum / pcmf32.size()) - (mean * mean);

    LOG_V("Audio Input Analysis:"
                    << " Samples=" << pcmf32.size()
                    << " Mean=" << mean
                    << " Variance=" << variance
                    << " Min=" << minVal
                    << " Max=" << maxVal
                    );

    // Prepare Whisper parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.language = "en";
    wparams.translate = false;
    wparams.n_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
    
    wparams.n_max_text_ctx = 64;
 
    // Diagnostic logging before transcription
    LOG_V("Preparing Whisper Transcription:"
                    << " Threads=" << wparams.n_threads
                    << " Max Text Context=" << wparams.n_max_text_ctx
                    );

    if (!_whisperContext) {
        LOG_E("Failed to initialize Whisper model!");
        return false;
    }

    int result = 0;
    // Attempt transcription
    result = whisper_full(
        _whisperContext,
        wparams, 
        pcmf32.data(), 
        pcmf32.size()
        );

    // Process results
    if (result == 0) {
        int numSegments = whisper_full_n_segments(_whisperContext);
        LOG_V("Transcription completed. Segments: " << numSegments);

        // Collect and log segments
        std::string fullTranscription;
        for (int i = 0; i < numSegments; ++i) {
            const char* text = whisper_full_get_segment_text(_whisperContext, i);
            if (text && strlen(text) > 0) {
                // Add proper spacing between segments
                if (!fullTranscription.empty()) {
                    fullTranscription += " ";
                }
                fullTranscription += std::string(text);
                LOG_V("Segment " << i << ": " << text);
            }
        }

        // Remove double spaces that might have been introduced
        fullTranscription = std::regex_replace(fullTranscription, std::regex("\\s+"), " ");

        _lastTranscriptionEnd = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                _lastTranscriptionEnd - _lastTranscriptionStart).count();
        LOG_I("'" << fullTranscription << "' time taken: " << duration << " ms");

        if (!fullTranscription.empty()) {
            // Remove text within brackets and the brackets themselves
            std::string cleanTranscription = std::regex_replace(fullTranscription, 
                std::regex("\\[.*?\\]|\\(.*?\\)|\\{.*?\\}"), "");
            
            if(!cleanTranscription.empty()) {
                _responseCallback.OnResponseComplete(true, cleanTranscription.c_str());
            } else {
                _responseCallback.OnResponseComplete(false, "No text transcribed");
            }
        }      
    } else {
        LOG_E("Whisper transcription failed. Error code: " << result);
        _responseCallback.OnResponseComplete(false, "Transcription failed with error");
    }

    // Reset processing flag
    _processingActive = false;

    return result == 0;
}

bool WhisperTranscriber::RunProcessingThread() {
    std::vector<uint8_t> audioBuffer;
    
    while (_running && _audioBuffer->availableToRead() > 0) {

        // Ensure audioBuffer is sized correctly to receive data
        audioBuffer.resize(_audioBuffer->availableToRead());
        if (_audioBuffer->read(audioBuffer.data(), audioBuffer.size())) {
            // Convert audio data to float PCM
            std::vector<float> pcmf32(audioBuffer.size() / 2);
            const int16_t* samples = reinterpret_cast<const int16_t*>(audioBuffer.data());
            for (size_t i = 0; i < pcmf32.size(); i++) {
                pcmf32[i] = static_cast<float>(samples[i]) / 32768.0f;
            }

            // Transcribe the audio
            if (!TranscribeAudioNonBlocking(pcmf32)) {
                LOG_E("Failed to transcribe audio");
                return false;
            }
        }
    }

    return true;
}

bool WhisperTranscriber::ValidateWhisperModel(const std::string& modelPath) {
    // Check file existence
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_E("Cannot open model file: " << modelPath);
        return false;
    }

    // Check file size (Whisper models are typically large)
    std::streamsize fileSize = file.tellg();
    file.close();

    // Typical Whisper model sizes range from 100MB to 1.5GB
    const int64_t minModelSize = static_cast<int64_t>(100) * 1024 * 1024;
    const int64_t maxModelSize = static_cast<int64_t>(2) * 1024 * 1024 * 1024;

    if (fileSize < minModelSize || fileSize > maxModelSize) {
        LOG_E("Unexpected model file size: " << fileSize << " bytes");
        return false;
    }

    return true;
}

bool WhisperTranscriber::InitializeWhisperModel(const std::string& modelPath) {
    // Open the file in binary mode
    FILE* file = fopen(modelPath.c_str(), "rb");
    if (!file) {
        LOG_E("Cannot open model file: " << modelPath);
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Log detailed file information
    LOG_I("Model file path: " << modelPath
                    << "Model file size: " << fileSize << " bytes"
                    );

    // Read first few bytes to check file signature
    unsigned char header[16];
    size_t bytesRead = fread(header, 1, sizeof(header), file);
    fclose(file);

    if (bytesRead < sizeof(header)) {
        LOG_E("Failed to read model file header");
        return false;
    }

    // Log header bytes for diagnostic purposes
    std::stringstream headerHex;
    headerHex << "Model file header (first 16 bytes): ";
    for (size_t i = 0; i < bytesRead; ++i) {
        headerHex << std::hex << std::setw(2) << std::setfill('0') 
                << static_cast<int>(header[i]) << " ";
    }
    LOG_V(headerHex.str());

    // Attempt model initialization with verbose error checking
    whisper_context_params context_params = whisper_context_default_params();
    
    // Try different GPU configuration options
    std::vector<bool> gpuOptions = { true };
    
    for (bool useGpu : gpuOptions) {
        context_params.use_gpu = useGpu;
        
        // Detailed logging before model initialization attempt
        LOG_I("Attempting to load model with GPU " 
                        << (useGpu ? "Enabled" : "Disabled")
                        );

        // Try to initialize the model
        whisper_context* localContext = whisper_init_from_file_with_params(
            modelPath.c_str(), 
            context_params
        );

        if (localContext) {
            _whisperContext = localContext;
            LOG_I("Model loaded successfully (GPU: " << (useGpu ? "Enabled" : "Disabled") << ")");
            return true;
        }

        LOG_W("Model load failed with GPU " << (useGpu ? "Enabled" : "Disabled"));
    }

    LOG_E("Failed to load Whisper model from: " << modelPath);
    return false;
}

whisper_context* WhisperTranscriber::TryAlternativeInitMethods(const std::string& modelPath) {
    // Method 1: Direct file initialization
    whisper_context* ctx = nullptr;

    // Method 2: Low-level initialization
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false;
    
    // If your Whisper.cpp version supports this
    ctx = whisper_init_from_file_with_params(modelPath.c_str(), params);
    
    return ctx;
}

void WhisperTranscriber::ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize) {
    if(_whisperContext == nullptr) {
        LOG_E("Whisper context is not initialized");
        return;
    }

    // Handle end-of-stream marker
    if (playoutBuffer == nullptr && kPlayoutBufferSize == (size_t)-1) {
        LOG_V("End of stream marker received");
        // Process any remaining accumulated buffer
        if (!_accumulatedByteBuffer.empty()) {
            LOG_V("Processing remaining " << _accumulatedByteBuffer.size() << " bytes");
            if (!_audioBuffer->write(_accumulatedByteBuffer.data(), _accumulatedByteBuffer.size())) {
                handleOverflow();
            }
            _accumulatedByteBuffer.clear();
            _samplesSinceVoiceStart = 0;
        }
        return;
    }

    // Convert bytes to samples for processing
    size_t numSamples = kPlayoutBufferSize / sizeof(int16_t);
    if (numSamples == 0) {
        LOG_V("Empty audio buffer received");
        return;
    }

    // Process the audio data
    _processingBuffer.resize(numSamples);
    std::memcpy(_processingBuffer.data(), playoutBuffer, kPlayoutBufferSize);

    // Voice detection logic
    bool voicePresent = false;
    const uint windowSize = kSampleRate / 8;
    
    if (_silenceFinder && _silenceFinder->avgAmplitude > 0) {
        float thresholdRatio = static_cast<float>(_silenceFinder->delta(
            _processingBuffer.data(), 
            std::min(windowSize, static_cast<uint>(_processingBuffer.size()))
        )) / _silenceFinder->avgAmplitude;

        // Voice detection state machine
        if (!_inVoiceSegment) {
            if (thresholdRatio > voiceStartThreshold) {
                _voiceState.consecutiveVoiceFrames++;
                _voiceState.consecutiveSilenceFrames = 0;
                
                if (_voiceState.consecutiveVoiceFrames >= kMinVoiceFrames) {
                    LOG_V("Voice segment started");
                    voicePresent = true;
                    _inVoiceSegment = true;
                }
            } else {
                _voiceState.consecutiveVoiceFrames = 0;
            }
        } else {
            if (thresholdRatio < voiceEndThreshold) {
                _voiceState.consecutiveSilenceFrames++;
                _voiceState.consecutiveVoiceFrames = 0;
                
                if (_voiceState.consecutiveSilenceFrames >= kMinSilenceFrames) {
                    LOG_V("Voice segment ended");
                    _inVoiceSegment = false;
                }
            } else {
                _voiceState.consecutiveSilenceFrames = 0;
                voicePresent = true;
            }
        }
    }

    // Buffer accumulation logic
    if (voicePresent || _inVoiceSegment) {
        if (!_inVoiceSegment) {
            _inVoiceSegment = true;
            _samplesSinceVoiceStart = 0;
            LOG_V("Starting new voice segment");
        }

        // Add data to accumulated buffer
        size_t currentSize = _accumulatedByteBuffer.size();
        _accumulatedByteBuffer.resize(currentSize + kPlayoutBufferSize);
        std::memcpy(_accumulatedByteBuffer.data() + currentSize, playoutBuffer, kPlayoutBufferSize);
        
        _samplesSinceVoiceStart += numSamples;
        
        // Check if we have enough data to process
        if (_accumulatedByteBuffer.size() >= kTargetSamples * sizeof(int16_t)) {
            LOG_V("Processing accumulated buffer of size: " << _accumulatedByteBuffer.size());
            if (!_audioBuffer->write(_accumulatedByteBuffer.data(), _accumulatedByteBuffer.size())) {
                handleOverflow();
            }
            _accumulatedByteBuffer.clear();
            _samplesSinceVoiceStart = 0;
        }
    }
}

void WhisperTranscriber::handleOverflow() {
    _overflowCount++;
    if(_overflowCount > 10) {
        LOG_I("Frequent buffer overflows, increasing buffer size");
        _audioBuffer->increaseWith(kRingBufferSizeIncrement); 
        _overflowCount = 0;
    }
}
bool WhisperTranscriber::start() {

    if(_whisperContext == nullptr) {
        LOG_E("Whisper context is not initialized");
        return false;
    }

    if (!_running) {
        _running = true;
        _processingThread = std::thread([this] {
             while (_running && RunProcessingThread()) {
             }
          });
    }

    return _running;
}

void WhisperTranscriber::stop() {

    if (_running) {
        _running = false;
        
        if (_processingThread.joinable()) {
            _processingThread.join();
        }

        // Clear any remaining accumulated buffer
        _accumulatedByteBuffer.clear();
    }
}
