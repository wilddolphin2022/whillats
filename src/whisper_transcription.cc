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
#include <cmath>
#include <vector>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>

#include <whisper.h>
#include "whisper_transcription.h"
#include "whisper_helpers.h"

bool WhisperTranscriber::vad_simple(
        const std::vector<float>& pcmf32,
        int sample_rate,
        int last_ms,
        float vad_thold,
        float freq_thold,
        bool verbose) {

    const int n_samples      = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        // compute FFT of the last n_samples_last samples
        const int n = 1 << (int) ceil(log2(n_samples_last));

        std::vector<float> hann(n_samples_last);
        std::vector<std::complex<float>> fft(n);

        // apply Hanning window
        for (int i = 0; i < n_samples_last; i++) {
            hann[i] = 0.5f * (1.0f - cos((2.0f * M_PI * i) / (n_samples_last - 1)));
        }

        for (int i = 0; i < n_samples_last; i++) {
            fft[i].real(pcmf32[n_samples - n_samples_last + i] * hann[i]);
            fft[i].imag(0.0f);
        }

        for (int i = n_samples_last; i < n; i++) {
            fft[i].real(0.0f);
            fft[i].imag(0.0f);
        }

        // FFT
        fft_forward(fft, n);

        // compute power spectrum
        std::vector<float> power(n/2 + 1);
        for (int i = 0; i <= n/2; i++) {
            power[i] = std::abs(fft[i]);
        }

        // compute average power in frequency range
        float avg_power_freq = 0.0f;
        int n_freq = 0;

        for (int i = 0; i <= n/2; i++) {
            const float freq = (float) i*sample_rate/n;
            if (freq >= freq_thold) {
                avg_power_freq += power[i];
                n_freq++;
            }
        }

        if (n_freq > 0) {
            avg_power_freq /= n_freq;
        }

        if (verbose) {
            LOG_V("avg_power_freq = " << avg_power_freq); 
        }

        if (avg_power_freq < vad_thold) {
            return false;
        }
    }

    // compute average power of the last n_samples_last samples
    float avg_power = 0.0f;
    float peak_power = 0.0f;
    for (int i = 0; i < n_samples_last; i++) {
        float sample_power = fabsf(pcmf32[n_samples - n_samples_last + i]);
        avg_power += sample_power;
        peak_power = std::max(peak_power, sample_power);
    }
    avg_power /= n_samples_last;

    if (verbose) {
        LOG_V("VAD stats: avg_power = " << avg_power 
              << ", peak_power = " << peak_power 
              << ", threshold = " << vad_thold);
    }

    return avg_power > vad_thold || peak_power > (vad_thold * 10);
}

void WhisperTranscriber::fft_forward(std::vector<std::complex<float>>& data, int n) {
    // bit reversal permutation
    int shift = 1;
    for (int i = 0; i < n; i++) {
        if (i < shift) {
            std::swap(data[i], data[shift]);
        }
        int bit = n >> 1;
        while (shift & bit) {
            shift >>= 1;
            bit >>= 1;
        }
        shift |= bit;
    }

    // butterfly algorithm
    for (int step = 2; step <= n; step <<= 1) {
        const int half = step >> 1;
        const float theta = -2.0f * M_PI / step;

        // recursive iteration
        for (int i = 0; i < n; i += step) {
            for (int j = 0; j < half; j++) {
                const std::complex<float> twiddle = std::polar(1.0f, theta * j);
                const std::complex<float> a = data[i + j];
                const std::complex<float> b = data[i + j + half] * twiddle;
                data[i + j]         = a + b;
                data[i + j + half]  = a - b;
            }
        }
    }
}

WhisperTranscriber::WhisperTranscriber(
    const char* model_path,
    WhillatsSetResponseCallback callback) 
    : _model_path(model_path),
      _responseCallback(callback),
      _whisperContext(nullptr),
      _running(false),
      _processingActive(false),
      _audioBuffer(new AudioRingBuffer<float>(kRingBufferSizeIncrement))
{
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

// Transcribe audio non-blocking 
bool WhisperTranscriber::TranscribeAudioNonBlocking(const std::vector<float>& pcmf32) {
    if (!_whisperContext) {
        LOG_E("Whisper context not initialized");
        return false;
    }

    LOG_V("Starting transcription of " << pcmf32.size() << " samples");

    // Ensure minimum duration of 1 second with proper padding
    std::vector<float> padded_audio;
    const size_t min_samples = WHISPER_SAMPLE_RATE;  // 1 second minimum
    
    // Pre-allocate the full size needed
    padded_audio.reserve(std::max(pcmf32.size(), min_samples));
    
    // Copy original audio
    padded_audio = pcmf32;
    
    // Add padding if needed
    if (padded_audio.size() < min_samples) {
        size_t padding_needed = min_samples - padded_audio.size();
        LOG_I("Padding audio with " << padding_needed << " samples of silence");
        
        // Add silence after the audio
        padded_audio.insert(padded_audio.end(), padding_needed, 0.0f);
    }

    LOG_V("Final audio size for transcription: " << padded_audio.size() << " samples");

    // Set up whisper parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    // Basic parameters for better results
    wparams.print_progress   = true;
    wparams.print_timestamps = true;
    wparams.translate        = false;
    wparams.no_context      = false;     // Allow context
    wparams.single_segment  = true;      // Process as single segment
    wparams.duration_ms     = 0;         // Process all available audio
    wparams.max_tokens      = 128;       // Increased token limit
    wparams.language        = "en";
    wparams.n_threads       = 4;
    wparams.audio_ctx       = 768;       // Default audio context
    wparams.suppress_blank  = true;      // Suppress blank outputs

    // Process audio with whisper
    int result = whisper_full(_whisperContext, wparams, padded_audio.data(), padded_audio.size());
    if (result != 0) {
        LOG_E("Whisper processing failed with code: " << result);
        return false;
    }

    // Get transcription result
    const int n_segments = whisper_full_n_segments(_whisperContext);
    LOG_V("Whisper found " << n_segments << " segments");

    if (n_segments > 0) {
        std::string full_text;
        for (int i = 0; i < n_segments; ++i) {
            const char* segment_text = whisper_full_get_segment_text(_whisperContext, i);
            LOG_V("Segment " << i << " text: " << (segment_text ? segment_text : "null"));
            
            if (segment_text && strlen(segment_text) > 0) {
                if (!full_text.empty()) {
                    full_text += " ";
                }
                full_text += segment_text;
            }
        }

        if (!full_text.empty()) {
            LOG_I("Transcribed text: " << full_text);
            _responseCallback.OnResponseComplete(true, full_text.c_str());
            return true;
        }
    }

    LOG_W("No transcription result produced");
    return false;
}

void WhisperTranscriber::ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize) {
    if(_whisperContext == nullptr) {
        LOG_E("Whisper context is not initialized");
        return;
    }

    // Handle end-of-stream marker
    if (playoutBuffer == nullptr && kPlayoutBufferSize == (size_t)-1) {
        LOG_I("End of stream marker received");
        
        // Signal processing thread to finish and wait for it
        _running = false;
        if (_processingThread.joinable()) {
            _processingThread.join();
        }

        // Process any remaining audio
        std::vector<float> audioBuffer;
        size_t samples_available = _audioBuffer->availableToRead();
        if (samples_available > 0) {
            audioBuffer.resize(samples_available);
            if (_audioBuffer->read(audioBuffer.data(), samples_available)) {
                LOG_I("Processing final " << samples_available << " samples");
                TranscribeAudioNonBlocking(audioBuffer);
            }
        }
        
        _responseCallback.OnResponseComplete(true, "End of stream processed");
        return;
    }

    // Convert bytes to samples and immediately to float
    size_t numSamples = kPlayoutBufferSize / sizeof(int16_t);
    if (numSamples == 0) {
        return;  // Skip empty buffers silently
    }

    // Convert int16 samples to float
    std::vector<float> pcmf32(numSamples);
    const int16_t* samples = reinterpret_cast<const int16_t*>(playoutBuffer);
    for (size_t i = 0; i < numSamples; i++) {
        pcmf32[i] = static_cast<float>(samples[i]) / 32768.0f;
    }

    // Write to ring buffer
    if (!_audioBuffer->write(pcmf32.data(), pcmf32.size())) {
        LOG_E("Failed to write to audio buffer");
    }
    LOG_V("Wrote to audio buffer"); 
}

bool WhisperTranscriber::RunProcessingThread() {
    while (_running) {
        std::vector<float> audioBuffer;
        bool shouldProcess = false;
        
        // Check available samples
        size_t samples_available = _audioBuffer->availableToRead();
        
        // Process when we have enough data (at least 10 seconds worth)
        if (samples_available >= WHISPER_SAMPLE_RATE * 10) {
            audioBuffer.resize(samples_available);
            if (_audioBuffer->read(audioBuffer.data(), samples_available)) {
                shouldProcess = true;
                LOG_V("Got " << samples_available << " samples to process");
            }
        } else {
            // Keep accumulating
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        if (shouldProcess) {
            // More sensitive VAD parameters for microphone input
            const float vad_thold = 0.0003f;    // Lowered from 0.01f
            const float freq_thold = 10.0f;     // Lowered from 20.0f
            const int last_ms = 1000;           // Increased window

            bool voicePresent = vad_simple(audioBuffer, WHISPER_SAMPLE_RATE, last_ms, vad_thold, freq_thold, true);
            LOG_V("VAD check: voice present = " << (voicePresent ? "true" : "false") 
                  << ", buffer size = " << audioBuffer.size() 
                  << ", threshold = " << vad_thold);

            if (voicePresent) {
                LOG_V("Voice detected, starting transcription with " << audioBuffer.size() << " samples");
                TranscribeAudioNonBlocking(audioBuffer);
            } else {
                LOG_V("No voice detected in buffer");
            }
        }
    }
    return true;
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
    }
}
