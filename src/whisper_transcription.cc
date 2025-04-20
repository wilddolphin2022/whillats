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

#include "whisper_transcription.h"
#include "whisper_helpers.h"

#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <regex>

#include <whisper.h>

WhisperTranscriber::WhisperTranscriber(const char* modelPath, WhillatsSetResponseCallback callback)
    : _audioBuffer(std::make_unique<AudioRingBuffer<float>>(WHISPER_SAMPLE_RATE * 60)),
      _ctx(nullptr),
      _state(nullptr),
      _responseCallback(callback),
      _segmentComplete(false),
      _nPast(0),
      _maxContext(224),
      _running(false) {
    if (!InitializeWhisperModel(modelPath)) {
        LOG_E("Failed to initialize Whisper");
        return;
    }

    _pastTokens.push_back(whisper_token_sot(_ctx));
    _pastTokens.push_back(whisper_token_transcribe(_ctx));
    _nPast = 2;
    LOG_I("Running v3");
}

WhisperTranscriber::~WhisperTranscriber() {
    stop();
    if (_state) whisper_free_state(_state);
    if (_ctx) whisper_free(_ctx);
}

bool WhisperTranscriber::InitializeWhisperModel(const std::string& modelPath) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;
    _ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!_ctx) {
        LOG_E("Failed to load Whisper model from " << modelPath.c_str());
        return false;
    }

    { std::lock_guard<std::mutex> lock(_state_mutex);
        _state = whisper_init_state(_ctx);
        if (!_state) {
            LOG_E("Failed to initialize Whisper state");
            whisper_free(_ctx);
            _ctx = nullptr;
            return false;
        }
    }

    LOG_I("Whisper initialized successfully with model: " << modelPath.c_str());
    return true;
}

void WhisperTranscriber::ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSizeInBytes) {
    // Assuming kPlayoutBufferSizeInBytes is the size in BYTES
    // and the data is 16-bit signed PCM, little-endian.
    if(kPlayoutBufferSizeInBytes == 0) {
        LOG_I("Shortcutting processing of audio buffer");
        ProcessRemainingAudio();
        return;
    }
    if (kPlayoutBufferSizeInBytes % 2 != 0) {
        LOG_W("Playout buffer size is not even, expected 16-bit PCM data. Size: " << kPlayoutBufferSizeInBytes);
        // Handle error appropriately, maybe return or log and continue carefully
        return; 
    }

    size_t numSamples = kPlayoutBufferSizeInBytes / 2;
    if (numSamples == 0) {
        LOG_V("Received empty audio buffer.");
        return;
    }
    
    std::vector<float> samples(numSamples);
    const int16_t* pcm16 = reinterpret_cast<const int16_t*>(playoutBuffer);

    for (size_t i = 0; i < numSamples; ++i) {
        // Convert int16_t sample to float [-1.0, 1.0]
        samples[i] = static_cast<float>(pcm16[i]) / 32768.0f; 
    }

    // Ensure normalization is correct, clip if necessary (though ideally shouldn't be needed if input is proper 16-bit PCM)
    for (size_t i = 0; i < numSamples; ++i) {
        samples[i] = std::max(-1.0f, std::min(1.0f, samples[i]));
    }

    if (!_audioBuffer->write(samples.data(), samples.size())) {
        LOG_W("Failed to write " << samples.size() << " samples to audio buffer");
    } else {
        // Log first few samples to verify conversion
        std::ostringstream first_samples_ss;
        size_t n_log = std::min((size_t)10, samples.size());
         first_samples_ss << std::fixed << std::setprecision(3);
        for(size_t i=0; i<n_log; ++i) {
             first_samples_ss << samples[i] << (i == n_log-1 ? "" : ", ");
        }

        if (kDebug) {
            LOG_V("Wrote " << samples.size() << " samples. First " << n_log << ": [" << first_samples_ss.str() << "]. Buffer available: " << _audioBuffer->availableToRead());
        }
    }
}

bool WhisperTranscriber::responseValidate(std::string& token_str) {
    // Regex to remove content within [] or ()
    std::regex bracket_regex(R"(\[[^\]]*\]|\([^)]*\))"); 
    std::string token_str_cleaned = std::regex_replace(token_str, bracket_regex, "");
    std::string token_spaces_check = token_str_cleaned;
    // Remove ALL whitespace (including between words)
    token_spaces_check.erase(std::remove_if(token_spaces_check.begin(), token_spaces_check.end(), ::isspace), token_spaces_check.end());
    if(token_spaces_check.empty()) {
        return false;
    }

    token_str = token_str_cleaned;
    return true;
}

bool WhisperTranscriber::TranscribeAudioNonBlocking(const std::vector<float>& samples) {
    float duration_ms = (samples.size() / static_cast<float>(WHISPER_SAMPLE_RATE)) * 1000.0f;
    LOG_V("Processing audio chunk of " << samples.size() << " samples (" << duration_ms << " ms)");

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads = 4;
    wparams.temperature = 0.8f;
    wparams.no_speech_thold = 0.4f;
    wparams.logprob_thold = -1.0f;
    wparams.language = _language.c_str();
    wparams.detect_language = _detectLanguage;

    // Manual language detection before transcription
    std::vector<float> lang_probs(whisper_lang_max_id(), 0.0f);
    if (whisper_lang_auto_detect(_ctx, 0, wparams.n_threads, lang_probs.data()) == 0) {
        int best_lang_id = 0;
        float best_prob = 0.0f;
        for (int i = 0; i < whisper_lang_max_id(); ++i) {
            if (lang_probs[i] > best_prob) {
                best_prob = lang_probs[i];
                best_lang_id = i;
            }
        }
        const char* detected_lang = whisper_lang_str(best_lang_id);
        LOG_I("Detected language: " << detected_lang << " with probability: " << best_prob);
        wparams.language = detected_lang;
    } else {
        LOG_W("Language detection failed, falling back to default language");
        wparams.language = "en";
    }
    wparams.detect_language = false;

    {
        std::lock_guard<std::mutex> lock(_state_mutex);
        if (whisper_full_with_state(_ctx, _state, wparams, samples.data(), samples.size()) != 0) {
            LOG_E("Whisper full processing failed");
            return false;
        }
    }

    int n_segments = whisper_full_n_segments_from_state(_state);
    LOG_V("Number of segments: " << n_segments);

    if (n_segments < 0) {
        LOG_E("Invalid segment count: " << n_segments);
        return false;
    }

    std::vector<whisper_token> new_tokens;
    int vocab_size = whisper_n_vocab(_ctx);
    for (int i_segment = 0; i_segment < n_segments; ++i_segment) {
        int n_tokens = whisper_full_n_tokens_from_state(_state, i_segment);
        LOG_V("Segment " << i_segment << " has " << n_tokens << " tokens");

        if (n_tokens < 0) {
            LOG_E("Invalid token count in segment " << i_segment << ": " << n_tokens);
            continue;
        }

        for (int i = 0; i < n_tokens; ++i) {
            whisper_token token = whisper_full_get_token_id_from_state(_state, i_segment, i);
            if (token >= 0 && token < vocab_size) {
                new_tokens.push_back(token);
                const char* token_str = whisper_token_to_str(_ctx, token);
                if (token_str) {
                    LOG_V("Segment " << i_segment << ", Token " << i << ": " << token << " (" << token_str << ")");
                } else {
                    LOG_W("Null token string for valid token " << token);
                }
            } else {
                LOG_W("Invalid token ID " << token << " (vocab size: " << vocab_size << ")");
            }
        }
    }

    if (!new_tokens.empty()) {
        LOG_V("Total decoded tokens: " << new_tokens.size());
        ProcessTokens(new_tokens);
    } else {
        LOG_V("No tokens decoded");
    }
    return true;
}

void WhisperTranscriber::ProcessTokens(const std::vector<whisper_token>& tokens) {
    std::string chunk_text;
    chunk_text.reserve(tokens.size() * 8);  // Pre-allocate

    for (size_t i = 0; i < tokens.size(); ++i) {
        const char* token_str = whisper_token_to_str(_ctx, tokens[i]);
        if (token_str) {
            chunk_text += token_str;
        } else {
            LOG_W("Null token string for token " << tokens[i] << " at index " << i);
        }
    }

    if (!chunk_text.empty() && responseValidate(chunk_text)) {
        LOG_I("Chunk transcribed text: " << chunk_text.c_str());

        // If this is a new segment, reset _fullTranscription
        if (_segmentComplete) {
            _fullTranscription.clear();
            _segmentComplete = false;
        }

        // Append to full transcription
        if (!_fullTranscription.empty()) {
            _fullTranscription += " ";
        }
        _fullTranscription += chunk_text;

        // Send only this chunk's text via callback
         _responseCallback.OnResponseComplete(true, chunk_text.c_str());

        // If chunk ends with silence or punctuation, mark segment complete
        if (chunk_text.back() == '.' || chunk_text.back() == '?' || chunk_text.back() == '!') {
            LOG_V("Segment complete. Full text: " << _fullTranscription.c_str());
            _segmentComplete = true;
        }
    } else if (!_fullTranscription.empty()) {
        // Silence after text indicates segment end
        _fullTranscription.clear();
        LOG_V("Segment complete due to silence. Full text: " << _fullTranscription.c_str());
        _segmentComplete = true;
    } else {
        LOG_V("No text transcribed from tokens in chunk");
    }
}

bool WhisperTranscriber::start() {
    if (_ctx == nullptr) {
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
        ProcessRemainingAudio();
    }
}

bool WhisperTranscriber::RunProcessingThread() {
    std::vector<float> chunk(kMinPhraseSamples);
    while (_running) {  // Add a stop condition if needed
        if (_audioBuffer->availableToRead() >= kMinPhraseSamples) {
            _audioBuffer->read(chunk.data(), kMinPhraseSamples);
            if (kDebug) {LOG_V("Read chunk size: " << kMinPhraseSamples << " samples. Buffer remaining: " << _audioBuffer->availableToRead());}
            if (vad_simple(chunk, WHISPER_SAMPLE_RATE, 600, kVADThreshold, 50.0f, true)) {
                TranscribeAudioNonBlocking(chunk);
            }
        } else {
            if (kDebug) {LOG_V("Not enough samples: " << _audioBuffer->availableToRead() << " < " << kMinPhraseSamples);}
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

void WhisperTranscriber::ProcessRemainingAudio() {
    LOG_I("Processing remaining audio in _audioBuffer");
    std::vector<float> chunk(kMinPhraseSamples);

    while (_audioBuffer->availableToRead() >= kMinPhraseSamples) {
        _audioBuffer->read(chunk.data(), kMinPhraseSamples);
        LOG_V("Processing remaining chunk size: " << kMinPhraseSamples << " samples. Buffer remaining: " << _audioBuffer->availableToRead());
        if (vad_simple(chunk, WHISPER_SAMPLE_RATE, 2000, 0.75f, 50.0f, true)) {
            TranscribeAudioNonBlocking(chunk);
        }
    }

    size_t remaining_samples = _audioBuffer->availableToRead();
    if (remaining_samples > 0) {
        LOG_V("Discarding " << remaining_samples << " leftover samples (less than " << kMinPhraseSamples << ")");
        std::vector<float> leftover(remaining_samples);
        _audioBuffer->read(leftover.data(), remaining_samples);
    }

    // Finalize current segment if any
    if (!_fullTranscription.empty() && !_segmentComplete) {
        LOG_I("Final segment text: " << _fullTranscription.c_str());
        _responseCallback.OnResponseComplete(true, _fullTranscription.c_str());
        _segmentComplete = true;
    }

    _fullTranscription.clear();  // Reset for next audio
    LOG_V("Finished processing remaining audio");
}

void WhisperTranscriber::fft_forward(std::vector<std::complex<float>>& data, int n) {
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

    for (int step = 2; step <= n; step <<= 1) {
        const int half = step >> 1;
        const float theta = -2.0f * M_PI / step;
        for (int i = 0; i < n; i += step) {
            for (int j = 0; j < half; j++) {
                const std::complex<float> twiddle = std::polar(1.0f, theta * j);
                const std::complex<float> a = data[i + j];
                const std::complex<float> b = data[i + j + half] * twiddle;
                data[i + j] = a + b;
                data[i + j + half] = a - b;
            }
        }
    }
}

bool WhisperTranscriber::vad_simple(const std::vector<float>& pcmf32, int sample_rate,
                                    int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples = pcmf32.size();
    const int n_samples_window = last_ms * sample_rate / 1000;  // 16000 for 1000ms

    if (n_samples < n_samples_window) {
        LOG_V("VAD: Too few samples (" << n_samples << " < " << n_samples_window << ")");
        return false;
    }

    float energy = 0.0f;
    float max_energy = 0.0f;
    float noise_floor = noise_profile.noise_level;

    for (int i = n_samples - n_samples_window; i < n_samples; i++) {
        float sample = fabsf(pcmf32[i]);
        energy += sample * sample;
        max_energy = std::max(max_energy, sample);
    }

    energy = sqrtf(energy / n_samples_window);
    noise_floor = std::max(noise_floor, 0.001f);
    float snr = 20 * log10f(std::max(energy, 1e-10f) / noise_floor);

    bool voice_detected = snr > 2.0f && max_energy > 0.002f;

    if (verbose && voice_detected) {
        LOG_V("VAD: energy=" << energy << ", SNR=" << snr << ", max=" << max_energy
              << ", noise_floor=" << noise_floor << ", detected=" << (voice_detected ? "yes" : "no"));
    }

    return voice_detected;
}