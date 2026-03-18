/*
 *  test_whillats — thin client test (uses whillats_server via IPC)
 *  Built with -fno-exceptions to match directcall environment.
 *  Requires WHILLATS_SERVER env var pointing to whillats_server binary.
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <thread>
#include <atomic>

#include "whillats.h"
#include "test_utils.h"

static std::vector<uint16_t> audio_buffer;
static std::atomic<bool> tts_done{false};
static std::atomic<bool> whisper_done{false};
static std::atomic<bool> llama_done{false};

void ttsAudioCallback(bool success, const uint16_t* buffer, size_t buffer_size, void*) {
    if (success && buffer && buffer_size > 0) {
        fprintf(stderr, "[test] TTS: %zu samples\n", buffer_size);
        audio_buffer.insert(audio_buffer.end(), buffer, buffer + buffer_size);
    } else if (!success) {
        fprintf(stderr, "[test] TTS done (total %zu samples)\n", audio_buffer.size());
        tts_done = true;
    }
}

void whisperResponseCallback(bool success, const char* response, void*) {
    fprintf(stderr, "[test] Whisper: %s\n", response);
    whisper_done = true;
}

void llamaResponseCallback(bool success, const char* response, void*) {
    fprintf(stderr, "[test] Llama: %s\n", response);
    llama_done = true;
}

void languageChangedCallback(bool success, const char* language, void*) {
    fprintf(stderr, "[test] Language: %s\n", language);
}

int main(int argc, char *argv[]) {
    Options opts = parseOptions(argc, argv);

    if (argc == 1 || opts.help) {
        fprintf(stderr, "%s\n", opts.help_string.c_str());
        return 1;
    }

    fprintf(stderr, "[test] Config: whisper=%s llama=%s\n",
            opts.whisper_model.c_str(), opts.llama_model.c_str());

    // Set model env vars early so whillats_server gets them in its config
    if (!opts.whisper_model.empty()) setenv("WHISPER_MODEL", opts.whisper_model.c_str(), 1);
    if (!opts.llama_model.empty()) setenv("LLAMA_MODEL", opts.llama_model.c_str(), 1);
    if (!opts.llama_mmproj.empty()) setenv("LLAMA_MMPROJ", opts.llama_mmproj.c_str(), 1);

    // ================================================================
    // TTS Test
    // ================================================================
    {
        audio_buffer.clear();
        tts_done = false;
        WhillatsSetAudioCallback callback(ttsAudioCallback, nullptr);
        WhillatsTTS tts(callback);

        if (tts.start()) {
            const char* text1 = "Hello, this is a test of text to speech synthesis.";
            fprintf(stderr, "[test] TTS: %s\n", text1);
            tts.queueText(text1, "en-US");
            while (!tts_done) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            writeWavFile("synthesized_audio.wav", audio_buffer, WhillatsTTS::getSampleRate());
            fprintf(stderr, "[test] Saved synthesized_audio.wav (%zu samples)\n", audio_buffer.size());

            tts_done = false;
            audio_buffer.clear();

            const char* long_text = "Hello, this is a test of text to speech synthesis. "
                "This is a longer test to ensure we have enough audio data. "
                "We are testing the whisper transcription system. "
                "The quick brown fox jumps over the lazy dog";
            fprintf(stderr, "[test] TTS long: %s\n", long_text);
            tts.queueText(long_text, "en");
            while (!tts_done) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            tts_done = false;

            tts.queueText("¿Cómo estás? ¿cómo te llamas?", "es");
            while (!tts_done) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            tts_done = false;

            tts.queueText("У вас есть меню на английском?", "ru");
            while (!tts_done) std::this_thread::sleep_for(std::chrono::milliseconds(100));

            writeWavFile("synthesized_audio_long.wav", audio_buffer, WhillatsTTS::getSampleRate());
            fprintf(stderr, "[test] Saved synthesized_audio_long.wav (%zu samples)\n", audio_buffer.size());
            tts.stop();
        } else {
            fprintf(stderr, "[test] TTS start FAILED\n");
        }
    }

    // ================================================================
    // Whisper Test
    // ================================================================
    if (opts.whisper && !audio_buffer.empty()) {
        WhillatsSetResponseCallback callback(whisperResponseCallback, nullptr);
        WhillatsSetLanguageCallback lang_cb(languageChangedCallback, nullptr);
        WhillatsTranscriber whisper(opts.whisper_model.c_str(), callback, lang_cb);

        if (whisper.start()) {
            fprintf(stderr, "[test] Whisper started\n");
            size_t samples_per_chunk = (WhillatsTTS::getSampleRate() * 10) / 1000;
            fprintf(stderr, "[test] Feeding %zu samples in %zu-sample chunks\n",
                    audio_buffer.size(), samples_per_chunk);

            for (size_t i = 0; i < audio_buffer.size(); i += samples_per_chunk) {
                size_t n = std::min(samples_per_chunk, audio_buffer.size() - i);
                whisper.processAudioBuffer(
                    reinterpret_cast<uint8_t*>(&audio_buffer[i]),
                    n * sizeof(uint16_t));
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            whisper.processAudioBuffer(nullptr, 0);

            for (int i = 0; i < 600 && !whisper_done; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (whisper_done)
                fprintf(stderr, "[test] Whisper PASSED\n");
            else
                fprintf(stderr, "[test] Whisper TIMEOUT\n");

            whisper.stop();
        } else {
            fprintf(stderr, "[test] Whisper start FAILED\n");
        }
    }

    // ================================================================
    // Llama Test
    // ================================================================
    if (opts.llama) {
        WhillatsSetResponseCallback callback(llamaResponseCallback, nullptr);
        WhillatsLlama llama(opts.llama_model.c_str(), opts.llama_mmproj.c_str(), callback);

        fprintf(stderr, "[test] Llama: starting with model %s\n", opts.llama_model.c_str());
        if (llama.start()) {
            fprintf(stderr, "[test] Llama started, asking: What is your name?\n");
            llama.askLlama("What is your name?");

            for (int i = 0; i < 600 && !llama_done; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (llama_done)
                fprintf(stderr, "[test] Llama PASSED\n");
            else
                fprintf(stderr, "[test] Llama TIMEOUT\n");

            llama.stop();
        } else {
            fprintf(stderr, "[test] Llama start FAILED\n");
        }
    }

    fprintf(stderr, "[test] Done\n");
    return 0;
}
