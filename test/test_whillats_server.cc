#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

#include "whillats_client.h"
#include "whillats_ipc.h"
#include "test_utils.h"

static std::atomic<bool> whisper_done{false};
static std::atomic<bool> llama_done{false};
static std::atomic<bool> tts_done{false};
static std::vector<uint16_t> tts_audio;
static std::string llama_full_response;

static void whisper_cb(bool success, const char* text, void*) {
    fprintf(stderr, "[test] Whisper: %s\n", text);
    whisper_done = true;
}

static void language_cb(bool success, const char* lang, void*) {
    fprintf(stderr, "[test] Language: %s\n", lang);
}

static void llama_cb(bool success, const char* text, void*) {
    fprintf(stderr, "[test] Llama: %s\n", text);
    if (text) llama_full_response += text;
    llama_done = true;
}

static void tts_cb(bool success, const uint16_t* buffer, size_t size, void*) {
    if (success && buffer && size > 0) {
        tts_audio.insert(tts_audio.end(), buffer, buffer + size);
    } else if (!success) {
        fprintf(stderr, "[test] TTS synthesis complete (%zu samples)\n", tts_audio.size());
        tts_done = true;
    }
}

static bool wait_for(std::atomic<bool>& flag, int timeout_sec) {
    for (int i = 0; i < timeout_sec * 10; ++i) {
        if (flag) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return flag.load();
}

static void feed_audio_realtime(WhillatsTranscriberClient& whisper,
                                const std::vector<uint16_t>& audio,
                                int sample_rate) {
    size_t samples_per_chunk = (sample_rate * 10) / 1000;  // 10ms
    for (size_t i = 0; i < audio.size(); i += samples_per_chunk) {
        size_t n = std::min(samples_per_chunk, audio.size() - i);
        whisper.processAudioBuffer(
            reinterpret_cast<uint8_t*>(const_cast<uint16_t*>(&audio[i])),
            n * sizeof(uint16_t));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    whisper.processAudioBuffer(nullptr, 0);
}

int main(int argc, char* argv[]) {
    const char* server_path = nullptr;
    const char* whisper_model = nullptr;
    const char* llama_model = nullptr;
    const char* piper_model = nullptr;
    const char* espeak_data = nullptr;
    bool test_tts = false, test_whisper = false, test_llama = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--server=") == 0) server_path = argv[i] + 9;
        else if (arg.find("--whisper_model=") == 0) whisper_model = argv[i] + 16;
        else if (arg.find("--llama_model=") == 0) llama_model = argv[i] + 14;
        else if (arg.find("--piper_model=") == 0) piper_model = argv[i] + 14;
        else if (arg.find("--espeak_data=") == 0) espeak_data = argv[i] + 14;
        else if (arg == "--tts") test_tts = true;
        else if (arg == "--whisper") test_whisper = true;
        else if (arg == "--llama") test_llama = true;
        else if (arg == "--all") { test_tts = test_whisper = test_llama = true; }
        else if (arg == "--help") {
            fprintf(stderr,
                "Usage: %s --server=PATH [--whisper_model=PATH] [--llama_model=PATH]\n"
                "  [--piper_model=PATH] [--espeak_data=PATH]\n"
                "  [--tts] [--whisper] [--llama] [--all]\n", argv[0]);
            return 0;
        }
    }

    if (!server_path) {
        fprintf(stderr, "Error: --server=PATH required\n");
        return 1;
    }

    whillats_ipc::ConfigMsg cfg{};
    if (whisper_model) strncpy(cfg.whisper_model, whisper_model, sizeof(cfg.whisper_model)-1);
    if (llama_model) strncpy(cfg.llama_model, llama_model, sizeof(cfg.llama_model)-1);
    if (piper_model) strncpy(cfg.piper_model, piper_model, sizeof(cfg.piper_model)-1);
    if (espeak_data) strncpy(cfg.espeak_data, espeak_data, sizeof(cfg.espeak_data)-1);
    strncpy(cfg.language, "en", sizeof(cfg.language)-1);
    cfg.whisper_threads = 4;
    cfg.llama_threads = 6;
    cfg.tts_threads = 2;

    WhillatsServerConnection conn;
    conn.setWhisperCallback(whisper_cb, nullptr);
    conn.setLanguageCallback(language_cb, nullptr);
    conn.setLlamaCallback(llama_cb, nullptr);
    conn.setTtsCallback(tts_cb, nullptr);

    if (!conn.start(server_path, cfg)) {
        fprintf(stderr, "FATAL: Failed to start server\n");
        return 1;
    }

    fprintf(stderr, "[test] Server connection established\n");
    int result = 0;

    // ================================================================
    // TTS Tests
    // ================================================================
    if (test_tts) {
        WhillatsTTSClient tts(conn);
        if (!tts.start()) {
            fprintf(stderr, "[test] TTS start FAILED\n");
            result = 1;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            // Short utterance
            fprintf(stderr, "\n=== TTS: Short utterance ===\n");
            tts_audio.clear(); tts_done = false;
            tts.queueText("Hello, this is a test of text to speech synthesis.", "en-US");
            if (wait_for(tts_done, 30)) {
                writeWavFile("synthesized_audio.wav", tts_audio, 16000);
                fprintf(stderr, "[test] TTS short PASSED (%zu samples) -> synthesized_audio.wav\n", tts_audio.size());
            } else {
                fprintf(stderr, "[test] TTS short FAILED (timeout)\n");
                result = 1;
            }

            // Long + multi-language
            fprintf(stderr, "\n=== TTS: Long + multi-language ===\n");
            tts_audio.clear(); tts_done = false;
            tts.queueText("Hello, this is a test of text to speech synthesis. "
                           "This is a longer test to ensure we have enough audio data. "
                           "We are testing the whisper transcription system. "
                           "The quick brown fox jumps over the lazy dog", "en");
            if (!wait_for(tts_done, 30)) { fprintf(stderr, "[test] TTS long EN FAILED\n"); result = 1; }

            tts_done = false;
            tts.queueText("¿Cómo estás? ¿cómo te llamas?", "es");
            if (!wait_for(tts_done, 30)) { fprintf(stderr, "[test] TTS ES FAILED\n"); result = 1; }

            tts_done = false;
            tts.queueText("У вас есть меню на английском?", "ru");
            if (!wait_for(tts_done, 30)) { fprintf(stderr, "[test] TTS RU FAILED\n"); result = 1; }

            tts_done = false;
            writeWavFile("synthesized_audio_long.wav", tts_audio, 16000);
            fprintf(stderr, "[test] TTS long PASSED (%zu samples) -> synthesized_audio_long.wav\n", tts_audio.size());
            tts.stop();
        }
    }

    // ================================================================
    // Whisper Test — feed TTS audio at ~real-time pace
    // ================================================================
    if (test_whisper) {
        if (tts_audio.empty()) {
            fprintf(stderr, "\n[test] Whisper skipped: no TTS audio. Run with --tts.\n");
        } else {
            WhillatsTranscriberClient whisper(conn);
            if (!whisper.start()) {
                fprintf(stderr, "[test] Whisper start FAILED\n");
                result = 1;
            } else {
                fprintf(stderr, "\n=== Whisper: Waiting for model load ===\n");
                std::this_thread::sleep_for(std::chrono::seconds(15));

                fprintf(stderr, "=== Whisper: Pass 1 — feeding %zu samples ===\n", tts_audio.size());
                whisper_done = false;
                feed_audio_realtime(whisper, tts_audio, 16000);

                if (wait_for(whisper_done, 90)) {
                    fprintf(stderr, "[test] Whisper PASSED\n");
                } else {
                    fprintf(stderr, "[test] Whisper FAILED (timeout)\n");
                    result = 1;
                }

                whisper.stop();
            }
        }
    }

    // ================================================================
    // Llama Test
    // ================================================================
    if (test_llama) {
        fprintf(stderr, "\n=== Llama: Loading model ===\n");
        WhillatsLlamaClient llama(conn);
        if (!llama.start()) {
            fprintf(stderr, "[test] Llama start FAILED\n");
            result = 1;
        } else {
            fprintf(stderr, "[test] Waiting for Llama model to load (may take 30s+ if other models loaded)...\n");
            std::this_thread::sleep_for(std::chrono::seconds(30));

            llama_full_response.clear(); llama_done = false;
            fprintf(stderr, "[test] Llama prompt: What is your name?\n");
            llama.askLlama("What is your name?");

            if (wait_for(llama_done, 60)) {
                fprintf(stderr, "[test] Llama PASSED: '%s'\n", llama_full_response.c_str());
            } else {
                fprintf(stderr, "[test] Llama FAILED (timeout)\n");
                result = 1;
            }

            llama.stop();
        }
    }

    conn.stop();
    fprintf(stderr, "\n=== Result: %s ===\n", result == 0 ? "ALL PASSED" : "SOME FAILED");
    return result;
}
