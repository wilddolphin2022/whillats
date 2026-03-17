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
static int tts_sample_rate = 16000;

static void whisper_cb(bool success, const char* text, void*) {
    fprintf(stderr, "[test] Whisper: %s\n", text);
    whisper_done = true;
}

static void language_cb(bool success, const char* lang, void*) {
    fprintf(stderr, "[test] Language: %s\n", lang);
}

static void llama_cb(bool success, const char* text, void*) {
    fprintf(stderr, "[test] Llama: %s\n", text);
    llama_done = true;
}

static void tts_cb(bool success, const uint16_t* buffer, size_t size, void*) {
    if (success && buffer && size > 0) {
        tts_audio.insert(tts_audio.end(), buffer, buffer + size);
        fprintf(stderr, "[test] TTS audio chunk: %zu samples\n", size);
    } else if (!success) {
        fprintf(stderr, "[test] TTS done (total %zu samples)\n", tts_audio.size());
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
            fprintf(stderr, "Usage: %s --server=PATH [--whisper_model=PATH] [--llama_model=PATH]\n"
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
    conn.setWhisperCallback({whisper_cb, nullptr});
    conn.setLanguageCallback({language_cb, nullptr});
    conn.setLlamaCallback({llama_cb, nullptr});
    conn.setTtsCallback({tts_cb, nullptr});

    if (!conn.start(server_path, cfg)) {
        fprintf(stderr, "Failed to start server\n");
        return 1;
    }

    fprintf(stderr, "[test] Server connection established\n");
    int result = 0;

    // --- Test TTS ---
    if (test_tts) {
        fprintf(stderr, "\n=== Testing TTS ===\n");
        WhillatsTTSClient tts(conn);
        if (tts.start()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            tts.queueText("Hello, this is a test of text to speech.", "en");
            if (wait_for(tts_done, 30)) {
                fprintf(stderr, "[test] TTS PASSED (%zu samples)\n", tts_audio.size());
                if (!tts_audio.empty()) {
                    writeWavFile("server_tts_test.wav", tts_audio, tts_sample_rate);
                    fprintf(stderr, "[test] Saved server_tts_test.wav\n");
                }
            } else {
                fprintf(stderr, "[test] TTS FAILED (timeout)\n");
                result = 1;
            }
            tts.stop();
        } else {
            fprintf(stderr, "[test] TTS start failed\n");
            result = 1;
        }
    }

    // --- Test Llama ---
    if (test_llama) {
        fprintf(stderr, "\n=== Testing Llama ===\n");
        WhillatsLlamaClient llama(conn);
        if (llama.start()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            llama.askLlama("What is your name?");
            if (wait_for(llama_done, 60)) {
                fprintf(stderr, "[test] Llama PASSED\n");
            } else {
                fprintf(stderr, "[test] Llama FAILED (timeout)\n");
                result = 1;
            }
            llama.stop();
        } else {
            fprintf(stderr, "[test] Llama start failed\n");
            result = 1;
        }
    }

    // --- Test TTS with Llama output ---
    if (test_tts && test_llama && result == 0) {
        fprintf(stderr, "\n=== Testing Llama -> TTS pipeline ===\n");
        tts_done = false;
        tts_audio.clear();
        llama_done = false;

        WhillatsTTSClient tts(conn);
        WhillatsLlamaClient llama(conn);
        tts.start();
        llama.start();
        std::this_thread::sleep_for(std::chrono::seconds(1));

        llama.askLlama("Say hello in one sentence.");
        if (wait_for(llama_done, 60)) {
            fprintf(stderr, "[test] Llama answered, waiting for TTS...\n");
        }
    }

    // --- Test Whisper ---
    if (test_whisper && !tts_audio.empty()) {
        fprintf(stderr, "\n=== Testing Whisper (using TTS audio) ===\n");
        WhillatsTranscriberClient whisper(conn);
        if (whisper.start()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            size_t chunk_samples = 160;
            for (size_t i = 0; i < tts_audio.size(); i += chunk_samples) {
                size_t n = std::min(chunk_samples, tts_audio.size() - i);
                whisper.processAudioBuffer(
                    reinterpret_cast<uint8_t*>(&tts_audio[i]),
                    n * sizeof(uint16_t));
            }
            whisper.processAudioBuffer(nullptr, 0);
            if (wait_for(whisper_done, 30)) {
                fprintf(stderr, "[test] Whisper PASSED\n");
            } else {
                fprintf(stderr, "[test] Whisper FAILED (timeout)\n");
                result = 1;
            }
            whisper.stop();
        }
    }

    conn.stop();
    fprintf(stderr, "\n=== Result: %s ===\n", result == 0 ? "ALL PASSED" : "SOME FAILED");
    return result;
}
