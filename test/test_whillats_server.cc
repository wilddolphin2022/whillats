#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "whillats_client.h"
#include "whillats_ipc.h"
#include "test_utils.h"

// Convert stb-loaded RGB/RGBA image to YUV420 planar YUVData
static YUVData rgb_to_yuv420(const uint8_t* rgb, int w, int h, int channels) {
    YUVData yuv;
    yuv.width  = w;
    yuv.height = h;
    yuv.y_size  = (size_t)(w * h);
    yuv.uv_size = (size_t)((w / 2) * (h / 2));

    auto yp = std::unique_ptr<uint8_t[]>(new uint8_t[yuv.y_size]);
    auto up = std::unique_ptr<uint8_t[]>(new uint8_t[yuv.uv_size]);
    auto vp = std::unique_ptr<uint8_t[]>(new uint8_t[yuv.uv_size]);

    // Y plane
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int idx = (row * w + col) * channels;
            int r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
            yp[row * w + col] = (uint8_t)((66*r + 129*g + 25*b + 128) / 256 + 16);
        }
    }
    // U/V planes (2x2 subsampled)
    for (int row = 0; row < h/2; ++row) {
        for (int col = 0; col < w/2; ++col) {
            int idx = (row*2 * w + col*2) * channels;
            int r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
            up[row * (w/2) + col] = (uint8_t)((-38*r - 74*g + 112*b + 128) / 256 + 128);
            vp[row * (w/2) + col] = (uint8_t)((112*r - 94*g - 18*b + 128) / 256 + 128);
        }
    }

    yuv.y = std::move(yp);
    yuv.u = std::move(up);
    yuv.v = std::move(vp);
    return yuv;
}

static std::atomic<bool> whisper_done{false};
static std::atomic<bool> llama_done{false};
static std::atomic<bool> tts_done{false};
static std::vector<uint16_t> tts_audio;
static std::string llama_full_response;
static std::string detected_language = "en";  // updated by language_cb

static void whisper_cb(bool success, const char* text, void*) {
    fprintf(stderr, "[test] Whisper: %s\n", text);
    whisper_done = true;
}

static void language_cb(bool success, const char* lang, void*) {
    if (lang && lang[0]) {
        detected_language = lang;
        fprintf(stderr, "[test] Language detected: %s\n", lang);
    }
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
    const char* mmproj_path = nullptr;
    const char* piper_model = nullptr;
    const char* espeak_data = nullptr;
    const char* image_path = nullptr;
    bool test_tts = false, test_whisper = false, test_llama = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--server=") == 0) server_path = argv[i] + 9;
        else if (arg.find("--whisper_model=") == 0) whisper_model = argv[i] + 16;
        else if (arg.find("--llama_model=") == 0) llama_model = argv[i] + 14;
        else if (arg.find("--mmproj_path=") == 0) mmproj_path = argv[i] + 14;
        else if (arg.find("--piper_model=") == 0) piper_model = argv[i] + 14;
        else if (arg.find("--espeak_data=") == 0) espeak_data = argv[i] + 14;
        else if (arg.find("--image=") == 0) image_path = argv[i] + 8;
        else if (arg == "--tts") test_tts = true;
        else if (arg == "--whisper") test_whisper = true;
        else if (arg == "--llama") test_llama = true;
        else if (arg == "--all") { test_tts = test_whisper = test_llama = true; }
        else if (arg == "--help") {
            fprintf(stderr,
                "Usage: %s --server=PATH [--whisper_model=PATH] [--llama_model=PATH]\n"
                "  [--mmproj_path=PATH] [--piper_model=PATH] [--espeak_data=PATH]\n"
                "  [--image=PATH] [--tts] [--whisper] [--llama] [--all]\n", argv[0]);
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
    if (mmproj_path) strncpy(cfg.llama_mmproj, mmproj_path, sizeof(cfg.llama_mmproj)-1);
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
            bool tts_long_ok = true;
            tts.queueText("Hello, this is a test of text to speech synthesis. "
                           "This is a longer test to ensure we have enough audio data. "
                           "We are testing the whisper transcription system. "
                           "The quick brown fox jumps over the lazy dog", "en");
            if (!wait_for(tts_done, 30)) { fprintf(stderr, "[test] TTS long EN FAILED\n"); result = 1; tts_long_ok = false; }

            tts_done = false;
            tts.queueText("¿Cómo estás? ¿cómo te llamas?", "es");
            if (!wait_for(tts_done, 30)) { fprintf(stderr, "[test] TTS ES FAILED\n"); result = 1; tts_long_ok = false; }

            tts_done = false;
            tts.queueText("У вас есть меню на английском?", "ru");
            if (!wait_for(tts_done, 30)) { fprintf(stderr, "[test] TTS RU FAILED\n"); result = 1; tts_long_ok = false; }

            tts_done = false;
            writeWavFile("synthesized_audio_long.wav", tts_audio, 16000);
            if (tts_long_ok)
                fprintf(stderr, "[test] TTS long PASSED (%zu samples) -> synthesized_audio_long.wav\n", tts_audio.size());
            else
                fprintf(stderr, "[test] TTS long FAILED (%zu samples collected)\n", tts_audio.size());
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
            fprintf(stderr, "[test] Waiting for Llama model to load (may take 60s+ for large multimodal models)...\n");
            std::this_thread::sleep_for(std::chrono::seconds(60));

            // --- Image recognition (if image path provided or default test/512.png exists) ---
            const char* img = image_path ? image_path : "test/512.png";
            int img_w = 0, img_h = 0, img_ch = 0;
            uint8_t* img_data = stbi_load(img, &img_w, &img_h, &img_ch, 3);
            if (img_data && img_w > 0 && img_h > 0) {
                fprintf(stderr, "\n=== Llama: Image recognition (%dx%d) ===\n", img_w, img_h);
                YUVData yuv = rgb_to_yuv420(img_data, img_w, img_h, 3);
                stbi_image_free(img_data);
                llama.receiveVideoFrame(yuv);

                llama_full_response.clear(); llama_done = false;
                const char* image_prompt = "Describe this image in detail.";
                fprintf(stderr, "[test] Llama image prompt: %s\n", image_prompt);
                llama.askLlama(image_prompt);

                if (wait_for(llama_done, 120)) {
                    fprintf(stderr, "[test] Llama image description PASSED: '%s'\n",
                            llama_full_response.c_str());

                    // Synthesize the image description via TTS
                    if (test_tts && !llama_full_response.empty()) {
                        fprintf(stderr, "\n=== TTS: Synthesizing Llama image description ===\n");
                        WhillatsTTSClient tts_llama(conn);
                        if (tts_llama.start()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(300));
                            tts_audio.clear(); tts_done = false;
                            fprintf(stderr, "[test] TTS language: %s\n", detected_language.c_str());
                            tts_llama.queueText(llama_full_response.c_str(), detected_language.c_str());
                            if (wait_for(tts_done, 60)) {
                                writeWavFile("llama_image_description.wav", tts_audio, 16000);
                                fprintf(stderr, "[test] TTS image description PASSED (%zu samples) -> llama_image_description.wav\n",
                                        tts_audio.size());
                            } else {
                                fprintf(stderr, "[test] TTS image description FAILED (timeout)\n");
                                result = 1;
                            }
                            tts_llama.stop();
                        }
                    }
                } else {
                    fprintf(stderr, "[test] Llama image recognition FAILED (timeout)\n");
                    result = 1;
                }
            } else {
                if (img_data) stbi_image_free(img_data);
                fprintf(stderr, "[test] Image not loaded (%s) — falling back to text prompt\n", img);

                // --- Text-only Llama test ---
                llama_full_response.clear(); llama_done = false;
                fprintf(stderr, "[test] Llama prompt: What is your name?\n");
                llama.askLlama("What is your name?");

                if (wait_for(llama_done, 120)) {
                    fprintf(stderr, "[test] Llama PASSED: '%s'\n", llama_full_response.c_str());
                } else {
                    fprintf(stderr, "[test] Llama FAILED (timeout)\n");
                    result = 1;
                }
            }

            llama.stop();
        }
    }

    conn.stop();
    fprintf(stderr, "\n=== Result: %s ===\n", result == 0 ? "ALL PASSED" : "SOME FAILED");
    return result;
}
