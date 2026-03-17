#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <unistd.h>

#include "whillats_ipc.h"
#include "whillats.h"
#include "whisper_helpers.h"

using namespace whillats_ipc;

static int g_write_fd = -1;
static std::atomic<bool> g_running{true};

static void send_response(uint8_t type, const char* text) {
    uint32_t len = text ? (uint32_t)strlen(text) : 0;
    write_msg(g_write_fd, type, text, len);
}

static void whisper_callback(bool success, const char* response, void*) {
    if (success && response)
        send_response(MSG_WHISPER_RESULT, response);
}

static void language_callback(bool success, const char* language, void*) {
    if (success && language)
        send_response(MSG_WHISPER_LANGUAGE, language);
}

static void llama_callback(bool success, const char* response, void*) {
    if (success && response)
        send_response(MSG_LLAMA_RESPONSE, response);
}

static void tts_audio_callback(bool success, const uint16_t* buffer, size_t buffer_size, void*) {
    if (success && buffer && buffer_size > 0) {
        TtsAudioMsg hdr;
        hdr.sample_rate = WhillatsTTS::getSampleRate();
        hdr.num_samples = (uint32_t)buffer_size;
        size_t payload_sz = sizeof(TtsAudioMsg) + buffer_size * sizeof(int16_t);
        std::vector<uint8_t> payload(payload_sz);
        memcpy(payload.data(), &hdr, sizeof(hdr));
        memcpy(payload.data() + sizeof(hdr), buffer, buffer_size * sizeof(int16_t));
        write_msg(g_write_fd, MSG_TTS_AUDIO, payload.data(), (uint32_t)payload_sz);
    } else if (!success) {
        write_msg(g_write_fd, MSG_TTS_DONE, nullptr, 0);
    }
}

int main(int argc, char* argv[]) {
    // Signal that we are the server process so libwhillats uses in-process mode
    setenv("WHILLATS_IS_SERVER", "1", 1);

    int read_fd  = STDIN_FILENO;
    g_write_fd   = STDOUT_FILENO;

    if (argc >= 3) {
        read_fd    = atoi(argv[1]);
        g_write_fd = atoi(argv[2]);
    }

    fprintf(stderr, "[whillats_server] Started (read_fd=%d, write_fd=%d)\n", read_fd, g_write_fd);

    WhillatsSetResponseCallback whisperCb(whisper_callback, nullptr);
    WhillatsSetLanguageCallback langCb(language_callback, nullptr);
    WhillatsSetResponseCallback llamaCb(llama_callback, nullptr);
    WhillatsSetAudioCallback    ttsCb(tts_audio_callback, nullptr);

    std::unique_ptr<WhillatsTranscriber> whisper;
    std::unique_ptr<WhillatsLlama>       llama;
    std::unique_ptr<WhillatsTTS>         tts;

    ConfigMsg cfg{};

    Header h;
    while (g_running && read_header(read_fd, h)) {
        std::vector<uint8_t> payload(h.len);
        if (h.len > 0 && !read_exact(read_fd, payload.data(), h.len)) break;

        switch (h.type) {

        case MSG_CONFIG: {
            if (h.len >= sizeof(ConfigMsg))
                memcpy(&cfg, payload.data(), sizeof(ConfigMsg));
            fprintf(stderr, "[whillats_server] Config: whisper=%s llama=%s piper=%s\n",
                    cfg.whisper_model, cfg.llama_model, cfg.piper_model);
            break;
        }

        case MSG_WHISPER_START: {
            if (!whisper && cfg.whisper_model[0]) {
                whisper = std::make_unique<WhillatsTranscriber>(cfg.whisper_model, whisperCb, langCb);
                if (cfg.whisper_threads > 0) whisper->setThreadCount(cfg.whisper_threads);
                if (cfg.language[0]) whisper->setLanguage(cfg.language);
                if (whisper->start())
                    fprintf(stderr, "[whillats_server] Whisper started\n");
                else
                    fprintf(stderr, "[whillats_server] Whisper failed to start\n");
            }
            break;
        }

        case MSG_WHISPER_STOP:
            if (whisper) { whisper->stop(); whisper.reset(); }
            break;

        case MSG_WHISPER_AUDIO:
            if (whisper) {
                if (h.len == 0)
                    whisper->processAudioBuffer(nullptr, 0);
                else
                    whisper->processAudioBuffer(payload.data(), payload.size());
            }
            break;

        case MSG_LLAMA_START: {
            if (!llama && cfg.llama_model[0]) {
                if (cfg.llama_mmproj[0])
                    llama = std::make_unique<WhillatsLlama>(cfg.llama_model, cfg.llama_mmproj, llamaCb);
                else
                    llama = std::make_unique<WhillatsLlama>(cfg.llama_model, llamaCb);
                if (cfg.llama_threads > 0) llama->setThreadCount(cfg.llama_threads);
                if (llama->start())
                    fprintf(stderr, "[whillats_server] Llama started\n");
                else
                    fprintf(stderr, "[whillats_server] Llama failed to start\n");
            }
            break;
        }

        case MSG_LLAMA_STOP:
            if (llama) { llama->stop(); llama.reset(); }
            break;

        case MSG_LLAMA_ASK: {
            if (llama && h.len > 0) {
                std::string prompt(reinterpret_cast<char*>(payload.data()), h.len);
                llama->askLlama(prompt.c_str());
            }
            break;
        }

        case MSG_LLAMA_VIDEO_FRAME: {
            if (llama && h.len > 16) {
                // Payload: [int32 w][int32 h][int32 y_size][int32 uv_size][y][u][v]
                int32_t w, ht, ys, uvs;
                memcpy(&w, payload.data(), 4);
                memcpy(&ht, payload.data()+4, 4);
                memcpy(&ys, payload.data()+8, 4);
                memcpy(&uvs, payload.data()+12, 4);
                size_t expected = 16 + ys + 2*uvs;
                if (h.len >= expected) {
                    YUVData yuv;
                    yuv.width = w; yuv.height = ht;
                    yuv.y_size = ys; yuv.uv_size = uvs;
                    yuv.y = std::make_unique<uint8_t[]>(ys);
                    yuv.u = std::make_unique<uint8_t[]>(uvs);
                    yuv.v = std::make_unique<uint8_t[]>(uvs);
                    memcpy(yuv.y.get(), payload.data()+16, ys);
                    memcpy(yuv.u.get(), payload.data()+16+ys, uvs);
                    memcpy(yuv.v.get(), payload.data()+16+ys+uvs, uvs);
                    llama->receiveVideoFrame(yuv);
                }
            }
            break;
        }

        case MSG_TTS_START: {
            if (!tts) {
                if (cfg.piper_model[0]) setenv("PIPER_MODEL", cfg.piper_model, 1);
                if (cfg.espeak_data[0]) setenv("ESPEAK_DATA_PATH", cfg.espeak_data, 1);
                fprintf(stderr, "[whillats_server] Creating TTS (isServer=%d)...\n",
                        getenv("WHILLATS_IS_SERVER") != nullptr);
                tts = std::make_unique<WhillatsTTS>(ttsCb);
                fprintf(stderr, "[whillats_server] TTS created, calling start()...\n");
                if (tts->start())
                    fprintf(stderr, "[whillats_server] TTS started (rate=%d)\n", WhillatsTTS::getSampleRate());
                else
                    fprintf(stderr, "[whillats_server] TTS failed to start\n");
            }
            break;
        }

        case MSG_TTS_STOP:
            if (tts) { tts->stop(); tts.reset(); }
            break;

        case MSG_TTS_SPEAK: {
            if (tts && h.len > 0) {
                // payload: [uint16 lang_len][lang bytes][text bytes]
                if (h.len >= 2) {
                    uint16_t lang_len;
                    memcpy(&lang_len, payload.data(), 2);
                    if (2 + lang_len <= h.len) {
                        std::string lang(reinterpret_cast<char*>(payload.data()+2), lang_len);
                        std::string text(reinterpret_cast<char*>(payload.data()+2+lang_len), h.len-2-lang_len);
                        tts->queueText(text.c_str(), lang.c_str());
                    }
                }
            }
            break;
        }

        case MSG_SHUTDOWN:
            g_running = false;
            break;

        default:
            fprintf(stderr, "[whillats_server] Unknown msg type 0x%02x len=%u\n", h.type, h.len);
            break;
        }
    }

    if (tts) tts->stop();
    if (llama) llama->stop();
    if (whisper) whisper->stop();

    fprintf(stderr, "[whillats_server] Exiting\n");
    return 0;
}
