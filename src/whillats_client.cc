#include "whillats_client.h"
#include "whillats_ipc.h"

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace whillats_ipc;

// --- WhillatsServerConnection ---

WhillatsServerConnection::WhillatsServerConnection() {}

WhillatsServerConnection::~WhillatsServerConnection() { stop(); }

bool WhillatsServerConnection::start(const std::string& server_path, const ConfigMsg& cfg) {
    int parent_to_child[2], child_to_parent[2];
    if (pipe(parent_to_child) != 0 || pipe(child_to_parent) != 0) {
        perror("[whillats_client] pipe");
        return false;
    }

    _serverPid = fork();
    if (_serverPid < 0) {
        perror("[whillats_client] fork");
        return false;
    }

    if (_serverPid == 0) {
        // Child
        close(parent_to_child[1]);
        close(child_to_parent[0]);

        char read_fd_str[16], write_fd_str[16];
        snprintf(read_fd_str, sizeof(read_fd_str), "%d", parent_to_child[0]);
        snprintf(write_fd_str, sizeof(write_fd_str), "%d", child_to_parent[1]);

        execl(server_path.c_str(), "whillats_server", read_fd_str, write_fd_str, nullptr);
        perror("[whillats_client] execl failed");
        _exit(1);
    }

    // Parent
    close(parent_to_child[0]);
    close(child_to_parent[1]);
    _writeFd = parent_to_child[1];
    _readFd  = child_to_parent[0];

    _running = true;

    // Send config
    write_msg(_writeFd, MSG_CONFIG, &cfg, sizeof(cfg));

    // Start reader thread
    _reader = std::thread([this]{ readerThread(); });

    fprintf(stderr, "[whillats_client] Server started pid=%d\n", _serverPid);
    return true;
}

void WhillatsServerConnection::stop() {
    if (!_running) return;
    _running = false;

    write_msg(_writeFd, MSG_SHUTDOWN, nullptr, 0);

    if (_writeFd >= 0) { close(_writeFd); _writeFd = -1; }

    if (_reader.joinable()) _reader.join();

    if (_readFd >= 0) { close(_readFd); _readFd = -1; }

    if (_serverPid > 0) {
        int status;
        waitpid(_serverPid, &status, 0);
        fprintf(stderr, "[whillats_client] Server exited status=%d\n", status);
        _serverPid = -1;
    }
}

bool WhillatsServerConnection::sendMsg(uint8_t type, const void* data, uint32_t len) {
    std::lock_guard<std::mutex> lock(_writeMutex);
    return write_msg(_writeFd, type, data, len);
}

void WhillatsServerConnection::readerThread() {
    Header h;
    while (_running && read_header(_readFd, h)) {
        std::vector<uint8_t> payload(h.len);
        if (h.len > 0 && !read_exact(_readFd, payload.data(), h.len)) break;

        switch (h.type) {

        case MSG_WHISPER_RESULT:
            if (_whisperCb.callback_) {
                std::string text(reinterpret_cast<char*>(payload.data()), h.len);
                _whisperCb.OnResponseComplete(true, text.c_str());
            }
            break;

        case MSG_WHISPER_LANGUAGE:
            if (_langCb.callback_) {
                std::string lang(reinterpret_cast<char*>(payload.data()), h.len);
                _langCb.OnLanguageChanged(true, lang.c_str());
            }
            break;

        case MSG_LLAMA_RESPONSE:
            if (_llamaCb.callback_) {
                std::string resp(reinterpret_cast<char*>(payload.data()), h.len);
                _llamaCb.OnResponseComplete(true, resp.c_str());
            }
            break;

        case MSG_TTS_AUDIO: {
            if (_ttsCb.callback_ && h.len >= sizeof(TtsAudioMsg)) {
                TtsAudioMsg hdr;
                memcpy(&hdr, payload.data(), sizeof(hdr));
                const uint16_t* samples = reinterpret_cast<const uint16_t*>(
                    payload.data() + sizeof(TtsAudioMsg));
                size_t n = hdr.num_samples;
                if (sizeof(TtsAudioMsg) + n * sizeof(int16_t) <= h.len) {
                    std::vector<uint16_t> buf(samples, samples + n);
                    _ttsCb.OnBufferComplete(true, buf);
                }
            }
            break;
        }

        case MSG_TTS_DONE:
            if (_ttsCb.callback_)
                _ttsCb.OnSynthesisComplete();
            break;

        default:
            break;
        }
    }
    fprintf(stderr, "[whillats_client] Reader thread exiting\n");
}

// --- WhillatsTranscriberClient ---

WhillatsTranscriberClient::WhillatsTranscriberClient(WhillatsServerConnection& conn)
    : _conn(conn) {}

WhillatsTranscriberClient::~WhillatsTranscriberClient() { stop(); }

bool WhillatsTranscriberClient::start() {
    _conn.sendMsg(MSG_WHISPER_START, nullptr, 0);
    _started = true;
    return true;
}

void WhillatsTranscriberClient::stop() {
    if (_started) {
        _conn.sendMsg(MSG_WHISPER_STOP, nullptr, 0);
        _started = false;
    }
}

void WhillatsTranscriberClient::processAudioBuffer(uint8_t* buffer, size_t size) {
    if (!_started) return;
    _conn.sendMsg(MSG_WHISPER_AUDIO, buffer, (uint32_t)size);
}

// --- WhillatsLlamaClient ---

WhillatsLlamaClient::WhillatsLlamaClient(WhillatsServerConnection& conn)
    : _conn(conn) {}

WhillatsLlamaClient::~WhillatsLlamaClient() { stop(); }

bool WhillatsLlamaClient::start() {
    _conn.sendMsg(MSG_LLAMA_START, nullptr, 0);
    _started = true;
    return true;
}

void WhillatsLlamaClient::stop() {
    if (_started) {
        _conn.sendMsg(MSG_LLAMA_STOP, nullptr, 0);
        _started = false;
    }
}

void WhillatsLlamaClient::askLlama(const char* prompt) {
    if (_started && prompt)
        _conn.sendMsg(MSG_LLAMA_ASK, prompt, (uint32_t)strlen(prompt));
}

void WhillatsLlamaClient::receiveVideoFrame(const YUVData& yuv) {
    if (!_started || !yuv.y || !yuv.u || !yuv.v) return;
    size_t payload_sz = 16 + yuv.y_size + 2 * yuv.uv_size;
    std::vector<uint8_t> buf(payload_sz);
    int32_t w = yuv.width, h = yuv.height;
    int32_t ys = (int32_t)yuv.y_size, uvs = (int32_t)yuv.uv_size;
    memcpy(buf.data(),    &w,  4);
    memcpy(buf.data()+4,  &h,  4);
    memcpy(buf.data()+8,  &ys, 4);
    memcpy(buf.data()+12, &uvs,4);
    memcpy(buf.data()+16, yuv.y.get(), yuv.y_size);
    memcpy(buf.data()+16+yuv.y_size, yuv.u.get(), yuv.uv_size);
    memcpy(buf.data()+16+yuv.y_size+yuv.uv_size, yuv.v.get(), yuv.uv_size);
    _conn.sendMsg(MSG_LLAMA_VIDEO_FRAME, buf.data(), (uint32_t)payload_sz);
}

// --- WhillatsTTSClient ---

WhillatsTTSClient::WhillatsTTSClient(WhillatsServerConnection& conn)
    : _conn(conn) {}

WhillatsTTSClient::~WhillatsTTSClient() { stop(); }

bool WhillatsTTSClient::start() {
    _conn.sendMsg(MSG_TTS_START, nullptr, 0);
    _started = true;
    return true;
}

void WhillatsTTSClient::stop() {
    if (_started) {
        _conn.sendMsg(MSG_TTS_STOP, nullptr, 0);
        _started = false;
    }
}

void WhillatsTTSClient::queueText(const char* text, const char* language) {
    if (!_started || !text) return;
    const char* lang = language ? language : "en";
    uint16_t lang_len = (uint16_t)strlen(lang);
    uint32_t text_len = (uint32_t)strlen(text);
    uint32_t total = 2 + lang_len + text_len;
    std::vector<uint8_t> buf(total);
    memcpy(buf.data(), &lang_len, 2);
    memcpy(buf.data()+2, lang, lang_len);
    memcpy(buf.data()+2+lang_len, text, text_len);
    _conn.sendMsg(MSG_TTS_SPEAK, buf.data(), total);
}
