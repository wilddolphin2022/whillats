/*
 *  Piper TTS subprocess implementation.
 *  Child process loads Piper/ONNX in pure libstdc++ context.
 *  Communication via pipes using a simple protocol:
 *    Parent->Child: [uint32_t text_len][text bytes]
 *    Child->Parent: [int32_t sample_rate][uint32_t num_samples][int16_t samples...]
 *    text_len=0 means shutdown
 */

#include "piper_subprocess.h"
#include "whisper_helpers.h"
#include <piper.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <cstring>
#include <cstdio>

static bool write_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) return false;
        p += n; len -= n;
    }
    return true;
}

static bool read_all(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n <= 0) return false;
        p += n; len -= n;
    }
    return true;
}

// Child process main loop
static void child_main(int read_fd, int write_fd,
                       const std::string& model_path,
                       const std::string& espeak_data) {
    piper_synthesizer* synth = piper_create(
        model_path.c_str(), nullptr, espeak_data.c_str());
    if (!synth) {
        fprintf(stderr, "[PiperChild] Failed to create synthesizer\n");
        _exit(1);
    }
    fprintf(stderr, "[PiperChild] Synthesizer ready\n");

    while (true) {
        uint32_t text_len = 0;
        if (!read_all(read_fd, &text_len, sizeof(text_len))) break;
        if (text_len == 0) break;

        std::string text(text_len, '\0');
        if (!read_all(read_fd, &text[0], text_len)) break;

        fprintf(stderr, "[PiperChild] Synthesizing: %s\n", text.c_str());

        int rc = piper_synthesize_start(synth, text.c_str(), nullptr);
        if (rc != PIPER_OK) {
            int32_t sr = 0; uint32_t ns = 0;
            write_all(write_fd, &sr, sizeof(sr));
            write_all(write_fd, &ns, sizeof(ns));
            continue;
        }

        std::vector<int16_t> all_audio;
        int sample_rate = 16000;
        piper_audio_chunk chunk;

        while (true) {
            rc = piper_synthesize_next(synth, &chunk);
            if (rc == PIPER_DONE) break;
            if (rc != PIPER_OK) break;
            if (chunk.samples && chunk.num_samples > 0) {
                sample_rate = chunk.sample_rate;
                for (size_t i = 0; i < chunk.num_samples; ++i) {
                    float v = chunk.samples[i] * 32767.0f;
                    if (v > 32767.0f) v = 32767.0f;
                    if (v < -32768.0f) v = -32768.0f;
                    all_audio.push_back(static_cast<int16_t>(v));
                }
            }
            if (chunk.is_last) break;
        }

        int32_t sr = static_cast<int32_t>(sample_rate);
        uint32_t ns = static_cast<uint32_t>(all_audio.size());
        write_all(write_fd, &sr, sizeof(sr));
        write_all(write_fd, &ns, sizeof(ns));
        if (ns > 0) {
            write_all(write_fd, all_audio.data(), ns * sizeof(int16_t));
        }
        fprintf(stderr, "[PiperChild] Sent %u samples at %dHz\n", ns, sr);
    }

    piper_free(synth);
    fprintf(stderr, "[PiperChild] Exiting\n");
    _exit(0);
}

PiperSubprocess::PiperSubprocess() {}

PiperSubprocess::~PiperSubprocess() { stop(); }

bool PiperSubprocess::start(const std::string& model_path,
                            const std::string& espeak_data) {
    if (_running) return true;

    int pipe_to[2], pipe_from[2];
    if (pipe(pipe_to) != 0 || pipe(pipe_from) != 0) {
        LOG_E("PiperSubprocess: pipe() failed");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        LOG_E("PiperSubprocess: fork() failed");
        return false;
    }

    if (pid == 0) {
        // Child
        close(pipe_to[1]);
        close(pipe_from[0]);
        child_main(pipe_to[0], pipe_from[1], model_path, espeak_data);
        _exit(0);
    }

    // Parent
    close(pipe_to[0]);
    close(pipe_from[1]);
    _toChild = pipe_to[1];
    _fromChild = pipe_from[0];
    _child = pid;
    _running = true;
    LOG_I("PiperSubprocess: Started child pid=" << pid);
    return true;
}

void PiperSubprocess::stop() {
    if (!_running) return;
    // Send shutdown (text_len=0)
    uint32_t zero = 0;
    write_all(_toChild, &zero, sizeof(zero));
    close(_toChild); _toChild = -1;
    close(_fromChild); _fromChild = -1;
    if (_child > 0) {
        int status;
        waitpid(_child, &status, 0);
        _child = -1;
    }
    _running = false;
    LOG_I("PiperSubprocess: Stopped");
}

std::vector<int16_t> PiperSubprocess::synthesize(const std::string& text) {
    if (!_running || text.empty()) return {};

    uint32_t text_len = static_cast<uint32_t>(text.size());
    if (!write_all(_toChild, &text_len, sizeof(text_len))) return {};
    if (!write_all(_toChild, text.data(), text_len)) return {};

    int32_t sr = 0;
    uint32_t ns = 0;
    if (!read_all(_fromChild, &sr, sizeof(sr))) return {};
    if (!read_all(_fromChild, &ns, sizeof(ns))) return {};

    if (sr > 0) _sampleRate = sr;
    if (ns == 0) return {};

    std::vector<int16_t> audio(ns);
    if (!read_all(_fromChild, audio.data(), ns * sizeof(int16_t))) return {};
    return audio;
}
