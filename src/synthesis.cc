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

#include <iostream>  // for LOG_I and std::endl
#include <mutex>     // for std::mutex and std::lock_guard
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <cstring>
#include <thread>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <string>    // for std::string

#include "synthesis.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"
// Helper to read exactly `size` bytes from fd into buf. Returns false on EOF or error.
bool Synthesis::readAll(int fd, void* buf, size_t size) {
    uint8_t* ptr = static_cast<uint8_t*>(buf);
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t r = ::read(fd, ptr, remaining);
        if (r <= 0) {
            return false;
        }
        ptr += r;
        remaining -= static_cast<size_t>(r);
    }
    return true;
}

Synthesis::Synthesis(WhillatsSetAudioCallback callback)
    : _callback(callback),
      _dylibPath(getDylibPath()),
      _synth_pid(-1),
      _running(false) {
    _pipe_to_synth[0] = _pipe_to_synth[1] = -1;
    _pipe_from_synth[0] = _pipe_from_synth[1] = -1;
}

Synthesis::~Synthesis() {
    stop();
}

bool Synthesis::start() {
    // Ensure the synthesis executable exists
    std::string full = _dylibPath + "/synthesis";
    if (access(full.c_str(), F_OK) == -1) {
        LOG_E("Synthesis executable not found at " << full);
        return false;
    }
    return true;
}

void Synthesis::stop() {
    // No persistent process to stop.
}

int Synthesis::getSampleRate() {
    return 16000;
}

void Synthesis::queueText(const std::string& text, const std::string& language) {
    if (!start()) return;
    // Spawn a synthesis process for this utterance
    std::string full = _dylibPath + "/synthesis";
    int pipe_to[2], pipe_from[2];
    if (pipe(pipe_to) == -1 || pipe(pipe_from) == -1) {
        LOG_E("Failed to create pipes for synthesis");
        return;
    }
    pid_t pid = fork();
    if (pid < 0) {
        LOG_E("Failed to fork synthesis process");
        return;
    } else if (pid == 0) {
        // Child: set up pipes and exec
        close(pipe_to[1]);
        dup2(pipe_to[0], STDIN_FILENO);
        close(pipe_to[0]);
        close(pipe_from[0]);
        dup2(pipe_from[1], STDOUT_FILENO);
        close(pipe_from[1]);
        execl(full.c_str(), full.c_str(), nullptr);
        _exit(1);
    }
    // Parent: close unused ends
    close(pipe_to[0]);
    close(pipe_from[1]);
    // Send text and language
    uint32_t sz = static_cast<uint32_t>(text.size());
    ::write(pipe_to[1], &sz, sizeof(sz));
    ::write(pipe_to[1], text.data(), sz);
    uint32_t lsz = static_cast<uint32_t>(language.size());
    ::write(pipe_to[1], &lsz, sizeof(lsz));
    ::write(pipe_to[1], language.data(), lsz);
    close(pipe_to[1]); // EOF for child
    // Read and dispatch audio buffers
    while (true) {
        uint32_t buf_size = 0;
        if (!readAll(pipe_from[0], &buf_size, sizeof(buf_size))) break;
        if (buf_size == 0) {
            _callback.OnSynthesisComplete();
            break;
        }
        size_t samples = buf_size / sizeof(uint16_t);
        std::vector<uint16_t> buffer(samples);
        if (!readAll(pipe_from[0], buffer.data(), buf_size)) break;
        _callback.OnBufferComplete(true, buffer);
    }
    close(pipe_from[0]);
    int status = 0;
    waitpid(pid, &status, 0);
}

// Stream a batch of text-language pairs continuously
void Synthesis::synthesizeBatch(const std::vector<std::pair<std::string, std::string>>& items) {
    if (!start()) return;
    // Launch sender thread to queue all items
    _sender_thread = std::thread([this, items]() {
        for (const auto& p : items) {
            queueText(p.first, p.second);
        }
        // After sending all, close the pipe to signal EOF
        close(_pipe_to_synth[1]);
    });
}

// Reader thread function implementation
void Synthesis::readerThreadFunction(int read_fd, WhillatsSetAudioCallback callback) {
    while (true) {
        uint32_t buf_size = 0;
        if (!readAll(read_fd, &buf_size, sizeof(buf_size))) break;
        if (buf_size == 0) {
            callback.OnSynthesisComplete();
            continue;
        }
        size_t samples = buf_size / sizeof(uint16_t);
        std::vector<uint16_t> buffer(samples);
        if (!readAll(read_fd, buffer.data(), buf_size)) break;
        callback.OnBufferComplete(true, buffer);
    }
}
