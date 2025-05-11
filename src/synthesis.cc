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
    LOG_I("Starting synthesis process");

    std::string full = _dylibPath + "/synthesis";
    LOG_I("Checking if synthesis executable exists at " << full);
    if (access(full.c_str(), F_OK) == -1) {
        LOG_E("Synthesis executable not found at " << full);
        return false;
    }
    LOG_I("Synthesis executable found at " << full);

    if (_running) 
      return true;

    if (pipe(_pipe_to_synth) == -1 || pipe(_pipe_from_synth) == -1) 
      return false;
      
    _synth_pid = fork();
    if (_synth_pid < 0) {
        LOG_E("Failed to fork synthesis process");
        return false;
    } else if (_synth_pid == 0) {
        // child: connect pipes and exec
        close(_pipe_to_synth[1]);
        dup2(_pipe_to_synth[0], STDIN_FILENO);
        close(_pipe_to_synth[0]);
        close(_pipe_from_synth[0]);
        dup2(_pipe_from_synth[1], STDOUT_FILENO);
        close(_pipe_from_synth[1]);

        std::string full = _dylibPath + "/synthesis";
        LOG_I("Running synthesis: " << full);
        execl(full.c_str(), "synthesis", nullptr);
        _exit(1);
    } else {
        LOG_I("Synthesis process started PID: " << _synth_pid);
        close(_pipe_to_synth[0]);
        close(_pipe_from_synth[1]);
        _running = true;
        return true;
    }
}

void Synthesis::stop() {
    if (!_running) return;
    // Signal the synthesis process to exit
    close(_pipe_to_synth[1]);
    // Join sender thread if batching
    if (_sender_thread.joinable()) {
        _sender_thread.join();
    }
    int status = 0;
    waitpid(_synth_pid, &status, 0);
    // Join reader thread
    if (_reader_thread.joinable()) {
        _reader_thread.join();
    }
    close(_pipe_from_synth[0]);
    _running = false;
}

int Synthesis::getSampleRate() {
    return 16000;
}

void Synthesis::queueText(const std::string& text, const std::string& language) {
    if (!_running) return;
    // Send text and language to synthesis process
    {
        std::lock_guard<std::mutex> lock(_write_mutex);
        uint32_t sz = static_cast<uint32_t>(text.size());
        ::write(_pipe_to_synth[1], &sz, sizeof(sz));
        ::write(_pipe_to_synth[1], text.data(), sz);
        uint32_t lsz = static_cast<uint32_t>(language.size());
        ::write(_pipe_to_synth[1], &lsz, sizeof(lsz));
        ::write(_pipe_to_synth[1], language.data(), lsz);
    }
    // Read and dispatch audio buffers synchronously
    while (true) {
        uint32_t buf_size = 0;
        if (!readAll(_pipe_from_synth[0], &buf_size, sizeof(buf_size))) {
            break;
        }
        if (buf_size == 0) {
            _callback.OnSynthesisComplete();
            break;
        }
        size_t samples = buf_size / sizeof(uint16_t);
        std::vector<uint16_t> buffer(samples);
        if (!readAll(_pipe_from_synth[0], buffer.data(), buf_size)) {
            break;
        }
        _callback.OnBufferComplete(true, buffer);
    }
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
