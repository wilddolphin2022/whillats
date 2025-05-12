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

#ifndef SYNTHESIS_H
#define SYNTHESIS_H

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <cstdint>
#include "whillats.h"
#include <queue>
#include <condition_variable>

// Class to manage external speech synthesis process and deliver audio buffers via callback
class Synthesis {
public:
    explicit Synthesis(WhillatsSetAudioCallback callback);
    ~Synthesis();

    bool start();
    void stop();
    void queueText(const std::string& text, const std::string& language);
    // Enqueue multiple text-language pairs and stream them continuously
    void synthesizeBatch(const std::vector<std::pair<std::string, std::string> >& items);

    static int getSampleRate();

private:
    WhillatsSetAudioCallback _callback;
    std::string _dylibPath;
    
    std::thread _worker_thread;
    std::queue<std::pair<std::string, std::string> > _text_queue;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    bool _worker_running;
    int _pipe_to_synth[2];
    int _pipe_from_synth[2];
    pid_t _synth_pid;
    std::thread _reader_thread;
    std::mutex _write_mutex;

    static bool readAll(int fd, void* buf, size_t size);
    static void readerThreadFunction(int read_fd, WhillatsSetAudioCallback callback);
    void workerFunction();
};

#endif // SYNTHESIS_H