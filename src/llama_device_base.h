/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef LLAMA_DEVICE_BASE_H
#define LLAMA_DEVICE_BASE_H

#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <chrono>

#include "whillats.h"
#include "whisper_helpers.h"
#include "llama_http_client.h"

struct Request {
    std::string              prompt;
    bool                     withImage;
    std::shared_ptr<YUVData> yuv;
};

class LlamaDeviceBase {
public:
    // server_url: e.g. "http://127.0.0.1:8080"
    LlamaDeviceBase(const char* server_url, WhillatsSetResponseCallback callback);
    virtual ~LlamaDeviceBase();

    bool start();
    void stop();
    bool isRunning() const { return _running; }
    void setThreadCount(int n) { /* unused — llama-server manages its own threads */ }

    void askLlama(const char* prompt);
    void receiveVideoFrame(const YUVData& yuv);

    bool hasMultimodalSupport() const { return true; }

private:
    std::string                     _server_url;
    std::unique_ptr<LlamaHttpClient> _http;
    WhillatsSetResponseCallback     _responseCallback;

    bool                         _running = false;
    std::thread                  _processingThread;
    std::deque<Request>          _requestQueue;
    std::mutex                   _queueMutex;
    std::condition_variable      _queueCondition;

    // Most-recent video frame for multimodal prompts
    std::shared_ptr<YUVData>     _lastFrame;
    std::mutex                   _frameMutex;

    bool RunProcessingThread();
    std::string yuvToJpegBase64(const YUVData& yuv);
};

#endif // LLAMA_DEVICE_BASE_H
