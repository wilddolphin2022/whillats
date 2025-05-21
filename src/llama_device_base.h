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

#ifndef LLAMA_DEVICE_BASE_H
#define LLAMA_DEVICE_BASE_H

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <queue>
#include <thread>
#include <functional>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <memory>

#include "whillats.h"
#include "whisper_helpers.h"
#include "llama.h"
#include "clip.h"
#include "mtmd.h" // Ensure mtmd.h is included

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
typedef int32_t llama_token;

class LlamaSimpleChat;

struct Request {
    std::string              prompt;
    bool                     withImage;
    std::shared_ptr<YUVData> yuv;   // nullptr for text-only, deep-copied frame if withImage
};

class LlamaDeviceBase {
public:
    LlamaDeviceBase(const char* model_path, const char* mmproj_path, WhillatsSetResponseCallback callback);
    virtual ~LlamaDeviceBase();

    bool start();
    void stop();

    void askLlama(const char *prompt);
    void askWithImage(const char *prompt, const YUVData& yuv);

private:
    bool _running;
    std::thread _processingThread;
    std::string _model_path;
    std::string _mmproj_path;

    WhillatsSetResponseCallback _responseCallback;
    
    void processPrompts();
    bool initialize();
    bool RunProcessingThread();

    std::unique_ptr<LlamaSimpleChat> _llama_chat;

    std::deque<Request>       _requestQueue;
    std::mutex                _queueMutex;
    std::condition_variable   _queueCondition;

    uint64_t                  _lastYuvHash = 0;
    
    std::vector<llama_token> context_tokens_;
    const size_t max_context_tokens_ = 2048;
    
    bool TrimContext();
    bool AppendToContext(const std::vector<llama_token>& new_tokens);
};

#endif // LLAMA_DEVICE_BASE_H