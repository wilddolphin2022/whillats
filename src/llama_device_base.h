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
#include <set>
#include <algorithm>
#include <memory>
#include <chrono>

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

class LlamaSimpleChat {
public:
  LlamaSimpleChat();
  ~LlamaSimpleChat();

  // Non-copyable and non-movable (std::atomic members)
  LlamaSimpleChat(const LlamaSimpleChat&) = delete;
  LlamaSimpleChat& operator=(const LlamaSimpleChat&) = delete;
  LlamaSimpleChat(LlamaSimpleChat&&) = delete;
  LlamaSimpleChat& operator=(LlamaSimpleChat&&) = delete;

  // Unified setters
  bool SetModelPaths(const std::string &path, const std::string &mmproj_path);
  bool SetNGL(int layers);
  bool SetContextSize(int size);
  void StopGeneration();

  bool Initialize();
  std::string generate(const std::string& request, WhillatsSetResponseCallback callback);
  std::string generateFromImage(YUVData* yuv, const std::string& prompt, WhillatsSetResponseCallback callback);

  bool InitializeContext();
  void FreeContext();

  bool LoadModel();

  std::string model_path_;
  std::string mmproj_path_;
  int ngl_ = 10;
  int n_predict_ = 4096;
  std::string prompt_ = "You are a helpful assistant.";

  llama_model* model_ = nullptr;
  const llama_vocab* vocab_ = nullptr;
  llama_context* ctx_ = nullptr;
  llama_sampler* smpl_ = nullptr;
  
  std::atomic<bool> continue_{false};

  bool isRepetitive(const std::string& text, size_t minPatternLength = 4);
  bool isCompleteSentence(const std::string &text);

  std::chrono::steady_clock::time_point _lastResponseStart;
  std::chrono::steady_clock::time_point _lastResponseEnd;

  // Vision via mtmd
  mtmd::context_ptr ctx_mtmd_;
  void DetectStoppingTokens();
  std::deque<llama_token> context_tokens_;
  int n_past_ = 0;
  std::set<llama_token> stopping_token_ids_;
  std::vector<std::string> stopping_token_strings_;
  bool ResetContextForImage();
};

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
    
    // New method to receive video frames
    void receiveVideoFrame(const YUVData& yuv);
    
    // Debug/monitoring methods
    size_t getImageQueueSize() const;
    bool hasMultimodalSupport() const { return _hasMultimodalModel; }
    void recheckMultimodalSupport();

private:
    bool _running;
    std::atomic<bool> _destructing_ {false};

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
    
    // Image management for video frames
    struct TimestampedImage {
        std::shared_ptr<YUVData> yuv;
        std::chrono::steady_clock::time_point timestamp;
        uint64_t hash;
    };
    
    std::deque<TimestampedImage> _imageQueue;
    mutable std::mutex           _imageMutex;
    bool                         _hasMultimodalModel;
    int                          _imageRetentionMs;
    
    // Helper methods
    bool TrimContext();
    bool AppendToContext(const std::vector<llama_token>& new_tokens);
    void cleanupOldImages();
    std::shared_ptr<YUVData> getRecentImage();
    bool detectMultimodalSupport();
};

#endif // LLAMA_DEVICE_BASE_H