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

#include "whillats.h"
#include "whisper_helpers.h"
#include "opencv2/opencv.hpp"

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
typedef int32_t llama_token;

class LlamaSimpleChat;

class LlamaDeviceBase {
public:
  LlamaDeviceBase(const char* model_path, WhillatsSetResponseCallback callback);
  virtual ~LlamaDeviceBase();

  bool start();
  void stop();

  void askLlama(const char *prompt);

  bool setImage(const uint8_t* yuvData, int width, int height);
  void askWithImage(const char *prompt, const uint8_t* yuvData, int width, int height);

private:
  bool _running;
  std::thread _processingThread;
  std::string _model_path;

  WhillatsSetResponseCallback _responseCallback;  // Add callback member
  
  void processPrompts();
  bool initialize();
  bool RunProcessingThread();

  std::unique_ptr<LlamaSimpleChat> _llama_chat;

  // Incoming ask text queue
  std::queue<std::string> _textQueue;
  std::mutex _queueMutex;
  std::condition_variable _queueCondition;


  // Add these new members
  std::vector<llama_token> context_tokens_;
  const size_t max_context_tokens_ = 2048; // Adjust based on your model   
  
  bool TrimContext();
  bool AppendToContext(const std::vector<llama_token>& new_tokens);
};

#endif // LLAMA_DEVICE_BASE_H