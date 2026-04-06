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

#pragma once

#include <string>
#include <atomic>
#include "whillats.h"

// Thin HTTP client for llama-server /v1/chat/completions (SSE streaming).
// Uses raw POSIX sockets — no external HTTP library required.
// llama-server is always localhost so no TLS is needed.
class LlamaHttpClient {
public:
    explicit LlamaHttpClient(const std::string& base_url);
    ~LlamaHttpClient() = default;

    // Poll /health until server is ready (or timeout_ms elapses).
    bool waitReady(int timeout_ms = 30000);

    // Streaming chat completion.
    // Calls cb.OnResponseComplete(true, token) for each streamed token.
    // Calls cb.OnResponseComplete(false, nullptr) when stream ends.
    void chat(const std::string& system_prompt,
              const std::string& user_prompt,
              WhillatsSetResponseCallback cb);

    // Multimodal: user_prompt + JPEG image encoded as base64.
    void chatWithImage(const std::string& system_prompt,
                       const std::string& user_prompt,
                       const std::string& jpeg_base64,
                       WhillatsSetResponseCallback cb);

    void stopGeneration() { _stop = true; }

private:
    std::string _host;
    int         _port;
    std::string _base_path;  // URL path prefix (usually "/")
    std::atomic<bool> _stop{false};

    bool parseUrl(const std::string& url);
    int  openSocket();
    bool getHealth();
    void doStreamingPost(const std::string& json_body,
                         WhillatsSetResponseCallback cb);
};
