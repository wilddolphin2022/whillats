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

#include "llama_device_base.h"
#include "whisper_helpers.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// YUV420 → JPEG → base64
// ---------------------------------------------------------------------------
static void stbi_write_to_vec(void* ctx, void* data, int size) {
    auto* vec = reinterpret_cast<std::vector<uint8_t>*>(ctx);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    vec->insert(vec->end(), p, p + size);
}

static const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode_vec(const std::vector<uint8_t>& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        uint32_t b = (uint32_t)data[i] << 16;
        if (i + 1 < data.size()) b |= (uint32_t)data[i+1] << 8;
        if (i + 2 < data.size()) b |= (uint32_t)data[i+2];
        out += kB64[(b >> 18) & 0x3f];
        out += kB64[(b >> 12) & 0x3f];
        out += (i + 1 < data.size()) ? kB64[(b >>  6) & 0x3f] : '=';
        out += (i + 2 < data.size()) ? kB64[(b      ) & 0x3f] : '=';
    }
    return out;
}

std::string LlamaDeviceBase::yuvToJpegBase64(const YUVData& yuv) {
    // Convert YUV420 to RGB
    int w = yuv.width, h = yuv.height;
    std::vector<uint8_t> rgb(w * h * 3);
    const uint8_t* Y  = yuv.y.get();
    const uint8_t* U  = yuv.u.get();
    const uint8_t* V  = yuv.v.get();

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int y_val = Y[row * w + col];
            int u_val = U[(row / 2) * (w / 2) + col / 2] - 128;
            int v_val = V[(row / 2) * (w / 2) + col / 2] - 128;

            int r = y_val + (int)(1.402f  * v_val);
            int g = y_val - (int)(0.344f  * u_val) - (int)(0.714f * v_val);
            int b = y_val + (int)(1.772f  * u_val);

            int idx = (row * w + col) * 3;
            rgb[idx]   = (uint8_t)std::max(0, std::min(255, r));
            rgb[idx+1] = (uint8_t)std::max(0, std::min(255, g));
            rgb[idx+2] = (uint8_t)std::max(0, std::min(255, b));
        }
    }

    std::vector<uint8_t> jpeg_buf;
    jpeg_buf.reserve(w * h);
    stbi_write_jpg_to_func(stbi_write_to_vec, &jpeg_buf,
                           w, h, 3, rgb.data(), 70 /* quality */);
    return base64_encode_vec(jpeg_buf);
}

// ---------------------------------------------------------------------------
// LlamaDeviceBase
// ---------------------------------------------------------------------------
LlamaDeviceBase::LlamaDeviceBase(const char* server_url,
                                 WhillatsSetResponseCallback callback)
    : _server_url(server_url ? server_url : "http://127.0.0.1:8080")
    , _http(std::make_unique<LlamaHttpClient>(_server_url))
    , _responseCallback(callback) {}

LlamaDeviceBase::~LlamaDeviceBase() {
    stop();
}

bool LlamaDeviceBase::start() {
    if (_running) return true;

    LOG_I("LlamaDeviceBase: connecting to llama-server at " << _server_url);
    if (!_http->waitReady(60000)) {
        LOG_E("LlamaDeviceBase: llama-server not available at " << _server_url);
        return false;
    }

    _running = true;
    _processingThread = std::thread([this]() { RunProcessingThread(); });
    LOG_I("LlamaDeviceBase: started (llama-server mode)");
    return true;
}

void LlamaDeviceBase::stop() {
    if (!_running) return;
    _running = false;
    _http->stopGeneration();
    _queueCondition.notify_all();
    if (_processingThread.joinable())
        _processingThread.join();
}

void LlamaDeviceBase::askLlama(const char* prompt) {
    if (!prompt || !_running) return;
    Request req;
    req.prompt = prompt;
    req.withImage = false;
    {
        std::lock_guard<std::mutex> lk(_frameMutex);
        if (_lastFrame) {
            req.withImage = true;
            req.yuv = _lastFrame;
        }
    }
    {
        std::lock_guard<std::mutex> lk(_queueMutex);
        _requestQueue.push_back(std::move(req));
    }
    _queueCondition.notify_one();
}

void LlamaDeviceBase::receiveVideoFrame(const YUVData& yuv) {
    auto frame = std::make_shared<YUVData>();
    frame->width  = yuv.width;
    frame->height = yuv.height;
    frame->y_size  = yuv.y_size;
    frame->uv_size = yuv.uv_size;
    frame->y = std::make_unique<uint8_t[]>(yuv.y_size);
    frame->u = std::make_unique<uint8_t[]>(yuv.uv_size);
    frame->v = std::make_unique<uint8_t[]>(yuv.uv_size);
    memcpy(frame->y.get(), yuv.y.get(), yuv.y_size);
    memcpy(frame->u.get(), yuv.u.get(), yuv.uv_size);
    memcpy(frame->v.get(), yuv.v.get(), yuv.uv_size);

    std::lock_guard<std::mutex> lk(_frameMutex);
    _lastFrame = std::move(frame);
}

bool LlamaDeviceBase::RunProcessingThread() {
    const std::string system_prompt =
        "You are a helpful multilingual voice assistant. "
        "Always respond in the same language the user speaks. "
        "Keep answers concise — suitable for voice output.";

    while (_running) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(_queueMutex);
            _queueCondition.wait(lk, [this] {
                return !_requestQueue.empty() || !_running;
            });
            if (!_running) break;
            req = std::move(_requestQueue.front());
            _requestQueue.pop_front();
        }

        LOG_I("LlamaDeviceBase: asking llama-server: " << req.prompt.substr(0, 80));
        if (req.withImage && req.yuv) {
            std::string b64 = yuvToJpegBase64(*req.yuv);
            _http->chatWithImage(system_prompt, req.prompt, b64, _responseCallback);
        } else {
            _http->chat(system_prompt, req.prompt, _responseCallback);
        }
    }
    return true;
}
