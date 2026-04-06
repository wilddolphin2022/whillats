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

#include "llama_http_client.h"
#include "whisper_helpers.h"

#include <cstring>
#include <cerrno>
#include <sstream>
#include <thread>
#include <chrono>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

// ---------------------------------------------------------------------------
// Tiny base64 encoder (RFC 4648) — no external dep
// ---------------------------------------------------------------------------
static const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t b = (uint32_t)data[i] << 16;
        if (i + 1 < len) b |= (uint32_t)data[i+1] << 8;
        if (i + 2 < len) b |= (uint32_t)data[i+2];
        out += kB64[(b >> 18) & 0x3f];
        out += kB64[(b >> 12) & 0x3f];
        out += (i + 1 < len) ? kB64[(b >>  6) & 0x3f] : '=';
        out += (i + 2 < len) ? kB64[(b      ) & 0x3f] : '=';
    }
    return out;
}

// ---------------------------------------------------------------------------
// Minimal JSON string escape
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// URL parser: "http://host:port/path" → host, port, path
// ---------------------------------------------------------------------------
bool LlamaHttpClient::parseUrl(const std::string& url) {
    // strip "http://"
    std::string rest = url;
    if (rest.substr(0, 7) == "http://")  rest = rest.substr(7);
    else if (rest.substr(0, 8) == "https://") rest = rest.substr(8);

    auto slash = rest.find('/');
    std::string authority = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    _base_path = (slash == std::string::npos) ? "/" : rest.substr(slash);

    auto colon = authority.find(':');
    if (colon != std::string::npos) {
        _host = authority.substr(0, colon);
        _port = std::stoi(authority.substr(colon + 1));
    } else {
        _host = authority;
        _port = 80;
    }
    return !_host.empty();
}

LlamaHttpClient::LlamaHttpClient(const std::string& base_url) {
    parseUrl(base_url);
}

// ---------------------------------------------------------------------------
// Open a connected TCP socket to _host:_port
// ---------------------------------------------------------------------------
int LlamaHttpClient::openSocket() {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(_port);
    if (getaddrinfo(_host.c_str(), port_str.c_str(), &hints, &res) != 0 || !res)
        return -1;

    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }

    // Disable Nagle for low-latency streaming
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    if (::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        ::close(fd); freeaddrinfo(res); return -1;
    }
    freeaddrinfo(res);
    return fd;
}

// ---------------------------------------------------------------------------
// GET /health → look for "ok" status
// ---------------------------------------------------------------------------
bool LlamaHttpClient::getHealth() {
    int fd = openSocket();
    if (fd < 0) return false;

    std::string req =
        "GET " + _base_path + "health HTTP/1.1\r\n"
        "Host: " + _host + ":" + std::to_string(_port) + "\r\n"
        "Connection: close\r\n\r\n";
    if (::write(fd, req.data(), req.size()) < 0) { ::close(fd); return false; }

    // Read response (just need status code or "ok" body)
    char buf[512] = {};
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) return false;
    std::string resp(buf, n);
    // Accept HTTP 200 or body containing "ok"
    return resp.find("200") != std::string::npos || resp.find("\"ok\"") != std::string::npos;
}

bool LlamaHttpClient::waitReady(int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (getHealth()) {
            LOG_I("llama-server ready at " << _host << ":" << _port);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    LOG_E("llama-server not ready after " << timeout_ms << "ms");
    return false;
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions with streaming SSE
// ---------------------------------------------------------------------------
void LlamaHttpClient::doStreamingPost(const std::string& json_body,
                                       WhillatsSetResponseCallback cb) {
    _stop = false;
    int fd = openSocket();
    if (fd < 0) {
        LOG_E("LlamaHttpClient: failed to connect " << _host << ":" << _port);
        cb.OnResponseComplete(false, nullptr);
        return;
    }

    // Build HTTP request
    std::string req =
        "POST " + _base_path + "v1/chat/completions HTTP/1.1\r\n"
        "Host: " + _host + ":" + std::to_string(_port) + "\r\n"
        "Content-Type: application/json\r\n"
        "Accept: text/event-stream\r\n"
        "Connection: close\r\n"
        "Content-Length: " + std::to_string(json_body.size()) + "\r\n"
        "\r\n" + json_body;

    if (::write(fd, req.data(), req.size()) < 0) {
        LOG_E("LlamaHttpClient: write failed");
        ::close(fd);
        cb.OnResponseComplete(false, nullptr);
        return;
    }

    // Read response: skip headers, then parse SSE lines
    // Buffer line-by-line
    std::string line_buf;
    bool headers_done = false;
    std::string header_accum;

    auto read_char = [&](char& c) -> bool {
        if (_stop) return false;
        return ::read(fd, &c, 1) == 1;
    };

    auto read_line = [&](std::string& line) -> bool {
        line.clear();
        char c;
        while (read_char(c)) {
            if (c == '\n') return true;
            if (c != '\r') line += c;
        }
        return !line.empty();  // EOF with partial line is OK
    };

    // Skip HTTP headers
    {
        std::string line;
        while (read_line(line)) {
            if (line.empty()) { headers_done = true; break; }
            // Check HTTP status on first line
            if (line.find("HTTP/") == 0 && line.find("200") == std::string::npos) {
                LOG_E("LlamaHttpClient: bad HTTP status: " << line);
                ::close(fd);
                cb.OnResponseComplete(false, nullptr);
                return;
            }
        }
    }

    if (!headers_done) {
        ::close(fd);
        cb.OnResponseComplete(false, nullptr);
        return;
    }

    // Parse SSE stream
    // Each event is: "data: <json>\n\n" or "data: [DONE]\n\n"
    std::string line;
    while (read_line(line) && !_stop) {
        if (line.empty()) continue;  // blank line between events
        if (line.substr(0, 6) != "data: ") continue;
        std::string data = line.substr(6);

        if (data == "[DONE]") break;

        // Extract "content" from delta: {"choices":[{"delta":{"content":"tok"}}]}
        // Must match "content":" but NOT "reasoning_content":" (thinking tokens).
        // Search for the exact key preceded by a non-word char (quote or comma or {).
        std::string token;
        size_t search_from = 0;
        while (true) {
            auto pos = data.find("\"content\":", search_from);
            if (pos == std::string::npos) break;
            // Reject if preceded by a word char (i.e. part of "reasoning_content")
            if (pos > 0 && data[pos-1] != '"' && data[pos-1] != ',' &&
                            data[pos-1] != '{' && data[pos-1] != ' ') {
                search_from = pos + 10;
                continue;
            }
            size_t val_pos = pos + 10; // after "content":
            // skip whitespace
            while (val_pos < data.size() && data[val_pos] == ' ') ++val_pos;
            // skip null values
            if (val_pos + 4 <= data.size() && data.substr(val_pos, 4) == "null") break;
            if (val_pos >= data.size() || data[val_pos] != '"') break;
            ++val_pos; // skip opening quote
            while (val_pos < data.size()) {
                char c = data[val_pos++];
                if (c == '"') break;
                if (c == '\\' && val_pos < data.size()) {
                    char esc = data[val_pos++];
                    switch (esc) {
                        case 'n': token += '\n'; break;
                        case 'r': token += '\r'; break;
                        case 't': token += '\t'; break;
                        default:  token += esc;  break;
                    }
                } else {
                    token += c;
                }
            }
            break;
        }
        if (!token.empty()) {
            cb.OnResponseComplete(true, token.c_str());
        }
    }

    ::close(fd);
    cb.OnResponseComplete(false, nullptr);  // signal done
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void LlamaHttpClient::chat(const std::string& system_prompt,
                            const std::string& user_prompt,
                            WhillatsSetResponseCallback cb) {
    std::string body =
        "{\"stream\":true,\"messages\":["
        "{\"role\":\"system\",\"content\":\"" + json_escape(system_prompt) + "\"},"
        "{\"role\":\"user\",\"content\":\"" + json_escape(user_prompt) + "\"}"
        "]}";
    doStreamingPost(body, cb);
}

void LlamaHttpClient::chatWithImage(const std::string& system_prompt,
                                     const std::string& user_prompt,
                                     const std::string& jpeg_base64,
                                     WhillatsSetResponseCallback cb) {
    // OpenAI vision format: content is an array with text + image_url
    std::string body =
        "{\"stream\":true,\"messages\":["
        "{\"role\":\"system\",\"content\":\"" + json_escape(system_prompt) + "\"},"
        "{\"role\":\"user\",\"content\":["
          "{\"type\":\"text\",\"text\":\"" + json_escape(user_prompt) + "\"},"
          "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,"
          + jpeg_base64 + "\"}}"
        "]}]}";
    doStreamingPost(body, cb);
}
