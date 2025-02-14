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

#pragma once

#include <mutex>
#include <iomanip>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <atomic>

// Define log levels
enum class LogLevel {
    VERBOSE,
    INFO,
    WARNING,
    ERROR
};

// Current log level - can be changed at runtime
static LogLevel g_currentLogLevel = LogLevel::INFO;

// Forward declare LogMessage for macro use
class LogMessage;

// Time measurement function
inline int64_t timeMillis() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// Logger class
class LogMessage {
public:
    LogMessage(const char* severity, const char* file, int line) {
        stream_ << "[" << severity << "] " 
               << timeMillis() << "ms "
               << file << ":" << line << ": ";
    }

    // Destructor to flush the message
    ~LogMessage() {
        std::cout << stream_.str();
    }

    std::ostream& stream() { return stream_; }

private:
    std::ostringstream stream_;
};

// Macro to enable/disable logging
#define ENABLE_LOGGING 1

#if ENABLE_LOGGING
    #define LOG_V(...) do { if (g_currentLogLevel <= LogLevel::VERBOSE) { LogMessage("VERBOSE", __FILE__, __LINE__).stream() << __VA_ARGS__ << std::endl; } } while(0)
    #define LOG_I(...) do { if (g_currentLogLevel <= LogLevel::INFO) { LogMessage("INFO", __FILE__, __LINE__).stream() << __VA_ARGS__ << std::endl; } } while(0)
    #define LOG_W(...) do { if (g_currentLogLevel <= LogLevel::WARNING) { LogMessage("WARNING", __FILE__, __LINE__).stream() << __VA_ARGS__ << std::endl; } } while(0)
    #define LOG_E(...) do { if (g_currentLogLevel <= LogLevel::ERROR) { LogMessage("ERROR", __FILE__, __LINE__).stream() << __VA_ARGS__ << std::endl; } } while(0)
#else
    #define LOG_V(...) ((void)0)
    #define LOG_I(...) ((void)0)
    #define LOG_W(...) ((void)0)
    #define LOG_E(...) ((void)0)
#endif

template<typename T>
class AudioRingBuffer {
public:
    AudioRingBuffer(size_t size) 
        : _buffer(size)
        , _writePos(0)
        , _readPos(0)
        , _available(0) {}

    bool write(const T* data, size_t size) {
        std::lock_guard<std::mutex> lock(_mutex);
        
        // If buffer is too small, resize it
        if (size > (_buffer.size() - _available)) {
            size_t newSize = _buffer.size() * 2;  // Double the size
            while (size > (newSize - _available)) {
                newSize *= 2;  // Keep doubling until we have enough space
            }
            
            LOG_V("Resizing ring buffer from " << _buffer.size() << " to " << newSize << " samples");
            
            // Create new buffer with larger size
            std::vector<T> newBuffer(newSize);
            
            // Copy existing data to new buffer
            if (_available > 0) {
                if (_writePos > _readPos) {
                    // Data is contiguous
                    std::memcpy(newBuffer.data(), &_buffer[_readPos], _available * sizeof(T));
                } else {
                    // Data is wrapped
                    size_t firstPart = _buffer.size() - _readPos;
                    std::memcpy(newBuffer.data(), &_buffer[_readPos], firstPart * sizeof(T));
                    std::memcpy(newBuffer.data() + firstPart, &_buffer[0], _writePos * sizeof(T));
                }
            }
            
            // Update buffer and positions
            _buffer = std::move(newBuffer);
            _readPos = 0;
            _writePos = _available;
        }

        // Now write the new data
        size_t firstWrite = std::min(size, _buffer.size() - _writePos);
        std::memcpy(&_buffer[_writePos], data, firstWrite * sizeof(T));

        if (firstWrite < size) {
            // Wrap around
            std::memcpy(&_buffer[0], data + firstWrite, (size - firstWrite) * sizeof(T));
        }

        _writePos = (_writePos + size) % _buffer.size();
        _available += size;
        return true;
    }

    bool read(T* data, size_t size) {
        std::lock_guard<std::mutex> lock(_mutex);
        
        if (size > _available) {
            return false;  // Not enough data
        }

        size_t firstRead = std::min(size, _buffer.size() - _readPos);
        std::memcpy(data, &_buffer[_readPos], firstRead * sizeof(T));

        if (firstRead < size) {
            // Wrap around
            std::memcpy(data + firstRead, &_buffer[0], (size - firstRead) * sizeof(T));
        }

        _readPos = (_readPos + size) % _buffer.size();
        _available -= size;
        return true;
    }

    size_t availableToRead() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _available;
    }

    void increaseWith(size_t additionalSize) {
        std::lock_guard<std::mutex> lock(_mutex);
        _buffer.resize(_buffer.size() + additionalSize);
    }

    size_t getAvailableSpace() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _buffer.size() - _available;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _readPos = 0;
        _writePos = 0;
        _available = 0;
    }

private:
    std::vector<T> _buffer;
    size_t _writePos;
    size_t _readPos;
    size_t _available;
    mutable std::mutex _mutex;
};

class HexPrinter {
public:
    HexPrinter() = default;
    ~HexPrinter() = default;

    static void Dump(const uint8_t* buffer, size_t length, size_t bytes_per_line = 16) {
        if (!buffer || length == 0) {
            // RTC_LOG(LS_WARNING) << "Invalid buffer or length";
            return;
        }

        std::stringstream output;
        output << std::hex << std::setfill('0');
        size_t display_length = std::max(length, bytes_per_line);

        for (size_t i = 0; i < display_length; ++i) {
            if (i < length) {
                uint8_t byte = buffer[i];
                if (std::isalnum(byte)) {
                    output << ' ' << static_cast<char>(byte) << ' ';
                } else {
                    output << std::setw(2) << static_cast<int>(byte) << ' ';
                }
            } else {
                output << ".. "; // Padding
            }

            if ((i + 1) % bytes_per_line == 0 && i < display_length - 1) {
                output << '\n';
            }
        }

        // RTC_LOG(LS_INFO) << "Buffer Dump (" << length << " bytes):\n" << output.str();
    }
};

template<typename T>
std::vector<T> convertDatatype(std::vector<float> float_vec)
{
    std::vector<T> vec;
    vec.reserve( float_vec.size() );    //  avoids unnecessary reallocations
    std::transform( float_vec.begin(), float_vec.end(),
        std::back_inserter( vec ),
        [](const float &arg) { return static_cast<T>(arg); } );
    return vec;
}

// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

