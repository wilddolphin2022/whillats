#pragma once

#include <espeak-ng/speak_lib.h>
#include <vector>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "whillats.h"

class AudioRingBuffer;
class ESpeakTTS {
public:
    ESpeakTTS(WhillatsSetAudioCallback callback);
    ~ESpeakTTS();

    
    // Add new methods
    bool Start();
    void Stop();
    void queueText(const std::string& text);

    static const int getSampleRate();
private:
    void synthesize(const char* text);

    static int internalSynthCallback(short* wav, int numsamples, espeak_EVENT* events);
    bool RunProcessingThread();
    
    std::chrono::steady_clock::time_point last_read_time_;
    std::unique_ptr<AudioRingBuffer> ring_buffer_;
    static const int SAMPLE_RATE = 16000;
    WhillatsSetAudioCallback _callback;

    std::vector<uint16_t> _buffer;

    // Add thread management
    bool _running{false};
    std::thread _processingThread;
    
    // Add text queue
    std::queue<std::string> _textQueue;
    std::mutex _queueMutex;
    std::condition_variable _queueCondition;
};