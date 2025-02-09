#pragma once

#include <espeak-ng/speak_lib.h>
#include <vector>
#include <chrono>
#include "whisper_helpers.h"  // From whisper_helpers.h

class ESpeakTTS {
public:
    ESpeakTTS();
    ~ESpeakTTS();

    void synthesize(const char* text, std::vector<uint16_t>& buffer);
    int getSampleRate() const;

private:
    static int internalSynthCallback(short* wav, int numsamples, espeak_EVENT* events);
    std::chrono::steady_clock::time_point last_read_time_;
    AudioRingBuffer ring_buffer_;
    static const int SAMPLE_RATE = 16000;
};