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

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

template<typename T>
class SilenceFinder {
public:
    SilenceFinder(T* data, size_t size, size_t samples) {
        reset(data, size, samples);
    }

    // Add reset method to reuse the instance with new buffer
    void reset(T* data, size_t size, size_t samples) {
        d = data;
        sBegin = 0;
        s = size;
        samp = samples;
        status = Undefined;
        // Recalculate average amplitude for the new buffer
        avgAmplitude = calculateAverageAmplitude(data, size);
    }

    std::vector<std::pair<size_t, size_t>> find(const float relativeThreshold, const size_t window) { 
        const T threshold = static_cast<T>(avgAmplitude * relativeThreshold);
        auto r = findSilence(d, s, threshold, window);
        regionsToTime(r);
        return r;
    }

    enum Status {
        Silent, Loud, Undefined
    };

    void toggleSilence(Status st, size_t pos, std::vector<std::pair<size_t, size_t>> & res) { 
        if (st == Silent) {
            if (status != Silent) sBegin = pos;  // Start of silence
            status = Silent;
        } else {
            if (status == Silent) {  // End of silence detected
                res.push_back(std::pair<size_t, size_t>(sBegin, pos));
                status = Loud;
            }
        }
    }

    void end(Status st, size_t pos, std::vector<std::pair<size_t, size_t>> & res) { 
        if (status == Silent) {  // If we're in silence at the end
            res.push_back(std::pair<size_t, size_t>(sBegin, pos));
        }
    }

    T calculateAverageAmplitude(T* data, size_t size) { 
        // Use RMS (Root Mean Square) instead of arithmetic mean
        double sum_squares = 0.0;
        for (size_t i = 0; i < size; ++i) {
            double sample = static_cast<double>(data[i]);
            sum_squares += sample * sample;
        }
        return static_cast<T>(std::sqrt(sum_squares / size));
    }

    static T delta(T * data, const size_t window) { 
        // Improve noise immunity by using RMS of window
        double sum_squares = 0.0;
        T max_amplitude = 0;
        
        for (size_t i = 0; i < window; ++i) { 
            double sample = static_cast<double>(std::abs(data[i]));
            sum_squares += sample * sample;
            max_amplitude = std::max(max_amplitude, static_cast<T>(sample));
        }
        
        T rms = static_cast<T>(std::sqrt(sum_squares / window));
        return std::max(rms, static_cast<T>(max_amplitude / 4));  // Consider both RMS and peak
    }

    std::vector<std::pair<size_t, size_t>> findSilence(T * data, const size_t size, const T threshold, const size_t win) { 
        std::vector<std::pair<size_t, size_t>> regions;
        if (size == 0 || win == 0) {
            return regions; // Return empty vector for invalid input
        }

        size_t window = win;
        size_t pos = 0;
        Status s = Undefined;

        while (pos < size) {  // Changed from <= to <
            // Use the minimum of window size or remaining data size
            size_t checkSize = std::min(window, size - pos); 
            if (delta(data + pos, checkSize) < threshold) {
                s = Silent;
            } else {
                s = Loud;
            }
            toggleSilence(s, pos, regions);
            pos += window;
            
            // Ensure we don't go past the buffer end
            if (pos > size) {
                pos = size;  // This will make the next iteration fail the loop condition
            }
        }

        // Handle the case where the last segment might not be a full window
        if (pos == size && status == Silent) {
            end(s, pos, regions); // Only call end if we were in a silent state at the end
        }

        return regions;
    }

    void regionsToTime(std::vector<std::pair<size_t, size_t>> & regions) { 
        for (auto & r : regions) {
            r.first /= samp;
            r.second /= samp;
        }
    }

    T* d;
    size_t sBegin, s, samp; 
    Status status;

public:    
    T avgAmplitude = 0; // Store average amplitude for relative threshold calculation
};

// Usage:
// SilenceFinder<int16_t> silenceFinder(int16Buffer.data(), int16Buffer.size(), kSampleRate);
// const float relativeThreshold = 0.05f; // 5% of average amplitude
// const uint windowSize = kSampleRate / 10; // 100ms window
// auto silenceRegions = silenceFinder.find(relativeThreshold, windowSize);

// Example usage:
// SilenceFinder<int16_t> silenceFinder(buffer1, size1, sampleRate);
// auto regions1 = silenceFinder.find(0.05f, sampleRate/10);
// 
// // Later, with new buffer:
// silenceFinder.reset(buffer2, size2, sampleRate);
// auto regions2 = silenceFinder.find(0.05f, sampleRate/10);