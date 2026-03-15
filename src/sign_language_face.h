/*
 *  (c) 2025, wilddolphin2022
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  ASL (American Sign Language) talking face.
 *  Instead of animating a mouth on a static image, this class switches
 *  between pre-rendered sign images (1024x1024) driven by word-level
 *  timing extracted from the audio/text.
 *
 *  Pipeline:
 *    setText("hello how are you") -> segments words with timing
 *    feedAudio(samples, count)    -> advances playback clock
 *    renderFrame(out)             -> picks current sign image, renders YUV
 *
 *  Sign lookup:
 *    1. Whole-word ASL signs (common vocabulary)
 *    2. Fingerspelling fallback (letter-by-letter for unknown words)
 *    3. Idle/rest pose when silent
 */

#ifndef SIGN_LANGUAGE_FACE_H
#define SIGN_LANGUAGE_FACE_H

#include "whillats_export.h"
#include "whillats.h"
#include <cstdint>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <functional>

struct SignImage {
    std::vector<uint8_t> rgb;
    int width  = 0;
    int height = 0;
};

struct WordTiming {
    std::string word;
    float start_sec = 0.0f;
    float end_sec   = 0.0f;
};

class WHILLATS_API SignLanguageFace {
public:
    SignLanguageFace();
    ~SignLanguageFace();

    void setOutputSize(int w, int h) { out_w_ = w; out_h_ = h; }
    int outputWidth()  const { return out_w_; }
    int outputHeight() const { return out_h_; }

    // Load a directory of ASL sign images.
    // Expected layout: dir/hello.png, dir/you.png, dir/a.png ... dir/z.png, dir/rest.png
    bool loadSignImages(const char* directory);

    // Register a single sign image for a word/letter.
    bool addSign(const std::string& key, const uint8_t* data, int size);

    // Programmatically generate ASL fingerspelling alphabet + common signs.
    // No external images needed — renders clean vector-style hand shapes.
    void generateBuiltinSigns(int size = 1024);

    // Set the text to be signed. Distributes words evenly across the
    // audio duration, or uses explicit timings if provided.
    void setText(const std::string& text, float total_duration_sec);
    void setWordTimings(const std::vector<WordTiming>& timings);

    // Feed audio to advance the playback clock.
    // PCM 16-bit signed mono, any sample rate.
    void feedAudio(const int16_t* samples, size_t count, int sample_rate = 24000);

    // Render the current sign frame as YUV420.
    bool renderFrame(YUVData& out);

    // Get current sign being displayed.
    std::string currentSign() const;

    // Reset playback position.
    void reset();

    // Callback for sign transitions (for logging/debugging).
    using SignCallback = std::function<void(const std::string& sign, float time_sec)>;
    void setSignCallback(SignCallback cb) { sign_callback_ = cb; }

private:
    void rgbToYuv420(const uint8_t* rgb, int w, int h, YUVData& yuv);
    void renderSign(const std::string& key, YUVData& out);
    void renderText(std::vector<uint8_t>& rgb, int imgW, int imgH,
                    const std::string& text, int cx, int cy, int fontSize,
                    uint8_t r, uint8_t g, uint8_t b);
    void drawChar(std::vector<uint8_t>& rgb, int imgW, int imgH,
                  char ch, int x, int y, int size,
                  uint8_t r, uint8_t g, uint8_t b);
    void generateLetterSign(char letter, int size);
    void generateWordSign(const std::string& word, int size);
    void drawCircle(std::vector<uint8_t>& rgb, int imgW, int imgH,
                    int cx, int cy, int radius,
                    uint8_t r, uint8_t g, uint8_t b);
    void drawLine(std::vector<uint8_t>& rgb, int imgW, int imgH,
                  int x0, int y0, int x1, int y1, int thickness,
                  uint8_t r, uint8_t g, uint8_t b);
    void drawFilledRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                        int x, int y, int w, int h,
                        uint8_t r, uint8_t g, uint8_t b);
    void drawRoundedRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                         int cx, int cy, int w, int h, int radius,
                         uint8_t r, uint8_t g, uint8_t b);

    std::unordered_map<std::string, SignImage> signs_;
    std::vector<WordTiming> word_timings_;
    
    int out_w_ = 1024;
    int out_h_ = 1024;

    std::atomic<float> playback_time_{0.0f};
    float total_duration_ = 0.0f;
    std::string current_sign_;

    SignCallback sign_callback_;
    mutable std::mutex render_mutex_;
};

#endif // SIGN_LANGUAGE_FACE_H
