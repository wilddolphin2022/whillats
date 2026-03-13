/*
 *  (c) 2025, wilddolphin2022
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 */

#ifndef TALKING_FACE_H
#define TALKING_FACE_H

#include "whillats_export.h"
#include "whillats.h"
#include <cstdint>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>

// Audio-driven animated face that renders the robot logo with mouth
// movement synced to TTS audio energy. Produces YUV420 frames suitable
// for injection into a WebRTC video track.
class WHILLATS_API TalkingFace {
public:
    TalkingFace();
    ~TalkingFace();

    bool loadImage(const char* imagePath);
    bool loadImageFromMemory(const uint8_t* data, int width, int height, int channels);

    // Feed TTS audio to drive the mouth animation.
    // PCM 16-bit signed mono at any sample rate.
    void feedAudio(const int16_t* samples, size_t count);

    // Render the current animation frame as YUV420.
    // Returns false if no image is loaded.
    bool renderFrame(YUVData& out);

    void setOutputSize(int w, int h) { out_w_ = w; out_h_ = h; }
    int outputWidth()  const { return out_w_; }
    int outputHeight() const { return out_h_; }

    // 0.0 = mouth closed, 1.0 = fully open
    float mouthOpenness() const { return mouth_openness_; }

private:
    void detectMouthRegion();
    void renderMouth(std::vector<uint8_t>& rgb, int w, int h, float openness);
    void rgbToYuv420(const uint8_t* rgb, int w, int h, YUVData& yuv);
    void drawFilledRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                        int x, int y, int w, int h,
                        uint8_t r, uint8_t g, uint8_t b);
    void drawRoundedRect(std::vector<uint8_t>& rgb, int imgW, int imgH,
                         int cx, int cy, int w, int h, int radius,
                         uint8_t r, uint8_t g, uint8_t b);

    std::vector<uint8_t> base_rgb_;
    int img_w_ = 0;
    int img_h_ = 0;

    int out_w_ = 640;
    int out_h_ = 480;

    // Mouth region in image coordinates (auto-detected or manual)
    int mouth_cx_ = 0;
    int mouth_cy_ = 0;
    int mouth_w_  = 0;
    int mouth_max_h_ = 0;

    // Navy blue color matching the logo
    static constexpr uint8_t kFaceR = 0x0a;
    static constexpr uint8_t kFaceG = 0x2e;
    static constexpr uint8_t kFaceB = 0x5c;

    // Background gray for "erasing" the old mouth
    uint8_t bg_r_ = 0xd8;
    uint8_t bg_g_ = 0xd8;
    uint8_t bg_b_ = 0xd8;

    std::atomic<float> mouth_openness_{0.0f};
    float smoothed_energy_ = 0.0f;

    mutable std::mutex render_mutex_;
};

#endif // TALKING_FACE_H
