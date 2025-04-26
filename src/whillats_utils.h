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

#ifndef WHILLATS_UTILS_H
#define WHILLATS_UTILS_H

#include "whillats.h"
#include "opencv2/opencv.hpp"

struct clip_image_u8 {
    int width;
    int height;
    uint8_t* data; // RGB, interleaved [R,G,B,R,G,B,...]
};

// Converts YUV I420 image to OpenCV Mat
cv::Mat preprocess_yuv_i420(uint8_t* yuv420pBuffer, const size_t yuv420pBufferSize, int width, int height);
cv::Mat preprocess_yuv_i420_file(const std::string& image_path, int width, int height);
std::string computeImageHash(const cv::Mat& image);

bool saveMatAsRGB(const cv::Mat& image, const std::string& filename);
bool saveClipImageU8AsRGB(const clip_image_u8* img_clip, const std::string& filename);
#endif // WHILLATS_UTILS_H
