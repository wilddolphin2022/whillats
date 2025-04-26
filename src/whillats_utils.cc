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

#include <fstream>
#include <vector>
#include <cstring>

#include "whillats.h"
#include "whillats_utils.h"
#include "whisper_helpers.h"

cv::Mat preprocess_yuv_i420(uint8_t* yuv420pBuffer, const size_t yuv420pBufferSize, int width, int height) {
    // Validate inputs
    if (!yuv420pBuffer) {
        LOG_E("ERROR: Null YUV buffer provided");
        return cv::Mat();
    }
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) {
        LOG_E("ERROR: Invalid dimensions: width=" << width << ", height=" << height << " (must be positive and even)");
        return cv::Mat();
    }

    // Calculate expected size for YUV I420
    size_t expected_size = static_cast<size_t>(width) * height * 3 / 2;
    if (yuv420pBufferSize != expected_size) {
        LOG_E("ERROR: YUV buffer size mismatch. Expected " << expected_size << " bytes, got " << yuv420pBufferSize);
        return cv::Mat();
    }

    // Create Mat for YUV I420 (height * 1.5, width, single channel)
    cv::Mat yuv_mat(height * 3 / 2, width, CV_8UC1, yuv420pBuffer);
    if (yuv_mat.empty() || !yuv_mat.isContinuous()) {
        LOG_E("ERROR: Failed to create YUV Mat or non-continuous data");
        return cv::Mat();
    }

    // Convert YUV I420 to RGB
    cv::Mat rgb_mat;
    cv::cvtColor(yuv_mat, rgb_mat, cv::COLOR_YUV2RGB_I420);

    if (rgb_mat.empty() || rgb_mat.type() != CV_8UC3 || rgb_mat.cols != width || rgb_mat.rows != height) {
        LOG_E("ERROR: Invalid RGB Mat after conversion. Size: " << rgb_mat.cols << "x" << rgb_mat.rows 
              << ", type: " << rgb_mat.type());
        return cv::Mat();
    }

    // Save the RGB image for debugging
    //cv::imwrite("debug_rgb.png", rgb_mat); 

    // Convert to float32 RGB (0–1 range)
    cv::Mat float_rgb;
    rgb_mat.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);
    if (float_rgb.empty() || float_rgb.type() != CV_32FC3) {
        LOG_E("ERROR: Failed to convert RGB to float32");
        return cv::Mat();
    }

    LOG_I("YUV image converted to RGB: " << float_rgb.cols << "x" << float_rgb.rows << ", channels: " << float_rgb.channels());
    return float_rgb;
}

cv::Mat preprocess_yuv_i420_file(const std::string& image_path, int width, int height) {
    // Validate inputs
    if (image_path.empty()) {
        LOG_E("ERROR: Empty YUV file path");
        return cv::Mat();
    }
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) {
        LOG_E("ERROR: Invalid dimensions: width=" << width << ", height=" << height << " (must be positive and even)");
        return cv::Mat();
    }

    LOG_I("DEBUG: Opening YUV file: " << image_path << ", expected dimensions: " << width << "x" << height);

    // Calculate expected file size for YUV I420
    size_t expected_size = static_cast<size_t>(width) * height * 3 / 2;

    // Read raw YUV data
    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_E("ERROR: Failed to open YUV file: " << image_path);
        return cv::Mat();
    }

    // Check file size
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size != expected_size) {
        LOG_E("ERROR: YUV file size mismatch. Expected " << expected_size << " bytes, got " << file_size);
        file.close();
        return cv::Mat();
    }

    // Read file into buffer
    std::vector<uint8_t> buffer(expected_size);
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
    if (file.gcount() != expected_size) {
        LOG_E("ERROR: Failed to read YUV file. Expected " << expected_size << " bytes, got " << file.gcount());
        file.close();
        return cv::Mat();
    }
    file.close();

    LOG_I("DEBUG: Read YUV file: " << image_path << ", size: " << file_size << " bytes");
    LOG_I("DEBUG: First 8 YUV bytes: " << std::hex << std::setfill('0') 
          << std::setw(2) << (int)buffer[0] << " " << std::setw(2) << (int)buffer[1] 
          << " ... " << std::setw(2) << (int)buffer[7] << std::dec);

    cv::Mat result = preprocess_yuv_i420(buffer.data(), expected_size, width, height);
    if (result.empty()) {
        LOG_E("ERROR: YUV preprocessing failed for file: " << image_path);
        return cv::Mat();
    }
    // Clone to ensure independent lifetime
    return result.clone();
}

bool saveMatAsRGB(const cv::Mat& image, const std::string& filename) {
    if (image.empty()) {
        LOG_E("ERROR: Cannot save empty image to " << filename);
        return false;
    }

    cv::Mat rgb_image;
    if (image.channels() == 1) {
        // Greyscale input
        if (image.type() == CV_8UC1) {
            cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
        } else if (image.type() == CV_32FC1) {
            cv::Mat temp;
            image.convertTo(temp, CV_8UC1, 255.0);
            cv::cvtColor(temp, rgb_image, cv::COLOR_GRAY2RGB);
        } else {
            LOG_E("ERROR: Unsupported greyscale image type: " << image.type());
            return false;
        }
    } else if (image.channels() == 3) {
        // RGB or BGR input
        if (image.type() == CV_8UC3) {
            cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        } else if (image.type() == CV_32FC3) {
            cv::Mat temp;
            image.convertTo(temp, CV_8UC3, 255.0);
            cv::cvtColor(temp, rgb_image, cv::COLOR_BGR2RGB);
        } else {
            LOG_E("ERROR: Unsupported RGB image type: " << image.type());
            return false;
        }
    } else {
        LOG_E("ERROR: Unsupported image channels: " << image.channels());
        return false;
    }

    bool success = cv::imwrite(filename, rgb_image);
    if (!success) {
        LOG_E("ERROR: Failed to save image to " << filename);
        return false;
    }
    LOG_I("DEBUG: Saved image to " << filename);
    return true;
}

// Compute a perceptual hash of the input image for caching
std::string computeImageHash(const cv::Mat& image) {
    // Quick empty check to avoid logging overhead
    if (image.empty()) {
        LOG_E("ERROR: Empty image provided for hashing");
        return "";
    }

    // Validate type
    cv::Mat hash_input;
    if (image.type() == CV_32FC3) {
        image.convertTo(hash_input, CV_8UC3, 255.0); // Convert float32 to 8-bit
    } else if (image.type() == CV_8UC3 || image.type() == CV_8UC1) {
        hash_input = image; // Avoid copy if already correct type
    } else {
        LOG_E("ERROR: Unsupported image type for hashing: " << image.type());
        return "";
    }

    // Resize to 32x32 in-place to minimize allocations
    cv::Mat resized(32, 32, hash_input.type());
    cv::resize(hash_input, resized, cv::Size(32, 32), 0, 0, cv::INTER_AREA);

    // Convert to grayscale
    cv::Mat gray;
    if (resized.type() == CV_8UC3) {
        cv::cvtColor(resized, gray, cv::COLOR_RGB2GRAY);
    } else {
        gray = resized; // No conversion needed for CV_8UC1
    }

    // Convert to float32 for DCT
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F);

    // Compute DCT
    cv::Mat dct_result;
    cv::dct(gray_float, dct_result);

    // Extract top-left 8x8 region efficiently
    cv::Mat dct_8x8 = dct_result(cv::Rect(0, 0, 8, 8));

    // Compute median of 8x8 coefficients
    float coefficients[64];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            coefficients[i * 8 + j] = dct_8x8.at<float>(i, j);
        }
    }
    std::sort(coefficients, coefficients + 64);
    float median = coefficients[31];

    // Generate 64-bit hash
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            hash <<= 1;
            if (dct_8x8.at<float>(i, j) > median) {
                hash |= 1;
            }
        }
    }

    // Convert to hexadecimal string
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << hash;
    std::string hash_str = ss.str();
    LOG_I("DEBUG: Computed image hash: " << hash_str);
    return hash_str;
}

bool saveClipImageU8AsRGB(const clip_image_u8* img_clip, const std::string& filename) {
    if (!img_clip || !img_clip->data || img_clip->width <= 0 || img_clip->height <= 0) {
        LOG_E("ERROR: Invalid clip_image_u8 for saving to " << filename);
        return false;
    }

    // Create cv::Mat from RGB data
    cv::Mat rgb_image(img_clip->height, img_clip->width, CV_8UC3, img_clip->data);
    if (rgb_image.empty()) {
        LOG_E("ERROR: Failed to create cv::Mat from clip_image_u8");
        return false;
    }

    // Save as PNG
    bool success = cv::imwrite(filename, rgb_image);
    if (!success) {
        LOG_E("ERROR: Failed to save clip_image_u8 to " << filename);
        return false;
    }
    LOG_I("DEBUG: Saved clip_image_u8 to " << filename);
    return true;
}