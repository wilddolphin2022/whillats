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
#include <filesystem>
#include <cstdint>
#include <memory>
#include <span>
#include <cstdio>

#include "whillats.h"
#include "whillats_utils.h"
#include "whisper_helpers.h"

// Compute perceptual hash for clip_image_u8
std::string compute_image_hash(const clip_image_u8& image) {
    // Validate input
    if (image.width <= 0 || image.height <= 0 || !image.data) {
        LOG_E("Invalid or empty image provided for hashing");
        return "";
    }

    // Resize to 32x32 (RGB)
    const int target_size = 32;
    std::vector<uint8_t> resized(target_size * target_size * 3); // RGB
    {
        // Simple area interpolation
        float x_ratio = static_cast<float>(image.width) / target_size;
        float y_ratio = static_cast<float>(image.height) / target_size;

        for (int y = 0; y < target_size; ++y) {
            for (int x = 0; x < target_size; ++x) {
                // Source pixel coordinates
                int src_x = static_cast<int>(x * x_ratio);
                int src_y = static_cast<int>(y * y_ratio);
                src_x = src_x < image.width ? src_x : image.width - 1;
                src_y = src_y < image.height ? src_y : image.height - 1;

                // Copy RGB
                size_t src_idx = (src_y * image.width + src_x) * 3;
                size_t dst_idx = (y * target_size + x) * 3;
                resized[dst_idx] = image.data[src_idx];     // R
                resized[dst_idx + 1] = image.data[src_idx + 1]; // G
                resized[dst_idx + 2] = image.data[src_idx + 2]; // B
            }
        }
    }

    // Convert to grayscale
    std::vector<float> gray(target_size * target_size);
    for (int i = 0; i < target_size * target_size; ++i) {
        size_t rgb_idx = i * 3;
        // ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B
        gray[i] = 0.299f * resized[rgb_idx] + 0.587f * resized[rgb_idx + 1] + 0.114f * resized[rgb_idx + 2];
    }

    // Compute 2D DCT
    std::vector<float> dct_result(target_size * target_size);
    const float pi = 3.14159265358979323846f;
    float sqrt_2n = std::sqrt(2.0f * target_size);

    // DCT-II (row-wise then column-wise)
    for (int u = 0; u < target_size; ++u) {
        for (int v = 0; v < target_size; ++v) {
            float sum = 0.0f;
            for (int x = 0; x < target_size; ++x) {
                for (int y = 0; y < target_size; ++y) {
                    float cos_x = std::cos(pi * (2 * x + 1) * u / (2.0f * target_size));
                    float cos_y = std::cos(pi * (2 * y + 1) * v / (2.0f * target_size));
                    sum += gray[y * target_size + x] * cos_x * cos_y;
                }
            }
            float cu = (u == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            float cv = (v == 0) ? 1.0f / std::sqrt(2.0f) : 1.0f;
            dct_result[v * target_size + u] = sum * 2.0f * cu * cv / target_size;
        }
    }

    // Extract top-left 8x8 region
    float coefficients[64];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            coefficients[i * 8 + j] = dct_result[i * target_size + j];
        }
    }

    // Compute median
    std::sort(coefficients, coefficients + 64);
    float median = coefficients[31];

    // Generate 64-bit hash
    uint64_t hash = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            hash <<= 1;
            if (dct_result[i * target_size + j] > median) {
                hash |= 1;
            }
        }
    }

    // Convert to hexadecimal string
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << hash;
    std::string hash_str = ss.str();
    LOG_I("DEBUG: Computed image hash: " + hash_str);
    return hash_str;
}

// Convert YUV I420 to clip_image_u8 (RGB, resized to 224x224)
clip_image_u8* yuv_to_clip(const YUVData& yuv) {
    if (!yuv.y || !yuv.u || !yuv.v) return nullptr;

    // Target size for clip_image_u8
    const int target_width = 224;
    const int target_height = 224;

    // Allocate clip_image_u8
    auto* clip = new clip_image_u8;
    clip->width = target_width;
    clip->height = target_height;
    size_t rgb_size = static_cast<size_t>(target_width) * target_height * 3;
    clip->data = (uint8_t*)malloc(rgb_size);
    if (!clip->data) {
        delete clip;
        return nullptr;
    }

    // Compute scaling ratios for resizing
    float x_ratio = static_cast<float>(yuv.width) / target_width;
    float y_ratio = static_cast<float>(yuv.height) / target_height;

    // Convert YUV to RGB with resizing
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            // Source pixel coordinates (nearest neighbor)
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            src_x = src_x < yuv.width ? src_x : yuv.width - 1;
            src_y = src_y < yuv.height ? src_y : yuv.height - 1;

            // Compute YUV indices
            size_t rgb_idx = (y * target_width + x) * 3;
            size_t y_idx = src_y * yuv.width + src_x;
            size_t uv_idx = (src_y / 2) * (yuv.width / 2) + (src_x / 2);

            // Get YUV values
            int Y = yuv.y[y_idx] - 16;
            int U = yuv.u[uv_idx] - 128;
            int V = yuv.v[uv_idx] - 128;

            // Fast integer-based YUV to RGB (ITU-R BT.601)
            int C = Y * 298 + 128;
            int D = U * 517;
            int E = V * 409;

            // Compute RGB
            int R = (C + E) >> 8;
            int G = (C - ((U * 100 + V * 208) >> 8)) >> 8;
            int B = (C + D) >> 8;

            // Clamp and store
            clip->data[rgb_idx]     = static_cast<uint8_t>(R < 0 ? 0 : R > 255 ? 255 : R);
            clip->data[rgb_idx + 1] = static_cast<uint8_t>(G < 0 ? 0 : G > 255 ? 255 : G);
            clip->data[rgb_idx + 2] = static_cast<uint8_t>(B < 0 ? 0 : B > 255 ? 255 : B);
        }
    }

    return clip;
}

void free_clip(clip_image_u8* clip) {
    if (clip) {
        free(clip->data);
        delete clip;
    }
}

// Load YUV I420 file into memory
YUVData* load_yuv(const char* filename, int width, int height) {
    // Validate file
    FILE* file = fopen(filename, "rb");
    if (!file) return nullptr;

    // Check file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    size_t expected_size = static_cast<size_t>(width * height) * 3 / 2; // Y + U + V
    if (file_size < expected_size) {
        fclose(file);
        return nullptr;
    }

    // Allocate YUVData
    auto* data = new YUVData;
    data->width = width;
    data->height = height;
    data->y_size = static_cast<size_t>(width) * height;
    data->uv_size = data->y_size / 4; // 4:2:0

    // Allocate memory
    data->y = std::make_unique<uint8_t[]>(data->y_size);
    data->u = std::make_unique<uint8_t[]>(data->uv_size);
    data->v = std::make_unique<uint8_t[]>(data->uv_size);

    // Read file
    if (fread(data->y.get(), 1, data->y_size, file) != data->y_size ||
        fread(data->u.get(), 1, data->uv_size, file) != data->uv_size ||
        fread(data->v.get(), 1, data->uv_size, file) != data->uv_size) {
        fclose(file);
        delete data;
        return nullptr;
    }

    fclose(file);
    return data;
}

// Function to free YUV data
void free_yuv(YUVData* data) {
    delete data; // unique_ptr handles memory cleanup
}

// Converts YUVData to a flat byte array in YUV I420 format (Y, U, V order)
std::vector<uint8_t> yuv_to_flat_array(const YUVData& yuv) {
    // Validate input
    if (!yuv.y || !yuv.u || !yuv.v || yuv.width <= 0 || yuv.height <= 0) {
        LOG_E("Invalid YUVData: null pointers or invalid dimensions");
        return std::vector<uint8_t>();
    }

    // Verify plane sizes
    size_t expected_y_size = static_cast<size_t>(yuv.width) * yuv.height;
    size_t expected_uv_size = expected_y_size / 4; // 4:2:0 subsampling
    if (yuv.y_size != expected_y_size || yuv.uv_size != expected_uv_size) {
        LOG_E("YUVData plane size mismatch: y_size=" << yuv.y_size << ", expected=" << expected_y_size
              << ", uv_size=" << yuv.uv_size << ", expected=" << expected_uv_size);
        return std::vector<uint8_t>();
    }

    // Calculate total size: Y + U + V
    size_t total_size = yuv.y_size + 2 * yuv.uv_size;

    // Allocate flat array
    std::vector<uint8_t> flat_array(total_size);

    // Copy Y plane
    std::memcpy(flat_array.data(), yuv.y.get(), yuv.y_size);

    // Copy U plane
    std::memcpy(flat_array.data() + yuv.y_size, yuv.u.get(), yuv.uv_size);

    // Copy V plane
    std::memcpy(flat_array.data() + yuv.y_size + yuv.uv_size, yuv.v.get(), yuv.uv_size);

    LOG_I("Converted YUVData to flat array: size=" << total_size << " bytes");
    return flat_array;
}

// Save clip_image_u8 as 24-bit RGB BMP file
bool save_clip_as_bmp(const clip_image_u8& clip, const char* filename) {
    // Validate input
    if (clip.width <= 0 || clip.height <= 0 || !clip.data) {
        LOG_E("Invalid or empty clip image for BMP saving");
        return false;
    }

    // Open file
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        LOG_E("Failed to open file for BMP writing");
        return false;
    }

    // Calculate BMP specifics
    int bytes_per_pixel = 3; // RGB, 24-bit
    int row_size = ((clip.width * bytes_per_pixel + 3) / 4) * 4; // Padded to 4-byte alignment
    int pixel_data_size = row_size * clip.height;
    int file_size = 14 + 40 + pixel_data_size; // File header + info header + pixel data

    // Write BITMAPFILEHEADER (14 bytes)
    uint8_t file_header[14] = {
        'B', 'M',                           // Signature
        static_cast<uint8_t>(file_size), static_cast<uint8_t>(file_size >> 8),
        static_cast<uint8_t>(file_size >> 16), static_cast<uint8_t>(file_size >> 24), // File size
        0, 0, 0, 0,                         // Reserved
        54, 0, 0, 0                         // Data offset (14 + 40)
    };
    if (fwrite(file_header, 1, 14, fp) != 14) {
        fclose(fp);
        LOG_E("Failed to write BMP file header");
        return false;
    }

    // Write BITMAPINFOHEADER (40 bytes)
    uint8_t info_header[40] = {
        40, 0, 0, 0,                        // Header size
        static_cast<uint8_t>(clip.width), static_cast<uint8_t>(clip.width >> 8),
        static_cast<uint8_t>(clip.width >> 16), static_cast<uint8_t>(clip.width >> 24), // Width
        static_cast<uint8_t>(clip.height), static_cast<uint8_t>(clip.height >> 8),
        static_cast<uint8_t>(clip.height >> 16), static_cast<uint8_t>(clip.height >> 24), // Height
        1, 0,                               // Planes
        24, 0,                              // Bits per pixel (24-bit RGB)
        0, 0, 0, 0,                         // Compression (BI_RGB, none)
        static_cast<uint8_t>(pixel_data_size), static_cast<uint8_t>(pixel_data_size >> 8),
        static_cast<uint8_t>(pixel_data_size >> 16), static_cast<uint8_t>(pixel_data_size >> 24), // Image size
        0, 0, 0, 0,                         // X pixels per meter (not specified)
        0, 0, 0, 0,                         // Y pixels per meter (not specified)
        0, 0, 0, 0,                         // Colors used (0 for 24-bit)
        0, 0, 0, 0                          // Important colors (0 for all)
    };
    if (fwrite(info_header, 1, 40, fp) != 40) {
        fclose(fp);
        LOG_E("Failed to write BMP info header");
        return false;
    }

    // Write pixel data (BGR order, padded rows)
    std::vector<uint8_t> row_buffer(row_size, 0); // Initialize with padding zeros
    for (int y = clip.height - 1; y >= 0; --y) { // BMP stores bottom-to-top
        for (int x = 0; x < clip.width; ++x) {
            size_t rgb_idx = (y * clip.width + x) * 3;
            size_t bmp_idx = x * 3;
            // Convert RGB to BGR
            row_buffer[bmp_idx] = clip.data[rgb_idx + 2]; // B
            row_buffer[bmp_idx + 1] = clip.data[rgb_idx + 1]; // G
            row_buffer[bmp_idx + 2] = clip.data[rgb_idx]; // R
        }
        if (fwrite(row_buffer.data(), 1, row_size, fp) != static_cast<size_t>(row_size)) {
            fclose(fp);
            LOG_E("Failed to write BMP pixel data");
            return false;
        }
    }

    // Clean up
    fclose(fp);
    return true;
}