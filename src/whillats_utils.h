/*
 *  (c) 2025, wilddolphin2025 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 */

#ifndef WHILLATS_UTILS_H
#define WHILLATS_UTILS_H

#include "whillats.h"
#include <vector>
#include <string>

clip_image_u8* yuv_to_clip(const YUVData& yuv);
void free_clip(clip_image_u8* clip);
bool load_yuv(YUVData& yuv, const char* filename, int width, int height);
std::vector<uint8_t> yuv_to_flat_array(const YUVData& yuv);
std::string compute_image_hash(const clip_image_u8& image);
bool save_clip_as_bmp(const clip_image_u8& clip, const char* filename);
std::string getDylibPath();

// Resample int16 audio from src_rate to dst_rate.
// Handles any rate pair (e.g. 24000->16000, 44100->16000, 48000->16000).
// Returns resampled buffer; leaves input unchanged if rates already match.
std::vector<int16_t> resampleAudio(const int16_t* data, size_t count,
                                   int src_rate, int dst_rate);

#endif // WHILLATS_UTILS_H
