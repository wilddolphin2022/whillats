/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
<<<<<<< HEAD
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
=======
>>>>>>> 61fe38f (build port to ios)
 */

#ifndef WHILLATS_UTILS_H
#define WHILLATS_UTILS_H

<<<<<<< HEAD
#include "whillats.h"

clip_image_u8* yuv_to_clip(const YUVData& yuv);
void free_clip(clip_image_u8* clip);
bool load_yuv(YUVData& yuv, const char* filename, int width, int height);
std::vector<uint8_t> yuv_to_flat_array(const YUVData& yuv);
std::string compute_image_hash(const clip_image_u8& image);
bool save_clip_as_bmp(const clip_image_u8& clip, const char* filename);
std::string getDylibPath();

#endif // WHILLATS_UTILS_H
=======
#include <cstddef>
#include <cstdint>

// Minimal YUV loader API used by SpeechAudioDeviceFactory
struct YUVData;

// Loads planar I420 YUV into YUVData from a file path with given dimensions.
// Returns true on success.
bool load_yuv(YUVData& out, const char* filepath, int width, int height);

#endif // WHILLATS_UTILS_H


>>>>>>> 61fe38f (build port to ios)
