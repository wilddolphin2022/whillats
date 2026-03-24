/* Single translation unit for stb_image (see stb_image.h header). Used by the
 * WebRTC in-tree GN build of speech_audio_device; the full whillats CMake build
 * gets STB_IMAGE_IMPLEMENTATION from llama.cpp mtmd-helper.cpp instead. */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
