/*
 *  (c) 2025, wilddolphin2025 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <vector>
#include <string>
#include <cstdint>
// Command line options
struct Options {
    bool help = false;
    bool tts = false;
    bool whisper = false;
    bool llama = false;
    std::string help_string;
    std::string whisper_model;
    std::string llama_model;
    std::string llama_mmproj;
    std::string test_image1;
    std::string test_image2;
    std::string styletts2_model_dir;
    std::string styletts2_style;
    std::string styletts2_predictor;
    bool orpheus = false;
    std::string orpheus_model;
    std::string snac_model;
    bool piper = false;
    std::string piper_model;
};

// Function to parse command line string to above options
Options parseOptions(int argc, char* argv[]);

// Function to get command line options to a string, to print or speak
std::string getUsage(const Options opts);

void writeWavFile(const std::string &filename, const std::vector<uint16_t> &audio_data, int sampleRate);
