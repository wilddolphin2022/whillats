/*
 *  (c) 2025, wilddolphin2022
 *  For WebRTCsays.ai project
 *
 *  Test: render ASL sign language video from audio input.
 *  Reads a WAV file, generates ASL sign frames synchronized to the audio,
 *  and writes a raw YUV4MPEG2 video file that can be muxed with ffmpeg:
 *
 *    ffmpeg -i asl_output.y4m -i input.wav -c:v libx264 -c:a aac -shortest asl_output.mp4
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "sign_language_face.h"

struct WavHeader {
    char     riff[4];
    uint32_t fileSize;
    char     wave[4];
    char     fmt[4];
    uint32_t fmtSize;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

static bool readWav(const char* path, std::vector<int16_t>& samples, int& sampleRate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    WavHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (std::memcmp(hdr.riff, "RIFF", 4) != 0 || std::memcmp(hdr.wave, "WAVE", 4) != 0) {
        return false;
    }

    sampleRate = hdr.sampleRate;

    // Skip to "data" chunk
    while (f) {
        char chunkId[4];
        uint32_t chunkSize;
        f.read(chunkId, 4);
        f.read(reinterpret_cast<char*>(&chunkSize), 4);
        if (std::memcmp(chunkId, "data", 4) == 0) {
            size_t numSamples = chunkSize / (hdr.bitsPerSample / 8);
            if (hdr.numChannels > 1) numSamples /= hdr.numChannels;
            samples.resize(numSamples);
            if (hdr.bitsPerSample == 16 && hdr.numChannels == 1) {
                f.read(reinterpret_cast<char*>(samples.data()), chunkSize);
            } else {
                // Read and convert to mono 16-bit
                std::vector<uint8_t> raw(chunkSize);
                f.read(reinterpret_cast<char*>(raw.data()), chunkSize);
                for (size_t i = 0; i < numSamples; i++) {
                    if (hdr.bitsPerSample == 16) {
                        int32_t sum = 0;
                        for (int ch = 0; ch < hdr.numChannels; ch++) {
                            sum += *reinterpret_cast<int16_t*>(&raw[(i * hdr.numChannels + ch) * 2]);
                        }
                        samples[i] = (int16_t)(sum / hdr.numChannels);
                    }
                }
            }
            return true;
        }
        f.seekg(chunkSize, std::ios::cur);
    }
    return false;
}

static void writeY4mHeader(std::ofstream& f, int w, int h, int fps) {
    f << "YUV4MPEG2 W" << w << " H" << h << " F" << fps << ":1 Ip A1:1 C420\n";
}

static void writeY4mFrame(std::ofstream& f, const YUVData& yuv) {
    f << "FRAME\n";
    f.write(reinterpret_cast<const char*>(yuv.y.get()), yuv.y_size);
    f.write(reinterpret_cast<const char*>(yuv.u.get()), yuv.uv_size);
    f.write(reinterpret_cast<const char*>(yuv.v.get()), yuv.uv_size);
}

int main(int argc, char* argv[]) {
    std::string wav_path = "synthesized_audio.wav";
    std::string text = "Hello, this is a test of text to speech synthesis.";
    std::string output_path = "asl_output.y4m";
    int video_size = 1024;
    int fps = 15;
    std::string sign_dir;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--wav" && i + 1 < argc) wav_path = argv[++i];
        else if (arg == "--text" && i + 1 < argc) text = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--size" && i + 1 < argc) video_size = std::atoi(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc) fps = std::atoi(argv[++i]);
        else if (arg == "--signs" && i + 1 < argc) sign_dir = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: test_asl_video [options]\n"
                      << "  --wav <path>      Input WAV file (default: synthesized_audio.wav)\n"
                      << "  --text <text>     Text to sign (default: test sentence)\n"
                      << "  --output <path>   Output Y4M file (default: asl_output.y4m)\n"
                      << "  --size <px>       Video size (default: 1024)\n"
                      << "  --fps <n>         Frame rate (default: 15)\n"
                      << "  --signs <dir>     Directory with custom ASL sign images\n";
            return 0;
        }
    }

    // Read audio
    std::vector<int16_t> samples;
    int sampleRate = 24000;
    if (!readWav(wav_path.c_str(), samples, sampleRate)) {
        std::cerr << "Failed to read WAV: " << wav_path << "\n";
        return 1;
    }
    float duration = (float)samples.size() / sampleRate;
    std::cout << "Audio: " << wav_path << " (" << duration << "s, " << sampleRate << "Hz, "
              << samples.size() << " samples)\n";

    // Setup sign language face
    SignLanguageFace face;
    face.setOutputSize(video_size, video_size);

    if (!sign_dir.empty()) {
        if (!face.loadSignImages(sign_dir.c_str())) {
            std::cerr << "Warning: no images loaded from " << sign_dir << ", using builtin signs\n";
        }
    }

    face.generateBuiltinSigns(video_size);

    face.setSignCallback([](const std::string& sign, float t) {
        std::cout << "[" << std::fixed << t << "s] Sign: " << sign << "\n";
    });

    face.setText(text, duration);
    face.reset();

    // Render video
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output: " << output_path << "\n";
        return 1;
    }

    writeY4mHeader(out, video_size, video_size, fps);

    int samples_per_frame = sampleRate / fps;
    int total_frames = (int)std::ceil(duration * fps);
    int audio_pos = 0;

    std::cout << "Rendering " << total_frames << " frames at " << fps << "fps ("
              << video_size << "x" << video_size << ")...\n";

    for (int frame = 0; frame < total_frames; frame++) {
        // Feed audio chunk for this frame
        int chunk_end = std::min(audio_pos + samples_per_frame, (int)samples.size());
        int chunk_size = chunk_end - audio_pos;
        if (chunk_size > 0) {
            face.feedAudio(samples.data() + audio_pos, chunk_size, sampleRate);
            audio_pos = chunk_end;
        }

        // Render frame
        YUVData yuv;
        if (face.renderFrame(yuv)) {
            writeY4mFrame(out, yuv);
        }

        if (frame % 30 == 0) {
            std::cout << "  frame " << frame << "/" << total_frames
                      << " (" << (100 * frame / total_frames) << "%)\n";
        }
    }

    out.close();
    std::cout << "Written: " << output_path << "\n";
    std::cout << "To create MP4:\n"
              << "  ffmpeg -i " << output_path << " -i " << wav_path
              << " -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest asl_output.mp4\n";

    return 0;
}
