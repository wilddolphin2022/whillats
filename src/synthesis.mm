// synthesis.mm
// Compile with: g++ -std=c++11 -x objective-c++ synthesis.mm -o synthesis -framework Foundation
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <sstream>

// Synthesize text using specified voice; returns PCM samples
std::vector<int16_t> synthesize_text(const std::string& text, const std::string& voice) {
    std::cerr << "Processing text: " << text << "\n";
    
    NSString* ns_text = [NSString stringWithUTF8String:text.c_str()];
    NSString* temp_file = @"/tmp/temp_say.wav";
    // Build command to invoke say with selected voice
    NSString* ns_voice = [NSString stringWithUTF8String:voice.c_str()];
    NSString* command = [NSString stringWithFormat:@"say -v %@ -o %@ --data-format=LEI16@16000 \"%@\"", ns_voice, temp_file, ns_text];
    
    std::cerr << "Running synthesis: " << [command UTF8String] << "\n";
    system([command UTF8String]);
    
    NSFileManager* file_manager = [NSFileManager defaultManager];
    if (![file_manager fileExistsAtPath:temp_file]) {
        std::cerr << "Synthesis failed to create WAV file\n";
        return {};
    }
    
    NSData* audio_data = [NSData dataWithContentsOfFile:temp_file];
    if (!audio_data) {
        std::cerr << "Failed to read WAV file\n";
        return {};
    }
    
    // Skip WAV header (44 bytes)
    const uint8_t* bytes = (const uint8_t*)[audio_data bytes];
    size_t data_size = [audio_data length] - 44;
    size_t num_samples = data_size / sizeof(int16_t);
    
    // Read full PCM data (no trimming)
    size_t data_bytes = num_samples * sizeof(int16_t);
    std::vector<int16_t> samples(num_samples);
    memcpy(samples.data(), bytes + 44, data_bytes);
    
    // Apply smooth fade-in and fade-out (Hann window) to reduce click artifact
    size_t out_samples = samples.size();
    // Determine fade duration: up to half utterance or ~50ms (16000/20 = 800 samples)
    const size_t max_fade = 16000 / 2; // ~50ms at 16kHz
    size_t fade_samples = std::min(out_samples / 2, max_fade);
    for (size_t i = 0; i < fade_samples; ++i) {
        // Hann window fade-in: 0 at start, ~1 at end
        float phase = static_cast<float>(i) / static_cast<float>(fade_samples - 1);
        float gain = 0.5f * (1.0f - std::cos(static_cast<float>(M_PI) * phase));
        samples[i] = static_cast<int16_t>(samples[i] * gain);
    }
    // Apply smooth fade-out (Hann window) to reduce click artifact at end
    for (size_t i = 0; i < fade_samples; ++i) {
        float phase = static_cast<float>(i) / static_cast<float>(fade_samples - 1);
        float gain = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * phase));
        size_t idx = out_samples - fade_samples + i;
        samples[idx] = static_cast<int16_t>(samples[idx] * gain);
    }
    
    std::cerr << "Synthesis produced " << num_samples << " samples, trimmed to " << samples.size() << "\n";
    [file_manager removeItemAtPath:temp_file error:nil];
    return samples;
}

int main(int argc, char *argv[]) {
    std::cerr << "Synthesis process started, input_fd: " << STDIN_FILENO << ", output_fd: " << STDOUT_FILENO << "\n";
    
    // Build mapping from language code to voice name
    std::unordered_map<std::string, std::string> voice_map;
    {
        FILE* pipe = popen("say -v \"?\"", "r");
        char buf[512];
        while (fgets(buf, sizeof(buf), pipe)) {
            std::string line(buf);
            std::istringstream iss(line);
            std::string vname, locale;
            if (!(iss >> vname >> locale)) continue;
            // Normalize locale: replace '_' with '-'
            std::replace(locale.begin(), locale.end(), '_', '-');
            if(locale == "en-US") {
                voice_map[locale] = "Samantha";
            } else if(voice_map.find(locale) == voice_map.end()) {
                voice_map[locale] = vname;
            } 
        }
        pclose(pipe);
    }
    // Add short locale mappings for convenience (e.g. "en", "es", "ru")
    {
        std::vector<std::pair<std::string, std::string>> additional;
        for (const auto& kv : voice_map) {
            auto pos = kv.first.find('-');
            if (pos != std::string::npos) {
                std::string short_loc = kv.first.substr(0, pos);
                if (voice_map.find(short_loc) == voice_map.end()) {
                    additional.emplace_back(short_loc, kv.second);
                }
            }
        }
        for (const auto& p : additional) {
            voice_map[p.first] = p.second;
        }
    }
    std::cerr << "TTS initialized\n";
    // Main loop: read text and language
    while (true) {
        uint32_t text_size;
        ssize_t bytes_read = read(STDIN_FILENO, &text_size, sizeof(text_size));
        if (bytes_read != sizeof(text_size)) {
            if (bytes_read == 0) {
                std::cerr << "EOF on input pipe\n";
                break;
            }
            std::cerr << "Failed to read text size, bytes read: " << bytes_read << ": " << strerror(errno) << "\n";
            return 1;
        }
        std::cerr << "Read text size: " << text_size << "\n";
        
        std::vector<char> text_buffer(text_size);
        bytes_read = read(STDIN_FILENO, text_buffer.data(), text_size);
        if (bytes_read != text_size) {
            std::cerr << "Failed to read text, bytes read: " << bytes_read << ": " << strerror(errno) << "\n";
            return 1;
        }
        std::string text(text_buffer.data(), text_size);
        std::cerr << "Received text: " << text << "\n";
        
        // Read language code after text
        uint32_t lang_size;
        if (read(STDIN_FILENO, &lang_size, sizeof(lang_size)) != sizeof(lang_size)) {
            std::cerr << "Failed to read language size" << std::endl;
            return 1;
        }
        std::vector<char> lang_buf(lang_size);
        if (read(STDIN_FILENO, lang_buf.data(), lang_size) != lang_size) {
            std::cerr << "Failed to read language" << std::endl;
            return 1;
        }
        std::string language(lang_buf.data(), lang_size);
        // Lookup voice for language, default to Samantha
        std::string voice = "Samantha";
        auto it = voice_map.find(language);
        if (it != voice_map.end()) voice = it->second;
        std::cerr << "Synthesizing text: " << text << " with voice: " << voice << "\n";
        std::vector<int16_t> samples = synthesize_text(text, voice);
        if (samples.empty()) {
            std::cerr << "Synthesis failed for text: " << text << "\n";
            continue;
        }
        
        // Send audio samples in smaller chunks to avoid blocking on large writes
        uint32_t total_bytes = samples.size() * sizeof(int16_t);
        std::cerr << "Sending " << samples.size() << " int16 samples at 16 kHz (" << total_bytes << " bytes)\n";
        const size_t CHUNK_SAMPLES = 8192;
        size_t offset = 0;
        while (offset < samples.size()) {
            size_t to_send_samples = std::min(samples.size() - offset, CHUNK_SAMPLES);
            uint32_t chunk_bytes = static_cast<uint32_t>(to_send_samples * sizeof(int16_t));
            // Write chunk size then chunk data
            if (write(STDOUT_FILENO, &chunk_bytes, sizeof(chunk_bytes)) != sizeof(chunk_bytes)) {
                std::cerr << "Failed to write chunk size: " << strerror(errno) << "\n";
                return 1;
            }
            if (write(STDOUT_FILENO, samples.data() + offset, chunk_bytes) != static_cast<ssize_t>(chunk_bytes)) {
                std::cerr << "Failed to write chunk: " << strerror(errno) << "\n";
                return 1;
            }
            std::cerr << "Chunk sent to client: " << chunk_bytes << " bytes\n";
            offset += to_send_samples;
        }
        // Send end-of-utterance sentinel (zero-length buffer)
        uint32_t zero = 0;
        if (write(STDOUT_FILENO, &zero, sizeof(zero)) != sizeof(zero)) {
            std::cerr << "Failed to write end-of-utterance sentinel: " << strerror(errno) << "\n";
            return 1;
        }
        std::cerr << "End-of-utterance sentinel sent\n";
    }
    
    std::cerr << "Synthesis process exiting\n";
    return 0;
}