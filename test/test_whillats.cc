#include <iostream>
#include <vector>
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "espeak_tts.h"
#include <fstream>

// Set log level
void setLogLevel(LogLevel level) {
    g_currentLogLevel = level;
}

ESpeakTTS tts;

void writeWavFile(const std::string& filename, const std::vector<uint16_t>& audio_data, int sampleRate) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }

    const int channels = 1;
    const int bitsPerSample = 16;
    const int dataSize = audio_data.size() * sizeof(uint16_t);
    const int headerSize = 44;
    const int fileSize = headerSize + dataSize;

    outFile.write("RIFF", 4);
    outFile.write(reinterpret_cast<const char*>(&fileSize), 4);
    outFile.write("WAVE", 4);
    outFile.write("fmt ", 4);
    int subchunk1Size = 16;
    outFile.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    int16_t audioFormat = 1; // PCM
    outFile.write(reinterpret_cast<const char*>(&audioFormat), 2);
    int16_t numChannels = channels;
    outFile.write(reinterpret_cast<const char*>(&numChannels), 2);
    outFile.write(reinterpret_cast<const char*>(&sampleRate), 4);
    int byteRate = sampleRate * channels * bitsPerSample / 8;
    outFile.write(reinterpret_cast<const char*>(&byteRate), 4);
    int16_t blockAlign = channels * bitsPerSample / 8;
    outFile.write(reinterpret_cast<const char*>(&blockAlign), 2);
    int16_t bitsPerSampleShort = bitsPerSample;
    outFile.write(reinterpret_cast<const char*>(&bitsPerSampleShort), 2);
    outFile.write("data", 4);
    outFile.write(reinterpret_cast<const char*>(&dataSize), 4);

    outFile.write(reinterpret_cast<const char*>(audio_data.data()), dataSize);
    outFile.close();
    std::cout << "Saved audio to " << filename << std::endl;
}

int main() {
    setLogLevel(LogLevel::INFO);

    // Test ESpeakTTS
    std::vector<uint16_t> audio_buffer;
    const char* test_text = "Hello, this is a test of text to speech synthesis.";
    std::cout << "Testing TTS with text: " << test_text << std::endl;
    tts.synthesize(test_text, audio_buffer);
    std::cout << "Generated " << audio_buffer.size() << " audio samples at " 
              << tts.getSampleRate() << "Hz" << std::endl;
    writeWavFile("synthesized_short_audio.wav", audio_buffer, tts.getSampleRate());

    std::vector<uint16_t> long_audio_buffer;
    const char* long_test_text = "Hello, this is a test of text to speech synthesis. "
                           "This is a longer test to ensure we have enough audio data. "
                           "We are testing the whisper transcription system. "
                           "The quick brown fox jumps over the lazy dog.";
    std::cout << "Testing TTS with text: " << long_test_text << std::endl;
    tts.synthesize(long_test_text, long_audio_buffer);
    std::cout << "Generated " << long_audio_buffer.size() << " audio samples at " 
              << tts.getSampleRate() << "Hz" << std::endl;
    writeWavFile("synthesized_long_audio.wav", long_audio_buffer, tts.getSampleRate());

    // Test WhisperTranscription
    const char* whisper_model = "models/ggml-base.bin";
    WhisperTranscriber whisper(nullptr, whisper_model);
    
    // Start the transcriber before processing audio
    if (!whisper.Start()) {
        std::cout << "Failed to start Whisper transcriber" << std::endl;
        return 1;
    }
    
    // Calculate chunk size for 10ms at the given sample rate (in samples, not bytes)
    size_t samples_per_chunk = (tts.getSampleRate() * 10) / 1000;
    std::cout << "Processing audio in " << samples_per_chunk << " sample chunks" << std::endl;
    
    // Process short audio
    std::cout << "\nProcessing short audio..." << std::endl;
    for (size_t i = 0; i < audio_buffer.size(); i += samples_per_chunk) {
        size_t chunk_size = std::min(samples_per_chunk, audio_buffer.size() - i);
        whisper.ProcessAudioBuffer((uint8_t*)(audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
    }
    whisper.ProcessAudioBuffer(nullptr, -1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    
    // Process long audio
    std::cout << "\nProcessing long audio..." << std::endl;
    for (size_t i = 0; i < long_audio_buffer.size(); i += samples_per_chunk) {
        size_t chunk_size = std::min(samples_per_chunk, long_audio_buffer.size() - i);
        whisper.ProcessAudioBuffer((uint8_t*)(long_audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
    }
    whisper.ProcessAudioBuffer(nullptr, -1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    // Stop the transcriber
    whisper.Stop();
    
    //std::cout << "Transcription: " << whisper.lastTranscription << std::endl;

    // Test LlamaDeviceBase
    // LlamaDeviceBase llama;
    // const char* model_path = "models/DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf";
    // std::cout << "Initializing LLama with model: " << model_path << std::endl;
    // if (llama.initialize(model_path)) {
    //     std::string prompt = "Please summarize this text in one sentence: " + transcription;
    //     std::cout << "Testing LLama with prompt: " << prompt << std::endl;
    //     std::string response = llama.generate(prompt);
    //     std::cout << "LLama response: " << response << std::endl;
    // } else {
    //     std::cout << "Failed to initialize LLama model" << std::endl;
    // }

    return 0;
}