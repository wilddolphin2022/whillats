#include <iostream>
#include <vector>
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "espeak_tts.h"

int main() {
    // Test ESpeakTTS
    ESpeakTTS tts;
    std::vector<uint16_t> audio_buffer;
    const char* test_text = "Hello, this is a test of text to speech synthesis.";
    std::cout << "Testing TTS with text: " << test_text << std::endl;
    tts.synthesize(test_text, audio_buffer);
    std::cout << "Generated " << audio_buffer.size() << " audio samples at " 
              << tts.getSampleRate() << "Hz" << std::endl;

    // Test WhisperTranscription
    WhisperTranscription whisper;
    std::string transcription;
    const char* whisper_model = "models/ggml-base.bin";
    std::cout << "Testing Whisper with model: " << whisper_model << std::endl;
    if (whisper.initialize(whisper_model)) {
        std::cout << "Whisper initialized successfully" << std::endl;
        // Use the audio buffer from TTS as input
        transcription = whisper.transcribe(audio_buffer.data(), audio_buffer.size());
        std::cout << "Transcription: " << transcription << std::endl;
    } else {
        std::cout << "Failed to initialize Whisper model" << std::endl;
    }

    // Test LlamaDeviceBase
    LlamaDeviceBase llama;
    const char* model_path = "models/DeepSeek-R1-Distill-Llama-8B-Q2_K_L.gguf";
    std::cout << "Initializing LLama with model: " << model_path << std::endl;
    if (llama.initialize(model_path)) {
        std::string prompt = "Please summarize this text in one sentence: " + transcription;
        std::cout << "Testing LLama with prompt: " << prompt << std::endl;
        std::string response = llama.generate(prompt);
        std::cout << "LLama response: " << response << std::endl;
    } else {
        std::cout << "Failed to initialize LLama model" << std::endl;
    }

    return 0;
}