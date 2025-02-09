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

    // Pad audio buffer with silence to reach target samples (12 seconds)
    const size_t targetSamples = 16000 * 12 * 2;  // 12 seconds at 16kHz, stereo
    size_t padding_size = (audio_buffer.size() < targetSamples) ? 
                         targetSamples - audio_buffer.size() : 0;
    if (padding_size > 0) {
        audio_buffer.resize(targetSamples, 0);  // Pad with zeros (silence)
        std::cout << "Padded audio with " << padding_size << " silent samples" << std::endl;
    }

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
    
    // Process all chunks
    for (size_t i = 0; i < audio_buffer.size(); i += samples_per_chunk) {
        size_t chunk_size = std::min(samples_per_chunk, audio_buffer.size() - i);
        whisper.ProcessAudioBuffer((uint8_t*)(audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Force process any remaining audio by sending a zero-length buffer
    whisper.ProcessAudioBuffer(nullptr, 0);
    
    // Wait a bit longer for processing to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Stop the transcriber
    whisper.Stop();
    
    std::cout << "Transcription: " << whisper.lastTranscription << std::endl;

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