#include <iostream>
#include <vector>
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "espeak_tts.h"
#include "test_utils.h"

// Set log level
void setLogLevel(LogLevel level)
{
  g_currentLogLevel = level;
}

ESpeakTTS tts;

int main(int argc, char *argv[])
{
  Options opts = parseOptions(argc, argv);

  if (argc == 1 || opts.help)
  {
    std::string usage = opts.help_string;
    LOG_E(usage);
    return 1;
  }

  LOG_I(getUsage(opts));

  setLogLevel(LogLevel::VERBOSE);

  // Test ESpeakTTS
  std::vector<uint16_t> audio_buffer;
  const char *test_text = "Hello, this is a test of text to speech synthesis.";
  std::cout << "Testing TTS with text: " << test_text << std::endl;
  tts.synthesize(test_text, audio_buffer);
  std::cout << "Generated " << audio_buffer.size() << " audio samples at "
            << tts.getSampleRate() << "Hz" << std::endl;
  writeWavFile("synthesized_short_audio.wav", audio_buffer, tts.getSampleRate());

  std::vector<uint16_t> long_audio_buffer;
  const char *long_test_text = "Hello, this is a test of text to speech synthesis. "
                               "This is a longer test to ensure we have enough audio data. "
                               "We are testing the whisper transcription system. "
                               "The quick brown fox jumps over the lazy dog.";
  std::cout << "Testing TTS with text: " << long_test_text << std::endl;
  tts.synthesize(long_test_text, long_audio_buffer);
  std::cout << "Generated " << long_audio_buffer.size() << " audio samples at "
            << tts.getSampleRate() << "Hz" << std::endl;
  writeWavFile("synthesized_long_audio.wav", long_audio_buffer, tts.getSampleRate());

  if (opts.whisper) {
    // Test WhisperTranscription
    bool whisper_done = false;
    WhisperTranscriber whisper(opts.whisper_model,
        WhillatsSetResponseCallback([&whisper_done](bool success, const std::string &response) {
          std::cout << "Whisper response via callback: " << response << std::endl;
          whisper_done = true; }
        ));

    // Start the transcriber before processing audio
    if (!whisper.Start())
    {
      std::cout << "Failed to start Whisper transcriber" << std::endl;
      return 1;
    }

    // Calculate chunk size for 10ms at the given sample rate (in samples, not bytes)
    size_t samples_per_chunk = (tts.getSampleRate() * 10) / 1000;
    std::cout << "Processing audio in " << samples_per_chunk << " sample chunks" << std::endl;

    // Process short audio
    std::cout << "\nProcessing short audio..." << std::endl;
    for (size_t i = 0; i < audio_buffer.size(); i += samples_per_chunk)
    {
      size_t chunk_size = std::min(samples_per_chunk, audio_buffer.size() - i);
      whisper.ProcessAudioBuffer((uint8_t *)(audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
    }
    whisper.ProcessAudioBuffer(nullptr, -1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    // Process long audio
    std::cout << "\nProcessing long audio..." << std::endl;
    for (size_t i = 0; i < long_audio_buffer.size(); i += samples_per_chunk)
    {
      size_t chunk_size = std::min(samples_per_chunk, long_audio_buffer.size() - i);
      whisper.ProcessAudioBuffer((uint8_t *)(long_audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
    }
    whisper.ProcessAudioBuffer(nullptr, -1);
    while (!whisper_done)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop the transcriber
    whisper.Stop();
  }

  if (opts.llama) {
    //  Test LlamaDeviceBase
    const char *model_path = "models/chat.gguf";

    bool llama_done = false;
    LlamaDeviceBase llama(opts.llama_model,
      WhillatsSetResponseCallback([&llama_done](bool success, const std::string &response) {
      std::cout << "LLama response via callback: " << response << std::endl; 
      llama_done = true; }
      ));

    LOG_I("Initializing Llama with model: " << model_path);
    if (llama.Start())
    {

      std::string prompt = "What will be 2+2?";
      LOG_I("Testing Llama with prompt: " << prompt);
      llama.askLlama(prompt);

      while (!llama_done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      llama.Stop();
    }
    else
    {
      LOG_E("Failed to initialize LLama model");
    }
  }
  return 0;
}
