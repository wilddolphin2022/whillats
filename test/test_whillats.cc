/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <iostream>
#include <vector>
#include "whillats.h"

#include "test_utils.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"

#ifdef __APPLE__ 
#define WHILLATS_USE_CF_RUNLOOP 1
#if WHILLATS_USE_CF_RUNLOOP
#include <CoreFoundation/CFRunLoop.h> // For CFRunLoopRunInMode
#endif
#endif

// Set log level
void setLogLevel(LogLevel level)
{
  g_currentLogLevel = level;
}

std::vector<uint16_t> audio_buffer;
bool tts_done = false;
bool whisper_done = false;
bool llama_done = false;
bool language_changed = false;

static size_t bufferCount = 0;

void ttsAudioCallback(bool success, const uint16_t* buffer, size_t buffer_size, void* user_data) {
    // Only handle actual audio data
    if (success) 
    {
        LOG_I("Generated " << buffer_size << " audio samples at " << WhillatsTTS::getSampleRate() << "Hz");
        if(buffer && buffer_size > 0) {
            audio_buffer.insert(audio_buffer.end(), buffer, buffer + buffer_size);
        }
    } else {
        // Signal end of synthesis
        LOG_I("TTS done");
        tts_done = true;
    }
}

void whisperResponseCallback(bool success, const char* response, void* user_data) {
    // Handle response here
    std::cout << "Whisper response via callback: " << response << std::endl;
    whisper_done = true; 
}

void llamaResponseCallback(bool success, const char* response, void* user_data) {
    // Handle response here
    std::cout << "Llama response via callback: " << response << std::endl;
    llama_done = true;   
}

void languageChangedCallback(bool success, const char* language, void* user_data) {
    // Handle response here
    std::cout << "Language changed via callback: " << language << std::endl;
    language_changed = true;
}

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
  opts.tts = true;

  setLogLevel(LogLevel::VERBOSE);

  if (opts.tts) {
    // Clear buffer before starting TTS
    audio_buffer.clear();
    WhillatsSetAudioCallback callback(ttsAudioCallback, nullptr);
    WhillatsTTS tts(callback); 
      
    if(tts.start()) {

      const char *test_text = "Hello, this is a test of text to speech synthesis.";
      std::cout << "Testing TTS with text: " << test_text << std::endl;

      // Queue and wait for first utterance
      tts.queueText(test_text, "en");
#if WHILLATS_USE_CF_RUNLOOP
      // Run the main run loop until .mm signals CFRunLoopStop
      CFRunLoopRun();
#else
      // Fallback: poll until completion
      while (!tts_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
#endif

      // Write accumulated audio for first utterance
      writeWavFile("synthesized_audio.wav", audio_buffer, WhillatsTTS::getSampleRate());
      // Prepare for next utterance
      tts_done = false;
      audio_buffer.clear();

      const char *long_test_text = "Hello, this is a test of text to speech synthesis. "
                                  "This is a longer test to ensure we have enough audio data. "
                                  "We are testing the whisper transcription system. "
                                  "The quick brown fox jumps over the lazy dog. "
                                  "¿Cómo estás? У вас есть меню на английском?";
      std::cout << "Testing TTS with text: " << long_test_text << std::endl;
      
      // Queue and wait for second (long) utterance
      tts.queueText(long_test_text, "en");
#if WHILLATS_USE_CF_RUNLOOP
      CFRunLoopRun();
#else
      while (!tts_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
#endif

      tts_done = false;
      // Write accumulated audio for second utterance
      writeWavFile("synthesized_audio_long.wav", audio_buffer, WhillatsTTS::getSampleRate());
      tts.stop();
    }
  }

  if (opts.whisper) {
    // Test WhisperTranscription
    WhillatsSetResponseCallback callback(whisperResponseCallback, nullptr);
    WhillatsSetLanguageCallback language_callback(languageChangedCallback, nullptr);
    WhillatsTranscriber whisper(opts.whisper_model.c_str(), callback, language_callback);

    // Start the transcriber before processing audio
    if (!whisper.start()) 
    {
      LOG_E("Failed to start Whisper transcriber");

    } else {
      LOG_I("Whisper transcriber started");

      // Calculate chunk size for 10ms at the given sample rate (in samples, not bytes)
      size_t samples_per_chunk = (WhillatsTTS::getSampleRate() * 10) / 1000;
      std::cout << "Processing audio in " << samples_per_chunk << " sample chunks" << std::endl;

      // Process audio
      LOG_V("Processing audio buffer size: " << audio_buffer.size() << "..." << std::endl);
      for (size_t i = 0; i < audio_buffer.size(); i += samples_per_chunk)
      {
        size_t chunk_size = std::min(samples_per_chunk, audio_buffer.size() - i);
        whisper.processAudioBuffer((uint8_t *)(audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
      }

      LOG_V("Short cutting audio buffer size: " << audio_buffer.size() << "..." << std::endl);
      whisper.processAudioBuffer(nullptr, 0);

      while (!whisper_done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      whisper_done = false;

      // Process long audio
      std::cout << "\nProcessing long audio..." << std::endl;
      for (size_t i = 0; i < audio_buffer.size(); i += samples_per_chunk)
      {
        size_t chunk_size = std::min(samples_per_chunk, audio_buffer.size() - i);
        whisper.processAudioBuffer((uint8_t *)(audio_buffer.data() + i), chunk_size * sizeof(uint16_t));
      }
      
      whisper.processAudioBuffer(nullptr, 0);
      while (!whisper_done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      // Stop the transcriber
      whisper.stop(); 
    }
  }

  if (opts.llama) {
    //  Test LlamaDeviceBase
    WhillatsSetResponseCallback callback(llamaResponseCallback, nullptr);
    WhillatsLlama llama(opts.llama_model.c_str(), opts.llama_mmproj.c_str(), callback);

    LOG_I("Initializing Llama with model: " << opts.llama_model);
    if (llama.start()) 
    {
      YUVData grey_yuv;
      load_yuv(grey_yuv, opts.test_image1.c_str(), 300, 300);

      llama.askWithImage("Describe the contents of the image in detail.", grey_yuv);
      
      // Wait for the first image processing to complete
      while (!llama_done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      llama_done = false;
      
      YUVData yuv;
      load_yuv(yuv, opts.test_image2.c_str(), 1754, 1240);

      //llama.setImage(*yuv);
      llama.askWithImage("Describe the contents of the image in detail.", yuv);
      
      // Wait for the second image processing to complete
      while (!llama_done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      llama_done = false;
      
      std::string prompt = "What is your name?";
      LOG_I("Testing Llama with prompt: " << prompt);
      llama.askLlama(prompt.c_str());
      
      // Wait for the text query response
      while (!llama_done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      llama_done = false;
      
      llama.stop();
    }
    else
    {
      LOG_E("Failed to initialize LLama model");
    }
  }
  return 0;
}
