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

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include "whillats.h"

#include "test_utils.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"
#include "script_engine.h"

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
bool script_done = false;
std::vector<std::string> script_events;

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

// Auto-detect and set espeak data path if not already set
void setupEspeakDataPath() {
    if (getenv("ESPEAK_DATA_PATH")) {
        return; // Already set
    }

    std::string bin_dir;

#if defined(__APPLE__)
    // macOS: use _NSGetExecutablePath
    char exe_path[1024];
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        // Resolve symlinks
        char real_path[1024];
        if (realpath(exe_path, real_path)) {
            bin_dir = std::string(real_path);
        } else {
            bin_dir = std::string(exe_path);
        }
    }
#else
    // Linux: use /proc/self/exe
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        bin_dir = std::string(exe_path);
    }
#endif

    if (!bin_dir.empty()) {
        size_t last_slash = bin_dir.find_last_of('/');
        if (last_slash != std::string::npos) {
            bin_dir = bin_dir.substr(0, last_slash);
            std::string data_path = bin_dir + "/espeak-ng-data";

            struct stat st;
            if (stat(data_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                setenv("ESPEAK_DATA_PATH", data_path.c_str(), 1);
                LOG_I("Auto-detected espeak data path: " << data_path);
                return;
            }
        }
    }

    LOG_W("Could not auto-detect espeak data path. Please set ESPEAK_DATA_PATH environment variable.");
}

int main(int argc, char *argv[])
{
  // Setup espeak data path first
  setupEspeakDataPath();
  
  Options opts = parseOptions(argc, argv);

  if (argc == 1 || opts.help)
  {
    std::string usage = opts.help_string;
    LOG_E(usage);
    return 1;
  }

  LOG_I(getUsage(opts));

#ifdef WHILLATS_STYLETTS2
  // Set StyleTTS2 env vars from command line options
  if (!opts.styletts2_model_dir.empty()) {
    setenv("STYLETTS2_MODEL_DIR", opts.styletts2_model_dir.c_str(), 1);
    LOG_I("Set STYLETTS2_MODEL_DIR=" << opts.styletts2_model_dir);
  }
  if (opts.styletts2_model_dir.empty() && !getenv("STYLETTS2_MODEL_DIR")) {
    LOG_W("StyleTTS2 enabled but no model dir specified. Use --styletts2_model_dir= or set STYLETTS2_MODEL_DIR");
  }
#endif

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
      tts.queueText(test_text, "en-US");
      // Fallback: poll until completion
      while (!tts_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      // Write accumulated audio for first utterance
      writeWavFile("synthesized_audio.wav", audio_buffer, WhillatsTTS::getSampleRate());
      // Prepare for next utterance: clear buffer and flag
      tts_done = false;
      audio_buffer.clear();

      const char *long_test_text = "Hello, this is a test of text to speech synthesis. "
                                  "This is a longer test to ensure we have enough audio data. "
                                  "We are testing the whisper transcription system. "
                                  "The quick brown fox jumps over the lazy dog";
                                  
      const char *spanish_test_text = "¿Cómo estás? ¿cómo te llamas?";
      const char *russian_test_text = "У вас есть меню на английском?";
      // Queue and wait for long English utterance
      std::cout << "Testing TTS with text: " << long_test_text << std::endl;
      tts.queueText(long_test_text, "en");
      while (!tts_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      tts_done = false;
      // Queue and wait for Spanish utterance
      std::cout << "Testing TTS with text: " << spanish_test_text << std::endl;
      tts.queueText(spanish_test_text, "es");
      while (!tts_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      tts_done = false;
      // Queue and wait for Russian utterance
      std::cout << "Testing TTS with text: " << russian_test_text << std::endl;
      tts.queueText(russian_test_text, "ru");
      while (!tts_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      
      tts_done = false;
      // Write accumulated audio for all utterances
      writeWavFile("synthesized_audio_long.wav", audio_buffer, WhillatsTTS::getSampleRate());
      // Stop TTS after all audio
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
      if(!opts.test_image1.empty())
      {
        YUVData grey_yuv;
        load_yuv(grey_yuv, opts.test_image1.c_str(), 300, 300);

        llama.askWithYUVRaw("Describe the contents of the image in detail.", 
                            grey_yuv.y.get(), grey_yuv.u.get(), grey_yuv.v.get(), 
                            grey_yuv.width, grey_yuv.height, 
                            grey_yuv.y_size, grey_yuv.uv_size);
        
        // Wait for the first image processing to complete
        while (!llama_done)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      llama_done = false;
      }

      if(!opts.test_image2.empty())
      {
        YUVData yuv;
        load_yuv(yuv, opts.test_image2.c_str(), 1754, 1240);

        //llama.setImage(*yuv);
        llama.askWithYUVRaw("Describe the contents of the image in detail.", 
                            yuv.y.get(), yuv.u.get(), yuv.v.get(), 
                            yuv.width, yuv.height, 
                            yuv.y_size, yuv.uv_size);
        
        // Wait for the second image processing to complete
        while (!llama_done)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        llama_done = false;
      }
      
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

  // --- Script Engine Tests ---
  if (opts.script) {
    LOG_I("=== Script Engine Tests ===");

    // Test 1: ScriptEngine YAML parsing (unit test, no TTS needed)
    {
      LOG_I("Test 1: YAML loading and parsing");
      ScriptEngine engine;

      std::string test_yaml = R"(
script:
  name: "Unit Test Script"
  language: "en"

steps:
  - id: greet
    action: speak
    text: "Hello from test."
    next: ask

  - id: ask
    action: listen
    prompt: "Say something."
    timeout_ms: 2000
    store_as: user_input
    on_match:
      - pattern: "yes|ok"
        next: confirmed
      - pattern: "no|cancel"
        next: denied
    next: echo

  - id: confirmed
    action: speak
    text: "You confirmed."
    next: done

  - id: denied
    action: speak
    text: "You denied."
    next: done

  - id: echo
    action: speak
    text: "You said: ${user_input}"
    next: done

  - id: done
    action: end
)";

      if (!engine.loadFromString(test_yaml)) {
        LOG_E("FAIL: Could not parse YAML");
        return 1;
      }
      LOG_I("  Script name: " << engine.config().name);
      LOG_I("  Steps: " << engine.config().steps.size());
      assert(engine.config().name == "Unit Test Script");
      assert(engine.config().steps.size() == 6);
      assert(engine.config().language == "en");
      LOG_I("  PASS: YAML parsing");
    }

    // Test 2: State machine transitions with simulated events
    {
      LOG_I("Test 2: State machine transitions");
      ScriptEngine engine;

      std::string yaml = R"(
script:
  name: "State Machine Test"
  language: "en"

steps:
  - id: greet
    action: speak
    text: "Hello."
    next: listen_step

  - id: listen_step
    action: listen
    timeout_ms: 2000
    store_as: input
    on_match:
      - pattern: "yes"
        next: yes_branch
      - pattern: "no"
        next: no_branch
    next: default_branch

  - id: yes_branch
    action: speak
    text: "You said yes."
    next: done

  - id: no_branch
    action: speak
    text: "You said no."
    next: done

  - id: default_branch
    action: speak
    text: "Default: ${input}"
    next: done

  - id: done
    action: end
)";

      engine.loadFromString(yaml);

      std::vector<std::string> events;
      engine.setEventHandler([&events](const ScriptEvent& e) {
        std::string type_str;
        switch (e.type) {
          case ScriptEventType::SPEAK_REQUEST: type_str = "speak"; break;
          case ScriptEventType::LISTEN_START: type_str = "listen"; break;
          case ScriptEventType::SCRIPT_END: type_str = "end"; break;
          case ScriptEventType::STEP_CHANGED: type_str = "step:" + e.step_id; break;
          default: type_str = "other"; break;
        }
        events.push_back(type_str + ":" + e.data.substr(0, 30));
      });

      engine.start();
      assert(engine.currentStepId() == "greet");

      // Simulate TTS completion -> moves to listen_step
      engine.onSpeechComplete();
      assert(engine.currentStepId() == "listen_step");

      // Simulate transcription "yes" -> should go to yes_branch
      engine.onTranscriptionReceived("yes");
      assert(engine.currentStepId() == "yes_branch");
      assert(engine.getVariable("input") == "yes");

      // Simulate TTS completion -> moves to done
      engine.onSpeechComplete();
      assert(engine.currentStepId() == "done");
      assert(!engine.isRunning());

      LOG_I("  Events captured: " << events.size());
      for (const auto& ev : events) {
        LOG_V("    " << ev);
      }
      LOG_I("  PASS: State machine with 'yes' branch");
    }

    // Test 3: "no" branch and variable expansion
    {
      LOG_I("Test 3: 'no' branch and variable expansion");
      ScriptEngine engine;

      std::string yaml = R"(
script:
  name: "Branch Test"
  language: "en"

steps:
  - id: start
    action: speak
    text: "Start."
    next: ask

  - id: ask
    action: listen
    timeout_ms: 2000
    store_as: answer
    on_match:
      - pattern: "yes"
        next: yes_path
      - pattern: "no"
        next: no_path
    next: fallback

  - id: yes_path
    action: speak
    text: "Yes path."
    next: end_step

  - id: no_path
    action: speak
    text: "No path."
    next: end_step

  - id: fallback
    action: speak
    text: "Fallback: ${answer}"
    next: end_step

  - id: end_step
    action: end
)";

      engine.loadFromString(yaml);
      std::string last_speak;
      engine.setEventHandler([&last_speak](const ScriptEvent& e) {
        if (e.type == ScriptEventType::SPEAK_REQUEST) last_speak = e.data;
      });

      engine.start();
      engine.onSpeechComplete(); // greet -> ask
      engine.onTranscriptionReceived("no way");
      assert(engine.getVariable("answer") == "no way");
      assert(engine.currentStepId() == "no_path");
      LOG_I("  PASS: 'no' match works");

      // Test fallback path
      engine.stop();
      engine.loadFromString(yaml);
      engine.start();
      engine.onSpeechComplete(); // start -> ask
      engine.onTranscriptionReceived("something random");
      assert(engine.currentStepId() == "fallback");
      assert(last_speak.find("something random") != std::string::npos);
      LOG_I("  PASS: Fallback with variable expansion");
    }

    // Test 4: Timeout handling
    {
      LOG_I("Test 4: Listen timeout");
      ScriptEngine engine;

      std::string yaml = R"(
script:
  name: "Timeout Test"
  language: "en"

steps:
  - id: start
    action: speak
    text: "Speak now."
    next: listen_step

  - id: listen_step
    action: listen
    timeout_ms: 100
    store_as: input
    next: after_timeout

  - id: after_timeout
    action: speak
    text: "Timed out."
    next: done

  - id: done
    action: end
)";

      engine.loadFromString(yaml);
      bool timed_out = false;
      engine.setEventHandler([&timed_out](const ScriptEvent& e) {
        if (e.type == ScriptEventType::LISTEN_TIMEOUT) timed_out = true;
      });

      engine.start();
      engine.onSpeechComplete(); // start -> listen_step

      // Wait for timeout
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      engine.checkTimeout();

      assert(timed_out);
      assert(engine.currentStepId() == "after_timeout");
      LOG_I("  PASS: Timeout triggers correctly");
    }

    // Test 5: Full script from YAML file (if path provided)
    if (!opts.script_path.empty()) {
      LOG_I("Test 5: Load script from file: " << opts.script_path);

      script_events.clear();
      script_done = false;

      auto scriptEventCb = [](const char* event_type, const char* step_id,
                               const char* data, void* user_data) {
        std::string evt = std::string(event_type) + ":" + std::string(step_id);
        script_events.push_back(evt);
        LOG_I("  Script event: " << event_type << " step=" << step_id
              << " data=" << (data ? std::string(data).substr(0, 40) : ""));
        if (std::string(event_type) == "end") {
          script_done = true;
        }
      };

      WhillatsSetAudioCallback audio_cb(ttsAudioCallback, nullptr);
      WhillatsSetResponseCallback response_cb(whisperResponseCallback, nullptr);

      WhillatsScript script(opts.script_path.c_str(),
                            audio_cb, response_cb,
                            scriptEventCb, nullptr);

      if (script.start(
            opts.whisper_model.empty() ? nullptr : opts.whisper_model.c_str(),
            opts.llama_model.empty() ? nullptr : opts.llama_model.c_str(),
            opts.llama_mmproj.empty() ? nullptr : opts.llama_mmproj.c_str())) {

        LOG_I("  Script started at step: " << script.currentStep());

        // Let it run for a few seconds (TTS will speak, then timeout on listen)
        int wait_ms = 0;
        while (!script_done && wait_ms < 60000) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          wait_ms += 200;
        }

        LOG_I("  Script finished. Events: " << script_events.size());
        for (const auto& ev : script_events) {
          LOG_I("    " << ev);
        }

        script.stop();
      } else {
        LOG_E("  FAIL: Could not start script");
      }
    }

    LOG_I("=== All Script Engine Tests Passed ===");
  }

  return 0;
}
