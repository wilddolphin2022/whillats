#include "test_utils.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "whisper_helpers.h"

Options parseOptions(int argc, char *argv[])
{
  Options opts;
  // Initialize defaults
  opts.help = false;
  opts.help_string = "Usage:\n"
                     "test_whillats [options]\n\n"
                     "Options:\n"
                     "  --tts, --no-tts                    Enable/disable tts (default: disabled)\n"
                     "  --whisper, --no-whisper            Enable/disable whisper (default: disabled)\n"
                     "  --llama, --no-llama                Enable/disable llama (default: disabled)\n"
                     "  --whisper_model=<path>             Path to whisper model\n"
                     "  --llama_model=<path>               Path to llama model\n"
                     "  --help                             Show this help message\n"
                     "\nExamples:\n"
                     "  test_whillats --whisper --whisper_model=model.bin\n"
                     "  test_whillats --llama --llama_model=model.bin\n";

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];

    // Handle parameters with values
    if (arg == "--help")
    {
      opts.help = true;
    }
    else if (arg == "--tts")
    {
      opts.tts = true;
    }
    else if (arg == "--no-tts")
    {
      opts.tts = false;
    }
    else if (arg == "--whisper")
    {
      opts.whisper = true;
    }
    else if (arg == "--no-whisper")
    {
      opts.whisper = false;
    }
    else if (arg == "--llama")
    {
      opts.llama = true;
    }
    else if (arg == "--no-llama")
    {
      opts.llama = false;
    }
    else if (arg.find("--whisper_model=") == 0)
    {
      opts.whisper_model = arg.substr(16); // Length of "-whisper_model="
      LOG_I("Whisper model path: " << opts.whisper_model);
      if (!opts.whisper)
        opts.whisper = true;
    }
    else if (arg.find("--llama_model=") == 0)
    {
      opts.llama_model = arg.substr(14); // Length of "-llama_model="
      LOG_I("Llama model path: " << opts.llama_model);
    }
  }

  // Load environment variables if paths not provided
  if (opts.whisper_model.empty())
  {
    if (const char *env_whisper = std::getenv("WHISPER_MODEL"))
    {
      opts.whisper_model = env_whisper;
    }
  }
  if (opts.llama_model.empty())
  {
    if (const char *env_llama = std::getenv("LLAMA_MODEL"))
    {
      opts.llama_model = env_llama;
    }
  }

  return opts;
}

std::string getUsage(const Options opts)
{
  std::stringstream usage;

  usage << "\nWhisper: " << (opts.whisper ? "enabled" : "disabled") << "\n";
  usage << "Llama: " << (opts.llama ? "enabled" : "disabled") << "\n";
  usage << "Whisper Model: " << opts.whisper_model << "\n";
  usage << "Llama Model: " << opts.llama_model << "\n";

  return usage.str();
}

void writeWavFile(const std::string &filename, const std::vector<uint16_t> &audio_data, int sampleRate)
{
  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile.is_open())
  {
    std::cout << "Failed to open file: " << filename << std::endl;
    return;
  }

  const int channels = 1;
  const int bitsPerSample = 16;
  const int dataSize = audio_data.size() * sizeof(uint16_t);
  const int headerSize = 44;
  const int fileSize = headerSize + dataSize;

  outFile.write("RIFF", 4);
  outFile.write(reinterpret_cast<const char *>(&fileSize), 4);
  outFile.write("WAVE", 4);
  outFile.write("fmt ", 4);
  int subchunk1Size = 16;
  outFile.write(reinterpret_cast<const char *>(&subchunk1Size), 4);
  int16_t audioFormat = 1; // PCM
  outFile.write(reinterpret_cast<const char *>(&audioFormat), 2);
  int16_t numChannels = channels;
  outFile.write(reinterpret_cast<const char *>(&numChannels), 2);
  outFile.write(reinterpret_cast<const char *>(&sampleRate), 4);
  int byteRate = sampleRate * channels * bitsPerSample / 8;
  outFile.write(reinterpret_cast<const char *>(&byteRate), 4);
  int16_t blockAlign = channels * bitsPerSample / 8;
  outFile.write(reinterpret_cast<const char *>(&blockAlign), 2);
  int16_t bitsPerSampleShort = bitsPerSample;
  outFile.write(reinterpret_cast<const char *>(&bitsPerSampleShort), 2);
  outFile.write("data", 4);
  outFile.write(reinterpret_cast<const char *>(&dataSize), 4);

  outFile.write(reinterpret_cast<const char *>(audio_data.data()), dataSize);
  outFile.close();
  std::cout << "Saved audio to " << filename << std::endl;
}
