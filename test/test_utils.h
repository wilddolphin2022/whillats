#include <vector>
#include <string>
#include <cstdint>
// Command line options
struct Options {
    bool help = false;
    bool whisper = false;
    bool llama = false;
    std::string help_string;
    std::string whisper_model;
    std::string llama_model;
};

// Function to parse command line string to above options
Options parseOptions(int argc, char* argv[]);

// Function to get command line options to a string, to print or speak
std::string getUsage(const Options opts);

void writeWavFile(const std::string &filename, const std::vector<uint16_t> &audio_data, int sampleRate);
