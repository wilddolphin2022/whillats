/*
 *  Piper TTS via subprocess to avoid libc++/libstdc++ ONNX crash.
 *  Forks a child that runs Piper in pure libstdc++ context.
 *  Parent sends text via pipe, child sends PCM audio back.
 */

#ifndef PIPER_SUBPROCESS_H
#define PIPER_SUBPROCESS_H

#include <cstdint>
#include <string>
#include <vector>
#include <sys/types.h>

class PiperSubprocess {
public:
    PiperSubprocess();
    ~PiperSubprocess();

    bool start(const std::string& model_path, const std::string& espeak_data);
    void stop();

    // Synthesize text, returns PCM int16 samples
    std::vector<int16_t> synthesize(const std::string& text);
    int getSampleRate() const { return _sampleRate; }

private:
    pid_t _child = -1;
    int _toChild = -1;    // pipe: parent writes text
    int _fromChild = -1;  // pipe: parent reads audio
    int _sampleRate = 16000;
    bool _running = false;
};

#endif
