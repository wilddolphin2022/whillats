#ifndef WHILLATS_SYNTH_H
#define WHILLATS_SYNTH_H

#include "whillats.h"

#include <string>
#include <functional>
#include <vector>

class WhillatsSetAudioCallback;
class WHILLATS_API WhillatsSpeechSynthesizerWrapper {
public:
    using CompletionCallback = std::function<void()>;

    WhillatsSpeechSynthesizerWrapper();
    ~WhillatsSpeechSynthesizerWrapper();

    void initialize(WhillatsSetAudioCallback* audioCallback, CompletionCallback completionCallback);
    void synthesize(const std::string& text);
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // WHILLATS_SYNTH_H
