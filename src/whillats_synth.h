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
    void synthesize(const std::string& text, const std::string& language);
    void stop();

    void enableSpeakerphone();
    void disableSpeakerphone();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    std::string _lastLanguage; 
    std::string _lastText;
};

#endif // WHILLATS_SYNTH_H
