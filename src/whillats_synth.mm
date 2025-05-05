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

#import <Foundation/Foundation.h>
#include "whillats_osx.h"
#include "whillats_synth.h"
#include "whisper_helpers.h"
#include <vector>
#include <memory>
#include <iostream>

// Define a single context struct for audio callback
struct CallbackContext {
    WhillatsSetAudioCallback* audioCallbackPtr;
};

// Static C bridge function for AudioCallback
static void AudioCallbackBridge(bool success, const uint16_t* buffer, size_t size, void* user_data) {
    CallbackContext* context = static_cast<CallbackContext*>(user_data);
    if (context && context->audioCallbackPtr) {
        // If there is valid data, deliver buffer; otherwise signal completion
        if (success && buffer && size > 0) {
            std::vector<uint16_t> bufferVec(buffer, buffer + size);
            context->audioCallbackPtr->OnBufferComplete(true, bufferVec);
        } else {
            // End of synthesis or error: invoke synthesis-complete callback
            context->audioCallbackPtr->OnSynthesisComplete();
        }
    }
}

// Static C bridge function for CompletionCallback
static void CompletionCallbackBridge(void* user_data) {
    CallbackContext* context = static_cast<CallbackContext*>(user_data);
    if (context && context->audioCallbackPtr) {
        context->audioCallbackPtr->OnSynthesisComplete();
    }
}

// Define Impl
struct WhillatsSpeechSynthesizerWrapper::Impl {
    WhillatsSpeechSynthesizerProcessor* processor;
    WhillatsSetAudioCallback* audioCallbackPtr;
    Impl() : processor(nil), audioCallbackPtr(nullptr) {}
};

WhillatsSpeechSynthesizerWrapper::WhillatsSpeechSynthesizerWrapper() :
    impl(std::make_unique<Impl>()) {}

WhillatsSpeechSynthesizerWrapper::~WhillatsSpeechSynthesizerWrapper() {
    stop();
}

void WhillatsSpeechSynthesizerWrapper::initialize(WhillatsSetAudioCallback* audioCallback) {
    // Clean up any prior processor
    if (impl->processor) stop();
    // Store callback pointer
    impl->audioCallbackPtr = audioCallback;
    // Create and populate the shared context
    CallbackContext* context = new CallbackContext();
    context->audioCallbackPtr = impl->audioCallbackPtr;
    // Initialize the OS X speech processor
    impl->processor = [[WhillatsSpeechSynthesizerProcessor alloc]
                       initWithAudioCallback:AudioCallbackBridge
                                 userData:context];
    if (!impl->processor) {
        LOG_E("[Whillats]: Failed to create SpeechSynthesizerProcessor");
        delete context;
    }
#if TARGET_OS_IOS
    [impl->processor enableSpeakerphone];
#endif
}

void WhillatsSpeechSynthesizerWrapper::synthesize(const std::string& text, const std::string& language) {
    NSString* nsText = [NSString stringWithUTF8String:text.c_str()];
    NSString* nsLanguage = [NSString stringWithUTF8String:language.c_str()];
    if (!nsText || !nsLanguage) return;
    _lastText = text;
    _lastLanguage = language;
    [impl->processor synthesizeText:nsText language:nsLanguage];
}

void WhillatsSpeechSynthesizerWrapper::stop() {
    if (impl->processor) {
        [impl->processor stop];
        CallbackContext* context = static_cast<CallbackContext*>([impl->processor userData]);
        if (context) delete context;
        impl->processor = nil;
    }
}

void WhillatsSpeechSynthesizerWrapper::setNotificationName(const char* name) {
    _notification_name = name;
}

#if TARGET_OS_IOS
void WhillatsSpeechSynthesizerWrapper::enableSpeakerphone() {
    if (impl->processor) {
        [impl->processor enableSpeakerphone];
    }
}

void WhillatsSpeechSynthesizerWrapper::disableSpeakerphone() {
    if (impl->processor) {
        [impl->processor disableSpeakerphone];
    }
}
#endif