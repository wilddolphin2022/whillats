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

#import <Foundation/Foundation.h> // Needed for NSString, nil, NSNotificationCenter

#include "whillats_osx.h"   // Include for macOS SpeechSynthesizerProcessor
#include "whillats_synth.h" // Include the header for WhillatsSpeechSynthesizerWrapper definition
#include "whisper_helpers.h"

#include <vector>          // Needed for std::vector used in callbacks
#include <memory>          // Needed for std::unique_ptr
#include <iostream>        // Needed for std::cout if used (e.g., in completion callback)

// Define a single context struct for both callbacks
struct CallbackContext {
    WhillatsSetAudioCallback* audioCallbackPtr; // Pointer to the C++ audio callback object
    std::function<void()> completionCallbackFunc; // C++ completion callback function
};

// Static C bridge function for AudioCallback
static void AudioCallbackBridge(bool success, const uint16_t* buffer, size_t size, void* user_data) {
    CallbackContext* context = static_cast<CallbackContext*>(user_data);
    if (context && context->audioCallbackPtr) {
        std::vector<uint16_t> bufferVec;
        if (success && buffer && size > 0) {
            bufferVec.assign(buffer, buffer + size);
        }
        context->audioCallbackPtr->OnBufferComplete(success, bufferVec);
    }
}

// Static C bridge function for CompletionCallback
static void CompletionCallbackBridge(void* user_data) {
    CallbackContext* context = static_cast<CallbackContext*>(user_data);
    if (context) {
        if (context->completionCallbackFunc) {
            context->completionCallbackFunc();
        }
    }
}

// Define Impl: remove thread-related fields
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

void WhillatsSpeechSynthesizerWrapper::initialize(WhillatsSetAudioCallback* audioCallback,
                                                 CompletionCallback completionCallback) {
    // Clean up any prior processor
    if (impl->processor) stop();
    // Store callback pointer
    impl->audioCallbackPtr = audioCallback;
    // Create C++ completion callback wrapper
    std::function<void()> cppCompletionCallback = [this, completionCallback]() {
        if (completionCallback) completionCallback();
    };
    // Create and populate the shared context
    CallbackContext* context = new CallbackContext();
    context->audioCallbackPtr = impl->audioCallbackPtr;
    context->completionCallbackFunc = cppCompletionCallback;
    // Initialize the OS X speech processor (spawns its own thread/runloop)
    impl->processor = [[WhillatsSpeechSynthesizerProcessor alloc]
                       initWithAudioCallback:AudioCallbackBridge
                                 userData:context
                       completionCallback:CompletionCallbackBridge];
    if (!impl->processor) {
        LOG_E("[Whillats]: Failed to create SpeechSynthesizerProcessor");
        delete context;
    }
#if TARGET_OS_IOS
    [impl->processor enableSpeakerphone];
#endif
}

void WhillatsSpeechSynthesizerWrapper::synthesize(const std::string& text, const std::string& language) {
    // Directly pass through to the OS X processor, which runs on its own NSThread
    NSString* nsText = [NSString stringWithUTF8String:text.c_str()];
    NSString* nsLanguage = [NSString stringWithUTF8String:language.c_str()];
    if (!nsText || !nsLanguage) return;
    _lastText = text;
    _lastLanguage = language;
    // Trigger processor
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