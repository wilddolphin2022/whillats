#include "whillats_synth.h" // Include the header for WhillatsSpeechSynthesizerWrapper definition
#include "whillats_ios.h"   // Include for SpeechSynthesizerProcessor, AudioCallback
#include "whillats_export.h"   // Include for SpeechSynthesizerProcessor, AudioCallback

#import <Foundation/Foundation.h> // Needed for NSString, nil, etc.
#include <vector>          // Needed for std::vector used in callbacks
#include <memory>          // Needed for std::unique_ptr
#include <iostream>        // Needed for std::cout if used (e.g., in completion callback)

// Ensure the guard matches the TTS_PLATFORMS definition from whillats_export.h
#if !TTS_PLATFORMS

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
    if (context && context->completionCallbackFunc) {
        context->completionCallbackFunc();
    }
}

struct WhillatsSpeechSynthesizerWrapper::Impl {
    WhillatsSpeechSynthesizerProcessor* processor; // Use correct ObjC type from whillats_ios.h
    // Store a raw pointer to the callback object managed elsewhere (e.g., in WhillatsTTS)
    WhillatsSetAudioCallback* audioCallbackPtr;

    // Constructor initializes the pointer to nullptr
    Impl() : processor(nil), audioCallbackPtr(nullptr) {}
};

WhillatsSpeechSynthesizerWrapper::WhillatsSpeechSynthesizerWrapper() :
    // Use the default Impl constructor
    impl(std::make_unique<Impl>()) {}

WhillatsSpeechSynthesizerWrapper::~WhillatsSpeechSynthesizerWrapper() {
    stop();
}

// Implementation now takes a pointer
void WhillatsSpeechSynthesizerWrapper::initialize(WhillatsSetAudioCallback* audioCallback,
                                                CompletionCallback completionCallback) {
    // Ensure processor is not already initialized or clean up previous one
    if (impl->processor) {
        stop(); // Clean up existing processor and context first
    }

    // Store the pointer
    impl->audioCallbackPtr = audioCallback;

    // Create the C++ completion function object
    std::function<void()> cppCompletionCallback = completionCallback ? completionCallback : [](){ 
        std::cout << "Default completion called." << std::endl; 
    };

    // Create and populate the shared context
    CallbackContext* context = new CallbackContext();
    context->audioCallbackPtr = impl->audioCallbackPtr; // Store pointer to C++ audio callback obj
    context->completionCallbackFunc = cppCompletionCallback; // Store C++ completion function

    // Create the Objective-C processor using the C bridge functions and the shared context
    impl->processor = [[WhillatsSpeechSynthesizerProcessor alloc] initWithAudioCallback:AudioCallbackBridge // Pass C function pointer
                                                                             userData:context // Pass shared context pointer
                                                                   completionCallback:CompletionCallbackBridge]; // Pass C function pointer

    if (!impl->processor) {
        std::cerr << "WhillatsSpeechSynthesizerWrapper: Failed to create SpeechSynthesizerProcessor" << std::endl;
        delete context; // Clean up context if processor creation failed
    }
}

void WhillatsSpeechSynthesizerWrapper::synthesize(const std::string& text) {
    if (impl->processor) {
        NSString* nsText = [NSString stringWithUTF8String:text.c_str()];
        if (nsText) { // Check if conversion was successful
            [impl->processor synthesizeText:nsText];
        } else {
            std::cerr << "WhillatsSpeechSynthesizerWrapper: Failed to convert text to NSString" << std::endl;
            // Optionally trigger an error state or callback
        }
    }
}

void WhillatsSpeechSynthesizerWrapper::stop() {
    if (impl->processor) {
        [impl->processor stop];
        // Clean up context
        CallbackContext* context = static_cast<CallbackContext*>([impl->processor userData]);
        if (context) {
            delete context;
        }
        impl->processor = nil;
    }
    // We don't own audioCallbackPtr, so don't reset it here unless the design changes
    // impl->audioCallbackPtr = nullptr;
}

#endif // !TTS_PLATFORMS