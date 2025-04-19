#include "whillats.h"
#include "whillats_synth.h"
#import "whillats_ios.h"

struct WhillatsSpeechSynthesizerWrapper::Impl {
    WhillatsSpeechSynthesizerProcessor* processor;
    std::unique_ptr<WhillatsSetAudioCallback> audioCallback;
    
    Impl(std::unique_ptr<WhillatsSetAudioCallback> cb) : processor(nil), 
        audioCallback(std::move(cb)) {}
};

WhillatsSpeechSynthesizerWrapper::WhillatsSpeechSynthesizerWrapper() : 
    impl(std::make_unique<Impl>(nullptr)) {}

WhillatsSpeechSynthesizerWrapper::~WhillatsSpeechSynthesizerWrapper() {
    stop();
}

void WhillatsSpeechSynthesizerWrapper::initialize(WhillatsSetAudioCallback audioCallback, 
    CompletionCallback completionCallback) {
        impl->audioCallback = audioCallback;
    
    AudioCallback objcAudioCallback = ^(bool success, const uint16_t* buffer, size_t size, void* userData) {
        if (impl->audioCallback) {
            std::vector<uint16_t> bufferVec(buffer, buffer + size);
            impl->audioCallback->OnBufferComplete(success, bufferVec);
        }
    };
    
    CompletionCallback objcCompletionCallback = completionCallback ? completionCallback : [](){};
    
    impl->processor = [[SpeechSynthesizerProcessor alloc] initWithAudioCallback:objcAudioCallback
                                                                      userData:nullptr // userData handled by WhillatsSetAudioCallback
                                                             completionCallback:^{
                                                                 objcCompletionCallback();
                                                             }];
}

void WhillatsSpeechSynthesizerWrapper::synthesize(const std::string& text, const std::string& language) {
    if (impl->processor) {
        _lastText = text;
        _lastLanguage = language;
        NSString* nsText = [NSString stringWithUTF8String:_lastText.c_str()];
        NSString* nsLanguage = [NSString stringWithUTF8String:_lastLanguage.c_str()];
        [impl->processor synthesizeText:nsText language:nsLanguage];
    }
}

void WhillatsSpeechSynthesizerWrapper::stop() {
    if (impl->processor) {
        [impl->processor stop];
        impl->processor = nil;
    }
}