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

#include "whillats.h"
#import "whillats_osx.h"
#import <Foundation/Foundation.h>
#import <AVFoundation/AVAudioConverter.h>
#import <CoreFoundation/CFRunLoop.h>  // For CFRunLoopStop
#import <AVFoundation/AVFoundation.h>
#import <dispatch/dispatch.h>

@interface WhillatsSpeechSynthesizerProcessor () <AVSpeechSynthesizerDelegate>
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) AVAudioEngine *audioEngine;
@property (nonatomic, strong) AVAudioConverter *converter;
@property (nonatomic, strong) AVAudioFormat *outputFormat;
@property (nonatomic, strong) AVAudioFormat *engineFormat;
@property (nonatomic, assign) AudioCallback audioCallback;
@end

@implementation WhillatsSpeechSynthesizerProcessor

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData {
    self = [super init];
    if (self) {
        NSLog(@"[Whillats] initWithAudioCallback: userData=%p", userData);
        _audioCallback = audioCallback;
        self.userData = userData;
        [self setupSynthesizer];
    }
    return self;
}

- (void)setupSynthesizer {
    _synthesizer = [[AVSpeechSynthesizer alloc] init];
    _synthesizer.delegate = self;
}

- (void)synthesizeText:(NSString *)text language:(NSString *)language {
    NSLog(@"[Whillats] synthesizeText called with text=%@, language=%@", text, language);
    AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
    utterance.voice = [AVSpeechSynthesisVoice voiceWithLanguage:language];
    utterance.rate = 0.5;
    if (@available(macOS 10.15, *)) {
        // Schedule synthesis on main queue so callbacks fire via the main runloop
        dispatch_async(dispatch_get_main_queue(), ^{
            NSLog(@"[Whillats] writeUtterance on main thread: %d", [NSThread isMainThread]);
            [self.synthesizer writeUtterance:utterance toBufferCallback:^(AVAudioBuffer *buffer) {
                AVAudioPCMBuffer *inputPCM = (AVAudioPCMBuffer *)buffer;
                NSLog(@"[Whillats] raw buffer frameLength=%u, sampleRate=%.0f", (unsigned int)inputPCM.frameLength,
                      inputPCM.format.sampleRate);
                if (inputPCM.frameLength > 0) {
                    // Downsample from native sample rate to 16kHz
                    AVAudioFormat *fromFormat = inputPCM.format;
                    AVAudioFormat *toFormat = [[AVAudioFormat alloc]
                        initWithCommonFormat:AVAudioPCMFormatFloat32
                                  sampleRate:16000
                                    channels:fromFormat.channelCount
                                 interleaved:NO];
                    AVAudioConverter *converter = [[AVAudioConverter alloc]
                        initFromFormat:fromFormat toFormat:toFormat];
                    AVAudioPCMBuffer *convertedPCM = [[AVAudioPCMBuffer alloc]
                        initWithPCMFormat:toFormat
                           frameCapacity:(AVAudioFrameCount)(inputPCM.frameLength * toFormat.sampleRate / fromFormat.sampleRate)];
                    convertedPCM.frameLength = convertedPCM.frameCapacity;
                    NSError *error = nil;
                    AVAudioConverterInputBlock block = ^AVAudioBuffer *(AVAudioPacketCount inPackets,
                                                                       AVAudioConverterInputStatus *outStatus) {
                        *outStatus = AVAudioConverterInputStatus_HaveData;
                        return inputPCM;
                    };
                    [converter convertToBuffer:convertedPCM error:&error withInputFromBlock:block];

                    float *floatData = convertedPCM.floatChannelData[0];
                    uint32_t count = convertedPCM.frameLength;
                    uint16_t *pcm16 = (uint16_t *)malloc(count * sizeof(uint16_t));
                    for (uint32_t i = 0; i < count; i++) {
                        int16_t sample = (int16_t)MAX(MIN(floatData[i] * 32767, 32767), -32768);
                        pcm16[i] = (uint16_t)sample;  // two's-complement bits
                    }
                    if (_audioCallback) {
                        _audioCallback(true, pcm16, count, _userData);
                    }
                    free(pcm16);
                } else {
                    if (_audioCallback) {
                        _audioCallback(false, NULL, 0, _userData);
                    }
                    // Stop run loop on utterance completion
                    CFRunLoopStop(CFRunLoopGetMain());
                }
            }];
            // Start the main runloop to process TTS callbacks until they stop it.
            CFRunLoopRun();
        }); // end dispatch_async to main queue
    } else {
        [self.synthesizer speakUtterance:utterance];
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
    }
}

- (void)stop {
    [_synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
}

#pragma mark - AVSpeechSynthesizerDelegate

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
      willSpeakRangeOfSpeechString:(NSRange)characterRange
                         utterance:(AVSpeechUtterance *)utterance {}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
         didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    NSLog(@"[Whillats] didFinishSpeechUtterance delegate");
    if (_audioCallback) {
        _audioCallback(false, NULL, 0, _userData);
    }
    // Also stop run loop if using delegate-based completion
    CFRunLoopStop(CFRunLoopGetMain());
}

@end
