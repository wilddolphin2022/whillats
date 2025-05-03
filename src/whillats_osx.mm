/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#import <Foundation/Foundation.h>
#import "whillats.h"
#import "whillats_osx.h"
#import <AVFoundation/AVFoundation.h>

@interface WhillatsSpeechSynthesizerProcessor () <AVSpeechSynthesizerDelegate>
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) AVAudioConverter *converter;
@property (nonatomic, strong) AVAudioFormat *outputFormat;    // 16kHz Int16 PCM
@property (nonatomic, assign) AudioCallback audioCallback;
@property (nonatomic, assign) CompletionCallback completionCallback;
@property (nonatomic, strong) NSThread *synthThread;
@end

@implementation WhillatsSpeechSynthesizerProcessor

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                              userData:(void *)userData
                     completionCallback:(CompletionCallback)completionCallback {
    self = [super init];
    if (self) {
        _audioCallback = audioCallback;
        _completionCallback = completionCallback;
        self.userData = userData;
        // Initialize AVSpeechSynthesizer
        _synthesizer = [[AVSpeechSynthesizer alloc] init];
        // Prepare output format: 16kHz Int16 PCM
        _outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16
                                                        sampleRate:16000
                                                          channels:1
                                                       interleaved:YES];
        _converter = nil;
        // Spawn dedicated thread with its own run loop
        _synthThread = [[NSThread alloc] initWithTarget:self selector:@selector(threadEntryPoint:) object:nil];
        [_synthThread start];
    }
    return self;
}

- (void)synthesizeText:(NSString *)text language:(NSString *)language {
    if (!self.synthThread) return;
    NSArray *args = @[text, language];
    [self performSelector:@selector(doSynthesizeText:) onThread:self.synthThread withObject:args waitUntilDone:NO];
}

- (void)stop {
    if (!self.synthThread) return;
    // Stop speaking on the synth thread
    [self performSelector:@selector(stopSpeakingInternal) onThread:self.synthThread withObject:nil waitUntilDone:YES];
    // Cancel and wake up the run loop
    [self.synthThread cancel];
    [self.synthThread performSelector:@selector(wakeUpRunLoop:) onThread:self.synthThread withObject:nil waitUntilDone:NO];
    self.synthThread = nil;
}

- (void)processBuffer:(AVAudioPCMBuffer *)buffer {
    if (!buffer) {
        if (self.audioCallback) self.audioCallback(false, NULL, 0, self.userData);
        return;
    }
    if (!_converter) {
        _converter = [[AVAudioConverter alloc] initFromFormat:buffer.format toFormat:_outputFormat];
    }
    AVAudioPCMBuffer *converted = [[AVAudioPCMBuffer alloc] initWithPCMFormat:_outputFormat
                                                               frameCapacity:(uint32_t)(buffer.frameLength * _outputFormat.sampleRate / buffer.format.sampleRate)];
    converted.frameLength = converted.frameCapacity;
    NSError *error = nil;
    [_converter convertToBuffer:converted error:&error withInputFromBlock:^AVAudioBuffer *(AVAudioPacketCount inPackets, AVAudioConverterInputStatus *outStatus) {
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return buffer;
    }];
    if (error) {
        if (self.audioCallback) self.audioCallback(false, NULL, 0, self.userData);
        return;
    }
    uint32_t totalSamples = converted.frameLength;
    uint32_t samplesPerChunk = _outputFormat.sampleRate * 10 / 1000;
    int16_t *pcmData = converted.int16ChannelData[0];
    for (uint32_t i = 0; i < totalSamples; i += samplesPerChunk) {
        uint32_t count = MIN(samplesPerChunk, totalSamples - i);
        if (self.audioCallback) self.audioCallback(true, (const uint16_t *)(pcmData + i), count, self.userData);
    }
}

// MARK: - Thread and synthesis helpers
- (void)threadEntryPoint:(id)unused {
    @autoreleasepool {
        NSThread *t = [NSThread currentThread];
        NSPort *port = [NSMachPort port];
        [[NSRunLoop currentRunLoop] addPort:port forMode:NSDefaultRunLoopMode];
        while (!t.isCancelled) {
            @autoreleasepool {
                [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate distantFuture]];
            }
        }
    }
}

- (void)doSynthesizeText:(NSArray *)args {
    NSString *text = args[0];
    NSString *language = args[1];
    AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
    // Map codes to BCP-47
    NSString *localeCode;
    if ([language isEqualToString:@"auto"]) {
        localeCode = @"en-US";
    } else if ([language isEqualToString:@"en"]) {
        localeCode = @"en-US";
    } else if ([language isEqualToString:@"zh"]) {
        localeCode = @"zh-CN";
    } else if ([language isEqualToString:@"ja"]) {
        localeCode = @"ja-JP";
    } else {
        localeCode = [NSString stringWithFormat:@"%@-%@", language, [language uppercaseString]];
    }
    utterance.voice = [AVSpeechSynthesisVoice voiceWithLanguage:localeCode];
    utterance.rate = 0.5;
    if ([self.synthesizer respondsToSelector:@selector(writeUtterance:toBufferCallback:)]) {
        [self.synthesizer writeUtterance:utterance toBufferCallback:^(AVAudioBuffer *buffer) {
            if (![buffer isKindOfClass:[AVAudioPCMBuffer class]]) return;
            AVAudioPCMBuffer *pcmBuffer = (AVAudioPCMBuffer *)buffer;
            if (pcmBuffer.frameLength == 0) {
                if (self.audioCallback) self.audioCallback(false, NULL, 0, self.userData);
                if (self.completionCallback) self.completionCallback(self.userData);
            } else {
                [self processBuffer:pcmBuffer];
            }
        }];
    } else {
        [self.synthesizer speakUtterance:utterance];
    }
}

- (void)stopSpeakingInternal {
    [self.synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
}

- (void)wakeUpRunLoop:(id)unused {
    // no-op: wakes up the run loop after cancellation
}

@end