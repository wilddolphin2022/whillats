/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#import "whillats.h"
#import "whillats_osx.h"
#import <AVFoundation/AVFoundation.h>

@interface WhillatsSpeechSynthesizerProcessor () <AVSpeechSynthesizerDelegate>
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) AVAudioConverter *converter;
@property (nonatomic, strong) AVAudioFormat *outputFormat;    // 16kHz Int16 PCM
@property (nonatomic, assign) AudioCallback audioCallback;
@property (nonatomic, assign) CompletionCallback completionCallback;
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
        _synthesizer = [[AVSpeechSynthesizer alloc] init];
        // Prepare output format: 16kHz Int16 PCM
        _outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16
                                                        sampleRate:16000
                                                          channels:1
                                                       interleaved:YES];
        _converter = nil;
    }
    return self;
}

- (void)synthesizeText:(NSString *)text language:(NSString *)language {
    AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
    utterance.voice = [AVSpeechSynthesisVoice voiceWithLanguage:language];
    utterance.rate = 0.5;
    if ([self.synthesizer respondsToSelector:@selector(writeUtterance:toBufferCallback:)]) {
        __weak typeof(self) weakSelf = self;
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            [weakSelf.synthesizer writeUtterance:utterance toBufferCallback:^(AVAudioBuffer *buffer) {
                if (![buffer isKindOfClass:[AVAudioPCMBuffer class]]) return;
                AVAudioPCMBuffer *pcmBuffer = (AVAudioPCMBuffer *)buffer;
                if (pcmBuffer.frameLength == 0) {
                    // End of stream
                    if (weakSelf.audioCallback) weakSelf.audioCallback(false, NULL, 0, weakSelf.userData);
                    if (weakSelf.completionCallback) weakSelf.completionCallback(weakSelf.userData);
                } else {
                    [weakSelf processBuffer:pcmBuffer];
                }
            }];
        });
    } else {
        // Fallback: speak without buffer capture
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.synthesizer speakUtterance:utterance];
        });
    }
}

- (void)stop {
    [self.synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
}

- (void)processBuffer:(AVAudioPCMBuffer *)buffer {
    NSLog(@"[Whillats] processBuffer called, frameLength=%u", (unsigned)buffer.frameLength);
    if (!buffer) {
        if (self.audioCallback) self.audioCallback(false, NULL, 0, self.userData);
        return;
    }
    // Convert to 16kHz Int16 PCM
    if (!self.converter) {
        self.converter = [[AVAudioConverter alloc] initFromFormat:buffer.format toFormat:self.outputFormat];
    }
    AVAudioPCMBuffer *converted = [[AVAudioPCMBuffer alloc] initWithPCMFormat:self.outputFormat
                                                               frameCapacity:(uint32_t)(buffer.frameLength * self.outputFormat.sampleRate / buffer.format.sampleRate)];
    converted.frameLength = converted.frameCapacity;
    NSError *convError = nil;
    [self.converter convertToBuffer:converted error:&convError withInputFromBlock:^AVAudioBuffer * _Nullable(AVAudioPacketCount inPackets, AVAudioConverterInputStatus *outStatus) {
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return buffer;
    }];
    if (convError) {
        if (self.audioCallback) self.audioCallback(false, NULL, 0, self.userData);
        return;
    }
    uint32_t totalSamples = converted.frameLength;
    uint32_t samplesPerChunk = self.outputFormat.sampleRate * 10 / 1000; // 10ms chunks
    int16_t *pcmData = converted.int16ChannelData[0];
    for (uint32_t pos = 0; pos < totalSamples; pos += samplesPerChunk) {
        uint32_t count = MIN(samplesPerChunk, totalSamples - pos);
        if (self.audioCallback) self.audioCallback(true, (const uint16_t *)(pcmData + pos), count, self.userData);
    }
}

@end