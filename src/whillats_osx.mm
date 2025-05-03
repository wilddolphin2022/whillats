/*
 *  (c) 2025, wilddolphin2022 
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#include "whillats.h"
#import "whillats_osx.h"
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>

@interface WhillatsSpeechSynthesizerProcessor () <AVSpeechSynthesizerDelegate>
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, assign) AudioCallback audioCallback;
@property (nonatomic, assign) CompletionCallback completionCallback;
@property (nonatomic, assign) BOOL shouldStop;
@property (nonatomic, assign) BOOL isSynthesizing;
@property (nonatomic, strong) NSMutableArray<NSData *> *audioBuffers;
@end

@implementation WhillatsSpeechSynthesizerProcessor

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData
                    completionCallback:(CompletionCallback)completionCallback {
    self = [super init];
    if (self) {
        _audioCallback = audioCallback;
        _userData = userData;
        _completionCallback = completionCallback;
        _shouldStop = NO;
        _isSynthesizing = NO;
        _audioBuffers = [NSMutableArray array];
        [self setupSynthesizer];
        [self logMacOSVersion];
    }
    return self;
}

- (void)logMacOSVersion {
    NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
    NSLog(@"macOS Version: %ld.%ld.%ld", (long)version.majorVersion, (long)version.minorVersion, (long)version.patchVersion);
}

- (void)setupSynthesizer {
    _synthesizer = [[AVSpeechSynthesizer alloc] init];
    _synthesizer.delegate = self;
}

- (void)synthesizeText:(NSString *)text language:(NSString *)language {
    if (_isSynthesizing || _shouldStop) {
        NSLog(@"Synthesis already in progress or stopped");
        return;
    }
    
    NSLog(@"Starting synthesizeText with text: %@, language: %@", text, language);
    _isSynthesizing = YES;
    [_audioBuffers removeAllObjects];
    
    @autoreleasepool {
        NSLog(@"Configuring utterance");
        AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
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
        AVSpeechSynthesisVoice *voice = [AVSpeechSynthesisVoice voiceWithLanguage:localeCode];
        if (!voice) {
            NSLog(@"Voice not found for locale: %@, falling back to en-US", localeCode);
            localeCode = @"en-US";
            voice = [AVSpeechSynthesisVoice voiceWithLanguage:localeCode];
        }
        utterance.voice = voice;
        utterance.rate = 0.5;
        utterance.volume = 1.0;
        
        NSLog(@"Utterance configured: text=%@, voice=%@, rate=%f", utterance.speechString, utterance.voice, utterance.rate);
        
        NSArray<AVSpeechSynthesisVoice *> *voices = [AVSpeechSynthesisVoice speechVoices];
        NSLog(@"Available voices: %@", voices);
        
        NSLog(@"Starting synthesis with writeUtterance");
        [self.synthesizer writeUtterance:utterance toBufferCallback:^(AVAudioBuffer *buffer) {
            NSLog(@"Buffer callback triggered");
            if (self.shouldStop) {
                NSLog(@"Buffer callback skipped due to stop");
                return;
            }
            
            // Cast to PCM and check length
            AVAudioPCMBuffer *pcmBuffer = (AVAudioPCMBuffer *)buffer;
            if (!pcmBuffer || pcmBuffer.frameLength == 0) {
                NSLog(@"Received empty or invalid buffer: %@", pcmBuffer);
                return;
            }
            // 1) Upsample to 48k float32 with proper EndOfStream signaling
            AVAudioFormat *float48Format = [[AVAudioFormat alloc]
                initWithCommonFormat:AVAudioPCMFormatFloat32
                          sampleRate:48000
                            channels:pcmBuffer.format.channelCount
                         interleaved:NO];
            AVAudioConverter *upsampler = [[AVAudioConverter alloc]
                initFromFormat:pcmBuffer.format toFormat:float48Format];
            AVAudioPCMBuffer *buffer48 = [[AVAudioPCMBuffer alloc]
                initWithPCMFormat:float48Format
                       frameCapacity:(uint32_t)(pcmBuffer.frameLength * float48Format.sampleRate / pcmBuffer.format.sampleRate)];
            buffer48.frameLength = buffer48.frameCapacity;
            NSError *upErr = nil;
            __block BOOL provided = NO;
            AVAudioConverterInputBlock inBlock = ^AVAudioBuffer *(AVAudioPacketCount inPackets, AVAudioConverterInputStatus *outStatus) {
                if (!provided) {
                    provided = YES;
                    *outStatus = AVAudioConverterInputStatus_HaveData;
                    return pcmBuffer;
                } else {
                    *outStatus = AVAudioConverterInputStatus_EndOfStream;
                    return nil;
                }
            };
            [upsampler convertToBuffer:buffer48 error:&upErr withInputFromBlock:inBlock];
            if (upErr) {
                NSLog(@"Upsampling error: %@", upErr);
            } else {
                NSLog(@"Raw 48k buffer: %u samples at %.0f Hz", buffer48.frameLength, buffer48.format.sampleRate);
                // Apply manual gain (reduce volume further)
                // float gain = 0.1f;
                // float *rawSamples = buffer48.floatChannelData[0];
                // for (uint32_t i = 0; i < buffer48.frameLength; i++) {
                //     rawSamples[i] *= gain;
                // }
                // pcmBuffer = buffer48;
            }
            // Now resample pcmBuffer (float32@48k) down to 16k + int16 conversion
            uint32_t sampleCount = pcmBuffer.frameLength;
            float *audioData = pcmBuffer.floatChannelData[0];
            double sourceSampleRate = pcmBuffer.format.sampleRate;
            NSLog(@"Using buffer for resampling: %u samples @ %.0f Hz", sampleCount, sourceSampleRate);

            if (sampleCount >= 5) {
                NSLog(@"First 5 samples post-gain: %f %f %f %f %f", audioData[0], audioData[1], audioData[2], audioData[3], audioData[4]);
            }

            uint32_t targetSampleRate = 16000;
            uint32_t targetSamples = (uint32_t)((double)sampleCount * targetSampleRate / sourceSampleRate);
            int16_t *resampledData = (int16_t *)malloc(targetSamples * sizeof(int16_t));
            for (uint32_t i = 0; i < targetSamples; i++) {
                double srcIndex = (double)i * sourceSampleRate / targetSampleRate;
                uint32_t index0 = (uint32_t)srcIndex;
                uint32_t index1 = MIN(index0 + 1, sampleCount - 1);
                float frac = srcIndex - index0;
                float sample0 = audioData[index0];
                float sample1 = audioData[index1];
                float interpolated = sample0 + frac * (sample1 - sample0);
                // Final int16 conversion
                resampledData[i] = (int16_t)(interpolated * 32767.0f);
            }
            
            uint32_t samplesPerFrame = 160;
            NSMutableArray *uint16Buffers = [NSMutableArray array];
            for (uint32_t i = 0; i < targetSamples; i += samplesPerFrame) {
                uint32_t samplesToCopy = MIN(samplesPerFrame, targetSamples - i);
                // Allocate signed PCM buffer
                int16_t *int16Buffer = (int16_t *)malloc(samplesToCopy * sizeof(int16_t));
                for (uint32_t j = 0; j < samplesToCopy; j++) {
                    // Use signed int16 samples directly without offset
                    int16Buffer[j] = resampledData[i + j];
                }
                [uint16Buffers addObject:[NSData dataWithBytesNoCopy:int16Buffer length:samplesToCopy * sizeof(int16_t) freeWhenDone:YES]];
            }
            free(resampledData);
            
            for (NSData *bufferData in uint16Buffers) {
                if (self.audioCallback) {
                    const int16_t *bufferPtr = (const int16_t *)bufferData.bytes;
                    size_t bufferLength = bufferData.length / sizeof(int16_t);
                    NSLog(@"Delivering buffer with %lu samples, first sample: %d", bufferLength, bufferPtr[0]);
                    // Pass signed PCM bits through unsigned pointer
                    self.audioCallback(true, (const uint16_t *)bufferPtr, bufferLength, self.userData);
                }
            }
        }];
        
        NSLog(@"Entering run loop");
        NSTimeInterval timeout = 10.0;
        NSDate *startTime = [NSDate date];
        CFRunLoopRef runLoop = CFRunLoopGetMain();
        while (_isSynthesizing && !_shouldStop) {
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, true);
            if ([[NSDate date] timeIntervalSince1970] - startTime.timeIntervalSince1970 > timeout) {
                NSLog(@"Synthesis timeout, forcing completion");
                [self.synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
                _isSynthesizing = NO;
                if (_completionCallback) {
                    NSLog(@"Calling completionCallback due to timeout");
                    _completionCallback(_userData);
                }
                break;
            }
        }
        
        NSLog(@"Synthesis complete, exiting run loop");
    }
}

- (void)stop {
    if (!_isSynthesizing) {
        NSLog(@"No synthesis to stop");
        return;
    }
    
    _shouldStop = YES;
    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
            NSLog(@"Stopping synthesis");
            [self.synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
            self->_isSynthesizing = NO;
            self->_shouldStop = NO;
            
            // Signal completion
            if (self->_completionCallback) {
                NSLog(@"Calling completionCallback");
                self->_completionCallback(self->_userData);
            }
        }
    });
}

#pragma mark - AVSpeechSynthesizerDelegate

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
         didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    if (!_isSynthesizing || _shouldStop) {
        NSLog(@"Delegate: Synthesis finished but ignored (stopped or not synthesizing)");
        return;
    }
    
    NSLog(@"Delegate: Synthesis finished");
    _isSynthesizing = NO;
    
    if (_completionCallback) {
        NSLog(@"Calling completionCallback");
        _completionCallback(_userData);
    }
}

@end