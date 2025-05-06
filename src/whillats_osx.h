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

// whillats_osx.h

#ifndef WHILLATS_OSX_H
#define WHILLATS_OSX_H

#import <AudioToolbox/AudioToolbox.h>

typedef void (*AudioCallback)(bool success, const uint16_t *audioData, size_t length, void *userData);
typedef void (*CompletionCallback)(void *userData);

@interface WhillatsSpeechSynthesizerProcessor : NSObject

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData
                    completionCallback:(CompletionCallback)completionCallback;

- (void)synthesizeText:(NSString *)text language:(NSString *)language;
- (void)stop;

@property (nonatomic, assign) void *userData;

@end

#endif // WHILLATS_OSX_H