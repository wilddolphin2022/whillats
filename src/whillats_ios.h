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

#ifndef WHILLATS_IOS_H_
#define WHILLATS_IOS_H_

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (*AudioCallback)(bool success, const uint16_t *buffer, size_t size, void *user_data);
typedef void (*CompletionCallback)(void* user_data);

@interface WhillatsSpeechSynthesizerProcessor : NSObject <AVSpeechSynthesizerDelegate>

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData
                    completionCallback:(CompletionCallback)completionCallback;
- (void)synthesizeText:(NSString *)text language:(NSString *)language;
- (void)stop;

// Speakerphone control
- (BOOL)enableSpeakerphone;
- (BOOL)disableSpeakerphone;

@property (nonatomic, readonly) void *userData; // Add getter for userData

@end

NS_ASSUME_NONNULL_END

#endif // WHILLATS_IOS_H_ 