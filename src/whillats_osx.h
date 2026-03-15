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

// whillats_osx.h

#ifndef WHILLATS_OSX_H
#define WHILLATS_OSX_H

#include "whillats.h"

#if TARGET_OS_IOS
    #import <Foundation/Foundation.h>
    #import <AVFoundation/AVFoundation.h>
#else
    #import <AppKit/AppKit.h>
    #import <AudioToolbox/AudioToolbox.h>
#endif

#import <AudioToolbox/AudioToolbox.h>

// C callback signature available in all contexts
typedef void (*AudioCallback)(bool success, const uint16_t *audioData, size_t length, void *userData);

#ifdef __OBJC__
#import <AppKit/AppKit.h>
#import <AudioToolbox/AudioToolbox.h>
#import <AVFoundation/AVFoundation.h>

@interface WhillatsSpeechSynthesizerProcessor : NSObject <AVSpeechSynthesizerDelegate>

- (instancetype)initWithAudioCallback:(AudioCallback)audioCallback
                             userData:(void *)userData;

- (void)synthesizeText:(NSString *)text language:(NSString *)language;
- (void)stop;

#if TARGET_OS_IOS
// Speakerphone control
- (BOOL)enableSpeakerphone;
- (BOOL)disableSpeakerphone;
#endif

@property (nonatomic, assign) void *userData;

@end
#endif // __OBJC__

#endif // WHILLATS_OSX_H