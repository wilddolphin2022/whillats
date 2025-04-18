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
- (void)synthesizeText:(NSString *)text;
- (void)stop;

@property (nonatomic, readonly) void *userData; // Add getter for userData

@end

NS_ASSUME_NONNULL_END

#endif // WHILLATS_IOS_H_ 