#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

#import "whillats_ios.h"

@interface WhillatsSpeechSynthesizerProcessor ()
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) AVAudioEngine *audioEngine;
@property (nonatomic, strong) AVAudioConverter *converter;
@property (nonatomic, strong) AVAudioFormat *outputFormat;
@property (nonatomic, assign) AudioCallback audioCallback;
@property (nonatomic, assign) void *userData;
@property (nonatomic, assign) CompletionCallback completionCallback;
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
        [self setupSynthesizer];
        [self setupAudioEngine];
    }
    return self;
}

@synthesize userData = _userData;

- (void)setupSynthesizer {
    _synthesizer = [[AVSpeechSynthesizer alloc] init];
    _synthesizer.delegate = self;
}

- (void)setupAudioEngine {
    _audioEngine = [[AVAudioEngine alloc] init];
    _outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16
                                                     sampleRate:16000
                                                       channels:1
                                                    interleaved:NO];
    
    AVAudioPlayerNode *playerNode = [[AVAudioPlayerNode alloc] init];
    [_audioEngine attachNode:playerNode];
    
    AVAudioMixerNode *mixerNode = _audioEngine.mainMixerNode;
    [_audioEngine connect:playerNode to:mixerNode format:_outputFormat];
    
    NSError *error;
    [_audioEngine startAndReturnError:&error];
    if (error) {
        NSLog(@"Audio engine start failed: %@", error);
        if (_audioCallback) {
            _audioCallback(false, nullptr, 0, _userData); // Signal failure, use nullptr instead of NULL
        }
    }
}

- (void)synthesizeText:(NSString *)text {
    dispatch_queue_t synthesisQueue = dispatch_queue_create("com.speech.synthesis", DISPATCH_QUEUE_SERIAL);
    dispatch_async(synthesisQueue, ^{
        AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
        utterance.voice = [AVSpeechSynthesisVoice voiceWithLanguage:@"en-US"];
        utterance.rate = 0.5;
        [self.synthesizer speakUtterance:utterance];
    });
}

- (void)stop {
    [_synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
    [_audioEngine stop];
}

#pragma mark - AVSpeechSynthesizerDelegate

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
      willSpeakRangeOfSpeechString:(NSRange)characterRange
                         utterance:(AVSpeechUtterance *)utterance {
    // Optional: Track progress
}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer
         didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    if (_completionCallback) {
        _completionCallback(_userData);
    }
}

- (void)processBuffer:(AVAudioPCMBuffer *)buffer {
    if (!buffer) {
        if (_audioCallback) {
            _audioCallback(false, nullptr, 0, _userData); // Signal failure, use nullptr
        }
        return;
    }
    
    // Convert buffer to 16 kHz, 16-bit, mono PCM
    AVAudioPCMBuffer *convertedBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:_outputFormat
                                                                     frameCapacity:buffer.frameLength];
    
    if (!_converter) {
        _converter = [[AVAudioConverter alloc] initFromFormat:buffer.format toFormat:_outputFormat];
    }
    
    NSError *error;
    AVAudioConverterOutputStatus outputStatus = [_converter convertToBuffer:convertedBuffer
                                                                     error:&error
                                                        withInputFromBlock:^AVAudioBuffer * _Nullable(AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus * _Nonnull outStatus) {
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return buffer;
    }];
    
    if (outputStatus != AVAudioConverterOutputStatus_HaveData || error) {
        NSLog(@"Buffer conversion failed: %@", error);
        if (_audioCallback) {
            _audioCallback(false, nullptr, 0, _userData); // Use nullptr
        }
        return;
    }
    
    // Chunk into 10ms buffers (160 samples at 16 kHz)
    uint32_t samplesPerFrame = 160; // 10ms = 0.01s * 16000 samples/s
    int16_t *pcmData = convertedBuffer.int16ChannelData[0];
    uint32_t totalSamples = convertedBuffer.frameLength;
    
    // Convert int16_t to uint16_t
    NSMutableArray *uint16Buffers = [NSMutableArray array];
    for (uint32_t i = 0; i < totalSamples; i += samplesPerFrame) {
        uint32_t samplesToCopy = MIN(samplesPerFrame, totalSamples - i);
        uint16_t *uint16Buffer = (uint16_t *)malloc(samplesToCopy * sizeof(uint16_t));
        for (uint32_t j = 0; j < samplesToCopy; j++) {
            // Convert signed int16_t to unsigned uint16_t (shift range)
            uint16Buffer[j] = (uint16_t)(pcmData[i + j] + 32768); // int16_t [-32768, 32767] -> uint16_t [0, 65535]
        }
        [uint16Buffers addObject:[NSData dataWithBytesNoCopy:uint16Buffer length:samplesToCopy * sizeof(uint16_t) freeWhenDone:YES]];
    }
    
    // Invoke callback for each buffer
    for (NSData *bufferData in uint16Buffers) {
        if (_audioCallback) {
            _audioCallback(true, (const uint16_t *)bufferData.bytes, bufferData.length / sizeof(uint16_t), _userData);
        }
    }
}

@end
