#import "whillats_ios.h"

@interface WhillatsSpeechSynthesizerProcessor ()
@property (nonatomic, strong) AVSpeechSynthesizer *synthesizer;
@property (nonatomic, strong) AVAudioEngine *audioEngine;
@property (nonatomic, strong) AVAudioConverter *converter;
@property (nonatomic, strong) AVAudioFormat *outputFormat;
@property (nonatomic, strong) AVAudioFormat *engineFormat; 
@property (nonatomic, assign) AudioCallback audioCallback;
@property (nonatomic, assign) void *userData;
@property (nonatomic, assign) CompletionCallback completionCallback;
@property (nonatomic, strong) AVAudioPlayerNode *playerNode;
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

- (void)setupSynthesizer {
    _synthesizer = [[AVSpeechSynthesizer alloc] init];
    _synthesizer.delegate = self;
}

- (void)setupAudioEngine {
    // Configure AVAudioSession
    AVAudioSession *session = [AVAudioSession sharedInstance];
    NSError *error;
    [session setCategory:AVAudioSessionCategoryPlayback
                    mode:AVAudioSessionModeDefault
                 options:0
                   error:&error];
    if (error) {
        NSLog(@"Failed to set AVAudioSession category: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    // Set preferred sample rate to 48 kHz
    [session setPreferredSampleRate:48000 error:&error];
    if (error) {
        NSLog(@"Failed to set preferred sample rate: %@", error);
    }
    
    [session setActive:YES error:&error];
    if (error) {
        NSLog(@"Failed to activate AVAudioSession: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    NSLog(@"Actual sample rate: %f", [session sampleRate]);
    
    _audioEngine = [[AVAudioEngine alloc] init];
    _outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16
                                                     sampleRate:16000
                                                       channels:1
                                                    interleaved:YES];
    
    _playerNode = [[AVAudioPlayerNode alloc] init];
    [_audioEngine attachNode:_playerNode];
    
    _engineFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                    sampleRate:48000
                                                      channels:1
                                                   interleaved:YES];
    [_audioEngine connect:_playerNode to:_audioEngine.mainMixerNode format:_engineFormat];
    
    __weak WhillatsSpeechSynthesizerProcessor *weakSelf = self;
    [_playerNode installTapOnBus:0
                      bufferSize:1024
                          format:_engineFormat
                           block:^(AVAudioPCMBuffer * _Nonnull buffer, AVAudioTime * _Nonnull when) {
        [weakSelf processBuffer:buffer];
    }];
    
    [_audioEngine prepare];
    
    [_audioEngine startAndReturnError:&error];
    if (error) {
        NSLog(@"Audio engine start failed: %@", error);
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
    }
}

- (void)synthesizeText:(NSString *)text {
    dispatch_queue_t synthesisQueue = dispatch_queue_create("com.speech.synthesis", DISPATCH_QUEUE_SERIAL);
    dispatch_async(synthesisQueue, ^{
        AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:text];
        utterance.voice = [AVSpeechSynthesisVoice voiceWithLanguage:@"en-US"];
        utterance.rate = 0.5;
        
        AVAudioPCMBuffer *silenceBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:_engineFormat
                                                                       frameCapacity:48000];
        silenceBuffer.frameLength = 48000;
        [_playerNode scheduleBuffer:silenceBuffer completionHandler:nil];
        [_playerNode play];
        
        [self.synthesizer speakUtterance:utterance];
    });
}

- (void)stop {
    [_synthesizer stopSpeakingAtBoundary:AVSpeechBoundaryImmediate];
    [_playerNode stop];
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
    [_playerNode stop];
    if (_completionCallback) {
        _completionCallback(_userData);
    }
}

- (void)processBuffer:(AVAudioPCMBuffer *)buffer {
    if (!buffer) {
        if (_audioCallback) {
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    AVAudioPCMBuffer *convertedBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:_outputFormat
                                                                     frameCapacity:(uint32_t)(buffer.frameLength * 16000.0 / 48000.0)];
    
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
            _audioCallback(false, NULL, 0, _userData);
        }
        return;
    }
    
    uint32_t samplesPerFrame = 160;
    int16_t *pcmData = convertedBuffer.int16ChannelData[0];
    uint32_t totalSamples = convertedBuffer.frameLength;
    
    NSMutableArray *uint16Buffers = [NSMutableArray array];
    for (uint32_t i = 0; i < totalSamples; i += samplesPerFrame) {
        uint32_t samplesToCopy = MIN(samplesPerFrame, totalSamples - i);
        uint16_t *uint16Buffer = (uint16_t *)malloc(samplesToCopy * sizeof(uint16_t));
        for (uint32_t j = 0; j < samplesToCopy; j++) {
            uint16Buffer[j] = (uint16_t)(pcmData[i + j] + 32768);
        }
        [uint16Buffers addObject:[NSData dataWithBytesNoCopy:uint16Buffer length:samplesToCopy * sizeof(uint16_t) freeWhenDone:YES]];
    }
    
    for (NSData *bufferData in uint16Buffers) {
        if (_audioCallback) {
            _audioCallback(true, (const uint16_t *)bufferData.bytes, bufferData.length / sizeof(uint16_t), _userData);
        }
    }
}

@synthesize userData = _userData;

@end