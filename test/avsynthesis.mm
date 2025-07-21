// synthesis.mm
// Compile with: g++ -std=c++11 -x objective-c++ avsynthesis.mm -o synthesis -framework Foundation -framework AVFoundation
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <Foundation/Foundation.h>
#include <AVFoundation/AVFoundation.h>

// Objective-C++ class to handle speech synthesis
@interface SpeechSynthesizerProcessor : NSObject <AVSpeechSynthesizerDelegate>
- (std::vector<int16_t>)synthesizeText:(const std::string&)text;
@end

@implementation SpeechSynthesizerProcessor {
    AVSpeechSynthesizer *synthesizer;
    NSMutableArray<NSData *> *audioBuffers;
    dispatch_semaphore_t synthesisSemaphore;
    BOOL synthesisCompleted;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        synthesizer = [[AVSpeechSynthesizer alloc] init];
        synthesizer.delegate = self;
        audioBuffers = [NSMutableArray array];
        synthesisSemaphore = dispatch_semaphore_create(0);
        synthesisCompleted = NO;
    }
    return self;
}

- (std::vector<int16_t>)synthesizeText:(const std::string&)text {
    std::cerr << "Processing text: " << text << "\n";
    [audioBuffers removeAllObjects];
    synthesisCompleted = NO;

    NSString *nsText = [NSString stringWithUTF8String:text.c_str()];
    AVSpeechUtterance *utterance = [[AVSpeechUtterance alloc] initWithString:nsText];
    AVSpeechSynthesisVoice *voice = [AVSpeechSynthesisVoice voiceWithLanguage:@"en-US"];
    utterance.voice = voice;
    utterance.rate = 0.5;
    utterance.volume = 1.0;

    std::cerr << "Configuring utterance: text=" << text << ", voice=en-US, rate=0.5\n";

    // Target format: 16 kHz, mono, int16 PCM
    AVAudioFormat *targetFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16
                                                                  sampleRate:16000
                                                                    channels:1
                                                                 interleaved:NO];

    [synthesizer writeUtterance:utterance toBufferCallback:^(AVAudioBuffer *buffer) {
        if (!buffer) {
            std::cerr << "Received null buffer - end of synthesis\n";
            synthesisCompleted = YES;
            dispatch_semaphore_signal(synthesisSemaphore);
            return;
        }

        AVAudioPCMBuffer *pcmBuffer = (AVAudioPCMBuffer *)buffer;
        if (pcmBuffer.frameLength == 0) {
            std::cerr << "Received empty buffer - end of synthesis\n";
            synthesisCompleted = YES;
            dispatch_semaphore_signal(synthesisSemaphore);
            return;
        }

        std::cerr << "Received buffer: " << pcmBuffer.frameLength << " samples at " << pcmBuffer.format.sampleRate << " Hz\n";

        // Convert to 16 kHz int16
        AVAudioConverter *converter = [[AVAudioConverter alloc] initFromFormat:pcmBuffer.format toFormat:targetFormat];
        AVAudioPCMBuffer *convertedBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:targetFormat
                                                                        frameCapacity:(UInt32)(pcmBuffer.frameLength * targetFormat.sampleRate / pcmBuffer.format.sampleRate)];
        convertedBuffer.frameLength = convertedBuffer.frameCapacity;

        NSError *error = nil;
        // Allow inputStatus to be mutated inside the block
        __block AVAudioConverterInputStatus inputStatus = AVAudioConverterInputStatus_HaveData;
        [converter convertToBuffer:convertedBuffer
                            error:&error
               withInputFromBlock:^AVAudioBuffer *(AVAudioPacketCount inNumberOfPackets, AVAudioConverterInputStatus *outStatus) {
                   *outStatus = inputStatus;
                   inputStatus = AVAudioConverterInputStatus_EndOfStream;
                   return pcmBuffer;
               }];

        if (error) {
            std::cerr << "Conversion error: " << [[error localizedDescription] UTF8String] << "\n";
            return;
        }

        // Store int16 samples
        int16_t *samples = convertedBuffer.int16ChannelData[0];
        NSData *bufferData = [NSData dataWithBytes:samples length:convertedBuffer.frameLength * sizeof(int16_t)];
        [audioBuffers addObject:bufferData];

        std::cerr << "Converted buffer: " << convertedBuffer.frameLength << " samples at 16000 Hz\n";
    }];

    // Wait for synthesis to complete
    dispatch_semaphore_wait(synthesisSemaphore, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
    if (!synthesisCompleted) {
        std::cerr << "Synthesis timed out\n";
        return {};
    }

    // Combine buffers
    size_t totalSamples = 0;
    for (NSData *data in audioBuffers) {
        totalSamples += data.length / sizeof(int16_t);
    }

    std::vector<int16_t> result(totalSamples);
    size_t offset = 0;
    for (NSData *data in audioBuffers) {
        size_t sampleCount = data.length / sizeof(int16_t);
        memcpy(result.data() + offset, data.bytes, data.length);
        offset += sampleCount;
    }

    // Trim initial 100 samples (6.25ms) to reduce click
    size_t trim_samples = std::min((size_t)100, totalSamples);
    if (trim_samples > 0) {
        std::vector<int16_t> trimmed(result.begin() + trim_samples, result.end());
        std::cerr << "Synthesized " << totalSamples << " samples, trimmed to " << trimmed.size() << "\n";
        return trimmed;
    }

    std::cerr << "Synthesized " << totalSamples << " samples\n";
    return result;
}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer didFinishSpeechUtterance:(AVSpeechUtterance *)utterance {
    std::cerr << "Synthesis completed\n";
    synthesisCompleted = YES;
    dispatch_semaphore_signal(synthesisSemaphore);
}

- (void)speechSynthesizer:(AVSpeechSynthesizer *)synthesizer didCancelSpeechUtterance:(AVSpeechUtterance *)utterance {
    std::cerr << "Synthesis cancelled\n";
    synthesisCompleted = YES;
    dispatch_semaphore_signal(synthesisSemaphore);
}

@end

int main() {
    std::cerr << "Synthesis process started, input_fd: " << STDIN_FILENO << ", output_fd: " << STDOUT_FILENO << "\n";
    
    @autoreleasepool {
        SpeechSynthesizerProcessor *processor = [[SpeechSynthesizerProcessor alloc] init];
        std::cerr << "TTS initialized\n";

        while (true) {
            uint32_t text_size;
            ssize_t bytes_read = read(STDIN_FILENO, &text_size, sizeof(text_size));
            if (bytes_read != sizeof(text_size)) {
                if (bytes_read == 0) {
                    std::cerr << "EOF on input pipe\n";
                    break;
                }
                std::cerr << "Failed to read text size, bytes read: " << bytes_read << ": " << strerror(errno) << "\n";
                return 1;
            }
            std::cerr << "Read text size: " << text_size << "\n";
            
            std::vector<char> text_buffer(text_size);
            bytes_read = read(STDIN_FILENO, text_buffer.data(), text_size);
            if (bytes_read != text_size) {
                std::cerr << "Failed to read text, bytes read: " << bytes_read << ": " << strerror(errno) << "\n";
                return 1;
            }
            std::string text(text_buffer.data(), text_size);
            std::cerr << "Received text: " << text << "\n";
            
            std::vector<int16_t> samples = [processor synthesizeText:text];
            if (samples.empty()) {
                std::cerr << "Synthesis failed for text: " << text << "\n";
                continue;
            }
            
            uint32_t buffer_size = samples.size() * sizeof(int16_t);
            std::cerr << "Sending " << samples.size() << " int16 samples at 16 kHz (" << buffer_size << " bytes)\n";
            
            if (write(STDOUT_FILENO, &buffer_size, sizeof(buffer_size)) != sizeof(buffer_size)) {
                std::cerr << "Failed to write buffer size: " << strerror(errno) << "\n";
                return 1;
            }
            if (write(STDOUT_FILENO, samples.data(), buffer_size) != buffer_size) {
                std::cerr << "Failed to write buffer: " << strerror(errno) << "\n";
                return 1;
            }
            std::cerr << "Buffer sent to client: " << buffer_size << " bytes\n";
        }
    }
    
    std::cerr << "Synthesis process exiting\n";
    return 0;
}