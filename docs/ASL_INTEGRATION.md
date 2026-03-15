# ASL Sign Language Face — Integration Guide

## Overview

`SignLanguageFace` is an alternative to `TalkingFace` for the directcall WebRTC application. Instead of animating a mouth on a static robot logo, it switches between full-frame ASL (American Sign Language) sign images synchronized to speech audio.

| Feature | TalkingFace | SignLanguageFace |
|---------|-------------|------------------|
| Visual output | Mouth animation on static image | Full-frame sign image switching |
| Audio input | RMS energy → mouth openness | Playback clock advancement |
| Text input | Not needed | Required (words to sign) |
| Image source | Single base image | Multiple sign images (builtin or custom) |
| Resolution | Any (scales to 640×480) | 1024×1024 native |

## Architecture

```
directcall app
  │
  ├── SpeechAudioDeviceFactory  (WebRTC module)
  │     ├── TalkingFace*  _talkingFace        ← current
  │     └── SignLanguageFace*  _signFace       ← new (add alongside)
  │
  ├── WhisperAudioDevice  (audio processing loop)
  │     └── ttsCallback → face->feedAudio()   ← drives animation
  │
  └── TalkingFaceRenderer / SignFaceRenderer   (video frame injection)
        └── face->renderFrame() → I420Buffer → WebRTC VideoTrack
```

## Integration into directcall

### Step 1: Add SignLanguageFace to SpeechAudioDeviceFactory

In `speech_audio_device_factory.h`, add alongside the existing TalkingFace:

```cpp
#include "modules/third_party/whillats/src/sign_language_face.h"

class SpeechAudioDeviceFactory {
public:
  // Existing
  static void SetTalkingFaceImage(const std::string& path);
  static TalkingFace* talkingFace() { return _talkingFace.get(); }

  // New: ASL sign language face
  static void SetSignLanguageFace(const std::string& sign_dir);
  static void SetSignLanguageText(const std::string& text, float duration);
  static SignLanguageFace* signFace() { return _signFace.get(); }

private:
  static std::unique_ptr<TalkingFace> _talkingFace;
  static std::unique_ptr<SignLanguageFace> _signFace;    // ← add
};
```

In `speech_audio_device_factory.cc`:

```cpp
std::unique_ptr<SignLanguageFace> SpeechAudioDeviceFactory::_signFace;

void SpeechAudioDeviceFactory::SetSignLanguageFace(const std::string& sign_dir) {
  _signFace = std::make_unique<SignLanguageFace>();
  _signFace->generateBuiltinSigns(1024);
  if (!sign_dir.empty()) {
    _signFace->loadSignImages(sign_dir.c_str());
  }
  _signFace->setOutputSize(640, 480);
}

void SpeechAudioDeviceFactory::SetSignLanguageText(
    const std::string& text, float duration) {
  if (_signFace) {
    _signFace->setText(text, duration);
  }
}
```

### Step 2: Add config option to directcall

In `option.h`, add a field:

```cpp
struct Options {
  // ...existing...
  std::string talking_face{};
  std::string sign_language_dir{};  // ← add: path to ASL sign images dir
  bool        sign_language{false}; // ← add: enable ASL mode
};
```

In `option.cc`, parse it:

```cpp
// JSON config
if (config_json.isMember("sign_language") && config_json["sign_language"].isBool()) {
    opts.sign_language = config_json["sign_language"].asBool();
}
if (config_json.isMember("sign_language_dir") && config_json["sign_language_dir"].isString()) {
    opts.sign_language_dir = expandHomePath(config_json["sign_language_dir"].asString());
}

// CLI
} else if (arg.find("--sign_language") == 0) {
    opts.sign_language = true;
} else if (arg.find("--sign_language_dir=") == 0) {
    opts.sign_language_dir = expandHomePath(stripQuotes(arg.substr(19)));
}
```

### Step 3: Create SignFaceRenderer in peer.cc

Add alongside `TalkingFaceRenderer`:

```cpp
#include "modules/third_party/whillats/src/sign_language_face.h"

class SignFaceRenderer {
 public:
  SignFaceRenderer(rtc::scoped_refptr<webrtc::FakeVideoTrackSource> src,
                   SignLanguageFace* face, int fps = 15)
      : src_(src), face_(face), interval_ms_(1000 / fps), running_(false) {}

  ~SignFaceRenderer() { Stop(); }

  void Start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread([this]() {
      while (running_) {
        YUVData yuv;
        if (face_->renderFrame(yuv)) {
          auto i420 = webrtc::I420Buffer::Create(yuv.width, yuv.height);
          memcpy(i420->MutableDataY(), yuv.y.get(), yuv.y_size);
          memcpy(i420->MutableDataU(), yuv.u.get(), yuv.uv_size);
          memcpy(i420->MutableDataV(), yuv.v.get(), yuv.uv_size);
          webrtc::VideoFrame frame =
              webrtc::VideoFrame::Builder()
                  .set_video_frame_buffer(i420)
                  .set_timestamp_us(rtc::TimeMicros())
                  .build();
          src_->InjectFrame(frame);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
      }
    });
  }

  void Stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
  }

 private:
  rtc::scoped_refptr<webrtc::FakeVideoTrackSource> src_;
  SignLanguageFace* face_;
  int interval_ms_;
  std::atomic<bool> running_;
  std::thread thread_;
};

rtc::scoped_refptr<webrtc::VideoTrackSourceInterface>
CreateSignLanguageVideoSource(DirectPeer* owner) {
  auto* face = webrtc::SpeechAudioDeviceFactory::signFace();
  if (!face) return nullptr;

  face->setOutputSize(640, 480);

  auto track_source = webrtc::FakeVideoTrackSource::Create(false);
  track_source->SetState(webrtc::MediaSourceInterface::kLive);

  auto* renderer = new SignFaceRenderer(track_source, face, 15);
  renderer->Start();
  (void)renderer;  // intentional leak, lives for process lifetime

  RTC_LOG(LS_INFO) << "ASL SignLanguageFace video source created (640x480@15fps)";
  return track_source;
}
```

### Step 4: Hook into the audio pipeline

In `whisper_audio_device.cc`, the TTS callback feeds audio to the face:

```cpp
// In ttsCallback — already existing for TalkingFace:
auto* face = SpeechAudioDeviceFactory::talkingFace();
if (face) {
  face->feedAudio(reinterpret_cast<const int16_t*>(buffer), buffer_size);
}

// Add for SignLanguageFace:
auto* sign = SpeechAudioDeviceFactory::signFace();
if (sign) {
  sign->feedAudio(reinterpret_cast<const int16_t*>(buffer), buffer_size,
                  16000);  // WebRTC uses 16kHz internally
}
```

### Step 5: Feed text to SignLanguageFace

When Llama generates a response, the text goes to TTS. At the same time,
feed it to SignLanguageFace so it knows what words to sign:

In `direct.cc` or the Llama response handler:

```cpp
// Existing: speak the text
SpeechAudioDeviceFactory::SpeakText(response, language);

// New: prepare ASL signs for this text
auto* sign = SpeechAudioDeviceFactory::signFace();
if (sign) {
  // Estimate duration: ~150ms per word
  std::istringstream iss(response);
  int word_count = std::distance(std::istream_iterator<std::string>(iss),
                                  std::istream_iterator<std::string>());
  float estimated_duration = word_count * 0.15f;
  sign->setText(response, estimated_duration);
  sign->reset();
}
```

### Step 6: Initialize in DirectApplication

In `direct.cc` around where `SetTalkingFaceImage` is called:

```cpp
// Existing
if (!opts_.talking_face.empty()) {
  webrtc::SpeechAudioDeviceFactory::SetTalkingFaceImage(opts_.talking_face);
}

// New: ASL mode
if (opts_.sign_language) {
  webrtc::SpeechAudioDeviceFactory::SetSignLanguageFace(opts_.sign_language_dir);
}
```

And in `client.cc` where the video source is created:

```cpp
if (opts_.sign_language) {
  video_source_ = CreateSignLanguageVideoSource(this);
} else if (!opts_.talking_face.empty()) {
  video_source_ = CreateTalkingFaceVideoSource(this);
}
```

## JSON Configuration

```json
{
  "mode": "callee",
  "sign_language": true,
  "sign_language_dir": "/opt/directcall/asl_signs",
  "whisper": true,
  "llama": true,
  "video": true,
  "whisper_model": "/opt/models/ggml-small.bin",
  "llama_model": "/opt/models/model.gguf"
}
```

If `sign_language_dir` is empty or omitted, built-in programmatically
generated signs are used (26 letters + ~20 common words).

## Custom Sign Images

Place 1024×1024 PNG/JPEG images in the sign directory:

```
/opt/directcall/asl_signs/
├── hello.png        # whole-word sign
├── thank.png
├── you.png
├── how.png
├── ...
├── a.png            # fingerspelling letters
├── b.png
├── ...
├── z.png
└── rest.png         # idle/neutral pose
```

Filename (without extension, lowercased) becomes the lookup key.
Words not found in the directory fall back to built-in generated signs.
Words not in any sign vocabulary are fingerspelled letter-by-letter.

## Standalone Testing

Build and run without directcall:

```bash
cd whillats/build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_asl_video -j8

./bin/Release/test_asl_video \
  --wav /path/to/audio.wav \
  --text "Hello, this is a test of text to speech synthesis." \
  --output asl_output.y4m \
  --size 1024 --fps 15

# Mux with audio
ffmpeg -i asl_output.y4m -i /path/to/audio.wav \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  asl_output.mp4
```

### test_asl_video options

| Flag | Default | Description |
|------|---------|-------------|
| `--wav` | `synthesized_audio.wav` | Input WAV file (PCM 16-bit mono) |
| `--text` | test sentence | Text to render as ASL signs |
| `--output` | `asl_output.y4m` | Output YUV4MPEG2 file |
| `--size` | 1024 | Video width and height |
| `--fps` | 15 | Frame rate |
| `--signs` | (none) | Directory with custom sign images |

## Key Differences from TalkingFace

1. **TalkingFace** only needs audio — it derives mouth shape from RMS energy.
   **SignLanguageFace** needs both audio (for timing) and text (for sign selection).

2. **TalkingFace** modifies pixels in-place on a base image.
   **SignLanguageFace** swaps the entire frame to a different pre-rendered sign.

3. **TalkingFace** runs at 24-30fps for smooth mouth movement.
   **SignLanguageFace** runs at 15fps since sign transitions are discrete, not continuous.

4. **TalkingFace** has no concept of words.
   **SignLanguageFace** tokenizes text, maps words to signs, and falls back to
   fingerspelling for unknown vocabulary.

## File Inventory (asl branch)

```
whillats/
├── src/
│   ├── sign_language_face.h     # SignLanguageFace class declaration
│   ├── sign_language_face.cc    # Implementation: rendering, sign generation, timing
│   ├── talking_face.h           # Original TalkingFace (unchanged)
│   └── talking_face.cc          # Original TalkingFace (unchanged)
├── test/
│   └── test_asl_video.cc        # Standalone test: WAV → ASL video
├── docs/
│   └── ASL_INTEGRATION.md       # This file
└── CMakeLists.txt               # Updated: builds sign_language_face + test_asl_video
```
