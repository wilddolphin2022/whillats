# Whillats Script Engine – Telephony Application Guide

## Overview

The Script Engine runs YAML-defined telephony applications (voicemail, IVR, call forwarding) on top of the Whillats TTS/STT/LLM stack. It provides a state machine that orchestrates **StyleTTS2** (speak), **Whisper** (listen), and **LLaMA** (ask AI) without writing C++ code.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   WebRTC Caller App                       │
│              (webrtcsays.ai / directcall)                 │
│                                                          │
│  Inbound audio ─────► WhillatsScript.feedAudio()         │
│  Outbound audio ◄──── WhillatsScript audio callback      │
│  Events ◄──────────── WhillatsScript event callback      │
│                       (forward, end, store, etc.)        │
├──────────────────────────────────────────────────────────┤
│                   WhillatsScript                          │
│                                                          │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ ScriptEngine │  │ WhillatsTTS│  │WhillatsTranscriber│  │
│  │ (YAML state  │  │ (StyleTTS2)│  │ (Whisper STT)    │  │
│  │  machine)    │  └────────────┘  └──────────────────┘  │
│  │              │  ┌────────────┐                        │
│  │              │  │WhillatsLlama│                       │
│  │              │  │ (LLaMA LLM)│                       │
│  └─────────────┘  └────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## YAML Script Format

```yaml
script:
  name: "My Application"
  language: "en"          # eSpeak phonemizer voice

steps:
  - id: step_name         # unique step identifier
    action: speak|listen|ask_ai|forward|store|end
    text: "Text to speak or AI prompt"
    prompt: "Spoken before listening"
    timeout_ms: 5000       # listen timeout in milliseconds
    store_as: variable     # save transcription/response to variable
    store_audio: true      # also save raw audio (for voicemail)
    next: next_step_id     # default next step
    target: "sip:..."     # forward destination
    on_match:              # regex branch on transcription
      - pattern: "yes|ok"
        next: confirmed
      - pattern: "no|cancel"
        next: denied
```

### Actions

| Action | What happens | Whillats component |
|--------|-------------|-------------------|
| `speak` | Synthesize `text` (with `${var}` expansion) and send as audio | StyleTTS2 TTS |
| `listen` | Optionally speak `prompt`, then transcribe caller audio until `timeout_ms` | Whisper STT |
| `ask_ai` | Send `text` as prompt to LLM, speak the response | LLaMA + TTS |
| `forward` | Emit forward event with `target` address | Event callback |
| `store` | Save `text` to variable `store_as` | Internal |
| `end` | Terminate the script | Event callback |

### Variable Expansion

Any `${variable_name}` in `text` or `prompt` is replaced with the stored value:

```yaml
  - id: echo
    action: speak
    text: "You said: ${caller_message}"
```

### Pattern Matching (Branching)

`on_match` uses regex (case-insensitive) against the transcribed text:

```yaml
  - id: confirm
    action: listen
    timeout_ms: 5000
    store_as: answer
    on_match:
      - pattern: "yes|correct|right|yeah"
        next: proceed
      - pattern: "no|wrong|nope"
        next: retry
    next: proceed   # fallback if no pattern matches
```

## C++ API

```cpp
#include "whillats.h"

// Event callback - receives script lifecycle events
void onScriptEvent(const char* event_type, const char* step_id,
                   const char* data, void* user_data) {
    // event_type: "speak", "listen", "timeout", "forward", "end", "error"
    if (strcmp(event_type, "forward") == 0) {
        // data contains the forward target
        initiateCallForward(data);
    }
    if (strcmp(event_type, "end") == 0) {
        // script finished, hang up
        disconnectCall();
    }
}

// Audio callback - receives synthesized TTS audio to send to caller
void onAudio(bool success, const uint16_t* buffer, size_t size, void* ud) {
    if (success && buffer) {
        sendAudioToWebRTC(buffer, size);
    }
}

// Setup
WhillatsSetAudioCallback audio_cb(onAudio, nullptr);
WhillatsSetResponseCallback resp_cb(nullptr, nullptr);

WhillatsScript script("scripts/voicemail.yml",
                      audio_cb, resp_cb,
                      onScriptEvent, nullptr);

// Start with Whisper model for STT (LLaMA optional)
script.start("/path/to/whisper-model.bin",
             "/path/to/llama-model.gguf",  // or nullptr
             nullptr);                      // mmproj, or nullptr

// Feed incoming caller audio (from WebRTC playout)
script.feedAudio(pcm_buffer, buffer_size);

// Query state
printf("Current step: %s\n", script.currentStep());
printf("Caller name: %s\n", script.getVariable("caller_name"));

// Cleanup
script.stop();
```

## Integration with webrtcsays.ai DirectCall

### Current Architecture (without scripts)

```
Remote caller → WebRTC → PlayThreadProcess()
    → Whisper STT → whisperResponseCallback()
    → LLaMA → llamaResponseCallback()
    → TTS → ttsAudioCallback()
    → RecThreadProcess() → WebRTC → Remote caller
```

The current `WhisperAudioDevice` in `modules/audio_device/speech/whisper_audio_device.cc` implements a hardcoded loop: transcribe → ask LLM → speak response.

### New Architecture (with scripts)

Replace the hardcoded loop with `WhillatsScript` that drives the conversation from a YAML file.

#### Step 1: Add script option to DirectCall

In `examples/direct/option.h`, add:

```cpp
struct Options {
    // ... existing fields ...
    std::string script_path;  // path to YAML script
};
```

In `examples/direct/option.cc`, parse `--script=<path>` and JSON key `"script"`.

#### Step 2: Create ScriptAudioDevice

New file: `modules/audio_device/speech/script_audio_device.cc`

```cpp
class ScriptAudioDevice : public AudioDeviceGeneric {
    std::unique_ptr<WhillatsScript> _script;
    // ...

    int32_t InitRecording() override {
        // TTS audio callback → SetRecordedBuffer → WebRTC send
        WhillatsSetAudioCallback audio_cb(ttsCallback, this);
        WhillatsSetResponseCallback resp_cb(nullptr, nullptr);
        _script = std::make_unique<WhillatsScript>(
            _script_path.c_str(), audio_cb, resp_cb,
            scriptEventCallback, this);
        _script->start(_whisper_model, _llama_model, _mmproj);
        return 0;
    }

    // Playout thread feeds caller audio to script
    void PlayThreadProcess() {
        _ptrAudioBuffer->RequestPlayoutData(samples);
        _ptrAudioBuffer->GetPlayoutData(playoutBuffer);
        _script->feedAudio(playoutBuffer, size);
    }

    // Recording thread sends TTS audio to WebRTC
    void RecThreadProcess() {
        // Pull from TTS buffer (same pattern as WhisperAudioDevice)
        if (_ttsBuffer.hasData()) {
            auto chunk = _ttsBuffer.read(samplesPerFrame);
            _ptrAudioBuffer->SetRecordedBuffer(chunk, samplesPerFrame);
            _ptrAudioBuffer->DeliverRecordedData();
        }
    }
};
```

#### Step 3: Wire up in DirectApplication

In `examples/direct/direct.cc`:

```cpp
if (!opts_.script_path.empty()) {
    SpeechAudioDeviceFactory::SetScriptPath(opts_.script_path);
    // Factory creates ScriptAudioDevice instead of WhisperAudioDevice
}
```

#### Step 4: Handle script events

```cpp
static void scriptEventCallback(const char* event, const char* step,
                                 const char* data, void* ud) {
    auto* device = static_cast<ScriptAudioDevice*>(ud);

    if (strcmp(event, "forward") == 0) {
        // Signal DirectApplication to forward the call
        device->onForwardRequested(data);
    }
    if (strcmp(event, "end") == 0) {
        // Script finished → disconnect
        device->onScriptComplete();
    }
}
```

### Config Example

`directcall.config.json`:

```json
{
    "mode": "callee",
    "whisper_model": "/opt/models/whisper-base.bin",
    "llama_model": "/opt/models/llama-3.2.gguf",
    "script": "/opt/scripts/voicemail.yml",
    "styletts2_model_dir": "/opt/models/styletts2",
    "user_name": "voicemail-bot"
}
```

### Deployment

```bash
# On the server
./directcall --config=/opt/config/voicemail.json

# Or with CLI args
./directcall \
    --mode=callee \
    --script=/opt/scripts/voicemail.yml \
    --whisper_model=/opt/models/whisper-base.bin \
    --user_name=voicemail-bot
```

## Example Scripts

### Voicemail (`scripts/voicemail.yml`)

Full voicemail flow:

1. **greet** – "Hello, no one is available..."
2. **ask_name** – Listen for caller name (8s timeout)
3. **confirm_name** – "I heard [name], correct?" → yes/no branch
4. **ask_message** – "Leave your message" (30s timeout, saves audio)
5. **playback** – Repeats the message back
6. **goodbye** – "Thank you [name], message saved. Goodbye."
7. **hangup** – End

### Call Forwarder (`scripts/forwarder.yml`)

IVR routing:

1. **greet** – "Welcome to call routing"
2. **ask_department** – Listen with regex: sales|support|billing
3. **forward_X** – "Connecting to [dept]..." → forward action

### AI Receptionist (example)

```yaml
script:
  name: "AI Receptionist"
  language: "en"

steps:
  - id: greet
    action: speak
    text: "Hello, thank you for calling. How can I help you today?"
    next: listen_request

  - id: listen_request
    action: listen
    timeout_ms: 10000
    store_as: request
    next: ai_respond

  - id: ai_respond
    action: ask_ai
    text: "You are a helpful receptionist. The caller said: ${request}. Respond briefly."
    store_as: ai_reply
    next: ask_more

  - id: ask_more
    action: listen
    prompt: "Is there anything else I can help with?"
    timeout_ms: 8000
    store_as: more
    on_match:
      - pattern: "no|nothing|bye|goodbye"
        next: goodbye
      - pattern: "yes|actually|also"
        next: listen_request
    next: goodbye

  - id: goodbye
    action: speak
    text: "Thank you for calling. Have a great day!"
    next: done

  - id: done
    action: end
```

## Testing

```bash
# Unit tests (no models needed)
./test_whillats --script

# Integration test with StyleTTS2 + YAML script
STYLETTS2_MODEL_DIR=./trained_models \
./test_whillats --script=scripts/test_simple.yml

# Full voicemail test (needs Whisper model for real STT)
STYLETTS2_MODEL_DIR=./trained_models \
./test_whillats --script=scripts/voicemail.yml \
    --whisper_model=/path/to/whisper-base.bin
```

## Limitations

- StyleTTS2 is English-trained; non-English text will be phonemized but accented
- `ask_ai` requires a LLaMA model to be loaded; falls back to "AI not available" otherwise
- `store_audio: true` is a flag for the event handler; actual audio storage must be implemented by the caller app
- `forward` emits an event; the WebRTC app must handle the actual call transfer
