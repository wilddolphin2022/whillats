#include <memory>
#include "whisper_transcription.h"
#include "llama_device_base.h"
#include "espeak_tts.h"

static WhisperTranscription _whisper;
static LlamaDeviceBase _llama;
static ESpeakTTS _tts;

void ProcessAudioBuffer(uint8_t* playoutBuffer, size_t kPlayoutBufferSize) {

}
