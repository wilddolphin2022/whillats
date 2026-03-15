/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#include "whillats.h"
#include "script_engine.h"
#include "whisper_helpers.h"
#include <thread>
#include <atomic>

struct WhillatsScriptImpl {
    std::string script_path;
    WhillatsSetAudioCallback audio_cb{nullptr, nullptr};
    WhillatsSetResponseCallback response_cb{nullptr, nullptr};
    ScriptEventCallback event_cb = nullptr;
    void* user_data = nullptr;

    std::unique_ptr<ScriptEngine> engine;
    std::unique_ptr<WhillatsTTS> tts;
    std::unique_ptr<WhillatsTranscriber> transcriber;
    std::unique_ptr<WhillatsLlama> llama;

    std::thread timeout_thread;
    std::atomic<bool> running{false};

    mutable std::string current_step_buf;
    mutable std::string variable_buf;

    void emitEvent(const char* type, const char* step, const char* data) {
        if (event_cb) {
            event_cb(type, step, data, user_data);
        }
    }
};

static void ttsCallback(bool success, const uint16_t* buffer, size_t buffer_size, void* user_data) {
    auto* impl = static_cast<WhillatsScriptImpl*>(user_data);
    impl->audio_cb.OnBufferComplete(success, 
        std::vector<uint16_t>(buffer, buffer + (buffer_size > 0 ? buffer_size : 0)));
    if (!success && buffer == nullptr) {
        impl->engine->onSpeechComplete();
    }
}

static void whisperCallback(bool success, const char* response, void* user_data) {
    auto* impl = static_cast<WhillatsScriptImpl*>(user_data);
    if (success && response && impl->engine->isRunning()) {
        impl->engine->onTranscriptionReceived(std::string(response));
    }
}

static void llamaCallback(bool success, const char* response, void* user_data) {
    auto* impl = static_cast<WhillatsScriptImpl*>(user_data);
    if (success && response && impl->engine->isRunning()) {
        impl->engine->onAiResponse(std::string(response));
    }
}

WhillatsScript::WhillatsScript(const char* script_path,
                               WhillatsSetAudioCallback audio_cb,
                               WhillatsSetResponseCallback response_cb,
                               ScriptEventCallback event_cb,
                               void* user_data)
    : _impl(std::make_unique<WhillatsScriptImpl>())
{
    _impl->script_path = script_path;
    _impl->audio_cb = audio_cb;
    _impl->response_cb = response_cb;
    _impl->event_cb = event_cb;
    _impl->user_data = user_data;
    _impl->engine = std::make_unique<ScriptEngine>();
}

WhillatsScript::~WhillatsScript() {
    stop();
}

bool WhillatsScript::start(const char* whisper_model,
                            const char* llama_model,
                            const char* llama_mmproj) {
    if (!_impl->engine->loadFromFile(_impl->script_path)) {
        LOG_E("WhillatsScript: Failed to load script: " << _impl->script_path);
        return false;
    }

    WhillatsSetAudioCallback tts_cb(ttsCallback, _impl.get());
    _impl->tts = std::make_unique<WhillatsTTS>(tts_cb);
    if (!_impl->tts->start()) {
        LOG_E("WhillatsScript: Failed to start TTS");
        return false;
    }

    if (whisper_model && strlen(whisper_model) > 0) {
        WhillatsSetResponseCallback whisper_cb(whisperCallback, _impl.get());
        _impl->transcriber = std::make_unique<WhillatsTranscriber>(whisper_model, whisper_cb);
        if (!_impl->transcriber->start()) {
            LOG_W("WhillatsScript: Failed to start Whisper transcriber");
            _impl->transcriber.reset();
        }
    }

    if (llama_model && strlen(llama_model) > 0) {
        WhillatsSetResponseCallback llama_cb(llamaCallback, _impl.get());
        if (llama_mmproj && strlen(llama_mmproj) > 0) {
            _impl->llama = std::make_unique<WhillatsLlama>(llama_model, llama_mmproj, llama_cb);
        } else {
            _impl->llama = std::make_unique<WhillatsLlama>(llama_model, llama_cb);
        }
        if (!_impl->llama->start()) {
            LOG_W("WhillatsScript: Failed to start LLaMA");
            _impl->llama.reset();
        }
    }

    std::string language = _impl->engine->config().language;
    _impl->engine->setEventHandler([this, language](const ScriptEvent& event) {
        switch (event.type) {
            case ScriptEventType::SPEAK_REQUEST:
                LOG_I("WhillatsScript: SPEAK -> " << event.data.substr(0, 60));
                _impl->tts->queueText(event.data.c_str(), language.c_str());
                _impl->emitEvent("speak", event.step_id.c_str(), event.data.c_str());
                break;

            case ScriptEventType::LISTEN_START:
                LOG_I("WhillatsScript: LISTEN (timeout=" << event.data << "ms)");
                _impl->emitEvent("listen", event.step_id.c_str(), event.data.c_str());
                break;

            case ScriptEventType::LISTEN_TIMEOUT:
                LOG_W("WhillatsScript: LISTEN TIMEOUT on step " << event.step_id);
                _impl->emitEvent("timeout", event.step_id.c_str(), "");
                break;

            case ScriptEventType::ASK_AI_REQUEST:
                LOG_I("WhillatsScript: ASK_AI -> " << event.data.substr(0, 60));
                if (_impl->llama) {
                    _impl->llama->askLlama(event.data.c_str());
                } else {
                    LOG_W("WhillatsScript: LLaMA not available, skipping ask_ai");
                    _impl->engine->onAiResponse("I'm sorry, AI is not available.");
                }
                _impl->emitEvent("ask_ai", event.step_id.c_str(), event.data.c_str());
                break;

            case ScriptEventType::FORWARD_REQUEST:
                LOG_I("WhillatsScript: FORWARD -> " << event.data);
                _impl->emitEvent("forward", event.step_id.c_str(), event.data.c_str());
                break;

            case ScriptEventType::STORE_DATA:
                LOG_I("WhillatsScript: STORE " << event.variable_name << " = " << event.data);
                _impl->emitEvent("store", event.step_id.c_str(), event.data.c_str());
                break;

            case ScriptEventType::SCRIPT_END:
                LOG_I("WhillatsScript: SCRIPT END");
                _impl->emitEvent("end", event.step_id.c_str(), "");
                break;

            case ScriptEventType::STEP_CHANGED:
                _impl->emitEvent("step", event.step_id.c_str(), "");
                break;

            case ScriptEventType::ERROR:
                LOG_E("WhillatsScript: ERROR -> " << event.data);
                _impl->emitEvent("error", event.step_id.c_str(), event.data.c_str());
                break;
        }
    });

    _impl->running = true;

    _impl->timeout_thread = std::thread([this] {
        while (_impl->running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (_impl->running) {
                _impl->engine->checkTimeout();
            }
        }
    });

    return _impl->engine->start();
}

void WhillatsScript::stop() {
    if (!_impl) return;
    _impl->running = false;

    if (_impl->timeout_thread.joinable()) {
        _impl->timeout_thread.join();
    }

    if (_impl->engine) _impl->engine->stop();
    if (_impl->tts) _impl->tts->stop();
    if (_impl->transcriber) _impl->transcriber->stop();
    if (_impl->llama) _impl->llama->stop();
}

void WhillatsScript::feedAudio(uint8_t* buffer, size_t size) {
    if (_impl->transcriber && _impl->engine->isRunning()) {
        _impl->transcriber->processAudioBuffer(buffer, size);
    }
}

const char* WhillatsScript::currentStep() const {
    if (!_impl || !_impl->engine) return "";
    _impl->current_step_buf = _impl->engine->currentStepId();
    return _impl->current_step_buf.c_str();
}

const char* WhillatsScript::getVariable(const char* name) const {
    if (!_impl || !_impl->engine || !name) return "";
    _impl->variable_buf = _impl->engine->getVariable(name);
    return _impl->variable_buf.c_str();
}

bool WhillatsScript::isRunning() const {
    return _impl && _impl->engine && _impl->engine->isRunning();
}
