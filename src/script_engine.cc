/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#include "script_engine.h"
#include "whisper_helpers.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>

const std::string ScriptEngine::_emptyString;

static StepAction parseAction(const std::string& s) {
    if (s == "speak")   return StepAction::SPEAK;
    if (s == "listen")  return StepAction::LISTEN;
    if (s == "ask_ai")  return StepAction::ASK_AI;
    if (s == "forward") return StepAction::FORWARD;
    if (s == "store")   return StepAction::STORE;
    if (s == "end")     return StepAction::END;
    return StepAction::END;
}

ScriptEngine::ScriptEngine() = default;
ScriptEngine::~ScriptEngine() { stop(); }

bool ScriptEngine::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_E("ScriptEngine: Cannot open script file: " << path);
        return false;
    }
    std::stringstream buf;
    buf << file.rdbuf();
    return parseYaml(buf.str());
}

bool ScriptEngine::loadFromString(const std::string& yaml_content) {
    return parseYaml(yaml_content);
}

bool ScriptEngine::parseYaml(const std::string& content) {
    try {
        YAML::Node root = YAML::Load(content);

        auto script = root["script"];
        if (!script) {
            LOG_E("ScriptEngine: Missing 'script' root key");
            return false;
        }

        _config.name = script["name"].as<std::string>("Unnamed Script");
        _config.language = script["language"].as<std::string>("en");

        auto steps = root["steps"];
        if (!steps || !steps.IsSequence()) {
            LOG_E("ScriptEngine: Missing or invalid 'steps' key");
            return false;
        }

        _config.steps.clear();
        for (const auto& node : steps) {
            ScriptStep step;
            step.id = node["id"].as<std::string>();
            step.action = parseAction(node["action"].as<std::string>("end"));
            step.text = node["text"].as<std::string>("");
            step.prompt = node["prompt"].as<std::string>("");
            step.store_as = node["store_as"].as<std::string>("");
            step.store_audio = node["store_audio"].as<bool>(false);
            step.timeout_ms = node["timeout_ms"].as<int>(5000);
            step.next_step = node["next"].as<std::string>("");
            step.forward_target = node["target"].as<std::string>("");

            if (node["on_match"] && node["on_match"].IsSequence()) {
                for (const auto& m : node["on_match"]) {
                    MatchRule rule;
                    rule.pattern = m["pattern"].as<std::string>("");
                    rule.next_step = m["next"].as<std::string>("");
                    step.on_match.push_back(rule);
                }
            }
            _config.steps.push_back(step);
        }

        LOG_I("ScriptEngine: Loaded script '" << _config.name
              << "' with " << _config.steps.size() << " steps");
        return !_config.steps.empty();

    } catch (const YAML::Exception& e) {
        LOG_E("ScriptEngine: YAML parse error: " << e.what());
        return false;
    }
}

void ScriptEngine::setEventHandler(ScriptEventHandler handler) {
    std::lock_guard<std::mutex> lock(_mutex);
    _eventHandler = handler;
}

bool ScriptEngine::start() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_config.steps.empty()) {
        LOG_E("ScriptEngine: No script loaded");
        return false;
    }
    _running = true;
    _variables.clear();
    _currentStepId = _config.steps.front().id;
    LOG_I("ScriptEngine: Started script '" << _config.name << "' at step: " << _currentStepId);
    executeCurrentStep();
    return true;
}

void ScriptEngine::stop() {
    std::lock_guard<std::mutex> lock(_mutex);
    _running = false;
    _waitingForSpeech = false;
    _waitingForListen = false;
    _waitingForAi = false;
}

void ScriptEngine::reset() {
    stop();
    _variables.clear();
    _currentStepId.clear();
}

const std::string& ScriptEngine::currentStepId() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _currentStepId;
}

const std::string& ScriptEngine::getVariable(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _variables.find(name);
    if (it != _variables.end()) return it->second;
    return _emptyString;
}

const std::unordered_map<std::string, std::string>& ScriptEngine::getAllVariables() const {
    return _variables;
}

std::string ScriptEngine::expandVariables(const std::string& text) const {
    std::string result = text;
    size_t pos = 0;
    while ((pos = result.find("${", pos)) != std::string::npos) {
        size_t end = result.find("}", pos + 2);
        if (end == std::string::npos) break;
        std::string var_name = result.substr(pos + 2, end - pos - 2);
        auto it = _variables.find(var_name);
        std::string replacement = (it != _variables.end()) ? it->second : "";
        result.replace(pos, end - pos + 1, replacement);
        pos += replacement.size();
    }
    return result;
}

const ScriptStep* ScriptEngine::findStep(const std::string& id) const {
    for (const auto& step : _config.steps) {
        if (step.id == id) return &step;
    }
    return nullptr;
}

std::string ScriptEngine::evaluateMatch(const std::string& input,
                                         const std::vector<MatchRule>& rules,
                                         const std::string& default_next) {
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);

    for (const auto& rule : rules) {
        try {
            std::regex re(rule.pattern, std::regex_constants::icase);
            if (std::regex_search(lower_input, re)) {
                LOG_V("ScriptEngine: Input matched pattern '" << rule.pattern << "'");
                return rule.next_step;
            }
        } catch (const std::regex_error&) {
            if (lower_input.find(rule.pattern) != std::string::npos) {
                return rule.next_step;
            }
        }
    }
    return default_next;
}

void ScriptEngine::executeCurrentStep() {
    const ScriptStep* step = findStep(_currentStepId);
    if (!step || !_running) return;

    LOG_I("ScriptEngine: Executing step '" << step->id << "' action=" << static_cast<int>(step->action));

    ScriptEvent event;
    event.step_id = step->id;

    switch (step->action) {
        case StepAction::SPEAK: {
            std::string expanded = expandVariables(step->text);
            _waitingForSpeech = true;

            if (!step->prompt.empty()) {
                std::string prompt_expanded = expandVariables(step->prompt);
                event.type = ScriptEventType::SPEAK_REQUEST;
                event.data = prompt_expanded;
                if (_eventHandler) _eventHandler(event);
            }

            event.type = ScriptEventType::SPEAK_REQUEST;
            event.data = expanded;
            if (_eventHandler) _eventHandler(event);
            break;
        }

        case StepAction::LISTEN: {
            if (!step->prompt.empty()) {
                std::string prompt_expanded = expandVariables(step->prompt);
                ScriptEvent speak_event;
                speak_event.type = ScriptEventType::SPEAK_REQUEST;
                speak_event.step_id = step->id;
                speak_event.data = prompt_expanded;
                _waitingForSpeech = true;
                if (_eventHandler) _eventHandler(speak_event);
            } else {
                _waitingForListen = true;
                _listenStartTime = std::chrono::steady_clock::now();
                event.type = ScriptEventType::LISTEN_START;
                event.data = std::to_string(step->timeout_ms);
                event.variable_name = step->store_as;
                if (_eventHandler) _eventHandler(event);
            }
            break;
        }

        case StepAction::ASK_AI: {
            std::string prompt = expandVariables(step->text);
            _waitingForAi = true;
            event.type = ScriptEventType::ASK_AI_REQUEST;
            event.data = prompt;
            if (_eventHandler) _eventHandler(event);
            break;
        }

        case StepAction::FORWARD: {
            event.type = ScriptEventType::FORWARD_REQUEST;
            event.data = step->forward_target;
            if (_eventHandler) _eventHandler(event);
            break;
        }

        case StepAction::STORE: {
            if (!step->store_as.empty() && !step->text.empty()) {
                _variables[step->store_as] = expandVariables(step->text);
            }
            event.type = ScriptEventType::STORE_DATA;
            event.variable_name = step->store_as;
            event.data = getVariable(step->store_as);
            if (_eventHandler) _eventHandler(event);
            if (!step->next_step.empty()) {
                advanceTo(step->next_step);
            }
            break;
        }

        case StepAction::END: {
            _running = false;
            event.type = ScriptEventType::SCRIPT_END;
            if (_eventHandler) _eventHandler(event);
            break;
        }
    }
}

void ScriptEngine::advanceTo(const std::string& step_id) {
    if (!_running) return;
    const ScriptStep* next = findStep(step_id);
    if (!next) {
        LOG_E("ScriptEngine: Step '" << step_id << "' not found, ending script");
        _running = false;
        ScriptEvent event;
        event.type = ScriptEventType::ERROR;
        event.data = "Step not found: " + step_id;
        if (_eventHandler) _eventHandler(event);
        return;
    }

    _currentStepId = step_id;
    ScriptEvent event;
    event.type = ScriptEventType::STEP_CHANGED;
    event.step_id = step_id;
    if (_eventHandler) _eventHandler(event);

    executeCurrentStep();
}

void ScriptEngine::onSpeechComplete() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_running || !_waitingForSpeech) return;
    _waitingForSpeech = false;

    const ScriptStep* step = findStep(_currentStepId);
    if (!step) return;

    if (step->action == StepAction::LISTEN) {
        _waitingForListen = true;
        _listenStartTime = std::chrono::steady_clock::now();
        ScriptEvent event;
        event.type = ScriptEventType::LISTEN_START;
        event.step_id = step->id;
        event.data = std::to_string(step->timeout_ms);
        event.variable_name = step->store_as;
        if (_eventHandler) _eventHandler(event);
    } else if (step->action == StepAction::SPEAK && !step->next_step.empty()) {
        advanceTo(step->next_step);
    }
}

void ScriptEngine::onTranscriptionReceived(const std::string& text) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_running || !_waitingForListen) return;
    _waitingForListen = false;

    const ScriptStep* step = findStep(_currentStepId);
    if (!step) return;

    if (!step->store_as.empty()) {
        _variables[step->store_as] = text;
        LOG_I("ScriptEngine: Stored '" << step->store_as << "' = '" << text << "'");
    }

    std::string next = evaluateMatch(text, step->on_match, step->next_step);
    if (!next.empty()) {
        advanceTo(next);
    } else {
        _running = false;
        ScriptEvent event;
        event.type = ScriptEventType::SCRIPT_END;
        event.step_id = step->id;
        if (_eventHandler) _eventHandler(event);
    }
}

void ScriptEngine::onAiResponse(const std::string& response) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_running || !_waitingForAi) return;
    _waitingForAi = false;

    const ScriptStep* step = findStep(_currentStepId);
    if (!step) return;

    if (!step->store_as.empty()) {
        _variables[step->store_as] = response;
    }

    ScriptEvent event;
    event.type = ScriptEventType::SPEAK_REQUEST;
    event.step_id = step->id;
    event.data = response;
    if (_eventHandler) _eventHandler(event);

    _waitingForSpeech = true;
}

void ScriptEngine::checkTimeout() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_running || !_waitingForListen) return;

    const ScriptStep* step = findStep(_currentStepId);
    if (!step) return;

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - _listenStartTime).count();

    if (elapsed >= step->timeout_ms) {
        LOG_W("ScriptEngine: Listen timeout on step '" << step->id << "'");
        _waitingForListen = false;

        ScriptEvent event;
        event.type = ScriptEventType::LISTEN_TIMEOUT;
        event.step_id = step->id;
        if (_eventHandler) _eventHandler(event);

        if (!step->next_step.empty()) {
            advanceTo(step->next_step);
        }
    }
}
