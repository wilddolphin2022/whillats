/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

#ifndef SCRIPT_ENGINE_H
#define SCRIPT_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <chrono>
#include <regex>

enum class StepAction {
    SPEAK,
    LISTEN,
    ASK_AI,
    FORWARD,
    STORE,
    END
};

struct MatchRule {
    std::string pattern;
    std::string next_step;
};

struct ScriptStep {
    std::string id;
    StepAction action;
    std::string text;
    std::string prompt;
    std::string store_as;
    bool store_audio = false;
    int timeout_ms = 5000;
    std::string next_step;
    std::string forward_target;
    std::vector<MatchRule> on_match;
};

struct ScriptConfig {
    std::string name;
    std::string language = "en";
    std::vector<ScriptStep> steps;
};

enum class ScriptEventType {
    SPEAK_REQUEST,
    LISTEN_START,
    LISTEN_TIMEOUT,
    ASK_AI_REQUEST,
    FORWARD_REQUEST,
    STORE_DATA,
    SCRIPT_END,
    STEP_CHANGED,
    ERROR
};

struct ScriptEvent {
    ScriptEventType type;
    std::string step_id;
    std::string data;
    std::string variable_name;
};

using ScriptEventHandler = std::function<void(const ScriptEvent&)>;

class ScriptEngine {
public:
    ScriptEngine();
    ~ScriptEngine();

    bool loadFromFile(const std::string& path);
    bool loadFromString(const std::string& yaml_content);

    void setEventHandler(ScriptEventHandler handler);

    bool start();
    void stop();
    void reset();

    void onTranscriptionReceived(const std::string& text);
    void onSpeechComplete();
    void onAiResponse(const std::string& response);
    void checkTimeout();

    const std::string& currentStepId() const;
    const std::string& getVariable(const std::string& name) const;
    const std::unordered_map<std::string, std::string>& getAllVariables() const;
    const ScriptConfig& config() const { return _config; }
    bool isRunning() const { return _running; }

private:
    bool parseYaml(const std::string& content);
    void executeCurrentStep();
    void advanceTo(const std::string& step_id);
    std::string expandVariables(const std::string& text) const;
    std::string evaluateMatch(const std::string& input, const std::vector<MatchRule>& rules, const std::string& default_next);
    const ScriptStep* findStep(const std::string& id) const;

    ScriptConfig _config;
    std::unordered_map<std::string, std::string> _variables;
    std::string _currentStepId;
    bool _running = false;
    bool _waitingForSpeech = false;
    bool _waitingForListen = false;
    bool _waitingForAi = false;
    std::chrono::steady_clock::time_point _listenStartTime;

    ScriptEventHandler _eventHandler;
    mutable std::mutex _mutex;

    static const std::string _emptyString;
};

#endif // SCRIPT_ENGINE_H
