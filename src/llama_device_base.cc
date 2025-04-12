#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <regex>

#include "llama.h"

// Assuming these are defined elsewhere
#include "llama_device_base.h"
#include "whisper_helpers.h"

class LlamaSimpleChat {
public:
    LlamaSimpleChat();
    ~LlamaSimpleChat();
    bool SetModelPath(const std::string &path);
    bool SetNGL(int layers);
    bool SetContextSize(int size);
    void StopGeneration();
    bool Initialize();
    std::string generate(const std::string &prompt, WhillatsSetResponseCallback callback);

    bool LoadModel();
    bool InitializeContext();
    void FreeContext();
    bool isRepetitive(const std::string &text, size_t minPatternLength = 10);
    bool isCompleteSentence(const std::string &text);

    std::string model_path_;
    int ngl_ = 0;
    int n_predict_ = 512; // Default context size
    std::string prompt_ = "You are a helpful assistant."; // Initial system prompt
    bool continue_ = false;

    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    llama_sampler *smpl_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    std::vector<llama_token> context_tokens_; // Persistent context
    int n_past_ = 0; // Track processed tokens

    std::chrono::steady_clock::time_point _lastResponseStart;
    std::chrono::steady_clock::time_point _lastResponseEnd;
};

LlamaSimpleChat::LlamaSimpleChat() = default;

LlamaSimpleChat::~LlamaSimpleChat() {
    if (smpl_) {
        llama_sampler_free(smpl_);
    }
    FreeContext();
    if (model_) {
        llama_model_free(model_);
    }
}

bool LlamaSimpleChat::SetModelPath(const std::string &path) {
    model_path_ = path;
    return true;
}

bool LlamaSimpleChat::SetNGL(int layers) {
    ngl_ = layers;
    return true;
}

bool LlamaSimpleChat::SetContextSize(int size) {
    n_predict_ = size;
    return true;
}

void LlamaSimpleChat::StopGeneration() {
    continue_ = false;
}

bool LlamaSimpleChat::Initialize() {
    ggml_backend_load_all();
    if (!LoadModel()) {
        LOG_E("Failed to load model.");
        return false;
    }
    if (!InitializeContext()) {
        LOG_E("Failed to initialize context.");
        return false;
    }
    if (smpl_) {
        llama_sampler_free(smpl_);
    }
    smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!smpl_) {
        LOG_E("Failed to initialize sampler.");
        return false;
    }
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return true;
}

bool LlamaSimpleChat::LoadModel() {
    if (model_path_.empty()) {
        LOG_E("Model path not set.");
        return false;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl_;
    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (!model_) {
        LOG_E("Unable to load model.");
        return false;
    }
    vocab_ = llama_model_get_vocab(model_);
    return true;
}

bool LlamaSimpleChat::InitializeContext() {
    if (ctx_) {
        FreeContext();
    }

    if (!model_ || !vocab_) {
        LOG_E("Model or vocab not loaded.");
        return false;
    }

    // Tokenize initial system prompt only on first initialization
    if (context_tokens_.empty()) {
        const int n_prompt = -llama_tokenize(vocab_, prompt_.c_str(), prompt_.size(), nullptr, 0, true, true);
        if (n_prompt < 0) {
            LOG_E("Failed to count prompt tokens.");
            return false;
        }
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab_, prompt_.c_str(), prompt_.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            LOG_E("Failed to tokenize the prompt.");
            return false;
        }
        context_tokens_ = prompt_tokens;
        n_past_ = 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_predict_;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = false;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        LOG_E("Failed to create the llama_context.");
        return false;
    }

    // Process initial context tokens
    if (!context_tokens_.empty() && n_past_ == 0) {
        int n_eval = context_tokens_.size();
        int n_batch = ctx_params.n_batch;
        for (int i = 0; i < n_eval; i += n_batch) {
            int n_tokens = std::min(n_batch, n_eval - i);
            struct llama_batch batch = llama_batch_get_one(&context_tokens_[i], n_tokens);
            if (llama_decode(ctx_, batch)) {
                LOG_E("Failed to decode initial context tokens.");
                FreeContext();
                return false;
            }
            n_past_ += n_tokens;
        }
        if (smpl_) {
            llama_sampler_reset(smpl_);
        }
    }

    return true;
}

void LlamaSimpleChat::FreeContext() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
}

bool LlamaSimpleChat::isRepetitive(const std::string &text, size_t minPatternLength) {
    if (text.length() < minPatternLength * 2) return false;
    for (size_t len = minPatternLength; len <= text.length() / 2; ++len) {
        std::string last = text.substr(text.length() - len);
        if (text.rfind(last, text.length() - len - 1) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool LlamaSimpleChat::isCompleteSentence(const std::string &text) {
    if (text.empty()) return false;
    char last_char = text.back();
    return (last_char == '.' || last_char == '!' || last_char == '?') &&
           !std::all_of(text.begin(), text.end(), isspace);
}

std::string LlamaSimpleChat::generate(const std::string &prompt, WhillatsSetResponseCallback callback) {
    if (!ctx_ || !vocab_ || !smpl_) {
        LOG_E("Context, vocab, or sampler not initialized.");
        return "";
    }

    // Tokenize the new prompt
    const int n_tokens = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(), nullptr, 0, false, false);
    if (n_tokens < 0) {
        LOG_E("Failed to count prompt tokens.");
        return "";
    }

    std::vector<llama_token> prompt_tokens(n_tokens);
    if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), false, false) < 0) {
        LOG_E("Failed to tokenize prompt.");
        return "";
    }

    // Append new tokens to context
    context_tokens_.insert(context_tokens_.end(), prompt_tokens.begin(), prompt_tokens.end());

    // Trim context if it exceeds the limit
    if (context_tokens_.size() > (size_t) n_predict_) {
        int excess = context_tokens_.size() - n_predict_;
        context_tokens_.erase(context_tokens_.begin(), context_tokens_.begin() + excess);
        n_past_ = std::max(0, n_past_ - excess);
        if (!InitializeContext()) {
            LOG_E("Failed to reinitialize context after trimming.");
            return "";
        }
    }

    // Process new tokens
    int n_new = prompt_tokens.size();
    if (n_new > 0) {
        struct llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_new);
        if (llama_decode(ctx_, batch)) {
            LOG_E("Failed to decode new prompt tokens.");
            return "";
        }
        n_past_ += n_new;
    }

    // Generation loop
    std::string response;
    std::string current_phrase;
    std::string recent_text;
    continue_ = true;

    const int max_response_tokens = 256;
    int generated_tokens = 0;
    int repetition_count = 0;

    _lastResponseStart = std::chrono::steady_clock::now();

    while (continue_ && generated_tokens < max_response_tokens) {
        if (!smpl_ || !ctx_) {
            LOG_E("Sampler or context became null during generation.");
            break;
        }

        // Sample from the last token's logits
        float *logits = llama_get_logits_ith(ctx_, -1);
        if (!logits) {
            LOG_E("Failed to get logits for sampling.");
            break;
        }

        int n_vocab = llama_vocab_n_tokens(vocab_);
        std::vector<llama_token_data> candidates(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            candidates[i] = {i, logits[i], 0.0f};
        }
        llama_token_data_array cur_p = {candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(smpl_, &cur_p);

        if (cur_p.size == 0 || cur_p.selected < 0 || cur_p.selected >= (int64_t)cur_p.size) {
            LOG_E("Invalid sampling result: empty array or out-of-bounds selection.");
            break;
        }

        llama_token new_token_id = cur_p.data[cur_p.selected].id;
        if (new_token_id == llama_vocab_eos(vocab_)) {
            LOG_V("Reached EOS token.");
            break;
        }

        // Validate token ID
        if (new_token_id < 0 || new_token_id >= n_vocab) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Invalid token ID sampled: %d (vocab size: %d)", new_token_id, n_vocab);
            LOG_E(msg);
            break;
        }

        // Convert token to text with a larger buffer
        char token_text[64]; // Increased from 8 to 64
        int token_text_len = llama_token_to_piece(vocab_, new_token_id, token_text, sizeof(token_text), 0, true);
        if (token_text_len < 0) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Failed to convert token %d to piece.", new_token_id);
            LOG_E(msg);
            break;
        }

        std::string piece(token_text, token_text_len);
        current_phrase += piece;
        recent_text += piece;

        if (recent_text.length() > 50) {
            recent_text = recent_text.substr(recent_text.length() - 50);
        }

        context_tokens_.push_back(new_token_id);
        llama_sampler_accept(smpl_, new_token_id);

        if (isRepetitive(recent_text)) {
            repetition_count++;
            if (repetition_count > 3) {
                LOG_V("Stopping due to repetitive output.");
                break;
            }
        } else {
            repetition_count = 0;
        }

        if (isCompleteSentence(current_phrase)) {
            callback.OnResponseComplete(true, current_phrase.c_str());
            LOG_I("Llama says: '" << current_phrase << "' in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - _lastResponseStart).count()
                      << " ms");
            response += current_phrase;
            current_phrase.clear();
        }

        struct llama_batch batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx_, batch)) {
            LOG_E("Failed to decode new token.");
            break;
        }
        n_past_++;

        generated_tokens++;
    }

    if (!current_phrase.empty() && isCompleteSentence(current_phrase)) {
        callback.OnResponseComplete(true, current_phrase.c_str());
        response += current_phrase;
        LOG_I("Llama says: '" << current_phrase << "' in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - _lastResponseStart).count()
                  << " ms");
    }

    return response;
}

//
// Llama device base
LlamaDeviceBase::LlamaDeviceBase(
    const char*model_path,
    WhillatsSetResponseCallback callback)
    : _model_path(model_path),
      _responseCallback(callback)
{
}

LlamaDeviceBase::~LlamaDeviceBase() { stop(); }

void LlamaDeviceBase::askLlama(const char *prompt)
{
  {
    std::unique_lock<std::mutex> lock(_queueMutex);
    if (prompt && *prompt)
    {
      _textQueue.push(std::string(prompt));
    }
  }
}

bool LlamaDeviceBase::start()
{
  if (!_running)
  {
    _llama_chat.reset(new LlamaSimpleChat());
    _llama_chat->SetModelPath(_model_path);
    if (_llama_chat && _llama_chat->Initialize())
    {
      LOG_V("Llama chat initialized!");
    }
    else
    {
      LOG_E("Failed to initialize Llama chat");
      return false;
    }

    _running = true;
    _processingThread = std::thread([this]
                                    {
  while (_running && RunProcessingThread()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } });
  }
  return _running;
}
void LlamaDeviceBase::stop()
{
  if (_running)
  {
    _running = false;

    if (_processingThread.joinable())
    {
      _processingThread.join();
    }
  }
}

// LlamaDeviceBase remains mostly unchanged, but ensure TrimContext and AppendToContext are used correctly
bool LlamaDeviceBase::RunProcessingThread()
{
  while (_running)
  {
    std::string textToAsk;
    bool shouldAsk = false;
    {
      std::unique_lock<std::mutex> lock(_queueMutex);
      if (!_textQueue.empty())
      {
        textToAsk = _textQueue.front();
        _textQueue.pop();
        shouldAsk = true;
      }
    }

    if (shouldAsk)
    {
      _llama_chat->_lastResponseStart = std::chrono::steady_clock::now();
      _llama_chat->generate(textToAsk, _responseCallback);
      textToAsk.clear();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}