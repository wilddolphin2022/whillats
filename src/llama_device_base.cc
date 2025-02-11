/*
 *  (c) 2025, wilddolphin2022
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2022
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <thread>

#include <llama.h>
#include "llama_device_base.h"
#include "whisper_helpers.h"

LlamaSimpleChat::LlamaSimpleChat() = default;

LlamaSimpleChat::~LlamaSimpleChat()
{
  if (smpl_)
  {
    llama_sampler_free(smpl_);
  }

  FreeContext();
  if (model_)
  {
    llama_model_free(model_);
  }
}

bool LlamaSimpleChat::SetModelPath(const std::string &path)
{
  model_path_ = path;
  return true;
}

bool LlamaSimpleChat::SetNGL(int layers)
{
  ngl_ = layers;
  return true;
}

bool LlamaSimpleChat::SetContextSize(int size)
{
  n_predict_ = size;
  return true;
}

void LlamaSimpleChat::StopGeneration()
{
  continue_ = false;
}

bool LlamaSimpleChat::Initialize()
{
  ggml_backend_load_all();
  return LoadModel() && InitializeContext();
}

bool LlamaSimpleChat::LoadModel()
{
  if (model_path_.empty())
  {
    LOG_E("Model path not set.");
    return false;
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = ngl_;
  model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
  if (!model_)
  {
    LOG_E("Unable to load model.");
    return false;
  }
  vocab_ = llama_model_get_vocab(model_);
  return true;
}

bool LlamaSimpleChat::InitializeContext()
{
  if (ctx_)
  {
    FreeContext();
  }

  if (!model_ || !vocab_)
  {
    LOG_E("Model or vocab not loaded.");
    return false;
  }

  // Tokenize the prompt
  const int n_prompt = -llama_tokenize(vocab_, prompt_.c_str(), prompt_.size(), NULL, 0, true, true);
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab_, prompt_.c_str(), prompt_.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
  {
    LOG_E("Failed to tokenize the prompt.");
    return false;
  }

  // Setup context parameters
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = n_prompt + n_predict_ - 1;
  ctx_params.n_batch = n_prompt;
  ctx_params.no_perf = false;

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_)
  {
    LOG_E("Failed to create the llama_context.");
    return false;
  }

  // Initialize sampler
  smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl_, llama_sampler_init_min_p(0.05f, 1));
  llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.8f));
  llama_sampler_chain_add(smpl_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  return true;
}

void LlamaSimpleChat::FreeContext()
{
  if (ctx_)
  {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
}

// Add helper function to detect repetition
bool LlamaSimpleChat::isRepetitive(const std::string &text, size_t minPatternLength)
{
  if (text.length() < minPatternLength * 2)
  {
    return false;
  }

  // Check for immediate repetition of phrases
  for (size_t len = minPatternLength; len <= text.length() / 2; ++len)
  {
    std::string last = text.substr(text.length() - len);
    size_t pos = text.rfind(last, text.length() - len - 1);
    if (pos != std::string::npos)
    {
      return true;
    }
  }
  return false;
}

// Add helper to check for confirmation patterns
bool LlamaSimpleChat::hasConfirmationPattern(const std::string &text)
{
  static const std::vector<std::string> patterns = {
      "yeah", "okay", "so", "right", "think",
      "that's", "correct", "makes sense"};

  size_t matches = 0;
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  for (const auto &pattern : patterns)
  {
    if (lower.find(pattern) != std::string::npos)
    {
      matches++;
      if (matches >= 3)
      { // If we see multiple confirmation words
        return true;
      }
    }
  }
  return false;
}

std::string LlamaSimpleChat::generate(const std::string &prompt, WhillatsSetResponseCallback callback)
{
  // Tokenize the prompt
  const struct llama_vocab *vocab = llama_model_get_vocab(model_);

  const int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, false);
  if (n_tokens < 0)
  {
    LOG_E("Failed to count prompt tokens");
    return "";
  }

  std::vector<llama_token> tokens(n_tokens);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, false) < 0)
  {
    LOG_E("Failed to tokenize prompt");
    return "";
  }

  // Create batch for prompt processing
  llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

  if (llama_decode(ctx_, batch))
  {
    LOG_E("Failed to process prompt");
    return "";
  }

  // Initialize generation
  std::string response;
  std::string current_phrase;
  std::string recent_text; // For pattern detection
  continue_ = true;

  const int max_response_tokens = 256;
  const int max_repetition_window = 50; // Characters to check for repetition
  int generated_tokens = 0;
  int unchanged_count = 0;    // Counter for unchanged text
  int confirmation_count = 0; // Counter for confirmation patterns

  // Initialize sampler chain if needed
  if (!smpl_)
  {
    auto params = llama_sampler_chain_default_params();
    smpl_ = llama_sampler_chain_init(params);
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
  }

  // Generation loop
  while (continue_ && generated_tokens < max_response_tokens)
  {
    if (!continue_)
    {
      break; // Immediate interrupt check
    }

    // Sample next token
    llama_token new_token_id = llama_sampler_sample(smpl_, ctx_, -1);

    if (new_token_id == llama_vocab_eos(vocab))
    {
      break;
    }

    // Convert token to text
    char token_text[8];
    int token_text_len = llama_token_to_piece(vocab, new_token_id, token_text, sizeof(token_text), 0, true);
    if (token_text_len < 0)
    {
      break;
    }

    // Process the generated piece
    std::string piece(token_text, token_text_len);
    current_phrase += piece;
    recent_text += piece;

    // Keep recent_text to a manageable size
    if (recent_text.length() > max_repetition_window)
    {
      recent_text = recent_text.substr(recent_text.length() - max_repetition_window);
    }

    // Check for natural response end conditions
    bool should_end = false;

    // 1. Check for repetitive patterns
    if (isRepetitive(recent_text))
    {
      unchanged_count++;
      if (unchanged_count > 3)
      { // Allow some repetition before breaking
        should_end = true;
      }
    }
    else
    {
      unchanged_count = 0;
    }

    // 2. Check for excessive confirmation patterns
    if (hasConfirmationPattern(current_phrase))
    {
      confirmation_count++;
      if (confirmation_count > 2)
      { // Break if too many confirmation patterns
        should_end = true;
      }
    }

    // Process completed phrases
    if (piece.find_first_of(".!?") != std::string::npos || should_end)
    {
      if (!current_phrase.empty())
      {
        callback.OnResponseComplete(true, current_phrase.c_str());

        _lastResponseEnd = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                _lastResponseEnd - _lastResponseStart).count();
        std::cout << "Llama says: '" << current_phrase << "' in " << duration << " ms";

      }
      response += current_phrase;
      current_phrase.clear();

      if (should_end)
      {
        break;
      }
    }

    // Prepare next token
    batch = llama_batch_get_one(&new_token_id, 1);
    if (llama_decode(ctx_, batch))
    {
      break;
    }

    generated_tokens++;
  }

  // Handle any remaining text
  if (!current_phrase.empty())
  {
    callback.OnResponseComplete(true, current_phrase.c_str());
    response += current_phrase;

    _lastResponseEnd = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            _lastResponseEnd - _lastResponseStart).count();
    std::cout << "Llama says: '" << current_phrase << "' in " << duration << " ms";
  }

  return response;
}

//
// Llama device base
//

LlamaDeviceBase::LlamaDeviceBase(
    const char*model_path,
    WhillatsSetResponseCallback callback)
    : _model_path(model_path),
      _responseCallback(callback)
{
}

LlamaDeviceBase::~LlamaDeviceBase() {}

void LlamaDeviceBase::askLlama(const char* prompt)
{
  {
    std::unique_lock<std::mutex> lock(_queueMutex);
    if (prompt && *prompt)
    {
      _textQueue.push(std::string(prompt));
    }
  }
}

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
        LOG_I("Llama was asked '" << textToAsk << "'");
      }
    }

    if (shouldAsk)
    {
      _llama_chat->_lastResponseStart = std::chrono::steady_clock::now();
      _llama_chat->generate(textToAsk, _responseCallback);
      textToAsk.clear();
    }

    // Sleep if no data available to read to prevent busy-waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  return true;
}

bool LlamaDeviceBase::start()
{
  if (!_running)
  {
    _llama_chat.reset(new LlamaSimpleChat());
    _llama_chat->SetModelPath(_model_path);
    if (_llama_chat && _llama_chat->Initialize())
    {
      LOG_I("Llama chat initialized!");
    } else {
      LOG_E("Failed to initialize Llama chat");
      return false;
    }

    _running = true;
    _processingThread = std::thread([this] {
      while (_running && RunProcessingThread()) {
      } 
    });
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

bool LlamaDeviceBase::TrimContext()
{
  if (context_tokens_.size() > max_context_tokens_)
  {
    // Keep the most recent tokens within the limit
    size_t excess = context_tokens_.size() - max_context_tokens_;
    context_tokens_.erase(context_tokens_.begin(), context_tokens_.begin() + excess);

    // Reinitialize context with trimmed tokens
    return _llama_chat->InitializeContext();
  }
  return true;
}

bool LlamaDeviceBase::AppendToContext(const std::vector<llama_token> &new_tokens)
{
  context_tokens_.insert(context_tokens_.end(), new_tokens.begin(), new_tokens.end());
  return TrimContext();
}
