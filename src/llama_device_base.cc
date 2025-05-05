#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <regex>
#include <set>

#include "llama.h"
#include "clip.h"
#include "llava.h"

// Assuming these are defined elsewhere
#include "llama_device_base.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"

// Clean response by removing special tokens and artifacts
std::string clean_response(const std::string& response) {
    std::string cleaned = response;
    // Remove <|eot_id|> (WebRTC-compatible regex)
    cleaned = std::regex_replace(cleaned, std::regex("<\\|eot_id\\|>"), "");
    // Remove repeated phrases
    size_t pos = cleaned.find("'t tell me what you're talking about");
    if (pos != std::string::npos) {
        cleaned = cleaned.substr(0, pos);
    }
    // Trim whitespace
    cleaned.erase(cleaned.find_last_not_of(" \n\r\t") + 1);
    return cleaned;
}

class LlamaSimpleChat {
public:
    LlamaSimpleChat();
    ~LlamaSimpleChat();
    bool SetModelPaths(const std::string &path, const std::string &mmproj_path);
    bool SetNGL(int layers);
    bool SetContextSize(int size);
    void StopGeneration();
    bool Initialize();

    std::string generate(const std::string &prompt, WhillatsSetResponseCallback callback);
    std::string generateFromImage(YUVData* yuv, const std::string& prompt, WhillatsSetResponseCallback callback);

    bool LoadModel();
    bool InitializeContext();
    void FreeContext();
    bool isRepetitive(const std::string &text, size_t minPatternLength = 10);
    bool isCompleteSentence(const std::string &text);

    std::string model_path_;
    std::string mmproj_path_;
    int ngl_ = 100;
    int n_predict_ = 4096; // Default context size
    std::string prompt_ = "You are a helpful assistant."; // Initial system prompt
    bool continue_ = false;

    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    llama_sampler *smpl_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    std::deque<llama_token> context_tokens_; // Persistent context
    int n_past_ = 0; // Track processed tokens
    std::set<llama_token> stopping_token_ids_; // Store IDs of stopping tokens like <|eot_id|>
    std::vector<std::string> stopping_token_strings_; // Store string representations of stopping tokens

    clip_ctx* ctx_clip_ = nullptr;
    llava_image_embed cached_embed_ = {nullptr, 0};
    llama_batch image_batch_ = {0, 0, 0};

    std::chrono::steady_clock::time_point _lastResponseStart;
    std::chrono::steady_clock::time_point _lastResponseEnd;

    // New method to detect stopping tokens in vocabulary
    void DetectStoppingTokens();
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

bool LlamaSimpleChat::SetModelPaths(const std::string &path, const std::string &mmproj_path) {
    model_path_ = path;
    mmproj_path_ = mmproj_path;
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

    llama_backend_init();

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
    
    // Detect stopping tokens in the model's vocabulary
    DetectStoppingTokens();
    return true;
}

bool LlamaSimpleChat::LoadModel() {
    if (model_path_.empty()) {
        LOG_E("Model path not set.");
        return false;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::min(ngl_, 100); 
    model_params.main_gpu = 0;
    
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
        context_tokens_ = std::deque<llama_token>(prompt_tokens.begin(), prompt_tokens.end());
        n_past_ = 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_predict_;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = false;
    ctx_params.n_threads = std::min((int)4, (int)std::thread::hardware_concurrency());

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        LOG_E("Failed to create the llama_context.");
        return false;
    }

    // Load MMProj
    if(!mmproj_path_.empty()) {
        ctx_clip_ = clip_model_load(mmproj_path_.c_str(), true);
        if (!ctx_clip_) {
            LOG_E("ERROR: Failed to load MMProj model\n");
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
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
    if (cached_embed_.embed) {
        free(cached_embed_.embed);
        cached_embed_.embed = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (ctx_clip_) {
        clip_free(ctx_clip_);
        ctx_clip_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    llama_backend_free();
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
        LOG_E("generate: context, vocab, or sampler not initialized.");
        return "";
    }

    // Ensure context has system prompt if empty
    if (context_tokens_.empty()) {
        const std::string system_prompt = "You are a helpful assistant.";
        const int n_prompt = -llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.size(), nullptr, 0, true, true);
        if (n_prompt > 0) {
            std::vector<llama_token> prompt_tokens(n_prompt);
            if (llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) >= 0) {
                context_tokens_ = std::deque<llama_token>(prompt_tokens.begin(), prompt_tokens.end());
                n_past_ = 0;
                // Process initial system prompt tokens
                int n_eval = context_tokens_.size();
                int n_batch = llama_context_default_params().n_batch;
                for (int i = 0; i < n_eval; i += n_batch) {
                    int n_tokens = std::min(n_batch, n_eval - i);
                    struct llama_batch batch = llama_batch_get_one(&context_tokens_[i], n_tokens);
                    if (llama_decode(ctx_, batch)) {
                        LOG_E("Failed to decode initial system prompt tokens.");
                        return "";
                    }
                    n_past_ += n_tokens;
                }
                LOG_I("Initialized context with system prompt, n_past_=" << n_past_);
            }
        }
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

    const int max_response_tokens = 100; // Reduced from 256 to prevent long outputs
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
        // Check if the new token is in stopping_token_ids_
        if (stopping_token_ids_.find(new_token_id) != stopping_token_ids_.end()) {
            LOG_V("Reached a stopping token with ID: " << new_token_id);
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

        // Check for stopping token strings in the output text and stop if found
        for (const auto& stop_str : stopping_token_strings_) {
            if (piece.find(stop_str) != std::string::npos) {
                LOG_V("Found stopping token string '" << stop_str << "' in output, stopping generation.");
                continue_ = false;
                break;
            }
        }
        if (!continue_) {
            break;
        }

        if (recent_text.length() > 50) {
            recent_text = recent_text.substr(recent_text.length() - 50);
        }

        context_tokens_.push_back(new_token_id);
        llama_sampler_accept(smpl_, new_token_id);

        if (isRepetitive(recent_text, 5)) { // Reduced minPatternLength from 10 to 5 for stricter check
            repetition_count++;
            if (repetition_count > 2) { // Reduced threshold from 5 to 2 to stop earlier
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

std::string LlamaSimpleChat::generateFromImage(YUVData* yuv, const std::string& prompt, WhillatsSetResponseCallback callback) {
 
    if (!ctx_ || !vocab_ || !smpl_ || !ctx_clip_) {
        LOG_E("Context, vocab, sampler, or clip context not initialized");
        return "";
    }

    if (!yuv) {
        LOG_E("Invalid YUV data");
        return "";
    }

    _lastResponseStart = std::chrono::steady_clock::now();
    
    // Reset context to system prompt for new image to avoid influence from previous outputs
    context_tokens_.clear();
    n_past_ = 0;
    const std::string system_prompt = "[INST] You are a helpful assistant for image description.";
    int n_ctx = llama_n_ctx(ctx_);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens > 0) {
        tokens.resize(n_tokens);
        context_tokens_ = std::deque<llama_token>(tokens.begin(), tokens.end());
        // Initialize a temporary batch for context initialization
        int max_batch_size = 64; // Low for RAM
        llama_batch temp_batch = llama_batch_init(max_batch_size, 0, 1);
        temp_batch.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; ++i) {
            temp_batch.token[i] = tokens[i];
            temp_batch.pos[i] = i;
            temp_batch.n_seq_id[i] = 1;
            temp_batch.seq_id[i][0] = 0;
            temp_batch.logits[i] = false;
        }
        if (llama_decode(ctx_, temp_batch) == 0) {
            n_past_ = n_tokens;
            LOG_I("Initialized context with system prompt, n_past_=" << n_past_);
        } else {
            LOG_E("Failed to initialize context with system prompt");
        }
        llama_batch_free(temp_batch);
    }
    
    // Initialize batch for image processing
    int max_batch_size = 64; // Low for RAM
    llama_batch batch = llama_batch_init(max_batch_size, 0, 1);

    llava_image_embed embed = {nullptr, 0};
    clip_image_u8* img_clip = nullptr;

    //Preprocess image and create embedding if needed
    img_clip = yuv_to_clip(*yuv);
    if (!img_clip) {
        llama_batch_free(batch);
        return "";
    }

    // Uncomment to save image
    save_clip_as_bmp(*img_clip, "image.bmp");

    float* image_embed_ptr = nullptr;
    int n_image_pos = 0;

    if (!llava_image_embed_make_with_clip_img(ctx_clip_, 2, img_clip, &image_embed_ptr, &n_image_pos)) {
        LOG_E("Failed to create image embedding");
        free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    if (n_image_pos <= 0) {
        LOG_E("ERROR: Invalid image embedding size: " << n_image_pos);
        free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }

    embed = {image_embed_ptr, n_image_pos};
    //cached_embed_.embed = image_embed_ptr;
    //cached_embed_.n_image_pos = n_image_pos;
    LOG_V("DEBUG: Created image embedding, n_image_pos=" << n_image_pos);
    
    // Process prompt
    n_ctx = llama_n_ctx(ctx_);

    // Prompt structure: [INST] <image> USER: prompt ASSISTANT:
    std::string text_before_image = "[INST] ";
    std::string text_after_image = "USER: " + prompt + " ASSISTANT: ";

    // 1. Process text before image
    std::vector<llama_token> tokens_before(n_ctx);
    int n_tokens_before = llama_tokenize(vocab_, text_before_image.c_str(), text_before_image.length(), tokens_before.data(), tokens_before.size(), true, false);
    if (n_tokens_before < 0) {
        LOG_E("ERROR: Failed to tokenize text before image");
        if (img_clip) 
            free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }

    tokens_before.resize(n_tokens_before);

    batch.n_tokens = n_tokens_before;
    for (int i = 0; i < n_tokens_before; ++i) {
        batch.token[i] = tokens_before[i];
        batch.pos[i] = n_past_ + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }

    if (llama_decode(ctx_, batch) != 0) {
        LOG_E("Failed to decode text before image");
        if (img_clip) free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }

    n_past_ += n_tokens_before;
    LOG_I("Decoded text before image, n_past=" << n_past_);

    // 2. Evaluate image embed if using image
    // Use a local embed structure like the working example
    llava_image_embed embed_to_eval = {nullptr, 0};
    // if (cached_embed_.embed) {
    //   embed_to_eval = cached_embed_; // Copy the cached embed data
    embed_to_eval = embed;

      // Add logging before the call
      if (!llava_eval_image_embed(ctx_, &embed_to_eval, max_batch_size, &n_past_)) {
          LOG_E("Failed to evaluate image embed");
          if (img_clip) free_clip(img_clip);
          llama_batch_free(batch);
          return "";
      }
      LOG_I("Evaluated image embed, n_past=" << n_past_);
    // }

    // 3. Process text after image
    std::vector<llama_token> tokens_after(n_ctx);
    int n_tokens_after = llama_tokenize(vocab_, text_after_image.c_str(), text_after_image.length(), tokens_after.data(), tokens_after.size(), false, false);
    if (n_tokens_after < 0) {
        LOG_E("Failed to tokenize text after image");
        if (img_clip) free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    tokens_after.resize(n_tokens_after);

    batch.n_tokens = n_tokens_after;
    for (int i = 0; i < n_tokens_after; ++i) {
        batch.token[i] = tokens_after[i];
        batch.pos[i] = n_past_ + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens_after - 1); // Logits for last token
    }
    if (llama_decode(ctx_, batch) != 0) {
        LOG_E("Failed to decode text after image"); 
        if (img_clip) free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    n_past_ += n_tokens_after;
    LOG_V("Decoded text after image, n_past=" << n_past_);

    // Clean up image resources
    if (img_clip) 
        free_clip(img_clip);

    // 4. Generation loop
    std::string current_phrase;
    int generated_tokens = 0;
    std::string response;
    int max_gen_tokens = 100; // Reduced to prevent over-generation
    int min_gen_tokens = 20; // Minimum tokens to generate before checking stopping conditions
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for (int i = 0; i < max_gen_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(sampler, ctx_, -1); // Sample last token's logits
        llama_sampler_accept(sampler, new_token);
        generated_tokens++;

        // Log raw token ID for debugging
        LOG_V("Raw token ID: " << new_token);

        // Only check stopping tokens after minimum tokens are generated
        if (generated_tokens > min_gen_tokens) {
            // Check for stopping tokens
            if (stopping_token_ids_.find(new_token) != stopping_token_ids_.end()) {
                LOG_V("Stopping token encountered with ID: " << new_token << " after " << generated_tokens << " tokens");
                break;
            }
        }

        // Convert token to text, skip special tokens
        char piece_buf[128];
        piece_buf[0] = '\0';
        int len = llama_token_to_piece(vocab_, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (len < 0) {
            LOG_E("Failed to convert token " << new_token << " to piece");
            break;
        }
        piece_buf[std::min(len, (int)sizeof(piece_buf) - 1)] = '\0';

        // Only check stopping token strings after minimum tokens
        std::string piece_str(piece_buf);
        current_phrase += piece_str;
        response += piece_str;

        if (generated_tokens > min_gen_tokens) {
            for (const auto& stop_str : stopping_token_strings_) {
                if (piece_str.find(stop_str) != std::string::npos) {
                    LOG_V("Stopping token string '" << stop_str << "' encountered in output after " << generated_tokens << " tokens");
                    continue_ = false;
                    break;
                }
            }
            if (!continue_) {
                break;
            }
        }

        LOG_V("Token " << new_token << ": '" << piece_buf);

        // Check if current_phrase is a complete sentence and long enough to send via callback
        if (isCompleteSentence(current_phrase) && current_phrase.length() > 50) {
            callback.OnResponseComplete(true, current_phrase.c_str());
            LOG_V("Partial image description (" << current_phrase.length() << " chars): " << current_phrase);
            current_phrase.clear();
        }

        // Stop if response is sufficiently long
        if (response.length() > 3000) { // Increased from 2000 to allow much longer descriptions
            LOG_V("Stopping due to response length (" << response.length() << " chars) after " << generated_tokens << " tokens");
            break;
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_past_;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx_, batch) != 0) {
            LOG_E("llama_decode failed during generation after " << generated_tokens << " tokens");
            break;
        }
        n_past_++;
    }

    // Clean up
    llama_sampler_free(sampler);
    llama_batch_free(image_batch_);
    if (embed.embed) {
        free(embed.embed);
        embed.embed = nullptr;
    }

    // Post-process response to remove artifacts
    if (!current_phrase.empty() && isCompleteSentence(current_phrase)) {
        callback.OnResponseComplete(true, current_phrase.c_str());
        response += current_phrase;
        LOG_V("Final partial image description: " << current_phrase);
    }
    std::string full_response = clean_response(response);
    if (!full_response.empty()) {
        callback.OnResponseComplete(true, full_response.c_str());
    }
    LOG_V("Llava done: '" << full_response << "' in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - _lastResponseStart).count()
                << " ms");

    return full_response;
}

// New method to detect stopping tokens in vocabulary
void LlamaSimpleChat::DetectStoppingTokens() {
    if (!vocab_) {
        LOG_E("Vocabulary not loaded, cannot detect stopping tokens.");
        return;
    }
    stopping_token_ids_.clear();
    stopping_token_strings_.clear();
    
    // List of known stopping token strings to check for
    std::vector<std::string> known_stopping_tokens = {"<|eot_id|>", "<|end_of_text|>", "<|end|>", "</s>"};
    
    int n_vocab = llama_vocab_n_tokens(vocab_);
    for (int token_id = 0; token_id < n_vocab; ++token_id) {
        char piece_buf[128];
        piece_buf[0] = '\0';
        int len = llama_token_to_piece(vocab_, token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (len < 0) {
            continue;
        }
        piece_buf[std::min(len, (int)sizeof(piece_buf) - 1)] = '\0';
        std::string token_str(piece_buf);
        
        for (const auto& stop_token : known_stopping_tokens) {
            if (token_str.find(stop_token) != std::string::npos) {
                stopping_token_ids_.insert(token_id);
                stopping_token_strings_.push_back(stop_token);
                LOG_V("Detected stopping token: " << stop_token << " with ID: " << token_id);
                break; // Avoid duplicate matches
            }
        }
    }
    
    // Always add EOS token as a stopping condition
    llama_token eos_token = llama_vocab_eos(vocab_);
    if (eos_token != -1) {
        stopping_token_ids_.insert(eos_token);
        LOG_V("Added EOS token as stopping condition with ID: " << eos_token);
    }
    
    if (stopping_token_ids_.empty()) {
        LOG_W("No stopping tokens detected in vocabulary.");
    }
}

//
// Llama device base
LlamaDeviceBase::LlamaDeviceBase(
    const char*model_path,
    const char* mmproj_path, 
    WhillatsSetResponseCallback callback)
    : _model_path(model_path),
      _mmproj_path(mmproj_path),
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
        LOG_I("Asking llama: " << prompt);
        _requestQueue.emplace_back(Request{std::string(prompt), false, nullptr});
        _queueCondition.notify_one();
    }
  }
}

void LlamaDeviceBase::askWithImage(const char *prompt, const YUVData& yuv) {
  {
    std::unique_lock<std::mutex> lock(_queueMutex);
    if (prompt && *prompt)
    {
      LOG_I("Asking llama with image: " << prompt);
      YUVData copy;
      copy.width   = yuv.width;   copy.height  = yuv.height;
      copy.y_size  = yuv.y_size;  copy.uv_size = yuv.uv_size;
      copy.y       = std::make_unique<uint8_t[]>(copy.y_size);
      copy.u       = std::make_unique<uint8_t[]>(copy.uv_size);
      copy.v       = std::make_unique<uint8_t[]>(copy.uv_size);
      std::memcpy(copy.y.get(), yuv.y.get(), copy.y_size);
      std::memcpy(copy.u.get(), yuv.u.get(), copy.uv_size);
      std::memcpy(copy.v.get(), yuv.v.get(), copy.uv_size);
      auto framePtr = std::make_shared<YUVData>(std::move(copy));
      _requestQueue.emplace_back(Request{std::string(prompt), true, framePtr});
      _queueCondition.notify_one();
    }
  }
}

bool LlamaDeviceBase::start() {
    if (!_running) {
        _llama_chat.reset(new LlamaSimpleChat());
        _llama_chat->SetModelPaths(_model_path, _mmproj_path);
        if (_llama_chat && _llama_chat->Initialize()) {
            LOG_V("Llama chat initialized!");
        } else {
            LOG_E("Failed to initialize Llama chat");
            return false;
        }

        _running = true;
        _processingThread = std::thread([this] {
            while (_running && RunProcessingThread()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
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


// LlamaDeviceBase remains mostly unchanged, but ensure TrimContext and AppendToContext are used correctly
bool LlamaDeviceBase::RunProcessingThread()
{
  while (_running)
  {
    std::string textToAsk;
    bool shouldAsk = false;
    {
      std::unique_lock<std::mutex> lock(_queueMutex);
      if (!_requestQueue.empty()) 
      {
        Request request = _requestQueue.front();
        _requestQueue.pop_front();
        shouldAsk = true;
        textToAsk = request.prompt;
        if (request.withImage) {
          _llama_chat->generateFromImage(request.yuv.get(), request.prompt, _responseCallback);
        } else {
          _llama_chat->generate(request.prompt, _responseCallback);
        }
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
