#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <regex>
#include <set>
#include <cstdint>
#include <cstring>
#include <memory>

#include "llama.h"
#include "clip.h"
#include "mtmd.h"
#include "llama_device_base.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"

// Clean response function (unchanged)
std::string clean_response(const std::string& response) {
    std::string cleaned = response;
    cleaned = std::regex_replace(cleaned, std::regex("<\\|eot_id\\|>"), "");
    size_t pos = cleaned.find("'t tell me what you're talking about");
    if (pos != std::string::npos) {
        cleaned = cleaned.substr(0, pos);
    }
    cleaned.erase(cleaned.find_last_not_of(" \n\r\t") + 1);
    return cleaned;
}

// fnv1a_hash_yuv function (included to fix undeclared identifier)
uint64_t fnv1a_hash_yuv(const YUVData &yuv) {
    const uint64_t FNV_offset_basis = 1469598103934665603ull;
    const uint64_t FNV_prime        = 1099511628211ull;
    uint64_t h = FNV_offset_basis;
    for (size_t i = 0; i < yuv.y_size;  ++i) h = (h ^ yuv.y[i]) * FNV_prime;
    for (size_t i = 0; i < yuv.uv_size; ++i) h = (h ^ yuv.u[i]) * FNV_prime;
    for (size_t i = 0; i < yuv.uv_size; ++i) h = (h ^ yuv.v[i]) * FNV_prime;
    return h;
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
    int n_predict_ = 4096;
    std::string prompt_ = "You are a helpful assistant.";
    bool continue_ = false;

    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    llama_sampler *smpl_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    std::deque<llama_token> context_tokens_;
    int n_past_ = 0;
    std::set<llama_token> stopping_token_ids_;
    std::vector<std::string> stopping_token_strings_;

    mtmd::context_ptr ctx_mtmd_; // Correct member name

    std::chrono::steady_clock::time_point _lastResponseStart;
    std::chrono::steady_clock::time_point _lastResponseEnd;

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

    // Load MMProj with mtmd
    if (!mmproj_path_.empty()) {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.n_threads = ctx_params.n_threads;
        mtmd_params.use_gpu = true;
        mtmd_params.verbosity = GGML_LOG_LEVEL_WARN;
        ctx_mtmd_.reset(mtmd_init_from_file(mmproj_path_.c_str(), model_, mtmd_params));
        if (!ctx_mtmd_) {
            LOG_E("Failed to load MMProj model with mtmd");
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
    }

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
    ctx_mtmd_.reset();
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

    context_tokens_.insert(context_tokens_.end(), prompt_tokens.begin(), prompt_tokens.end());

    if (context_tokens_.size() > (size_t) n_predict_) {
        int excess = context_tokens_.size() - n_predict_;
        context_tokens_.erase(context_tokens_.begin(), context_tokens_.begin() + excess);
        n_past_ = std::max(0, n_past_ - excess);
        if (!InitializeContext()) {
            LOG_E("Failed to reinitialize context after trimming.");
            return "";
        }
    }

    int n_new = prompt_tokens.size();
    if (n_new > 0) {
        struct llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_new);
        if (llama_decode(ctx_, batch)) {
            LOG_E("Failed to decode new prompt tokens.");
            return "";
        }
        n_past_ += n_new;
    }

    std::string response;
    std::string current_phrase;
    std::string recent_text;
    continue_ = true;

    const int max_response_tokens = 100;
    int generated_tokens = 0;
    int repetition_count = 0;

    _lastResponseStart = std::chrono::steady_clock::now();

    while (continue_ && generated_tokens < max_response_tokens) {
        if (!smpl_ || !ctx_) {
            LOG_E("Sampler or context became null during generation.");
            break;
        }

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
        if (stopping_token_ids_.find(new_token_id) != stopping_token_ids_.end()) {
            LOG_V("Reached a stopping token with ID: " << new_token_id);
            break;
        }

        if (new_token_id < 0 || new_token_id >= n_vocab) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Invalid token ID sampled: %d (vocab size: %d)", new_token_id, n_vocab);
            LOG_E(msg);
            break;
        }

        char token_text[64];
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

        if (isRepetitive(recent_text, 5)) {
            repetition_count++;
            if (repetition_count > 2) {
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
    // Ensure generation loop is enabled
    continue_ = true;

    if (!ctx_ || !vocab_ || !smpl_ || !ctx_mtmd_) {
        LOG_E("Context, vocab, sampler, or mtmd context not initialized");
        return "";
    }

    if (!yuv || !yuv->y || !yuv->u || !yuv->v) {
        LOG_E("Invalid YUV data");
        return "";
    }

    _lastResponseStart = std::chrono::steady_clock::now();

    // Reset context
    context_tokens_.clear();
    n_past_ = 0;
    const std::string system_prompt = "[INST] You are a helpful assistant for image description.";
    int n_ctx = llama_n_ctx(ctx_);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens <= 0) {
        LOG_E("Failed to tokenize system prompt");
        return "";
    }
    tokens.resize(n_tokens);
    context_tokens_ = std::deque<llama_token>(tokens.begin(), tokens.end());

    // Process system prompt
    int max_batch_size = 32; // Reduced for iOS compatibility
    llama_batch batch = llama_batch_init(max_batch_size, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    if (llama_decode(ctx_, batch) != 0) {
        LOG_E("Failed to decode system prompt");
        llama_batch_free(batch);
        return "";
    }
    n_past_ = n_tokens;
    LOG_I("Initialized context with system prompt, n_past_=" << n_past_);

    // Convert YUV to clip_image_u8
    auto start_encode = std::chrono::steady_clock::now();
    clip_image_u8* img_clip = yuv_to_clip(*yuv);
    if (!img_clip) {
        LOG_E("Failed to convert YUV to clip image");
        llama_batch_free(batch);
        return "";
    }
    LOG_I("Converted YUV to clip image in " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_encode).count() << " ms");

    // Create mtmd_bitmap
    mtmd::bitmap bitmap(img_clip->width, img_clip->height, img_clip->data);
    if (!bitmap.ptr) {
        LOG_E("Failed to create mtmd_bitmap");
        free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    std::string bitmap_id = "image_" + std::to_string(fnv1a_hash_yuv(*yuv));
    bitmap.set_id(bitmap_id.c_str());
    LOG_V("Created mtmd_bitmap with ID: " << bitmap_id);

    // Prepare prompt with image marker
    std::string full_prompt = "[INST] " + std::string(MTMD_DEFAULT_IMAGE_MARKER) + " USER: " + prompt + " ASSISTANT: ";
    mtmd_input_text input_text = { full_prompt.c_str(), true, false };
    std::vector<const mtmd_bitmap*> bitmaps = { bitmap.ptr.get() };

    // Tokenize
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) {
        LOG_E("Failed to initialize input chunks");
        free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    auto start_tokenize = std::chrono::steady_clock::now();
    int32_t tokenize_result = mtmd_tokenize(ctx_mtmd_.get(), chunks.ptr.get(), &input_text, bitmaps.data(), bitmaps.size());
    if (tokenize_result != 0) {
        LOG_E("mtmd_tokenize failed with error code: " << tokenize_result);
        free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    LOG_I("Tokenized input in " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_tokenize).count() << " ms");

    // Evaluate chunks
    auto start_eval = std::chrono::steady_clock::now();
    llama_pos new_n_past = n_past_;
    int32_t eval_result = mtmd_helper_eval_chunks(ctx_mtmd_.get(), ctx_, chunks.ptr.get(), n_past_, 0, max_batch_size, true, &new_n_past);
    if (eval_result != 0) {
        LOG_E("mtmd_helper_eval_chunks failed with error code: " << eval_result);
        free_clip(img_clip);
        llama_batch_free(batch);
        return "";
    }
    n_past_ = new_n_past;
    LOG_I("Evaluated chunks in " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_eval).count() 
          << " ms, n_past_=" << n_past_);

    // Update context tokens
    size_t total_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
    LOG_V("Total tokens in chunks: " << total_tokens);
    std::vector<llama_token> all_tokens;
    all_tokens.reserve(total_tokens);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens_chunk = 0;
            const llama_token* tokens_chunk = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens_chunk);
            all_tokens.insert(all_tokens.end(), tokens_chunk, tokens_chunk + n_tokens_chunk);
            LOG_V("Added " << n_tokens_chunk << " text tokens from chunk " << i);
        } else {
            LOG_V("Processed image chunk " << i);
        }
    }
    context_tokens_.insert(context_tokens_.end(), all_tokens.begin(), all_tokens.end());

    // Clean up
    free_clip(img_clip);
    llama_batch_free(batch);

    // Check context size
    if (context_tokens_.size() > (size_t)n_predict_) {
        LOG_W("Context size " << context_tokens_.size() << " exceeds limit " << n_predict_ << ", trimming");
        int excess = context_tokens_.size() - n_predict_;
        context_tokens_.erase(context_tokens_.begin(), context_tokens_.begin() + excess);
        n_past_ = std::max(0, n_past_ - excess);
        if (!InitializeContext()) {
            LOG_E("Failed to reinitialize context after trimming");
            return "";
        }
    }

    // Generation loop
    std::string current_phrase;
    std::string response;
    int generated_tokens = 0;
    int max_gen_tokens = 100;
    int min_gen_tokens = 20;
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    for (int i = 0; i < max_gen_tokens && continue_; ++i) {
        auto start_sample = std::chrono::steady_clock::now();
        llama_token new_token = llama_sampler_sample(sampler, ctx_, -1);
        llama_sampler_accept(sampler, new_token);
        generated_tokens++;

        LOG_V("Raw token ID: " << new_token);

        if (generated_tokens > min_gen_tokens && stopping_token_ids_.find(new_token) != stopping_token_ids_.end()) {
            LOG_V("Stopping token encountered with ID: " << new_token << " after " << generated_tokens << " tokens");
            break;
        }

        char piece_buf[128];
        int len = llama_token_to_piece(vocab_, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (len < 0) {
            LOG_E("Failed to convert token " << new_token << " to piece");
            break;
        }
        piece_buf[std::min(len, (int)sizeof(piece_buf) - 1)] = '\0';
        std::string piece_str(piece_buf);
        current_phrase += piece_str;
        response += piece_str;

        if (generated_tokens > min_gen_tokens) {
            for (const auto& stop_str : stopping_token_strings_) {
                if (piece_str.find(stop_str) != std::string::npos) {
                    LOG_V("Stopping token string '" << stop_str << "' encountered after " << generated_tokens << " tokens");
                    continue_ = false;
                    break;
                }
            }
            if (!continue_) {
                break;
            }
        }

        LOG_V("Token " << new_token << ": '" << piece_buf << "'");

        if (isCompleteSentence(current_phrase) && current_phrase.length() > 50) {
            callback.OnResponseComplete(true, current_phrase.c_str());
            LOG_V("Partial image description (" << current_phrase.length() << " chars): " << current_phrase);
            current_phrase.clear();
        }

        if (response.length() > 3000) {
            LOG_V("Stopping due to response length (" << response.length() << " chars) after " << generated_tokens << " tokens");
            break;
        }

        batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_past_;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx_, batch) != 0) {
            LOG_E("llama_decode failed during generation after " << generated_tokens << " tokens");
            llama_batch_free(batch);
            break;
        }
        n_past_++;
        llama_batch_free(batch);
        LOG_V("Sampled and decoded token " << generated_tokens << " in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - start_sample).count() << " ms");
    }

    llama_sampler_free(sampler);

    if (!current_phrase.empty() && isCompleteSentence(current_phrase)) {
        callback.OnResponseComplete(true, current_phrase.c_str());
        response += current_phrase;
        LOG_V("Final partial image description: " << current_phrase);
    }
    std::string full_response = clean_response(response);
    if (!full_response.empty()) {
        callback.OnResponseComplete(true, full_response.c_str());
    } else {
        callback.OnResponseComplete(false, "");
    }
    LOG_I("Mtmd done: '" << full_response << "' in "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - _lastResponseStart).count()
          << " ms");

    return full_response;
}

void LlamaSimpleChat::DetectStoppingTokens() {
    if (!vocab_) {
        LOG_E("Vocabulary not loaded, cannot detect stopping tokens.");
        return;
    }
    stopping_token_ids_.clear();
    stopping_token_strings_.clear();
    
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
                break;
            }
        }
    }
    
    llama_token eos_token = llama_vocab_eos(vocab_);
    if (eos_token != -1) {
        stopping_token_ids_.insert(eos_token);
        LOG_V("Added EOS token as stopping condition with ID: " << eos_token);
    }
    
    if (stopping_token_ids_.empty()) {
        LOG_W("No stopping tokens detected in vocabulary.");
    }
}

// LlamaDeviceBase implementation (unchanged from provided)
LlamaDeviceBase::LlamaDeviceBase(
    const char* model_path,
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
    std::unique_lock<std::mutex> lock(_queueMutex);
    if (prompt && *prompt)
    {
        LOG_I("Asking llama: " << prompt);
        _requestQueue.emplace_back(Request{std::string(prompt), false, nullptr});
        _queueCondition.notify_one();
    }
}

void LlamaDeviceBase::askWithImage(const char *prompt, const YUVData& yuv) {
    std::unique_lock<std::mutex> lock(_queueMutex);
    if (prompt && *prompt)
    {
        LOG_I("Asking llama with image: " << prompt);
        YUVData copy;
        copy.width   = yuv.width;
        copy.height  = yuv.height;
        copy.y_size  = yuv.y_size;
        copy.uv_size = yuv.uv_size;
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

bool LlamaDeviceBase::RunProcessingThread()
{
    while (_running) {
        Request req;
        {
            std::unique_lock<std::mutex> lock(_queueMutex);
            _queueCondition.wait(lock, [&]{ return !_requestQueue.empty() || !_running; });
            if (!_running && _requestQueue.empty()) break;
            req = std::move(_requestQueue.front());
            _requestQueue.pop_front();
        }

        if (req.withImage && req.yuv) {
            uint64_t h = fnv1a_hash_yuv(*req.yuv);
            if (h != _lastYuvHash) {
                _lastYuvHash = h;
                _llama_chat->_lastResponseStart = std::chrono::steady_clock::now();
                _llama_chat->generateFromImage(req.yuv.get(), req.prompt, _responseCallback);
            }
        } else {
            _llama_chat->_lastResponseStart = std::chrono::steady_clock::now();
            _llama_chat->generate(req.prompt, _responseCallback);
        }
    }
    return true;
}

bool LlamaDeviceBase::TrimContext() {
    if (context_tokens_.size() > max_context_tokens_) {
        context_tokens_.erase(context_tokens_.begin(), context_tokens_.begin() + (context_tokens_.size() - max_context_tokens_));
        return true;
    }
    return false;
}

bool LlamaDeviceBase::AppendToContext(const std::vector<llama_token>& new_tokens) {
    context_tokens_.insert(context_tokens_.end(), new_tokens.begin(), new_tokens.end());
    return TrimContext();
}