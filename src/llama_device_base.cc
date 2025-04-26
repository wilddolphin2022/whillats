#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <regex>

#include "llama.h"
#include "clip.h"
#include "llava.h"

// Assuming these are defined elsewhere
#include "llama_device_base.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"

// Convert CV::Mat (float32 RGB, 0–1 range) to clip_image_u8
clip_image_u8* mat_to_clip_image_u8(const cv::Mat& img_float_rgb) {
    if (img_float_rgb.empty() || img_float_rgb.type() != CV_32FC3 || !img_float_rgb.isContinuous()) {
        LOG_E("ERROR: Invalid input image for conversion");
        return nullptr;
    }

    clip_image_u8* img_clip = clip_image_u8_init();
    if (!img_clip) {
        LOG_E("ERROR: Failed to initialize clip_image_u8");
        return nullptr;
    }

    // Convert directly to uchar in-place to avoid extra Mat allocation
    cv::Mat img_uchar_rgb(img_float_rgb.size(), CV_8UC3);
    img_float_rgb.convertTo(img_uchar_rgb, CV_8UC3, 255.0);

    clip_build_img_from_pixels(img_uchar_rgb.data, img_uchar_rgb.cols, img_uchar_rgb.rows, img_clip);
    return img_clip;
}

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

    bool LoadModel();
    bool InitializeContext();
    void FreeContext();
    bool isRepetitive(const std::string &text, size_t minPatternLength = 10);
    bool isCompleteSentence(const std::string &text);

    bool setImage(cv::Mat& image);
    std::string processWithImage(const std::string& prompt);
    cv::Mat _image;
    std::string last_image_hash_;   

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

    clip_ctx* ctx_clip_ = nullptr;
    llava_image_embed cached_embed_ = {nullptr, 0};

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
    // ??? ggml_backend_load_all();

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
    ctx_clip_ = clip_model_load(mmproj_path_.c_str(), true);
    if (!ctx_clip_) {
        LOG_E("ERROR: Failed to load MMProj model\n");
        llama_model_free(model_);
        model_ = nullptr;
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
            if (repetition_count > 5) {
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

// Set image and generate embedding
bool LlamaSimpleChat::setImage(cv::Mat& image) {
    if (!ctx_ || !vocab_ || !smpl_ || !ctx_clip_) {
        LOG_E("setImage: context, vocab, sampler, or clip context not initialized.");
        return false;
    }
    if (image.empty()) {
        LOG_E("ERROR: Empty image passed to setImage");
        return false;
    }
    LOG_I("DEBUG: Input image size: " << image.cols << "x" << image.rows 
          << ", type: " << image.type() << ", channels: " << image.channels());

    saveMatAsRGB(image, "debug_input_image.png");

    std::string current_hash = computeImageHash(image);
    if (current_hash.empty()) {
        LOG_E("ERROR: Failed to compute image hash; proceeding without cache");
    } else if (current_hash == last_image_hash_ && !last_image_hash_.empty()) {
        LOG_I("DEBUG: Reusing cached embedding for hash: " << current_hash);
        return true;
    }

    if (cached_embed_.embed) {
        free(cached_embed_.embed);
        cached_embed_.embed = nullptr;
        cached_embed_.n_image_pos = 0;
    }

    cv::Mat processed_image;
    cv::resize(image, processed_image, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
    if (processed_image.empty()) {
        LOG_E("ERROR: Failed to resize image");
        return false;
    }

    cv::Mat rgb_image;
    if (processed_image.channels() == 3 && processed_image.type() == CV_8UC3) {
        cv::cvtColor(processed_image, rgb_image, cv::COLOR_BGR2RGB);
        rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);
    } else if (processed_image.type() == CV_32FC3) {
        rgb_image = processed_image;
    } else if (processed_image.channels() == 1) {
        cv::Mat normalized;
        if (processed_image.type() == CV_8UC1) {
            cv::equalizeHist(processed_image, normalized);
            normalized.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
        } else if (processed_image.type() == CV_32FC1) {
            cv::Mat temp;
            processed_image.convertTo(temp, CV_8UC1, 255.0);
            cv::equalizeHist(temp, temp);
            temp.convertTo(normalized, CV_32FC1, 1.0 / 255.0);
        } else {
            LOG_E("ERROR: Unsupported greyscale image type: " << processed_image.type());
            return false;
        }
        cv::normalize(normalized, normalized, 0.0, 1.0, cv::NORM_MINMAX);
        cv::cvtColor(normalized, rgb_image, cv::COLOR_GRAY2RGB);
    } else {
        LOG_E("ERROR: Unsupported image type: " << processed_image.type() << ", channels: " << processed_image.channels());
        return false;
    }

    cv::normalize(rgb_image, rgb_image, 0.0, 1.0, cv::NORM_MINMAX);

    // cv::Mat rgb_save;
    // rgb_image.convertTo(rgb_save, CV_8UC3, 255.0);
    // cv::imwrite("debug_processed_image.png", rgb_save);
    // LOG_I("DEBUG: Saved processed RGB image to debug_processed_image.png");

    clip_image_u8* img_clip = mat_to_clip_image_u8(rgb_image);
    if (!img_clip) {
        LOG_E("ERROR: Failed to convert image to clip_image_u8");
        return false;
    }

    //saveClipImageU8AsRGB(img_clip, "debug_clip_image.png");

    float* image_embed_ptr = nullptr;
    int n_image_pos = 0;
    if (!llava_image_embed_make_with_clip_img(ctx_clip_, std::thread::hardware_concurrency(), img_clip, &image_embed_ptr, &n_image_pos)) {
        LOG_E("ERROR: Failed to create image embedding");
        clip_image_u8_free(img_clip);
        return false;
    }
    if (n_image_pos <= 0) {
        LOG_E("ERROR: Invalid image embedding size: " << n_image_pos);
        clip_image_u8_free(img_clip);
        return false;
    }

    cached_embed_.embed = image_embed_ptr;
    cached_embed_.n_image_pos = n_image_pos;

    std::stringstream embed_log;
    embed_log << "DEBUG: First 10 embedding values: ";
    for (int i = 0; i < std::min(10, n_image_pos); ++i) {
        embed_log << cached_embed_.embed[i] << " ";
    }
    LOG_I(embed_log.str());

    clip_image_u8_free(img_clip);
    LOG_I("DEBUG: Created image embedding, n_image_pos=" << n_image_pos);

    last_image_hash_ = current_hash;
    LOG_I("DEBUG: Updated last_image_hash_: " << last_image_hash_);
    return true;
}

// Process text prompt with optional image context
std::string LlamaSimpleChat::processWithImage(const std::string& prompt) {
    if (!ctx_ || !vocab_ || !smpl_ || !ctx_clip_) {
        LOG_E("ERROR: Context, vocab, sampler, or clip context not initialized");
        return "";
    }

    if (!cached_embed_.embed || cached_embed_.n_image_pos <= 0) {
        LOG_E("ERROR: No valid image embedding available. Call setImage first");
        return "";
    }

    llama_kv_self_clear(ctx_);
    context_tokens_.clear();
    n_past_ = 0;
    LOG_I("DEBUG: Cleared KV cache and context, n_past=" << n_past_);

    int max_batch_size = llama_n_batch(ctx_);
    llama_batch batch = llama_batch_init(max_batch_size, 0, 1);
    LOG_I("DEBUG: Initialized batch with max_batch_size=" << max_batch_size);

    // Evaluate image embedding
    int n_image_pos = cached_embed_.n_image_pos;
    for (int i = 0; i < n_image_pos; i += max_batch_size) {
        int n_tokens_chunk = std::min(max_batch_size, n_image_pos - i);
        llava_image_embed chunk_embed = cached_embed_;
        chunk_embed.n_image_pos = n_tokens_chunk;
        chunk_embed.embed += i;

        if (!llava_eval_image_embed(ctx_, &chunk_embed, n_tokens_chunk, &n_past_)) {
            LOG_E("ERROR: Failed to evaluate image embed chunk " << i << " to " << i + n_tokens_chunk);
            llama_batch_free(batch);
            return "";
        }
        LOG_I("DEBUG: Evaluated image embed chunk " << i << " to " << i + n_tokens_chunk << ", n_past=" << n_past_);
    }

    // Process prompt
    std::string full_prompt = prompt + "\n\n";
    LOG_I("DEBUG: Full prompt: '" << full_prompt << "'");

    std::vector<llama_token> tokens(n_predict_);
    int n_tokens = llama_tokenize(vocab_, full_prompt.c_str(), full_prompt.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        LOG_E("ERROR: Failed to tokenize prompt");
        llama_batch_free(batch);
        return "";
    }
    tokens.resize(n_tokens);

    // Log tokenized prompt
    std::stringstream token_log;
    token_log << "DEBUG: Prompt tokens (" << n_tokens << "): ";
    for (int i = 0; i < n_tokens; ++i) {
        char piece_buf[128];
        int len = llama_token_to_piece(vocab_, tokens[i], piece_buf, sizeof(piece_buf), 0, true);
        piece_buf[std::min(len, (int)sizeof(piece_buf) - 1)] = '\0';
        token_log << tokens[i] << "='" << piece_buf << "' ";
    }
    LOG_I(token_log.str());

    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = n_past_ + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }
    if (llama_decode(ctx_, batch) != 0) {
        LOG_E("ERROR: Failed to decode prompt");
        llama_batch_free(batch);
        return "";
    }
    n_past_ += n_tokens;
    context_tokens_.insert(context_tokens_.end(), tokens.begin(), tokens.begin() + n_tokens);
    LOG_I("DEBUG: Decoded prompt, n_past=" << n_past_);

    // Initialize generation with BOS token
    batch.n_tokens = 1;
    batch.token[0] = llama_vocab_bos(vocab_);
    batch.pos[0] = n_past_;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    if (llama_decode(ctx_, batch) != 0) {
        LOG_E("ERROR: Failed to decode initial BOS token");
        llama_batch_free(batch);
        return "";
    }
    n_past_++;
    context_tokens_.push_back(batch.token[0]);
    LOG_I("DEBUG: Decoded BOS token, n_past=" << n_past_);

    // Generation loop
    std::string response;
    response.reserve(1024);
    int max_gen_tokens = 256;
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40)); // Prevent low-probability tokens
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());  // Deterministic sampling

    for (int i = 0; i < max_gen_tokens; ++i) {
        float* logits = llama_get_logits(ctx_);
        if (!logits) {
            LOG_E("ERROR: No logits available for sampling");
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            return "";
        }

        // Validate context
        if (n_past_ >= n_predict_) {
            LOG_E("ERROR: Context size exceeded: n_past=" << n_past_ << ", n_predict=" << n_predict_);
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            return "";
        }

        llama_token new_token = llama_sampler_sample(sampler, ctx_, -1);
        llama_sampler_accept(sampler, new_token);

        if (i < 5) {
            int n_vocab = llama_vocab_n_tokens(vocab_);
            std::vector<std::pair<float, int>> probs;
            for (int j = 0; j < n_vocab; ++j) {
                probs.emplace_back(logits[j], j);
            }
            std::sort(probs.begin(), probs.end(), std::greater<>());
            std::stringstream prob_log;
            prob_log << "DEBUG: Top 3 tokens at iteration " << i << ": ";
            for (int j = 0; j < 3; ++j) {
                char piece_buf[128];
                int len = llama_token_to_piece(vocab_, probs[j].second, piece_buf, sizeof(piece_buf), 0, true);
                piece_buf[std::min(len, (int)sizeof(piece_buf) - 1)] = '\0';
                prob_log << "{token=" << probs[j].second << ", prob=" << probs[j].first << ", text='" << piece_buf << "'} ";
            }
            LOG_I(prob_log.str());
        }

        if (i < 10 && (new_token == llama_vocab_eos(vocab_) || new_token == 128001)) {
            LOG_I("DEBUG: Skipping EOS token " << new_token << " at iteration " << i);
            continue;
        }

        if (new_token == llama_vocab_eos(vocab_) || new_token == 128001) {
            LOG_I("DEBUG: EOS token encountered");
            break;
        }

        char piece_buf[128];
        int len = llama_token_to_piece(vocab_, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (len < 0) {
            LOG_E("ERROR: Failed to convert token " << new_token << " to piece");
            break;
        }
        piece_buf[std::min(len, (int)sizeof(piece_buf) - 1)] = '\0';

        if (std::string(piece_buf).find("<|eot_id|>") != std::string::npos) {
            LOG_I("DEBUG: EOT token encountered");
            break;
        }

        response += piece_buf;
        LOG_I("DEBUG: Token " << new_token << ": '" << piece_buf << "'");

        if (response.length() > 1000) {
            LOG_I("DEBUG: Response length limit reached");
            break;
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_past_;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx_, batch) != 0) {
            LOG_E("ERROR: llama_decode failed during generation");
            break;
        }
        n_past_++;
        context_tokens_.push_back(new_token);
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    std::string final_response = clean_response(response);
    LOG_I("DEBUG: Final response: '" << final_response << "'");
    return final_response;
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
      LOG_I("Asking llama: " << prompt);
      _textQueue.push(std::string(prompt));
    }
  }
}

bool LlamaDeviceBase::start() {
    if (!_running) {
        _llama_chat.reset(new LlamaSimpleChat());
        _llama_chat->SetModelPaths(_model_path, "/Users/ykiryanov/Public/models/llava-llama-3-8b-v1_1-mmproj-f16.gguf");
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

bool LlamaDeviceBase::setImage(cv::Mat& image)
{
  if (_llama_chat && _llama_chat->setImage(image))
  {
      LOG_V("Llama chat image set!");
      return true;
  }
  return false;
}

void LlamaDeviceBase::processWithImage(const char *prompt) {
    if (_llama_chat) {
        _llama_chat->processWithImage(prompt);
    }
}