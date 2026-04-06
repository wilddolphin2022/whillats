/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

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
#include <cstdlib>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include "llama.h"
#include "clip.h"
#include "mtmd.h"
#include "llama_device_base.h"
#include "whisper_helpers.h"
#include "whillats_utils.h"
#include "mtmd-helper.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Strip markdown formatting so TTS speaks clean text
// e.g. "**bold**" → "bold", "*item*" → "item", "# Title" → "Title"
// ---------------------------------------------------------------------------
static std::string strip_markdown(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        // Skip heading markers at start of token (# ## ###)
        if (i == 0 || out.empty() || out.back() == '\n') {
            while (i < s.size() && s[i] == '#') ++i;
            while (i < s.size() && s[i] == ' ') ++i;
            continue;
        }
        char c = s[i];
        // Bold/italic: ** or * or _ — skip the markers, keep the text
        if ((c == '*' || c == '_') ) {
            // peek ahead to find matching marker and strip both
            size_t j = i + 1;
            bool double_marker = (j < s.size() && s[j] == c);
            if (double_marker) j++;
            // find closing
            size_t close = s.find(double_marker ? std::string(2, c) : std::string(1, c), j);
            if (close != std::string::npos) {
                // copy inner text
                out += s.substr(j, close - j);
                i = close + (double_marker ? 2 : 1);
                continue;
            }
            // no closing marker found — just skip the marker char
            i++;
            continue;
        }
        // Inline code: `text` → text
        if (c == '`') {
            size_t close = s.find('`', i + 1);
            if (close != std::string::npos) {
                out += s.substr(i + 1, close - i - 1);
                i = close + 1;
                continue;
            }
            i++;
            continue;
        }
        // Block quote marker at line start
        if (c == '>' && (i == 0 || s[i-1] == '\n')) {
            i++;
            while (i < s.size() && s[i] == ' ') ++i;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Clean model output: remove control tokens + markdown
// ---------------------------------------------------------------------------
static std::string clean_response(const std::string& response) {
    std::string cleaned = response;
    // Remove special tokens
    static const std::regex re_ctrl(
        "<\\|eot_id\\|>|<\\|im_end\\|>|<\\|endoftext\\|>|<end_of_turn>"
        "|<start_of_turn>[^<]*>?|<turn\\|>|<think>[\\s\\S]*?</think>");
    cleaned = std::regex_replace(cleaned, re_ctrl, "");
    // Strip markdown
    cleaned = strip_markdown(cleaned);
    // Trim trailing whitespace
    auto last = cleaned.find_last_not_of(" \n\r\t");
    if (last != std::string::npos)
        cleaned = cleaned.substr(0, last + 1);
    else
        cleaned.clear();
    return cleaned;
}

uint64_t fnv1a_hash_yuv(const YUVData &yuv) {
    const uint64_t FNV_offset_basis = 1469598103934665603ull;
    const uint64_t FNV_prime        = 1099511628211ull;
    uint64_t h = FNV_offset_basis;
    for (size_t i = 0; i < yuv.y_size;  ++i) h = (h ^ yuv.y[i]) * FNV_prime;
    for (size_t i = 0; i < yuv.uv_size; ++i) h = (h ^ yuv.u[i]) * FNV_prime;
    for (size_t i = 0; i < yuv.uv_size; ++i) h = (h ^ yuv.v[i]) * FNV_prime;
    return h;
}

// ============================================================================
// LlamaSimpleChat
// ============================================================================

LlamaSimpleChat::LlamaSimpleChat() = default;

LlamaSimpleChat::~LlamaSimpleChat() {
    if (smpl_) llama_sampler_free(smpl_);
    FreeContext();
    if (model_) llama_model_free(model_);
}

bool LlamaSimpleChat::SetModelPaths(const std::string &path, const std::string &mmproj_path) {
    model_path_ = path;
    mmproj_path_ = mmproj_path;
    return true;
}

bool LlamaSimpleChat::SetNGL(int layers) { ngl_ = layers; return true; }
bool LlamaSimpleChat::SetContextSize(int size) { n_predict_ = size; return true; }
void LlamaSimpleChat::StopGeneration() { continue_ = false; }

bool LlamaSimpleChat::Initialize() {
    llama_backend_init();
    if (!LoadModel()) { LOG_E("Failed to load model."); return false; }
    if (!InitializeContext()) { LOG_E("Failed to initialize context."); return false; }
    if (smpl_) llama_sampler_free(smpl_);
    smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!smpl_) { LOG_E("Failed to initialize sampler."); return false; }
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    DetectStoppingTokens();
    DetectChatFormat();
    return true;
}

bool LlamaSimpleChat::LoadModel() {
    if (model_path_.empty()) { LOG_E("Model path not set."); return false; }

    llama_model_params model_params = llama_model_default_params();
#ifdef GGML_USE_METAL
    const char *env_ngl = std::getenv("LLAMA_METAL_GPU_LAYERS");
    bool env_valid = false;
    if (env_ngl) {
        int env_val = std::atoi(env_ngl);
        if (env_val >= 0) { model_params.n_gpu_layers = env_val; env_valid = true; }
    }
    if (!env_valid) {
        size_t mem_bytes = 0;
#ifdef __APPLE__
        size_t len = sizeof(mem_bytes);
        sysctlbyname("hw.memsize", &mem_bytes, &len, nullptr, 0);
#endif
        size_t mem_gb = mem_bytes / (1024ULL * 1024ULL * 1024ULL);
        int max_layers = (mem_gb >= 32) ? 32 : (mem_gb >= 24) ? 24 :
                         (mem_gb >= 16) ? 16 : (mem_gb >= 12) ? 12 :
                         (mem_gb >= 8)  ? 8  : 0;
        model_params.n_gpu_layers = std::min(ngl_, max_layers);
    }
#elif defined(GGML_USE_CUDA)
    size_t free_mem, total_mem;
    cudaError_t cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    int max_layers = 0;
    if (cuda_err == cudaSuccess) {
        size_t available_mb = free_mem / (1024 * 1024);
        size_t total_mb = total_mem / (1024 * 1024);
        if (total_mb <= 4096) max_layers = 0;
        else if (available_mb >= 11000) max_layers = 30;
        else if (available_mb >= 9000)  max_layers = 20;
        else if (available_mb >= 7000)  max_layers = 5;
        else if (available_mb >= 5000)  max_layers = 2;
        else max_layers = 0;
    }
    model_params.n_gpu_layers = std::min(ngl_, max_layers);
    model_params.main_gpu = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
#else
    model_params.n_gpu_layers = 0;
#endif

    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (!model_) { LOG_E("Unable to load model."); return false; }
    vocab_ = llama_model_get_vocab(model_);
    return true;
}

bool LlamaSimpleChat::InitializeContext() {
    if (ctx_) FreeContext();
    if (!model_ || !vocab_) { LOG_E("Model or vocab not loaded."); return false; }

    if (context_tokens_.empty()) {
        const int n_prompt = -llama_tokenize(vocab_, prompt_.c_str(), prompt_.size(),
                                              nullptr, 0, true, true);
        if (n_prompt < 0) { LOG_E("Failed to count prompt tokens."); return false; }
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab_, prompt_.c_str(), prompt_.size(),
                           prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            LOG_E("Failed to tokenize prompt."); return false;
        }
        context_tokens_ = std::deque<llama_token>(prompt_tokens.begin(), prompt_tokens.end());
        n_past_ = 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
#ifdef GGML_USE_METAL
    ctx_params.n_ctx   = std::min(n_predict_, 1024);
    ctx_params.n_batch = 256;
#else
    ctx_params.n_ctx   = std::min(n_predict_, 2048);
    ctx_params.n_batch = 512;
#endif
    ctx_params.no_perf = false;

#if defined(__APPLE__)
    int perf_cores = 0;
    size_t len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &len, NULL, 0) == 0 && perf_cores > 0)
        ctx_params.n_threads = perf_cores;
    else
        ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
#else
    ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
#endif
    if (n_threads_ > 0) ctx_params.n_threads = n_threads_;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) { LOG_E("Failed to create llama_context."); return false; }

    if (!mmproj_path_.empty()) {
        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.n_threads = ctx_params.n_threads;
        mtmd_params.use_gpu   = true;
        ctx_mtmd_.reset(mtmd_init_from_file(mmproj_path_.c_str(), model_, mtmd_params));
        if (!ctx_mtmd_) {
            LOG_E("Failed to load MMProj model");
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
    }

    if (!context_tokens_.empty() && n_past_ == 0) {
        int n_eval  = context_tokens_.size();
        int n_batch = ctx_params.n_batch;
        for (int i = 0; i < n_eval; i += n_batch) {
            int n_tokens = std::min(n_batch, n_eval - i);
            struct llama_batch batch = llama_batch_get_one(&context_tokens_[i], n_tokens);
            if (llama_decode(ctx_, batch)) { LOG_E("Failed to decode context."); FreeContext(); return false; }
            n_past_ += n_tokens;
        }
        if (smpl_) llama_sampler_reset(smpl_);
    }
    return true;
}

void LlamaSimpleChat::FreeContext() {
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    ctx_mtmd_.reset();
}

bool LlamaSimpleChat::ResetContextForImage() {
    if (!ctx_) { LOG_E("Context not initialized"); return false; }
    llama_memory_t memory = llama_get_memory(ctx_);
    llama_memory_clear(memory, true);
    context_tokens_.clear();
    n_past_ = 0;
    if (smpl_) llama_sampler_reset(smpl_);
    return true;
}

bool LlamaSimpleChat::isRepetitive(const std::string &text, size_t minPatternLength) {
    if (text.length() < minPatternLength * 2) return false;
    for (size_t len = minPatternLength; len <= text.length() / 2; ++len) {
        std::string last = text.substr(text.length() - len);
        if (text.rfind(last, text.length() - len - 1) != std::string::npos) return true;
    }
    return false;
}

bool LlamaSimpleChat::isCompleteSentence(const std::string &text) {
    if (text.empty()) return false;
    char last = text.back();
    return (last == '.' || last == '!' || last == '?') &&
           !std::all_of(text.begin(), text.end(), isspace);
}

// ---------------------------------------------------------------------------
// System prompt: Gemma-4 format, no markdown, voice-friendly
// ---------------------------------------------------------------------------
static std::string make_system_prompt(LlamaSimpleChat::ChatFormat fmt) {
    const std::string instruction =
        "You are a helpful multilingual voice assistant. "
        "Always respond in the same language the user speaks. "
        "Keep answers short and conversational. "
        "Never use markdown, bullet points, asterisks, or any special formatting. "
        "Speak in plain sentences only.";

    if (fmt == LlamaSimpleChat::ChatFormat::GEMMA) {
        // Gemma has no system role — embed as a priming exchange
        return "<start_of_turn>user\n" + instruction + "<end_of_turn>\n"
               "<start_of_turn>model\nUnderstood. I will respond naturally in plain speech.<end_of_turn>\n";
    } else if (fmt == LlamaSimpleChat::ChatFormat::CHATML) {
        return "<|im_start|>system\n" + instruction + "<|im_end|>\n";
    } else {
        return "<|start_header_id|>system<|end_header_id|>\n\n" + instruction + "<|eot_id|>\n";
    }
}

// ---------------------------------------------------------------------------
// generate() — text-only inference, sentence-by-sentence streaming to TTS
// ---------------------------------------------------------------------------
std::string LlamaSimpleChat::generate(const std::string &prompt,
                                       WhillatsSetResponseCallback callback) {
    if (!ctx_ || !vocab_ || !smpl_) {
        LOG_E("generate: context/vocab/sampler not ready.");
        return "";
    }

    // Initialize context with system prompt on first call
    if (context_tokens_.empty()) {
        // DetectChatFormat already run; build proper system prompt
        std::string system_prompt = make_system_prompt(chat_format_);
        const int n_sp = -llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.size(),
                                          nullptr, 0, true, true);
        if (n_sp > 0) {
            std::vector<llama_token> sp_tokens(n_sp);
            if (llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.size(),
                               sp_tokens.data(), sp_tokens.size(), true, true) >= 0) {
                context_tokens_ = std::deque<llama_token>(sp_tokens.begin(), sp_tokens.end());
                n_past_ = 0;
                int n_batch = llama_context_default_params().n_batch;
                for (int i = 0; i < (int)context_tokens_.size(); i += n_batch) {
                    int n_tok = std::min(n_batch, (int)context_tokens_.size() - i);
                    struct llama_batch batch = llama_batch_get_one(&context_tokens_[i], n_tok);
                    if (llama_decode(ctx_, batch)) { LOG_E("Failed to decode system prompt."); return ""; }
                    n_past_ += n_tok;
                }
            }
        }
    }

    // Wrap prompt in chat template
    std::string wrapped;
    if (chat_format_ == ChatFormat::GEMMA) {
        wrapped = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
    } else if (chat_format_ == ChatFormat::CHATML) {
        wrapped = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        wrapped = "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
                  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    const int n_tokens = -llama_tokenize(vocab_, wrapped.c_str(), wrapped.size(), nullptr, 0, false, false);
    if (n_tokens < 0) { LOG_E("Failed to count prompt tokens."); return ""; }
    std::vector<llama_token> prompt_tokens(n_tokens);
    if (llama_tokenize(vocab_, wrapped.c_str(), wrapped.size(),
                       prompt_tokens.data(), prompt_tokens.size(), false, false) < 0) {
        LOG_E("Failed to tokenize prompt."); return "";
    }

    context_tokens_.insert(context_tokens_.end(), prompt_tokens.begin(), prompt_tokens.end());

    // Trim context if too long
    if ((int)context_tokens_.size() > n_predict_) {
        int excess = context_tokens_.size() - n_predict_;
        context_tokens_.erase(context_tokens_.begin(), context_tokens_.begin() + excess);
        n_past_ = std::max(0, n_past_ - excess);
        if (!InitializeContext()) { LOG_E("Context reinit failed."); return ""; }
    }

    // Decode new prompt tokens
    if (!prompt_tokens.empty()) {
        struct llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());
        if (llama_decode(ctx_, batch)) { LOG_E("Failed to decode prompt."); return ""; }
        n_past_ += (int)prompt_tokens.size();
    }

    std::string response;
    std::string current_phrase;
    std::string recent_text;
    continue_ = true;
    int generated = 0;
    int repetition_count = 0;
    const int max_tokens = 512;

    _lastResponseStart = std::chrono::steady_clock::now();

    while (continue_ && generated < max_tokens) {
        if (!smpl_ || !ctx_) break;

        float *logits = llama_get_logits_ith(ctx_, -1);
        if (!logits) break;

        int n_vocab = llama_vocab_n_tokens(vocab_);
        std::vector<llama_token_data> candidates(n_vocab);
        for (int i = 0; i < n_vocab; ++i) candidates[i] = {i, logits[i], 0.0f};
        llama_token_data_array cur_p = {candidates.data(), (size_t)n_vocab, -1, false};
        llama_sampler_apply(smpl_, &cur_p);

        if (cur_p.size == 0 || cur_p.selected < 0 || cur_p.selected >= (int64_t)cur_p.size) break;

        llama_token new_token_id = cur_p.data[cur_p.selected].id;
        if (llama_vocab_is_eog(vocab_, new_token_id) ||
            stopping_token_ids_.find(new_token_id) != stopping_token_ids_.end()) break;

        char piece_buf[64];
        int piece_len = llama_token_to_piece(vocab_, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len < 0) break;

        std::string piece(piece_buf, piece_len);

        // Check for stopping strings
        bool stop = false;
        for (const auto &ss : stopping_token_strings_) {
            if (piece.find(ss) != std::string::npos) { stop = true; break; }
        }
        if (stop) break;

        current_phrase += piece;
        recent_text    += piece;
        if (recent_text.length() > 50) recent_text = recent_text.substr(recent_text.length() - 50);

        context_tokens_.push_back(new_token_id);
        llama_sampler_accept(smpl_, new_token_id);

        if (isRepetitive(recent_text, 5) && ++repetition_count > 2) break;
        else if (!isRepetitive(recent_text, 5)) repetition_count = 0;

        // Flush complete sentences to TTS immediately
        if (isCompleteSentence(current_phrase)) {
            std::string cleaned = clean_response(current_phrase);
            if (!cleaned.empty()) {
                callback.OnResponseComplete(true, cleaned.c_str());
                LOG_I("LlamaSimpleChat: '" << cleaned << "' in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - _lastResponseStart).count() << "ms");
            }
            response += current_phrase;
            current_phrase.clear();
        }

        struct llama_batch batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx_, batch)) { LOG_E("Failed to decode token."); break; }
        n_past_++;
        generated++;
    }

    if (!current_phrase.empty()) {
        response += current_phrase;
        std::string cleaned = clean_response(current_phrase);
        if (!cleaned.empty()) callback.OnResponseComplete(true, cleaned.c_str());
    }

    return clean_response(response);
}

// ---------------------------------------------------------------------------
// generateFromImage() — multimodal inference via mtmd
// ---------------------------------------------------------------------------
std::string LlamaSimpleChat::generateFromImage(YUVData* yuv, const std::string& prompt,
                                                WhillatsSetResponseCallback callback) {
    continue_ = true;
    if (!ctx_ || !vocab_ || !smpl_ || !ctx_mtmd_) {
        LOG_E("generateFromImage: not initialized");
        return "";
    }
    if (!yuv || !yuv->y || !yuv->u || !yuv->v) { LOG_E("Invalid YUV"); return ""; }

    _lastResponseStart = std::chrono::steady_clock::now();
    if (!ResetContextForImage()) { LOG_E("ResetContextForImage failed"); return ""; }

    const std::string sys = "You are a helpful visual assistant. "
                            "Describe what you see briefly and answer the user's question. "
                            "Use plain speech only, no markdown or formatting symbols.";
    int n_ctx = llama_n_ctx(ctx_);
    std::vector<llama_token> sys_tokens(n_ctx);
    int n_sys = llama_tokenize(vocab_, sys.c_str(), sys.size(), sys_tokens.data(), sys_tokens.size(), true, false);
    if (n_sys <= 0) { LOG_E("Failed to tokenize system prompt"); return ""; }
    sys_tokens.resize(n_sys);
    context_tokens_ = std::deque<llama_token>(sys_tokens.begin(), sys_tokens.end());

    int max_batch = 512;
#ifdef GGML_USE_METAL
    max_batch = 256;
#endif
    llama_batch batch = llama_batch_init(max_batch, 0, 1);
    if (!batch.token) { LOG_E("Failed to init batch"); return ""; }
    batch.n_tokens = n_sys;
    for (int i = 0; i < n_sys; ++i) {
        batch.token[i]      = sys_tokens[i];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = (i == n_sys - 1);
    }
    if (llama_decode(ctx_, batch)) { LOG_E("System prompt decode failed"); llama_batch_free(batch); return ""; }
    n_past_ = n_sys;

    // Convert YUV → clip_image_u8
    clip_image_u8* img_clip = yuv_to_clip(*yuv);
    if (!img_clip) { LOG_E("YUV→clip failed"); llama_batch_free(batch); return ""; }

    mtmd::bitmap bitmap(img_clip->width, img_clip->height, img_clip->data);
    free_clip(img_clip);
    if (!bitmap.ptr) { LOG_E("Failed to create bitmap"); llama_batch_free(batch); return ""; }

    std::string full_prompt;
    if (chat_format_ == ChatFormat::GEMMA) {
        full_prompt = "<start_of_turn>user\n" + std::string(MTMD_DEFAULT_IMAGE_MARKER)
                    + " " + prompt + "<end_of_turn>\n<start_of_turn>model\n";
    } else if (chat_format_ == ChatFormat::CHATML) {
        full_prompt = "<|im_start|>user\n" + std::string(MTMD_DEFAULT_IMAGE_MARKER)
                    + " " + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        full_prompt = "<|start_header_id|>user<|end_header_id|>\n\n"
                    + std::string(MTMD_DEFAULT_IMAGE_MARKER) + " " + prompt
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    mtmd_input_text input_text = { full_prompt.c_str(), true, false };
    std::vector<const mtmd_bitmap*> bitmaps = { bitmap.ptr.get() };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) { LOG_E("Failed to init chunks"); llama_batch_free(batch); return ""; }
    if (mtmd_tokenize(ctx_mtmd_.get(), chunks.ptr.get(), &input_text, bitmaps.data(), bitmaps.size()) != 0) {
        LOG_E("mtmd_tokenize failed"); llama_batch_free(batch); return "";
    }

    llama_pos new_n_past = n_past_;
    if (mtmd_helper_eval_chunks(ctx_mtmd_.get(), ctx_, chunks.ptr.get(), n_past_, 0, max_batch, true, &new_n_past) != 0) {
        LOG_E("mtmd eval failed"); llama_batch_free(batch); return "";
    }
    n_past_ = new_n_past;
    llama_batch_free(batch);
    if (smpl_) llama_sampler_reset(smpl_);

    std::string response;
    std::string current_phrase;
    std::string recent_text;
    int generated = 0;
    int repetition_count = 0;
    const int max_tokens = 128;
    const int min_gen    = 10;

    while (continue_ && generated < max_tokens) {
        if (!smpl_ || !ctx_) break;
        float *logits = llama_get_logits_ith(ctx_, -1);
        if (!logits) break;

        int n_vocab = llama_vocab_n_tokens(vocab_);
        std::vector<llama_token_data> candidates(n_vocab);
        for (int i = 0; i < n_vocab; ++i) candidates[i] = {i, logits[i], 0.0f};
        llama_token_data_array cur_p = {candidates.data(), (size_t)n_vocab, -1, false};
        llama_sampler_apply(smpl_, &cur_p);
        if (cur_p.size == 0 || cur_p.selected < 0) break;

        llama_token new_token_id = cur_p.data[cur_p.selected].id;
        if (llama_vocab_is_eog(vocab_, new_token_id)) break;
        if (generated > min_gen && stopping_token_ids_.find(new_token_id) != stopping_token_ids_.end()) break;

        char piece_buf[64];
        int piece_len = llama_token_to_piece(vocab_, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len < 0) break;
        std::string piece(piece_buf, piece_len);

        if (generated > min_gen) {
            bool stop = false;
            for (const auto &ss : stopping_token_strings_)
                if (piece.find(ss) != std::string::npos) { stop = true; break; }
            if (stop) break;
        }

        current_phrase += piece;
        recent_text    += piece;
        if (recent_text.length() > 50) recent_text = recent_text.substr(recent_text.length() - 50);

        context_tokens_.push_back(new_token_id);
        llama_sampler_accept(smpl_, new_token_id);

        if (isRepetitive(recent_text, 5) && ++repetition_count > 2) break;
        else if (!isRepetitive(recent_text, 5)) repetition_count = 0;

        if (isCompleteSentence(current_phrase)) {
            std::string cleaned = clean_response(current_phrase);
            if (!cleaned.empty()) callback.OnResponseComplete(true, cleaned.c_str());
            response += current_phrase;
            current_phrase.clear();
        }

        llama_batch tok_batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx_, tok_batch)) { LOG_E("Decode failed in image loop."); break; }
        n_past_++;
        generated++;
    }

    if (!current_phrase.empty()) {
        response += current_phrase;
        std::string cleaned = clean_response(current_phrase);
        if (!cleaned.empty()) callback.OnResponseComplete(true, cleaned.c_str());
    }

    return clean_response(response);
}

void LlamaSimpleChat::DetectStoppingTokens() {
    if (!vocab_) return;
    stopping_token_ids_.clear();
    stopping_token_strings_.clear();

    static const std::vector<std::string> known = {
        "<|eot_id|>", "<|end_of_text|>", "<|end|>", "</s>",
        "<|im_end|>", "<|endoftext|>", "</think>",
        "<end_of_turn>", "<turn|>"
    };

    int n_vocab = llama_vocab_n_tokens(vocab_);
    for (int id = 0; id < n_vocab; ++id) {
        char buf[128] = {};
        int len = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
        if (len < 0) continue;
        buf[std::min(len, (int)sizeof(buf)-1)] = '\0';
        std::string ts(buf);
        for (const auto &s : known) {
            if (ts.find(s) != std::string::npos) {
                stopping_token_ids_.insert(id);
                stopping_token_strings_.push_back(s);
                break;
            }
        }
    }
    llama_token eos = llama_vocab_eos(vocab_);
    if (eos != -1) stopping_token_ids_.insert(eos);
}

void LlamaSimpleChat::DetectChatFormat() {
    if (!vocab_) return;
    auto probe = [&](const char* text) -> bool {
        llama_token buf;
        return llama_tokenize(vocab_, text, strlen(text), &buf, 1, false, true) == 1;
    };
    if (probe("<start_of_turn>")) { chat_format_ = ChatFormat::GEMMA;  LOG_I("Chat format: Gemma"); return; }
    if (probe("<|im_start|>"))    { chat_format_ = ChatFormat::CHATML; LOG_I("Chat format: ChatML"); return; }
    if (probe("<|start_header_id|>")) { chat_format_ = ChatFormat::LLAMA3; LOG_I("Chat format: Llama3"); return; }
    chat_format_ = ChatFormat::CHATML;
    LOG_I("Chat format: ChatML (default)");
}

// ============================================================================
// LlamaDeviceBase
// ============================================================================

LlamaDeviceBase::LlamaDeviceBase(const char* model_path,
                                  const char* mmproj_path,
                                  WhillatsSetResponseCallback callback)
    : _model_path(model_path ? model_path : "")
    , _mmproj_path(mmproj_path ? mmproj_path : "")
    , _responseCallback(callback)
    , _hasMultimodalModel(false)
    , _imageRetentionMs(5000)
{
    LOG_I("LlamaDeviceBase: model=" << _model_path
          << " mmproj=" << (_mmproj_path.empty() ? "(none)" : _mmproj_path));
}

LlamaDeviceBase::~LlamaDeviceBase() {
    if (_destructing_.exchange(true)) return;
    stop();
}

bool LlamaDeviceBase::start() {
    if (_running) return true;

    _chat = std::make_unique<LlamaSimpleChat>();
    _chat->SetModelPaths(_model_path, _mmproj_path);
    _chat->SetNGL(32);

    LOG_I("LlamaDeviceBase: loading model...");
    if (!_chat->Initialize()) {
        LOG_E("LlamaDeviceBase: model init failed");
        _chat.reset();
        return false;
    }

    // Detect multimodal support
    if (!_mmproj_path.empty() && _chat->ctx_mtmd_) {
        _hasMultimodalModel = mtmd_support_vision(_chat->ctx_mtmd_.get());
        LOG_I("LlamaDeviceBase: multimodal=" << (_hasMultimodalModel ? "yes" : "no"));
    }

    _running = true;
    _processingThread = std::thread([this]() { RunProcessingThread(); });
    LOG_I("LlamaDeviceBase: started");
    return true;
}

void LlamaDeviceBase::stop() {
    if (!_running) return;
    _running = false;
    if (_chat) _chat->StopGeneration();
    _queueCondition.notify_all();
    if (_processingThread.joinable()) {
        if (std::this_thread::get_id() != _processingThread.get_id())
            _processingThread.join();
    }
}

void LlamaDeviceBase::askLlama(const char* prompt) {
    if (!prompt || !*prompt || !_running) return;
    if (_chat) _chat->StopGeneration();

    Request req;
    req.prompt = prompt;
    req.yuv    = nullptr;
    if (_hasMultimodalModel) {
        std::lock_guard<std::mutex> lk(_frameMutex);
        auto now = std::chrono::steady_clock::now();
        if (_lastFrame && (now - _lastFrameTime) <= std::chrono::milliseconds(_imageRetentionMs)) {
            req.yuv = _lastFrame;
        }
    }

    {
        std::lock_guard<std::mutex> lk(_queueMutex);
        _requestQueue.clear(); // cancel pending requests
        _requestQueue.push_back(std::move(req));
    }
    _queueCondition.notify_one();
}

void LlamaDeviceBase::receiveVideoFrame(const YUVData& yuv) {
    auto frame = std::make_shared<YUVData>();
    frame->width  = yuv.width;
    frame->height = yuv.height;
    frame->y_size  = yuv.y_size;
    frame->uv_size = yuv.uv_size;
    frame->y = std::make_unique<uint8_t[]>(yuv.y_size);
    frame->u = std::make_unique<uint8_t[]>(yuv.uv_size);
    frame->v = std::make_unique<uint8_t[]>(yuv.uv_size);
    memcpy(frame->y.get(), yuv.y.get(), yuv.y_size);
    memcpy(frame->u.get(), yuv.u.get(), yuv.uv_size);
    memcpy(frame->v.get(), yuv.v.get(), yuv.uv_size);

    std::lock_guard<std::mutex> lk(_frameMutex);
    _lastFrame     = std::move(frame);
    _lastFrameTime = std::chrono::steady_clock::now();
}

bool LlamaDeviceBase::RunProcessingThread() {
    while (_running) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(_queueMutex);
            _queueCondition.wait(lk, [this] {
                return !_requestQueue.empty() || !_running;
            });
            if (!_running) break;
            if (_requestQueue.empty()) continue;
            req = std::move(_requestQueue.front());
            _requestQueue.pop_front();
        }

        LOG_I("LlamaDeviceBase: prompt='" << req.prompt.substr(0, 80) << "'"
              << (req.yuv ? " [+image]" : ""));

        if (req.yuv && _hasMultimodalModel) {
            _chat->generateFromImage(req.yuv.get(), req.prompt, _responseCallback);
        } else {
            _chat->generate(req.prompt, _responseCallback);
        }
        _responseCallback.OnResponseComplete(false, nullptr); // signal done
    }
    return true;
}
