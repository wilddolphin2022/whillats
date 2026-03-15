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

// Implementation of LlamaSimpleChat declared in llama_device_base.h

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
#ifdef GGML_USE_METAL
    // On Apple Silicon we use the Metal backend. While the unified memory pool is
    // large, the per-command-buffer memory available to the GPU is noticeably
    // lower than on discrete GPUs.  Trying to off-load the full model (all 32
    // layers for an 8B model) regularly causes `command buffer … out of memory`
    // errors on machines such as the Mac mini M4.

    // Allow the user to override the number of GPU layers through the
    // environment variable `LLAMA_METAL_GPU_LAYERS`.  If it is present and
    // parses to a non-negative integer, use that value *exactly*.
    const char *env_ngl = std::getenv("LLAMA_METAL_GPU_LAYERS");
    bool env_valid = false;
    if (env_ngl) {
        int env_val = std::atoi(env_ngl);
        if (env_val >= 0) {
            model_params.n_gpu_layers = env_val;
            env_valid = true;
            LOG_I("[Metal] n_gpu_layers overridden by env → " << env_val);
        }
    }

    // If the env var wasn't provided, or contained an invalid value, apply a
    // memory-based heuristic.
    if (!env_valid) {
        // Heuristic: cap the number of GPU layers based on the amount of
        // physical memory.  The unified memory size is an upper bound – we use
        // conservative limits to keep peak GPU working-set under ~6 GiB which
        // has proven stable on 8-10 GiB GPUs.

        size_t mem_bytes = 0;
#ifdef __APPLE__
        size_t len = sizeof(mem_bytes);
        sysctlbyname("hw.memsize", &mem_bytes, &len, nullptr, 0);
#endif

        size_t mem_gb = mem_bytes / (1024ULL * 1024ULL * 1024ULL);

        int max_layers = 0;
        if (mem_gb >= 32) {
            max_layers = 32; // Plenty of memory – allow full offload
        } else if (mem_gb >= 24) {
            max_layers = 24;
        } else if (mem_gb >= 16) {
            max_layers = 16;
        } else if (mem_gb >= 12) {
            max_layers = 12;
        } else if (mem_gb >= 8) {
            max_layers = 8;
        } else {
            max_layers = 0; // fall back to CPU-only if very low memory
        }

        model_params.n_gpu_layers = std::min(ngl_, max_layers);
        LOG_I("[Metal] hw.memsize=" << mem_gb << " GiB, limiting GPU layers to " << model_params.n_gpu_layers);
    }
#elif defined(GGML_USE_CUDA)
    // CUDA on Linux - dynamically detect available memory
    size_t free_mem, total_mem;
    cudaError_t cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    
    int max_layers = 0;
    if (cuda_err == cudaSuccess) {
        size_t available_mb = free_mem / (1024 * 1024);
        size_t total_mb = total_mem / (1024 * 1024);
        
        LOG_I("CUDA Memory: " << available_mb << "MB free, " << total_mb << "MB total");
        
        // For RTX 3050 (4GB VRAM) and similar cards, force CPU-only for large models
        // This prevents CUDA allocation failures
        if (total_mb <= 4096) {  // 4GB or less VRAM
            LOG_I("Detected low VRAM GPU (" << total_mb << "MB), forcing CPU-only mode for stability");
            max_layers = 0;
        } else if (available_mb >= 11000) {
            max_layers = 30;  // 11+ GiB free
        } else if (available_mb >= 9000) {
            max_layers = 20;  // 9–11 GiB free
        } else if (available_mb >= 7000) {
            max_layers = 5;   // 7–9 GiB free
        } else if (available_mb >= 5000) {
            max_layers = 2;   // 5–7 GiB free
        } else {
            max_layers = 0;   // Force CPU-only if very low memory
        }
        LOG_I("Setting max GPU layers to " << max_layers << " (total VRAM: " << total_mb << "MB, available: " << available_mb << "MB)");
    } else {
        LOG_W("Failed to query CUDA memory, falling back to CPU-only mode");
        max_layers = 0;  // Conservative fallback - CPU only
    }
    
    model_params.n_gpu_layers = std::min(ngl_, max_layers);
    model_params.main_gpu = 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;  // Split by layer for better memory management
#else
    // CPU-only mode when no GPU backend available
    model_params.n_gpu_layers = 0;
#endif
    
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
    // Use a smaller context window on Metal to reduce memory traffic and speed
    // up evaluation.  1k tokens is more than enough for a single image + prompt
    // interaction and halves KV-cache bandwidth compared to 2k.
#ifdef GGML_USE_METAL
    ctx_params.n_ctx = std::min(n_predict_, 1024);
    ctx_params.n_batch = 256;
#else
    ctx_params.n_ctx = std::min(n_predict_, 2048);  // Default for other back-ends
    ctx_params.n_batch = 512;
#endif
    ctx_params.no_perf = false;
    // Use as many physical cores as are available on the machine instead of
    // the previous hard-cap of 4.  On Apple Silicon machines like the M4 Mac
    // mini this unlocks the additional high-performance cores and noticeably
    // reduces the per-batch evaluation latency for image decoding.
#if defined(__APPLE__)
    // macOS exposes both performance and efficiency cores via sysctl.  Query
    // the "hw.perflevel0.logicalcpu" key first; if that fails, fall back to
    // the generic std::thread::hardware_concurrency().
    int perf_cores = 0;
    size_t len = sizeof(perf_cores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &perf_cores, &len, NULL, 0) == 0 && perf_cores > 0) {
        ctx_params.n_threads = perf_cores;
    } else {
        ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
    }
#else
    ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
#endif

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

bool LlamaSimpleChat::ResetContextForImage() {
    if (!ctx_) {
        LOG_E("Context not initialized");
        return false;
    }
    
    // Clear KV cache to reset conversation state
    llama_memory_t memory = llama_get_memory(ctx_);
    llama_memory_clear(memory, true);
    
    // Reset our tracking variables
    context_tokens_.clear();
    n_past_ = 0;
    
    // Reset sampler state
    if (smpl_) {
        llama_sampler_reset(smpl_);
    }
    
    LOG_V("Reset context for image processing");
    return true;
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

    if (!current_phrase.empty()) {
        response += current_phrase;
        callback.OnResponseComplete(true, current_phrase.c_str());
    }

    std::string full_response = clean_response(response);
    auto t0 = std::chrono::steady_clock::now();
    LOG_I("Image+answer in "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now()-t0).count()
          << " ms");

    return full_response;
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

    // Reset context state for fresh image processing
    if (!ResetContextForImage()) {
        LOG_E("Failed to reset context for image processing");
        return "";
    }
    
    // Initialize with a simple system prompt for image description
    const std::string system_prompt = "You are a helpful assistant for image description.";
    int n_ctx = llama_n_ctx(ctx_);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(vocab_, system_prompt.c_str(), system_prompt.length(), tokens.data(), tokens.size(), true, false);
    if (n_tokens <= 0) {
        LOG_E("Failed to tokenize system prompt");
        return "";
    }
    tokens.resize(n_tokens);
    context_tokens_ = std::deque<llama_token>(tokens.begin(), tokens.end());

    // Process system prompt with proper sequence setup
    int max_batch_size = 512;
    #ifdef GGML_USE_METAL
    // Match ctx_params.n_batch to avoid n_tokens_batch > n_batch assertion
    max_batch_size = 256;
    #endif
    llama_batch batch = llama_batch_init(max_batch_size, 0, 1);
    if (!batch.token) {
        LOG_E("Failed to initialize batch");
        return "";
    }
    
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i; // Start from position 0 since context is cleared
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1); // Only last token needs logits
    }
    
    int decode_result = llama_decode(ctx_, batch);
    if (decode_result != 0) {
        LOG_E("Failed to decode system prompt, error code: " << decode_result);
        llama_batch_free(batch);
        return "";
    }
    n_past_ = n_tokens; // Set to the number of tokens processed, not add
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

    // Compose the prompt using the same chat-template that llama.cpp employs.
    // <|start_header_id|>user<|end_header_id|>  <image + question>  <|eot_id|>
    // <|start_header_id|>assistant<|end_header_id|>
    const std::string header_user      = "<|start_header_id|>user<|end_header_id|>\n\n";
    const std::string header_assistant = "<|start_header_id|>assistant<|end_header_id|>\n\n";

    std::string full_prompt = header_user + std::string(MTMD_DEFAULT_IMAGE_MARKER) + " " + prompt + "<|eot_id|>" + header_assistant;
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

    // Generation loop - reuse the existing sampler (smpl_) for consistency with text generation
    if (smpl_) {
        llama_sampler_reset(smpl_);
    }

    std::string current_phrase;
    std::string response;
    std::string recent_text;
    continue_ = true;

    const int max_response_tokens = 64;
    int generated_tokens = 0;
    int repetition_count = 0;
    const int min_gen_tokens = 10;

    while (continue_ && generated_tokens < max_response_tokens) {
        if (!smpl_ || !ctx_) {
            LOG_E("Sampler or context became null during image generation.");
            break;
        }

        float *logits = llama_get_logits_ith(ctx_, -1);
        if (!logits) {
            LOG_E("Failed to get logits for sampling (image).");
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
            LOG_E("Invalid sampling result for image loop.");
            break;
        }

        llama_token new_token_id = cur_p.data[cur_p.selected].id;
        if (generated_tokens > min_gen_tokens && stopping_token_ids_.find(new_token_id) != stopping_token_ids_.end()) {
            LOG_V("Image generation reached stopping token ID: " << new_token_id);
            break;
        }

        char token_text[64];
        int token_text_len = llama_token_to_piece(vocab_, new_token_id, token_text, sizeof(token_text), 0, true);
        if (token_text_len < 0) {
            LOG_E("Failed to convert token " << new_token_id << " to piece (image loop).");
            break;
        }
        std::string piece(token_text, token_text_len);
        current_phrase += piece;
        recent_text += piece;

        if (generated_tokens > min_gen_tokens) {
            for (const auto &stop_str : stopping_token_strings_) {
                if (piece.find(stop_str) != std::string::npos) {
                    LOG_V("Stopping token string '" << stop_str << "' encountered in image loop.");
                    continue_ = false;
                    break;
                }
            }
            if (!continue_) break;
        }

        if (recent_text.length() > 50) {
            recent_text = recent_text.substr(recent_text.length() - 50);
        }

        context_tokens_.push_back(new_token_id);
        llama_sampler_accept(smpl_, new_token_id);

        if (isRepetitive(recent_text, 5)) {
            repetition_count++;
            if (repetition_count > 2) {
                LOG_V("Stopping due to repetitive output during image generation.");
                break;
            }
        } else {
            repetition_count = 0;
        }

        if (isCompleteSentence(current_phrase)) {
            callback.OnResponseComplete(true, current_phrase.c_str());
            LOG_V("Partial image description: " << current_phrase);
            response += current_phrase;
            current_phrase.clear();
        }

        // feed back token
        llama_batch batch_tok = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx_, batch_tok)) {
            LOG_E("llama_decode failed during image generation.");
            break;
        }
        n_past_++;
        generated_tokens++;
    }

    if (!current_phrase.empty()) {
        response += current_phrase;
        callback.OnResponseComplete(true, current_phrase.c_str());
    }

    std::string full_response = clean_response(response);
    auto t0 = std::chrono::steady_clock::now();
    LOG_I("Image+answer in "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now()-t0).count()
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

// LlamaDeviceBase implementation
LlamaDeviceBase::LlamaDeviceBase(
    const char* model_path,
    const char* mmproj_path, 
    WhillatsSetResponseCallback callback)
    : _model_path(model_path),
      _mmproj_path(mmproj_path),
      _responseCallback(callback),
      _hasMultimodalModel(false), // Will be detected after model loading
      _imageRetentionMs(5000) // Keep images for 5 seconds
{
    LOG_I("LlamaDeviceBase initialized - Model: " << _model_path 
          << ", MMProj: " << (_mmproj_path.empty() ? "none" : _mmproj_path)
          << ", Multimodal support: " << (_hasMultimodalModel ? "enabled" : "will be detected")
          << ", Image retention: " << _imageRetentionMs << "ms");
}

LlamaDeviceBase::~LlamaDeviceBase() {
    if (_destructing_.exchange(true)) {
       return;
    }
    stop();
}

void LlamaDeviceBase::receiveVideoFrame(const YUVData& yuv) {
    uint64_t hash = fnv1a_hash_yuv(yuv);
    
    // Debug logging to diagnose the issue
    LOG_V("receiveVideoFrame called - hasMultimodalModel: " << _hasMultimodalModel 
          << ", frame hash: " << hash);
    
    // Allow temporary storage even if multimodal support not confirmed yet
    // This helps with the chicken-and-egg problem during initialization
    std::unique_lock<std::mutex> lock(_imageMutex);
    
    // Don't store duplicate frames
    if (!_imageQueue.empty() && _imageQueue.back().hash == hash) {
        return;
    }
    
    // Create a copy of the YUV data
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
    
    TimestampedImage timestampedImage;
    timestampedImage.yuv = framePtr;
    timestampedImage.timestamp = std::chrono::steady_clock::now();
    timestampedImage.hash = hash;
    
    _imageQueue.push_back(timestampedImage);
    
    // Keep only the most recent images (limit to 3 frames)
    while (_imageQueue.size() > 3) {
        _imageQueue.pop_front();
    }
    
    LOG_V("Stored video frame with hash: " << hash << ", queue size: " << _imageQueue.size()
          << ", hasMultimodalModel: " << _hasMultimodalModel);
}

void LlamaDeviceBase::cleanupOldImages() {
    std::unique_lock<std::mutex> lock(_imageMutex);
    auto now = std::chrono::steady_clock::now();
    auto retention_duration = std::chrono::milliseconds(_imageRetentionMs);
    
    while (!_imageQueue.empty()) {
        auto& front = _imageQueue.front();
        if (now - front.timestamp > retention_duration) {
            LOG_V("Removing expired image with hash: " << front.hash);
            _imageQueue.pop_front();
        } else {
            break; // Images are stored in chronological order
        }
    }
}

std::shared_ptr<YUVData> LlamaDeviceBase::getRecentImage() {
    std::unique_lock<std::mutex> lock(_imageMutex);
    
    if (_imageQueue.empty()) {
        return nullptr;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto retention_duration = std::chrono::milliseconds(_imageRetentionMs);
    
    // Get the most recent image that's still valid
    auto& latest = _imageQueue.back();
    if (now - latest.timestamp <= retention_duration) {
        LOG_V("Using recent image with hash: " << latest.hash);
        return latest.yuv;
    }
    
    return nullptr;
}

size_t LlamaDeviceBase::getImageQueueSize() const {
    std::unique_lock<std::mutex> lock(_imageMutex);
    return _imageQueue.size();
}

bool LlamaDeviceBase::detectMultimodalSupport() {
    LOG_I("detectMultimodalSupport called - mmproj_path: '" << _mmproj_path << "'");
    
    // Check if we have mmproj path provided
    if (_mmproj_path.empty()) {
        LOG_V("No MMProj path provided - multimodal support disabled");
        return false;
    }
    
    LOG_I("Checking LlamaSimpleChat state - _llama_chat: " << (_llama_chat ? "valid" : "nullptr"));
    if (_llama_chat) {
        LOG_I("LlamaSimpleChat details - model_: " << (_llama_chat->model_ ? "loaded" : "not loaded")
              << ", ctx_mtmd_: " << (_llama_chat->ctx_mtmd_ ? "loaded" : "not loaded"));
    }
    
    // Primary check: verify that the LlamaSimpleChat has successfully loaded multimodal context
    // and that it actually supports vision input using the proper mtmd API
    if (_llama_chat && _llama_chat->ctx_mtmd_) {
        bool supportsVision = mtmd_support_vision(_llama_chat->ctx_mtmd_.get());
        bool supportsAudio = mtmd_support_audio(_llama_chat->ctx_mtmd_.get());
        LOG_I("mtmd context loaded - vision support: " << (supportsVision ? "yes" : "no") 
              << ", audio support: " << (supportsAudio ? "yes" : "no"));
        
        if (supportsVision) {
            LOG_I("Multimodal support confirmed - vision is supported");
            return true;
        } else {
            LOG_W("mtmd context loaded but vision not supported - multimodal disabled");
            return false;
        }
    }
    
    // Secondary check: verify model is loaded and mmproj file exists
    if (_llama_chat && _llama_chat->model_) {
        // Verify that mmproj file exists and is readable
        FILE* test_file = fopen(_mmproj_path.c_str(), "rb");
        if (test_file) {
            fclose(test_file);
            LOG_W("MMProj file exists but mtmd context not loaded - multimodal may be partially supported");
            return false; // Conservative approach - require successful mtmd loading
        } else {
            LOG_E("MMProj file not accessible: " << _mmproj_path);
            return false;
        }
    }
    
    // Check if at least mmproj file exists for potential future loading
    FILE* test_file = fopen(_mmproj_path.c_str(), "rb");
    if (test_file) {
        fclose(test_file);
        LOG_V("MMProj file exists, but model not yet loaded - multimodal support pending");
        return false; // Will be re-evaluated after model loading
    }
    
    LOG_E("MMProj file not found: " << _mmproj_path << " - multimodal support disabled");
    return false;
}

void LlamaDeviceBase::recheckMultimodalSupport() {
    bool previousState = _hasMultimodalModel;
    _hasMultimodalModel = detectMultimodalSupport();
    
    if (previousState != _hasMultimodalModel) {
        LOG_I("Multimodal support status changed: " << (previousState ? "enabled" : "disabled") 
              << " -> " << (_hasMultimodalModel ? "enabled" : "disabled"));
        
        // Clear image queue if multimodal support was disabled
        if (!_hasMultimodalModel && !_imageQueue.empty()) {
            std::unique_lock<std::mutex> lock(_imageMutex);
            _imageQueue.clear();
            LOG_I("Cleared image queue due to multimodal support being disabled");
        }
    }
}

void LlamaDeviceBase::askLlama(const char *prompt)
{
    if (!prompt || !*prompt) {
        return;
    }
    
    // Stop any ongoing generation before processing new request
    if (_llama_chat) {
        _llama_chat->StopGeneration();
        LOG_I("Stopped ongoing generation for new request");
    }
    
    // Clean up old images first
    cleanupOldImages();
    
    // Debug logging for troubleshooting
    size_t queueSize = getImageQueueSize();
    LOG_I("askLlama called - prompt: '" << prompt << "', hasMultimodalModel: " << _hasMultimodalModel 
          << ", image queue size: " << queueSize);
    
    // Check if we have a recent image and multimodal model
    std::shared_ptr<YUVData> recentImage = nullptr;
    if (_hasMultimodalModel) {
        recentImage = getRecentImage();
        LOG_I("Multimodal model available, getRecentImage returned: " << (recentImage ? "valid image" : "nullptr"));
    } else {
        LOG_I("Multimodal model not available - forcing text-only mode");
    }
    
    std::unique_lock<std::mutex> lock(_queueMutex);
    
    // Cancel any requests that are still waiting so the new one pre-empts them.
    if (!_requestQueue.empty()) {
        LOG_V("Clearing " << _requestQueue.size() << " pending request(s) from the queue");
        _requestQueue.clear();
    }

    if (recentImage) {
        LOG_I("Asking llama with recent image: " << prompt);
        _requestQueue.emplace_back(Request{std::string(prompt), true, recentImage});
    } else {
        LOG_I("Asking llama (text-only): " << prompt);
        _requestQueue.emplace_back(Request{std::string(prompt), false, nullptr});
    }
    
    _queueCondition.notify_one();
}

void LlamaDeviceBase::askWithImage(const char *prompt, const YUVData& yuv) {
    if (!prompt || !*prompt) {
        return;
    }
    
    // Stop any ongoing generation before processing new image request
    if (_llama_chat) {
        _llama_chat->StopGeneration();
        LOG_I("Stopped ongoing generation for new image request");
    }
    
    std::unique_lock<std::mutex> lock(_queueMutex);
    
    // Cancel any requests that are still waiting so the new one pre-empts them.
    if (!_requestQueue.empty()) {
        LOG_V("Clearing " << _requestQueue.size() << " pending request(s) from the queue");
        _requestQueue.clear();
    }

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

bool LlamaDeviceBase::start() {
    if (!_running) {
        _llama_chat.reset(new LlamaSimpleChat());
        _llama_chat->SetModelPaths(_model_path, _mmproj_path);
        // Off-load as many layers as the GPU can take (8 GiB RTX 4060 handles the full 32-layer model)
        _llama_chat->SetNGL(32);
        if (_llama_chat && _llama_chat->Initialize()) {
            LOG_V("Llama chat initialized!");
            
            // Detect multimodal support after successful initialization
            _hasMultimodalModel = detectMultimodalSupport();
            LOG_I("Multimodal support detection result: " << (_hasMultimodalModel ? "enabled" : "disabled"));
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
    // Ensure only one thread performs shutdown logic at a time.
    static std::mutex stop_mutex;
    std::lock_guard<std::mutex> stop_lock(stop_mutex);

    if (!_running) {
        return;  // already stopped – idempotent
    }

    _running = false;
    _queueCondition.notify_all();  // wake worker so it can exit

    if (_processingThread.joinable()) {
        if (std::this_thread::get_id() == _processingThread.get_id()) {
            // We are *inside* the worker thread – we cannot join ourselves.
            // Simply return; the thread will fall out of its loop and finish
            // after this call returns.
            LOG_V("stop() invoked from inside processing thread – not joining");
        } else {
            LOG_V("stop() joining processing thread");
            _processingThread.join();
        }
    }

    // Clear any pending data once the thread is done.
    {
        std::unique_lock<std::mutex> lock(_imageMutex);
        _imageQueue.clear();
    }
}

bool LlamaDeviceBase::RunProcessingThread()
{
    auto lastCleanup = std::chrono::steady_clock::now();
    const auto cleanupInterval = std::chrono::seconds(1); // Clean up every second
    
    while (_running) {
        Request req;
        bool hasRequest = false;
        
        {
            std::unique_lock<std::mutex> lock(_queueMutex);
            auto waitResult = _queueCondition.wait_for(lock, std::chrono::milliseconds(100), 
                [&]{ return !_requestQueue.empty() || !_running; });
            
            if (!_running && _requestQueue.empty()) break;
            
            if (waitResult && !_requestQueue.empty()) {
                req = std::move(_requestQueue.front());
                _requestQueue.pop_front();
                hasRequest = true;
            }
        }
        
        // Periodic cleanup of old images and multimodal status check
        auto now = std::chrono::steady_clock::now();
        if (now - lastCleanup > cleanupInterval) {
            cleanupOldImages();
            
            // Periodically recheck multimodal support (in case it changes after model loading)
            if (!_hasMultimodalModel && !_mmproj_path.empty()) {
                recheckMultimodalSupport();
            }
            
            lastCleanup = now;
        }
        
        // Process request if we have one
        if (hasRequest) {
            if (req.withImage && req.yuv && _hasMultimodalModel) {
                uint64_t h = fnv1a_hash_yuv(*req.yuv);
                if (h != _lastYuvHash) {
                    // Stop any ongoing generation before starting new image processing
                    if (_llama_chat) {
                        _llama_chat->StopGeneration();
                        LOG_I("Stopped ongoing generation for new frame processing");
                    }
                    
                    _lastYuvHash = h;
                    _llama_chat->_lastResponseStart = std::chrono::steady_clock::now();
                    auto t0 = std::chrono::steady_clock::now();
                    _llama_chat->generateFromImage(req.yuv.get(), req.prompt, _responseCallback);
                    LOG_I("Image+answer in "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now()-t0).count()
                          << " ms");
                } else {
                    LOG_V("Skipping duplicate image with hash: " << h);
                }
            } else {
                // Stop any ongoing generation before starting new text processing
                if (_llama_chat) {
                    _llama_chat->StopGeneration();
                    LOG_I("Stopped ongoing generation for new text processing");
                }
                
                _llama_chat->_lastResponseStart = std::chrono::steady_clock::now();
                _llama_chat->generate(req.prompt, _responseCallback);
            }
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