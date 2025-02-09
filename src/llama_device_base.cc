#include "llama_device_base.h"
#include "llama.h"
#include <vector>

LlamaDeviceBase::LlamaDeviceBase() : ctx_(nullptr), model_(nullptr) {}

LlamaDeviceBase::~LlamaDeviceBase() {
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
}

bool LlamaDeviceBase::initialize(const std::string& model_path) {
    struct llama_model_params model_params = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        return false;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // context window
    ctx_params.n_batch = 512; // batch size for prompt processing
    ctx_params.n_threads = 4; // number of threads to use for generation
    
    ctx_ = llama_init_from_model(model_, ctx_params);
    return ctx_ != nullptr;
}

std::string LlamaDeviceBase::generate(const std::string& prompt) {
    if (!ctx_ || prompt.empty()) {
        return "";
    }

    std::vector<llama_token> tokens(prompt.length());
    int n_tokens = llama_tokenize(llama_model_get_vocab(model_), 
                                prompt.c_str(), prompt.length(), 
                                tokens.data(), tokens.size(), 
                                true, false);
    
    if (n_tokens < 0) {
        return "";
    }

    struct llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx_, batch) != 0) {
        return "";
    }

    std::string result;
    const int max_tokens = 256;
    
    for (int i = 0; i < max_tokens; ++i) {
        float* logits = llama_get_logits(ctx_);
        llama_token_data_array candidates = { nullptr, 0, 0, false };
        candidates.data = new llama_token_data[llama_vocab_n_tokens(llama_model_get_vocab(model_))];
        candidates.size = llama_vocab_n_tokens(llama_model_get_vocab(model_));
        
        for (int j = 0; j < candidates.size; j++) {
            candidates.data[j].id = j;
            candidates.data[j].logit = logits[j];
            candidates.data[j].p = 0.0f;
        }
        
        llama_token token_id = candidates.data[0].id;
        delete[] candidates.data;
        
        if (token_id == llama_vocab_eos(llama_model_get_vocab(model_))) {
            break;
        }

        const char* token_str = llama_vocab_get_text(llama_model_get_vocab(model_), token_id);
        result += token_str;

        batch = llama_batch_get_one(&token_id, 1);
        if (llama_decode(ctx_, batch) != 0) {
            break;
        }
    }

    return result;
}