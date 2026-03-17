/*
 *  (c) 2025, wilddolphin2025
 *  For WebRTCsays.ai project
 *  https://github.com/wilddolphin2025
 *
 *  Orpheus TTS implementation using llama.cpp + SNAC ONNX decoder.
 */

#include "orpheus_tts.h"
#include "whillats_utils.h"
#include "whisper_helpers.h"
#include <llama.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>

#ifdef WHILLATS_STYLETTS2
#include <onnxruntime_cxx_api.h>
#endif

struct OrpheusTTS::SnacDecoder {
#ifdef WHILLATS_STYLETTS2
    Ort::Env env{nullptr};
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::RunOptions> run_options;
#endif
};

OrpheusTTS::OrpheusTTS(WhillatsSetAudioCallback callback)
    : _callback(callback) {}

OrpheusTTS::~OrpheusTTS() {
    stop();
}

bool OrpheusTTS::start(const std::string& orpheus_model_path,
                       const std::string& snac_onnx_path) {
    if (_initialized) return true;

    // Load Orpheus llama model
    llama_model_params model_params = llama_model_default_params();
    _model = llama_model_load_from_file(orpheus_model_path.c_str(), model_params);
    if (!_model) {
        LOG_E("OrpheusTTS: Failed to load model: " << orpheus_model_path);
        return false;
    }
    LOG_I("OrpheusTTS: Model loaded: " << orpheus_model_path);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx) {
        LOG_E("OrpheusTTS: Failed to create context");
        llama_model_free(_model);
        _model = nullptr;
        return false;
    }

    // Create sampler
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    _sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(0));

#ifdef WHILLATS_STYLETTS2
    _snac = std::make_unique<SnacDecoder>();
    try {
        _snac->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "orpheus_snac");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        _snac->session = std::make_unique<Ort::Session>(
            _snac->env, snac_onnx_path.c_str(), opts);
        _snac->run_options = std::make_unique<Ort::RunOptions>();
        LOG_I("OrpheusTTS: SNAC decoder loaded: " << snac_onnx_path);
    } catch (const Ort::Exception& e) {
        LOG_E("OrpheusTTS: Failed to load SNAC decoder: " << e.what());
        _snac.reset();
        llama_free(_ctx); _ctx = nullptr;
        llama_model_free(_model); _model = nullptr;
        return false;
    }
#else
    LOG_E("OrpheusTTS: SNAC decoder requires WHILLATS_STYLETTS2 (ONNX Runtime)");
    llama_free(_ctx); _ctx = nullptr;
    llama_model_free(_model); _model = nullptr;
    return false;
#endif

    _initialized = true;
    _running = true;
    _processingThread = std::thread([this]() {
        while (_running) {
            if (!runProcessingThread()) break;
        }
    });
    LOG_I("OrpheusTTS: Started");
    return true;
}

void OrpheusTTS::stop() {
    if (_running) {
        _running = false;
        _queueCondition.notify_all();
        if (_processingThread.joinable())
            _processingThread.join();
    }
    _snac.reset();
    if (_sampler) { llama_sampler_free(_sampler); _sampler = nullptr; }
    if (_ctx) { llama_free(_ctx); _ctx = nullptr; }
    if (_model) { llama_model_free(_model); _model = nullptr; }
    _initialized = false;
    LOG_I("OrpheusTTS: Stopped");
}

void OrpheusTTS::queueText(const char* text, const char* voice) {
    if (!_initialized || !text) return;
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _textQueue.push({std::string(text), std::string(voice ? voice : "tara")});
    }
    _queueCondition.notify_one();
}

int OrpheusTTS::tokenToId(const std::string& token_text, int index) {
    // Orpheus custom tokens: "<custom_token_NNNNN>"
    if (token_text.size() > 15 && token_text.substr(0, 14) == "<custom_token_" &&
        token_text.back() == '>') {
        try {
            int number = std::stoi(token_text.substr(14, token_text.size() - 15));
            return number - 10 - ((index % TOKENS_PER_FRAME) * 4096);
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

std::vector<int16_t> OrpheusTTS::convertToAudio(const std::vector<int>& tokens) {
#ifdef WHILLATS_STYLETTS2
    if (!_snac || !_snac->session || tokens.size() < static_cast<size_t>(TOKENS_PER_CHUNK))
        return {};

    int num_frames = static_cast<int>(tokens.size()) / TOKENS_PER_FRAME;

    // Split into 3 codebook layers per Orpheus spec
    std::vector<int32_t> codes_0, codes_1, codes_2;
    for (int j = 0; j < num_frames; ++j) {
        int i = TOKENS_PER_FRAME * j;
        codes_0.push_back(tokens[i]);
        codes_1.push_back(tokens[i + 1]);
        codes_1.push_back(tokens[i + 4]);
        codes_2.push_back(tokens[i + 2]);
        codes_2.push_back(tokens[i + 3]);
        codes_2.push_back(tokens[i + 5]);
        codes_2.push_back(tokens[i + 6]);
    }

    // Validate ranges
    for (auto v : codes_0) if (v < 0 || v > 4096) return {};
    for (auto v : codes_1) if (v < 0 || v > 4096) return {};
    for (auto v : codes_2) if (v < 0 || v > 4096) return {};

    // Cast to int64 for ONNX
    std::vector<int64_t> c0(codes_0.begin(), codes_0.end());
    std::vector<int64_t> c1(codes_1.begin(), codes_1.end());
    std::vector<int64_t> c2(codes_2.begin(), codes_2.end());

    std::vector<int64_t> shape0 = {1, static_cast<int64_t>(c0.size())};
    std::vector<int64_t> shape1 = {1, static_cast<int64_t>(c1.size())};
    std::vector<int64_t> shape2 = {1, static_cast<int64_t>(c2.size())};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, c0.data(), c0.size(), shape0.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, c1.data(), c1.size(), shape1.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem, c2.data(), c2.size(), shape2.data(), 2));

    // Get input names from session
    size_t n_inputs = _snac->session->GetInputCount();
    std::vector<std::string> input_name_strs(n_inputs);
    std::vector<const char*> input_names(n_inputs);
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < n_inputs; ++i) {
        auto name = _snac->session->GetInputNameAllocated(i, alloc);
        input_name_strs[i] = name.get();
        input_names[i] = input_name_strs[i].c_str();
    }

    size_t n_outputs = _snac->session->GetOutputCount();
    std::vector<std::string> output_name_strs(n_outputs);
    std::vector<const char*> output_names(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        auto name = _snac->session->GetOutputNameAllocated(i, alloc);
        output_name_strs[i] = name.get();
        output_names[i] = output_name_strs[i].c_str();
    }

    auto result = _snac->session->Run(
        *_snac->run_options,
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());

    const float* audio_data = result[0].GetTensorData<float>();
    auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    for (auto d : shape) total *= d;

    // Use the full output - the static model outputs the correct length
    int64_t audio_len = total;
    int64_t start = 0;
    fprintf(stderr, "[ORPHEUS] SNAC output shape:");
    for (auto d : shape) fprintf(stderr, " %lld", (long long)d);
    fprintf(stderr, " total=%lld\n", (long long)total);

    std::vector<int16_t> pcm(audio_len);
    for (int64_t i = 0; i < audio_len; ++i) {
        float v = audio_data[start + i] * 32767.0f;
        v = std::max(-32768.0f, std::min(32767.0f, v));
        pcm[i] = static_cast<int16_t>(v);
    }

    return pcm;
#else
    return {};
#endif
}

bool OrpheusTTS::runProcessingThread() {
    std::string text;
    std::string voice;

    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        _queueCondition.wait_for(lock, std::chrono::milliseconds(100),
            [this] { return !_textQueue.empty() || !_running; });
        if (!_running) return false;
        if (_textQueue.empty()) return true;
        text = _textQueue.front().first;
        voice = _textQueue.front().second;
        _textQueue.pop();
    }

    LOG_I("OrpheusTTS: Synthesizing (" << text.size() << " chars, voice=" << voice << "): "
          << text.substr(0, 60) << (text.size() > 60 ? "..." : ""));

    // Orpheus finetuned prompt: <|begin_of_text|> voice: text<|eot_id|>
    // Try both formats depending on vocab
    const llama_vocab* vocab_check = llama_model_get_vocab(_model);
    llama_token audio_token_id = -1;
    {
        const char* audio_str = "<|audio|>";
        llama_token tmp[4];
        int n = llama_tokenize(vocab_check, audio_str, strlen(audio_str), tmp, 4, false, true);
        if (n == 1 && tmp[0] > 128000) audio_token_id = tmp[0];
    }

    std::string prompt;
    if (audio_token_id >= 0) {
        prompt = "<|audio|>" + voice + ": " + text + "<|eot_id|> ";
    } else {
        // Mungert-style: BOS is added by tokenizer, then " voice: text<|eot_id|> "
        prompt = " " + voice + ": " + text + "<|eot_id|> ";
    }
    fprintf(stderr, "[ORPHEUS] Using prompt format: %s\n",
            audio_token_id >= 0 ? "<|audio|>" : "<|begin_of_text|>");

    const llama_vocab* vocab = llama_model_get_vocab(_model);
    std::vector<llama_token> prompt_tokens(prompt.size() + 32);
    // add_special=true for BOS, parse_special=true to recognize <|audio|> etc.
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                  prompt_tokens.data(), prompt_tokens.size(),
                                  true, true);
    if (n_tokens < 0) {
        LOG_E("OrpheusTTS: Tokenization failed");
        _callback.OnSynthesisComplete();
        return true;
    }
    prompt_tokens.resize(n_tokens);
    fprintf(stderr, "[ORPHEUS] Prompt: '%s' -> %d tokens:", prompt.c_str(), n_tokens);
    for (int t = 0; t < std::min(n_tokens, 20); ++t) {
        fprintf(stderr, " %d", prompt_tokens[t]);
    }
    fprintf(stderr, "\n");

    // Clear KV cache for fresh generation
    llama_memory_clear(llama_get_memory(_ctx), true);

    // Process prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_tokens);
    if (llama_decode(_ctx, batch) != 0) {
        LOG_E("OrpheusTTS: Prompt decode failed");
        _callback.OnSynthesisComplete();
        return true;
    }

    // Generate audio tokens
    std::vector<int> audio_tokens;
    int token_count = 0;
    int max_tokens = 2048;
    std::vector<int16_t> all_audio;

    llama_token eos_token = 128001; // <|end_of_text|>

    for (int i = 0; i < max_tokens && _running; ++i) {
        llama_token new_token = llama_sampler_sample(_sampler, _ctx, -1);

        if (new_token == eos_token) {
            fprintf(stderr, "[ORPHEUS] EOS token %d at position %d\n", new_token, i);
            break;
        }
        if (new_token == 128009) {
            fprintf(stderr, "[ORPHEUS] eot_id token at position %d (continuing)\n", i);
        }

        // Decode the token text
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (len <= 0) continue;
        std::string token_text(buf, len);

        if (i < 20 || (i < 100 && i % 20 == 0)) {
            fprintf(stderr, "[ORPHEUS] token[%d] id=%d text='%s' code=%d\n",
                    i, new_token, token_text.c_str(), tokenToId(token_text, token_count));
        }

        // Convert to audio codebook id
        int code_id = tokenToId(token_text, token_count);
        if (code_id >= 0 && code_id <= 4096) {
            audio_tokens.push_back(code_id);
            token_count++;

            // Every 28 tokens (4 frames), decode to audio
            if (token_count % TOKENS_PER_CHUNK == 0 && token_count > TOKENS_PER_CHUNK) {
                std::vector<int> chunk(audio_tokens.end() - TOKENS_PER_CHUNK,
                                       audio_tokens.end());
                auto pcm = convertToAudio(chunk);
                if (!pcm.empty()) {
                    all_audio.insert(all_audio.end(), pcm.begin(), pcm.end());
                }
            }
        }

        // Prepare next decode
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(_ctx, batch) != 0) {
            LOG_E("OrpheusTTS: Decode failed at token " << i);
            break;
        }
    }

    LOG_I("OrpheusTTS: Generated " << token_count << " audio tokens, "
          << all_audio.size() << " PCM samples at " << SNAC_SAMPLE_RATE << "Hz");

    if (!all_audio.empty()) {
        // Resample from 24kHz to 16kHz
        auto resampled = resampleAudio(all_audio.data(), all_audio.size(),
                                       SNAC_SAMPLE_RATE, OUTPUT_SAMPLE_RATE);
        LOG_V("OrpheusTTS: Resampled to " << resampled.size() << " samples at " << OUTPUT_SAMPLE_RATE << "Hz");

        std::vector<uint16_t> u16(resampled.begin(), resampled.end());
        _callback.OnBufferComplete(true, u16);
    }

    _callback.OnSynthesisComplete();
    return true;
}

const int OrpheusTTS::getSampleRate() {
    return OUTPUT_SAMPLE_RATE;
}
