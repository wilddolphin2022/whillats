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
 *
 *  StyleTTS2 integration based on StyleTTS2-onnx-cpp by DDATT
 *  https://github.com/DDATT/StyleTTS2-onnx-cpp (MIT License)
 */

#include "styletts2_tts.h"
#include <iostream>
#include <numeric>
#include <cstring>
#include <cstdlib>
#include <thread>

struct MaskResult {
    std::vector<bool> text_mask;
    std::vector<int32_t> attention_mask;
    std::vector<int64_t> shape;
};

static MaskResult generateMasks(const std::vector<int64_t>& lengths) {
    if (lengths.empty()) return {};

    int64_t batch_size = static_cast<int64_t>(lengths.size());
    int64_t max_len = *std::max_element(lengths.begin(), lengths.end());

    MaskResult result;
    result.shape = {batch_size, max_len};
    result.text_mask.resize(batch_size * max_len);
    result.attention_mask.resize(batch_size * max_len);

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t current_len = lengths[i];
        for (int64_t j = 0; j < max_len; ++j) {
            int64_t flat_index = i * max_len + j;
            bool is_padding = (j + 1) > current_len;
            result.text_mask[flat_index] = is_padding;
            result.attention_mask[flat_index] = is_padding ? 0 : 1;
        }
    }
    return result;
}

StyleTTS2TTS::StyleTTS2TTS(WhillatsSetAudioCallback callback,
                           const std::string& modelDir,
                           const std::string& espeakDataDir,
                           bool useCuda)
    : _callback(callback),
      _modelDir(modelDir),
      _espeakDataDir(espeakDataDir),
      _useCuda(useCuda)
{
    // IPA symbol-to-id map (from StyleTTS2 vocabulary)
    _symbolToId = {
        {U'_', 0}, {U';', 1}, {U':', 2}, {U',', 3}, {U'.', 4}, {U'!', 5}, {U'?', 6},
        {U'\u00A1', 7}, {U'\u00BF', 8}, {U'\u2014', 9}, {U'\u2026', 10},
        {U'"', 11}, {U'\u00AB', 12}, {U'\u00BB', 13}, {U'\u201C', 14}, {U'\u201D', 15},
        {U' ', 16},
        {U'A', 17}, {U'B', 18}, {U'C', 19}, {U'D', 20}, {U'E', 21}, {U'F', 22},
        {U'G', 23}, {U'H', 24}, {U'I', 25}, {U'J', 26}, {U'K', 27}, {U'L', 28},
        {U'M', 29}, {U'N', 30}, {U'O', 31}, {U'P', 32}, {U'Q', 33}, {U'R', 34},
        {U'S', 35}, {U'T', 36}, {U'U', 37}, {U'V', 38}, {U'W', 39}, {U'X', 40},
        {U'Y', 41}, {U'Z', 42},
        {U'a', 43}, {U'b', 44}, {U'c', 45}, {U'd', 46}, {U'e', 47}, {U'f', 48},
        {U'g', 49}, {U'h', 50}, {U'i', 51}, {U'j', 52}, {U'k', 53}, {U'l', 54},
        {U'm', 55}, {U'n', 56}, {U'o', 57}, {U'p', 58}, {U'q', 59}, {U'r', 60},
        {U's', 61}, {U't', 62}, {U'u', 63}, {U'v', 64}, {U'w', 65}, {U'x', 66},
        {U'y', 67}, {U'z', 68},
        {U'\u0251', 69}, {U'\u0250', 70}, {U'\u0252', 71}, {U'\u00E6', 72},
        {U'\u0253', 73}, {U'\u0299', 74}, {U'\u03B2', 75}, {U'\u0254', 76},
        {U'\u0255', 77}, {U'\u00E7', 78}, {U'\u0257', 79}, {U'\u0256', 80},
        {U'\u00F0', 81}, {U'\u02A4', 82}, {U'\u0259', 83}, {U'\u0258', 84},
        {U'\u025A', 85}, {U'\u025B', 86}, {U'\u025C', 87}, {U'\u025D', 88},
        {U'\u025E', 89}, {U'\u025F', 90}, {U'\u0284', 91}, {U'\u0261', 92},
        {U'\u0260', 93}, {U'\u0262', 94}, {U'\u029B', 95}, {U'\u0266', 96},
        {U'\u0267', 97}, {U'\u0127', 98}, {U'\u0265', 99}, {U'\u029C', 100},
        {U'\u0268', 101}, {U'\u026A', 102}, {U'\u029D', 103}, {U'\u026D', 104},
        {U'\u026C', 105}, {U'\u026B', 106}, {U'\u026E', 107}, {U'\u029F', 108},
        {U'\u0271', 109}, {U'\u026F', 110}, {U'\u0270', 111}, {U'\u014B', 112},
        {U'\u0273', 113}, {U'\u0272', 114}, {U'\u0274', 115}, {U'\u00F8', 116},
        {U'\u0275', 117}, {U'\u0278', 118}, {U'\u03B8', 119}, {U'\u0153', 120},
        {U'\u0276', 121}, {U'\u0298', 122}, {U'\u0279', 123}, {U'\u027A', 124},
        {U'\u027E', 125}, {U'\u027B', 126}, {U'\u0280', 127}, {U'\u0281', 128},
        {U'\u027D', 129}, {U'\u0282', 130}, {U'\u0283', 131}, {U'\u0288', 132},
        {U'\u02A7', 133}, {U'\u0289', 134}, {U'\u028A', 135}, {U'\u028B', 136},
        {U'\u2C71', 137}, {U'\u028C', 138}, {U'\u0263', 139}, {U'\u0264', 140},
        {U'\u028D', 141}, {U'\u03C7', 142}, {U'\u028E', 143}, {U'\u028F', 144},
        {U'\u0291', 145}, {U'\u0290', 146}, {U'\u0292', 147}, {U'\u0294', 148},
        {U'\u02A1', 149}, {U'\u0295', 150}, {U'\u02A2', 151}, {U'\u01C0', 152},
        {U'\u01C1', 153}, {U'\u01C2', 154}, {U'\u01C3', 155}, {U'\u02C8', 156},
        {U'\u02CC', 157}, {U'\u02D0', 158}, {U'\u02D1', 159}, {U'\u02BC', 160},
        {U'\u02B4', 161}, {U'\u02B0', 162}, {U'\u02B1', 163}, {U'\u02B2', 164},
        {U'\u02B7', 165}, {U'\u02E0', 166}, {U'\u02E4', 167}, {U'\u02DE', 168},
        {U'\u2193', 169}, {U'\u2191', 170}, {U'\u2192', 171}, {U'\u2197', 172},
        {U'\u2198', 173},
        {U'\u0329', 175},
        {U'\u1D7B', 177}
    };
}

StyleTTS2TTS::~StyleTTS2TTS() {
    stop();
    if (_phonemizerReady) {
        espeak_Terminate();
        _phonemizerReady = false;
    }
}

void StyleTTS2TTS::initPhonemizer(const std::string& voice, const std::string& espeakData) {
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0,
                                   espeakData.c_str(), 0);
    if (result < 0) {
        LOG_E("StyleTTS2: Failed to initialize eSpeak-ng phonemizer");
        return;
    }
    int setvoice = espeak_SetVoiceByName(voice.c_str());
    if (setvoice != 0) {
        LOG_E("StyleTTS2: Failed to set eSpeak-ng voice: " << voice);
        return;
    }
    _phonemizerReady = true;
    LOG_I("StyleTTS2: eSpeak-ng phonemizer initialized with voice: " << voice);
}

std::string StyleTTS2TTS::phonemize(const std::string& text) {
    if (!_phonemizerReady) return "";

    std::string textCopy(text);
    const char* inputTextPointer = textCopy.c_str();
    std::string result;

    while (inputTextPointer != nullptr) {
        std::string clausePhonemes(espeak_TextToPhonemes(
            (const void**)&inputTextPointer,
            espeakCHARS_AUTO,
            0x02));  // IPA mode
        result += clausePhonemes;
        result += ", ";
    }

    if (result.size() >= 2) {
        result.pop_back();
        result.pop_back();
    }
    result += ".";
    return result;
}

std::vector<char32_t> StyleTTS2TTS::utf8ToCodepoints(const std::string& str) {
    std::vector<char32_t> result;
    size_t i = 0;
    while (i < str.size()) {
        char32_t cp = 0;
        unsigned char c = static_cast<unsigned char>(str[i]);
        int bytes = 0;

        if (c < 0x80) {
            cp = c; bytes = 1;
        } else if (c < 0xC0) {
            i++; continue;
        } else if (c < 0xE0) {
            cp = c & 0x1F; bytes = 2;
        } else if (c < 0xF0) {
            cp = c & 0x0F; bytes = 3;
        } else {
            cp = c & 0x07; bytes = 4;
        }

        for (int j = 1; j < bytes && (i + j) < str.size(); j++) {
            cp = (cp << 6) | (static_cast<unsigned char>(str[i + j]) & 0x3F);
        }
        result.push_back(cp);
        i += bytes;
    }
    return result;
}

std::vector<int64_t> StyleTTS2TTS::textToSequence(const std::string& text) {
    std::vector<int64_t> sequence;
    std::string phonemes = phonemize(text);
    auto codepoints = utf8ToCodepoints(phonemes);

    for (char32_t cp : codepoints) {
        auto it = _symbolToId.find(cp);
        if (it != _symbolToId.end()) {
            sequence.push_back(it->second);
        }
    }
    return sequence;
}

std::vector<float> StyleTTS2TTS::loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_E("StyleTTS2: Cannot open file: " << filename);
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    return {};
}

void StyleTTS2TTS::loadStyle(const std::string& styleFile, const std::string& predictorFile) {
    _styleEmbedding = loadBinaryFile(styleFile);
    if (_styleEmbedding.empty()) {
        LOG_E("StyleTTS2: Failed to load style embedding from " << styleFile);
        return;
    }

    _predictorEmbedding = loadBinaryFile(predictorFile);
    if (_predictorEmbedding.empty()) {
        LOG_E("StyleTTS2: Failed to load predictor embedding from " << predictorFile);
        return;
    }

    LOG_I("StyleTTS2: Loaded style embeddings (style=" << _styleEmbedding.size()
          << " predictor=" << _predictorEmbedding.size() << " floats)");
}

bool StyleTTS2TTS::start() {
    if (_initialized) {
        LOG_W("StyleTTS2: Already initialized");
        return _running ? false : true;
    }

    LOG_I("StyleTTS2: Initializing with model dir: " << _modelDir);

    // Initialize ONNX Runtime
    _env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "StyleTTS2");
    _env.DisableTelemetryEvents();

    if (_useCuda) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        _sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        LOG_I("StyleTTS2: CUDA execution provider enabled");
    }

    _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    int ttsThreads = (_nThreads > 0) ? _nThreads : std::max(1u, std::thread::hardware_concurrency());
    _sessionOptions.SetIntraOpNumThreads(ttsThreads);
    _sessionOptions.DisableProfiling();

    // Load ONNX models
    std::string plBertPath = _modelDir + "/plbert_simp.onnx";
    std::string bertEncoderPath = _modelDir + "/bert_encoder.onnx";
    std::string modelPath = _modelDir + "/final_simp.onnx";

    try {
        _plBert = std::make_unique<Ort::Session>(_env, plBertPath.c_str(), _sessionOptions);
        _bertEncoder = std::make_unique<Ort::Session>(_env, bertEncoderPath.c_str(), _sessionOptions);
        _model = std::make_unique<Ort::Session>(_env, modelPath.c_str(), _sessionOptions);
    } catch (const Ort::Exception& e) {
        LOG_E("StyleTTS2: Failed to load ONNX models: " << e.what());
        return false;
    }

    LOG_I("StyleTTS2: ONNX models loaded successfully");

    // Initialize phonemizer
    initPhonemizer("en-us", _espeakDataDir);
    if (!_phonemizerReady) {
        LOG_E("StyleTTS2: Phonemizer initialization failed");
        return false;
    }

    // Load default style embeddings if available
    std::string defaultStyle = _modelDir + "/ref_s.bin";
    std::string defaultPredictor = _modelDir + "/ref_p.bin";
    std::ifstream testStyle(defaultStyle);
    std::ifstream testPred(defaultPredictor);
    if (testStyle.good() && testPred.good()) {
        loadStyle(defaultStyle, defaultPredictor);
    } else {
        LOG_W("StyleTTS2: No default style embeddings found at "
              << defaultStyle << " - call loadStyle() before synthesis");
    }

    _initialized = true;

    // Start processing thread
    if (!_running) {
        _running = true;
        _processingThread = std::thread([this] {
            while (_running && runProcessingThread()) {}
        });
        LOG_I("StyleTTS2: Processing thread started");
    }

    return true;
}

void StyleTTS2TTS::stop() {
    if (_running) {
        _running = false;
        _queueCondition.notify_all();
        if (_processingThread.joinable()) {
            _processingThread.join();
        }
        LOG_I("StyleTTS2: Stopped");
    }
}

void StyleTTS2TTS::queueText(const std::string& text, const std::string& language) {
    if (text.empty()) return;
    if (!_initialized) {
        LOG_E("StyleTTS2: Not initialized, cannot queue text");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _textQueue.push(std::make_pair(text, language));
    }
    _queueCondition.notify_one();
}

std::vector<int16_t> StyleTTS2TTS::synthesize(const std::string& text, float speed) {
    if (!_initialized || _styleEmbedding.empty() || _predictorEmbedding.empty()) {
        LOG_E("StyleTTS2: Cannot synthesize - not initialized or style not loaded");
        return {};
    }

    std::vector<int16_t> audioBuffer;
    std::vector<int64_t> tokens = textToSequence(text);
    if (tokens.empty()) {
        LOG_W("StyleTTS2: No phoneme tokens generated for text");
        return {};
    }

    tokens.insert(tokens.begin(), 0);

    std::vector<int64_t> textLength = {static_cast<int64_t>(tokens.size())};
    MaskResult masks = generateMasks(textLength);

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // PL-BERT inference
    std::vector<Ort::Value> plBertInputTensors;
    std::vector<int64_t> phonemeIdsShape{1, static_cast<int64_t>(tokens.size())};

    plBertInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, tokens.data(), tokens.size(),
        phonemeIdsShape.data(), phonemeIdsShape.size()));
    plBertInputTensors.push_back(Ort::Value::CreateTensor<int32_t>(
        memoryInfo, masks.attention_mask.data(), masks.attention_mask.size(),
        phonemeIdsShape.data(), phonemeIdsShape.size()));

    std::array<const char*, 2> plBertInputNames = {"input_ids", "attention_mask"};
    std::array<const char*, 1> plBertOutputNames = {"bert_dur"};

    auto bertDur = _plBert->Run(
        Ort::RunOptions{nullptr}, plBertInputNames.data(),
        plBertInputTensors.data(), plBertInputTensors.size(),
        plBertOutputNames.data(), plBertOutputNames.size());

    const float* bertDurData = bertDur.front().GetTensorData<float>();
    auto bertDurShape = bertDur.front().GetTensorTypeAndShapeInfo().GetShape();

    int64_t totalElements = 1;
    for (auto dim : bertDurShape) totalElements *= dim;
    std::vector<float> bertDurVector(bertDurData, bertDurData + totalElements);

    // BERT encoder inference
    std::vector<Ort::Value> bertEncoderInputTensors;
    bertEncoderInputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, bertDurVector.data(), bertDurVector.size(),
        bertDurShape.data(), bertDurShape.size()));

    std::array<const char*, 1> bertEncoderInputNames = {"input"};
    std::array<const char*, 1> bertEncoderOutputNames = {"d_en"};

    auto dEn = _bertEncoder->Run(
        Ort::RunOptions{nullptr}, bertEncoderInputNames.data(),
        bertEncoderInputTensors.data(), bertEncoderInputTensors.size(),
        bertEncoderOutputNames.data(), bertEncoderOutputNames.size());

    const float* dEnData = dEn.front().GetTensorData<float>();
    auto dEnShape = dEn.front().GetTensorTypeAndShapeInfo().GetShape();

    totalElements = 1;
    for (auto dim : dEnShape) totalElements *= dim;

    int64_t batch = dEnShape[0];
    int64_t dim1 = dEnShape[1];
    int64_t dim2 = dEnShape[2];

    // Transpose [batch, dim1, dim2] -> [batch, dim2, dim1]
    std::vector<float> dEnVector(totalElements);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < dim1; i++) {
            for (int64_t j = 0; j < dim2; j++) {
                int64_t oldIdx = b * (dim1 * dim2) + i * dim2 + j;
                int64_t newIdx = b * (dim2 * dim1) + j * dim1 + i;
                dEnVector[newIdx] = dEnData[oldIdx];
            }
        }
    }

    std::vector<int64_t> dEnShapeTransposed = {batch, dim2, dim1};
    std::vector<int64_t> refShape = {1, 128};
    std::vector<float> speedVec = {speed};

    // Final model inference
    std::vector<Ort::Value> finalModelInputs;
    finalModelInputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, tokens.data(), tokens.size(),
        phonemeIdsShape.data(), phonemeIdsShape.size()));
    finalModelInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, dEnVector.data(), dEnVector.size(),
        dEnShapeTransposed.data(), dEnShapeTransposed.size()));
    finalModelInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, _predictorEmbedding.data(), _predictorEmbedding.size(),
        refShape.data(), refShape.size()));
    finalModelInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, _styleEmbedding.data(), _styleEmbedding.size(),
        refShape.data(), refShape.size()));

    std::vector<int64_t> scalarShape = {1};
    finalModelInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, speedVec.data(), speedVec.size(),
        scalarShape.data(), scalarShape.size()));

    std::array<const char*, 5> finalInputNames = {"tokens", "d_en", "ref", "s", "speed"};
    std::array<const char*, 1> finalOutputNames = {"output_wav"};

    auto audioOutput = _model->Run(
        Ort::RunOptions{nullptr}, finalInputNames.data(),
        finalModelInputs.data(), finalModelInputs.size(),
        finalOutputNames.data(), finalOutputNames.size());

    const float* audioOutputData = audioOutput.front().GetTensorData<float>();
    auto audioOutputShape = audioOutput.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioOutputCount = audioOutputShape[audioOutputShape.size() - 1];

    audioBuffer.reserve(audioOutputCount);
    for (int64_t i = 0; i < audioOutputCount; i++) {
        float clamped = std::max(
            static_cast<float>(std::numeric_limits<int16_t>::min()),
            std::min(audioOutputData[i] * MAX_WAV_VALUE,
                     static_cast<float>(std::numeric_limits<int16_t>::max())));
        audioBuffer.push_back(static_cast<int16_t>(clamped));
    }

    return audioBuffer;
}

bool StyleTTS2TTS::runProcessingThread() {
    std::string textToSynth;
    std::string language;
    bool shouldSynth = false;

    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (_queueCondition.wait_for(lock, std::chrono::milliseconds(100),
            [this] { return !_textQueue.empty() || !_running; })) {

            if (!_running) return false;

            if (!_textQueue.empty()) {
                textToSynth = _textQueue.front().first;
                language = _textQueue.front().second;
                _textQueue.pop();
                shouldSynth = true;
            }
        }
    }

    if (shouldSynth) {
        LOG_I("StyleTTS2: Synthesizing: " << textToSynth.substr(0, 60)
              << (textToSynth.size() > 60 ? "..." : ""));

        auto audio = synthesize(textToSynth, 1.0f);

        if (!audio.empty()) {
            // Convert int16_t to uint16_t for the callback
            std::vector<uint16_t> audioU16(audio.begin(), audio.end());
            LOG_V("StyleTTS2: Generated " << audioU16.size() << " samples at " << SAMPLE_RATE << "Hz");
            _callback.OnBufferComplete(true, audioU16);
        } else {
            LOG_W("StyleTTS2: No audio generated for text");
        }

        _callback.OnSynthesisComplete();
    }

    return true;
}

const int StyleTTS2TTS::getSampleRate() {
    return SAMPLE_RATE;
}
