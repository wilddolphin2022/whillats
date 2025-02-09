#pragma once

#include <string>
#include "llama.h"

class LlamaDeviceBase {
public:
    LlamaDeviceBase();
    virtual ~LlamaDeviceBase();

    bool initialize(const std::string& model_path);
    std::string generate(const std::string& prompt);

protected:
    llama_context* ctx_;
    llama_model* model_;
};