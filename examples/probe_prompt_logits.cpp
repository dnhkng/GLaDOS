// Build against the same llama.cpp headers and shared libraries as the server.
// Given LLAMA_CPP=/path/to/llama.cpp and LIB_DIR=$LLAMA_CPP/build-current/bin:
// g++ -O2 -std=c++17 examples/probe_prompt_logits.cpp -I$LLAMA_CPP/include
//     -I$LLAMA_CPP/ggml/include -I$LLAMA_CPP/vendor/nlohmann -L$LIB_DIR
//     -Wl,-rpath,$LIB_DIR -lllama -lggml -lggml-base -o /tmp/glados-probe-prompt-logits
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    std::vector<llama_token> result(text.size() + 16);
    int count = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), special, true);
    if (count < 0) {
        result.resize(-count);
        count = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), special, true);
    }
    if (count < 0) throw std::runtime_error("Tokenization failed");
    result.resize(count);
    return result;
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> result(256);
    int count = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    if (count < 0) {
        result.resize(-count);
        count = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    }
    if (count < 0) throw std::runtime_error("Token rendering failed");
    return std::string(result.data(), count);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "Usage: probe_prompt_logits MODEL GPU_LAYERS THREADS\n";
        return 2;
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = std::stoi(argv[2]);
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_outputs_max = 64;
    cp.n_threads = cp.n_threads_batch = std::stoi(argv[3]);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.swa_full = true;
    cp.no_perf = false;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 4; }
    const auto vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    auto batch = llama_batch_init(cp.n_batch, 0, 1);
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto request = json::parse(line);
            auto tokens = tokenize(vocab, request.at("prompt"), true);
            const int truncate = request.value("truncate_last", 0);
            if (truncate < 0 || truncate >= static_cast<int>(tokens.size())) {
                throw std::runtime_error("Invalid truncation");
            }
            tokens.resize(tokens.size() - truncate);
            const int scramble = request.value("scramble_last", 0);
            if (scramble < 0 || scramble >= static_cast<int>(tokens.size())) {
                throw std::runtime_error("Invalid future-token control");
            }
            if (scramble) {
                auto replacement = tokenize(vocab, "X", false);
                for (int i = tokens.size() - scramble; i < static_cast<int>(tokens.size()); ++i) {
                    tokens[i] = replacement[0];
                }
            }
            const int window = request.value("window", 16);
            const int start = std::max(0, static_cast<int>(tokens.size()) - window);
            if (window < 1 || window > 64 || tokens.size() >= cp.n_ctx) {
                throw std::runtime_error("Invalid window or prompt exceeds context");
            }
            std::vector<llama_token> labels;
            for (const auto & label : request.at("labels")) {
                auto ids = tokenize(vocab, label, false);
                if (ids.size() != 1) throw std::runtime_error("Choice label must be one token");
                labels.push_back(ids[0]);
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            json positions = json::array();
            const auto begin = std::chrono::steady_clock::now();
            for (int offset = 0; offset < static_cast<int>(tokens.size()); offset += cp.n_batch) {
                batch.n_tokens = std::min(static_cast<int>(cp.n_batch), static_cast<int>(tokens.size()) - offset);
                for (int i = 0; i < batch.n_tokens; ++i) {
                    batch.token[i] = tokens[offset + i];
                    batch.pos[i] = offset + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = offset + i >= start;
                }
                if (llama_decode(ctx, batch) != 0) throw std::runtime_error("Prompt evaluation failed");
                for (int i = 0; i < batch.n_tokens; ++i) {
                    if (!batch.logits[i]) continue;
                    const float * logits = llama_get_logits_ith(ctx, i);
                    if (!logits) throw std::runtime_error("Missing position logits");
                    double maximum = *std::max_element(logits, logits + nv);
                    double normalizer = 0;
                    for (int j = 0; j < nv; ++j) normalizer += std::exp(logits[j] - maximum);
                    double label_max = -INFINITY;
                    for (auto id : labels) label_max = std::max(label_max, static_cast<double>(logits[id]));
                    double label_z = 0;
                    for (auto id : labels) label_z += std::exp(logits[id] - label_max);
                    json candidates = json::array();
                    double mass = 0;
                    for (size_t j = 0; j < labels.size(); ++j) {
                        const auto id = labels[j];
                        const double probability = std::exp(logits[id] - maximum) / normalizer;
                        mass += probability;
                        const int rank = 1 + std::count_if(logits, logits + nv, [&](float v) {return v > logits[id];});
                        candidates.push_back({{"label", request["labels"][j]}, {"id", id},
                            {"logit", logits[id]}, {"conditional_probability", std::exp(logits[id] - label_max) / label_z},
                            {"vocab_probability", probability}, {"vocab_rank", rank}});
                    }
                    std::vector<int> order(nv);
                    std::iota(order.begin(), order.end(), 0);
                    std::partial_sort(order.begin(), order.begin() + 12, order.end(),
                        [&](int a, int b) {return logits[a] > logits[b];});
                    json top = json::array();
                    for (int j = 0; j < 12; ++j) {
                        const int id = order[j];
                        top.push_back({{"id", id}, {"piece", piece(vocab, id)},
                            {"probability", std::exp(logits[id] - maximum) / normalizer}});
                    }
                    positions.push_back({{"position", offset + i},
                        {"offset", offset + i - static_cast<int>(tokens.size()) + 1},
                        {"token_id", tokens[offset+i]}, {"piece", piece(vocab, tokens[offset+i])},
                        {"label_mass", mass}, {"candidates", candidates}, {"top_tokens", top}});
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::cout << json({{"id", request.at("id")}, {"n_tokens", tokens.size()},
                {"elapsed_ms", ms}, {"positions", positions}}).dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
        } catch (const std::exception & error) {
            std::cout << json({{"error", error.what()}}).dump() << std::endl;
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
