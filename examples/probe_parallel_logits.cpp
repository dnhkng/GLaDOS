// Build with the same command as probe_prompt_logits.cpp, replacing the source name.
// JSONL input: {"id":"group", "requests":[{"id":"request", "prompt":"...", "labels":["A","B"]}]}
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool bos) {
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), bos, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), bos, true);
    }
    if (n <= 0) throw std::runtime_error("Tokenization failed or empty input");
    tokens.resize(n);
    return tokens;
}

static json extract(const float * logits, int n_vocab, const json & labels,
                    const std::vector<llama_token> & ids) {
    double maximum = -INFINITY;
    for (auto id : ids) maximum = std::max(maximum, static_cast<double>(logits[id]));
    double z = 0;
    for (auto id : ids) z += std::exp(logits[id] - maximum);
    const double vocab_max = *std::max_element(logits, logits + n_vocab);
    double vocab_z = 0;
    for (int i = 0; i < n_vocab; ++i) vocab_z += std::exp(logits[i] - vocab_max);
    json result = json::array();
    for (size_t i = 0; i < ids.size(); ++i) {
        result.push_back({{"label", labels[i]}, {"token_id", ids[i]}, {"logit", logits[ids[i]]},
                          {"conditional_probability", std::exp(logits[ids[i]] - maximum) / z},
                          {"vocab_probability", std::exp(logits[ids[i]] - vocab_max) / vocab_z}});
    }
    return result;
}

struct Batch {
    llama_batch value;
    explicit Batch(int capacity) : value(llama_batch_init(capacity, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
};

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "Usage: probe_parallel_logits MODEL GPU_LAYERS THREADS SLOTS\n";
        return 2;
    }
    try {
        const int slots = std::stoi(argv[4]);
        const int threads = std::stoi(argv[3]);
        if (slots < 1 || slots > 16 || threads < 1) throw std::runtime_error("Invalid slots or threads");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = std::stoi(argv[2]);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("Model loading failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048 * slots;
        cp.n_batch = cp.n_ubatch = 512;
        cp.n_seq_max = slots;
        cp.n_outputs_max = slots;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.swa_full = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("Context creation failed");
        const auto vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        Batch batch(cp.n_batch);
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto group = json::parse(line);
                const auto & requests = group.at("requests");
                if (!requests.is_array() || requests.empty() || requests.size() > static_cast<size_t>(slots)) {
                    throw std::runtime_error("Group must contain between 1 and SLOTS requests");
                }
                std::vector<std::vector<llama_token>> tokens, labels;
                for (const auto & request : requests) {
                    tokens.push_back(tokenize(vocab, request.at("prompt"), true));
                    if (tokens.back().size() >= 2048) throw std::runtime_error("Prompt exceeds slot context");
                    std::vector<llama_token> ids;
                    for (const auto & label : request.at("labels")) {
                        const auto ids_label = tokenize(vocab, label, false);
                        if (ids_label.size() != 1) throw std::runtime_error("Labels must be single tokens");
                        ids.push_back(ids_label[0]);
                    }
                    if (ids.empty()) throw std::runtime_error("No labels supplied");
                    labels.push_back(std::move(ids));
                }
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                std::vector<size_t> cursors(requests.size(), 0);
                json results = json::array();
                for (const auto & request : requests) results.push_back({{"id", request.at("id")}});
                int decode_calls = 0, mixed_sequence_calls = 0, max_sequences_per_call = 0;
                int max_logit_rows_per_call = 0;
                const auto begin = std::chrono::steady_clock::now();
                while (true) {
                    batch.value.n_tokens = 0;
                    std::vector<std::pair<int, int>> outputs;
                    int participating = 0;
                    // Every active sequence contributes up to 32 tokens per call.
                    for (size_t seq = 0; seq < requests.size(); ++seq) {
                        if (cursors[seq] == tokens[seq].size()) continue;
                        ++participating;
                        const size_t end = std::min(tokens[seq].size(), cursors[seq] + 32);
                        while (cursors[seq] < end) {
                            const int i = batch.value.n_tokens++;
                            const int position = cursors[seq]++;
                            batch.value.token[i] = tokens[seq][position];
                            batch.value.pos[i] = position;
                            batch.value.n_seq_id[i] = 1;
                            batch.value.seq_id[i][0] = seq;
                            const bool last = cursors[seq] == tokens[seq].size();
                            batch.value.logits[i] = last;
                            if (last) outputs.emplace_back(i, seq);
                        }
                    }
                    if (!batch.value.n_tokens) break;
                    if (llama_decode(ctx.get(), batch.value) != 0) throw std::runtime_error("Decode failed");
                    ++decode_calls;
                    mixed_sequence_calls += participating > 1;
                    max_sequences_per_call = std::max(max_sequences_per_call, participating);
                    max_logit_rows_per_call = std::max(max_logit_rows_per_call, static_cast<int>(outputs.size()));
                    for (auto [index, seq] : outputs) {
                        const auto logits = llama_get_logits_ith(ctx.get(), index);
                        if (!logits) throw std::runtime_error("Missing request logits");
                        // Copy scores before the next decode overwrites the context output buffer.
                        results[seq]["candidates"] = extract(logits, n_vocab, requests[seq]["labels"], labels[seq]);
                        results[seq]["n_tokens"] = tokens[seq].size();
                        results[seq]["sequence_id"] = seq;
                    }
                }
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                std::cout << json({{"id", group.at("id")}, {"results", results}, {"elapsed_ms", ms},
                    {"decode_calls", decode_calls}, {"mixed_sequence_calls", mixed_sequence_calls},
                    {"max_sequences_per_call", max_sequences_per_call},
                    {"max_logit_rows_per_call", max_logit_rows_per_call},
                    {"generated_tokens", 0}}).dump() << std::endl;
            } catch (const std::exception & e) {
                std::cout << json({{"error", e.what()}}).dump() << std::endl;
            }
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 3;
    }
    llama_backend_free();
    return 0;
}
