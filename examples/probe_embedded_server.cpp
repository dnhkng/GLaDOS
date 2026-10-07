// Embed upstream server-context without an HTTP listener or downstream patches.
// Requires SERVER_TASK_TYPE_DECISION (tested at llama.cpp 6c73b3e12).
// Usage: probe_embedded_server MODEL [GPU_LAYERS] [score|completion]
#include "server-context.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) return 2;
    const std::string mode = argc == 4 ? argv[3] : "score";
    if (mode != "score" && mode != "completion") return 2;
    const bool score_only = mode == "score";
    ggml_backend_load_all();
    common_init();
    llama_backend_init();
    common_params params;
    params.model.path = argv[1];
    params.n_gpu_layers = argc >= 3 ? std::stoi(argv[2]) : -2;
    params.no_op_offload = params.n_gpu_layers == 0;
    params.fit_params = false;
    params.n_ctx = 8192;
    params.n_parallel = 4;
    params.n_batch = params.n_ubatch = 512;
    params.n_outputs_max = 4;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 8;
    params.cont_batching = true;
    params.swa_full = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_ram_mib = 0;
    params.cache_idle_slots = false;
    params.enable_reasoning = 0;
    json report;
    int status = 0;
    {
        server_context server;
        if (!server.load_model(params)) return 3;
        report["build_info"] = server.get_meta().build_info;
        report["model"] = argv[1];
        report["slots"] = 4;
        report["gpu_layers"] = params.n_gpu_layers;
        report["http_listener_started"] = false;
        report["unmodified_server_engine"] = true;
        report["fourth_request_type"] = score_only ? "SERVER_TASK_TYPE_DECISION" : "SERVER_TASK_TYPE_COMPLETION";
        report["fourth_request_generated_tokens"] = score_only ? 0 : 1;
        const auto vocab = llama_model_get_vocab(llama_get_model(server.get_llama_context()));
        std::vector<llama_token> label_ids;
        for (const std::string label : {"A", "B", "C"}) {
            llama_token token;
            if (llama_tokenize(vocab, label.data(), label.size(), &token, 1, false, true) != 1) return 6;
            label_ids.push_back(token);
        }
        std::vector<std::unique_ptr<server_response_reader>> readers;
        for (int i = 0; i < 4; ++i) {
            readers.emplace_back(new server_response_reader(server.get_response_reader()));
            readers.back()->polling_interval_seconds = 1;
        }
        std::mutex mutex;
        std::condition_variable ready;
        std::array<int, 4> partials{};
        std::array<bool, 4> finished{};
        std::array<json, 4> finals;
        std::atomic<bool> abort{false};
        const auto begin = std::chrono::steady_clock::now();
        const auto deadline = begin + std::chrono::seconds(120);
        auto elapsed = [&] {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        };
        auto submit = [&](int i, const std::string & prompt, int n_predict) {
            server_task task(i == 3 && score_only ? SERVER_TASK_TYPE_DECISION : SERVER_TASK_TYPE_COMPLETION);
            task.id = readers[i]->get_new_id();
            task.id_slot = i;
            task.cli = true;
            task.cli_prompt = prompt;
            task.params.stream = i != 3 || !score_only;
            task.params.n_predict = n_predict;
            task.params.sampling.temp = 0;
            task.params.return_tokens = true;
            if (i == 3 && score_only) task.decision.labels = label_ids;
            if (i == 3 && !score_only) task.params.sampling.n_probs = 8;
            readers[i]->post_task(std::move(task));
        };
        auto collect = [&](int i) {
            while (readers[i]->has_next()) {
                auto result = readers[i]->next([&] {
                    return abort.load() || std::chrono::steady_clock::now() >= deadline;
                });
                if (!result) break;
                std::lock_guard<std::mutex> lock(mutex);
                if (result->is_error()) {
                    finals[i] = result->to_json();
                    abort = true;
                } else if (result->is_stop()) {
                    finished[i] = true;
                    finals[i] = result->to_json();
                    finals[i]["received_at_ms"] = elapsed();
                    if (i == 3 && score_only) {
                        report["fourth_score_response_ms"] = elapsed();
                        report["first_three_finished_at_fourth_output"] = json::array({finished[0], finished[1], finished[2]});
                        const auto scores = dynamic_cast<server_task_result_decision *>(result.get());
                        if (!scores || scores->scores.size() != label_ids.size()) abort = true;
                        else report["fourth_selected_label"] = std::string(1, 'A' + std::distance(
                            scores->scores.begin(), std::max_element(scores->scores.begin(), scores->scores.end())));
                    }
                } else {
                    const auto partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
                    if (!partial || partial->is_begin || partial->is_progress || partial->n_decoded == 0) continue;
                    ++partials[i];
                    if (i == 3) {
                        report["fourth_first_partial_ms"] = elapsed();
                        report["first_three_finished_at_fourth_output"] = json::array({finished[0], finished[1], finished[2]});
                        report["fourth_first_token_response"] = result->to_json();
                        report["fourth_selected_label"] = partial->content;
                    }
                }
                ready.notify_all();
                if (abort) break;
            }
        };
        std::vector<std::thread> collectors;
        const std::string header = "<|turn>user\n";
        const std::string suffix = "<turn|>\n<|turn>model\n";
        for (int i = 0; i < 3; ++i) {
            submit(i, header + "List integers from " + std::to_string(i * 1000 + 1)
                      + " to " + std::to_string(i * 1000 + 1000) + ", separated by commas. Do not stop early." + suffix, 48);
            collectors.emplace_back(collect, i);
        }
        std::thread engine([&] { server.start_loop(); });
        {
            std::unique_lock<std::mutex> lock(mutex);
            const bool active = ready.wait_until(lock, deadline, [&] {
                return abort.load() || (partials[0] && partials[1] && partials[2]);
            });
            if (!active || abort || finished[0] || finished[1] || finished[2]) {
                abort = true;
                status = 4;
            } else {
                report["fourth_submitted_ms"] = elapsed();
                report["first_three_partials_at_fourth_submission"] = json::array({partials[0], partials[1], partials[2]});
                report["first_three_finished_at_fourth_submission"] = json::array({finished[0], finished[1], finished[2]});
            }
        }
        if (!abort) {
            submit(3, header + "Choose exactly one letter. A: red. B: blue. C: green. "
                      "What colour is a clear daytime sky? Output only the letter." + suffix, score_only ? 0 : 1);
            collectors.emplace_back(collect, 3);
        }
        for (auto & collector : collectors) collector.join();
        server.terminate();
        engine.join();
        report["final_results"] = json::array();
        for (const auto & result : finals) report["final_results"].push_back(result);
        report["partial_counts"] = json::array({partials[0], partials[1], partials[2], partials[3]});
        const bool all_finished = finished[0] && finished[1] && finished[2] && finished[3];
        report["all_requests_completed"] = all_finished;
        report["late_request_completed_before_existing"] = all_finished
            && finals[3]["received_at_ms"].get<double>() < finals[0]["received_at_ms"].get<double>()
            && finals[3]["received_at_ms"].get<double>() < finals[1]["received_at_ms"].get<double>()
            && finals[3]["received_at_ms"].get<double>() < finals[2]["received_at_ms"].get<double>();
        report["elapsed_ms"] = elapsed();
        if (!all_finished || abort) status = 5;
    }
    std::cout << report.dump(2) << '\n';
    llama_backend_free();
    return status;
}
