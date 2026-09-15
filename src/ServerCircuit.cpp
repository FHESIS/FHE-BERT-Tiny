#include "ServerCircuit.h"
#include <chrono>

using namespace std::chrono;

namespace server_circuit {

namespace {

Ctxt classifier(FHEController& controller, Ctxt input, bool verbose, utils::ServerLog& log) {
    auto log_start = log.stage_start("classifier");

    Ptxt weight = controller.read_plain_input("./weights-sst2/classifier_weight.txt", input->GetLevel());
    Ptxt bias = controller.read_plain_expanded_input("./weights-sst2/classifier_bias.txt", input->GetLevel());

    Ctxt output = controller.mult(input, weight);

    output = controller.rotsum(output, 128, 1);

    output = controller.add(output, bias);

    vector<double> mask;
    for (int i = 0; i < controller.num_slots; i++) {
        mask.push_back(0);
    }

    mask[0] = 1;
    mask[128] = 1;

    output = controller.mult(output, controller.encrypt(mask, output->GetLevel()));

    output = controller.add(output, controller.rotate(controller.rotate(output, -1), 128));

    log.stage_end("classifier", log_start, {{"level", std::to_string(output->GetLevel())}});

    return output;
}

Ctxt pooler(FHEController& controller, Ctxt input, bool verbose, utils::ServerLog& log) {
    auto start = high_resolution_clock::now();
    auto log_start = log.stage_start("pooler");

    double tanhScale = 1 / 30.0;

    Ptxt weight = controller.read_plain_input("./weights-sst2/pooler_dense_weight.txt", input->GetLevel(), tanhScale);
    Ptxt bias = controller.read_plain_repeated_input("./weights-sst2/pooler_dense_bias.txt", input->GetLevel(), tanhScale);

    Ctxt output = controller.mult(input, weight);

    output = controller.rotsum(output, 128, 128);

    output = controller.add(output, bias);

    output = controller.bootstrap(output, 0, verbose); // Meta-BTS (numIterations=2) for extra precision

    output = controller.eval_tanh_function(output, -1, 1, tanhScale, 300);

    if (verbose) cout << "The evaluation of Pooler took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("pooler", log_start, {{"level", std::to_string(output->GetLevel())}});

    return output;
}

Ctxt encoder2(FHEController& controller, vector<Ctxt> inputs, bool verbose, utils::ServerLog& log) {
    auto start = high_resolution_clock::now();
    auto log_start = log.stage_start("encoder2.self_attention");

    Ptxt query_w = controller.read_plain_input("./weights-sst2/layer1_attself_query_weight.txt", inputs[0]->GetLevel());
    Ptxt query_b = controller.read_plain_repeated_input("./weights-sst2/layer1_attself_query_bias.txt", inputs[0]->GetLevel());
    Ptxt key_w = controller.read_plain_input("./weights-sst2/layer1_attself_key_weight.txt", inputs[0]->GetLevel());
    Ptxt key_b = controller.read_plain_repeated_input("./weights-sst2/layer1_attself_key_bias.txt", inputs[0]->GetLevel());

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);

    scores = controller.bootstrap(scores, 0, verbose); // Meta-BTS (numIterations=2) for extra precision

    scores = controller.eval_exp(scores, inputs.size());

    scores = controller.mult(scores, 1 / 500.0); //Here values are scaled down in order to achieve better accuracy with bootstrapping
    scores = controller.bootstrap(scores, 0, verbose); // Meta-BTS (numIterations=2) for extra precision
    scores = controller.mult(scores, 500.0);

    Ctxt scores_sum = controller.rotsum(scores, 128, 128);

    Ctxt scores_denominator = controller.eval_inverse_naive_2(scores_sum, 3, 145000, 1);

    scores_denominator = controller.bootstrap(scores_denominator, 0, verbose); // Meta-BTS (numIterations=2) for extra precision

    scores = controller.mult(scores, scores_denominator);

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());

    Ptxt value_w = controller.read_plain_input("./weights-sst2/layer1_attself_value_weight.txt", inputs[0]->GetLevel());
    Ptxt value_b = controller.read_plain_repeated_input("./weights-sst2/layer1_attself_value_bias.txt", inputs[0]->GetLevel());

    vector<Ctxt> V = controller.matmulRE(inputs, value_w, value_b);

    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    if (verbose) cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("encoder2.self_attention", log_start, {{"level", std::to_string(output[0]->GetLevel())}});
    // Here the precision is 0.9868

    /*
     * I remove all the ciphertexts except the first corresponding to the CLS token
     */
    Ctxt copyFirst = output[0];
    output.clear();
    output.push_back(copyFirst);

    start = high_resolution_clock::now();
    log_start = log.stage_start("encoder2.self_output");

    Ptxt dense_w = controller.read_plain_input("./weights-sst2/layer1_selfoutput_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("./weights-sst2/layer1_selfoutput_bias.txt", output[0]->GetLevel() + 1); // Bias only do 12 reps

    output = controller.matmulCR(output, dense_w, dense_b);

    for (size_t i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    Ctxt wrappedOutput = controller.wrapUpExpanded(output);

    Ptxt precomputed_mean = controller.read_plain_repeated_input("./weights-sst2/layer1_selfoutput_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    wrappedOutput = controller.bootstrap(wrappedOutput, 0, verbose); // Meta-BTS (numIterations=2) for extra precision

    Ptxt vy = controller.read_plain_input("./weights-sst2/layer1_selfoutput_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    Ptxt bias = controller.read_plain_expanded_input("./weights-sst2/layer1_selfoutput_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    Ctxt output_copy = wrappedOutput->Clone(); // Required at the last layernorm

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    if (verbose) cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("encoder2.self_output", log_start, {{"level", std::to_string(output[0]->GetLevel())}});
    // Up to this point I get 0.9828 precision


    start = high_resolution_clock::now();
    log_start = log.stage_start("encoder2.intermediate_output");

    double GELU_max_abs_value = 1 / 17.0;

    Ptxt intermediate_w_1 = controller.read_plain_input("./weights-sst2/layer1_intermediate_weight1.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("./weights-sst2/layer1_intermediate_weight2.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("./weights-sst2/layer1_intermediate_weight3.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("./weights-sst2/layer1_intermediate_weight4.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    Ptxt intermediate_bias = controller.read_plain_input("./weights-sst2/layer1_intermediate_bias.txt", output[0]->GetLevel() + 1, GELU_max_abs_value);

    output = controller.matmulRElarge(output, dense_weights, intermediate_bias);

    output = controller.generate_containers(output, nullptr);

    for (size_t i = 0; i < output.size(); i++) {
        output[i] = controller.eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 59);
        output[i] = controller.bootstrap(output[i], 0, verbose); // Meta-BTS (numIterations=2) for extra precision
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(output, output.size());

    if (verbose) cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.

    Ptxt output_w_1 = controller.read_plain_input("./weights-sst2/layer1_output_weight1.txt", output[0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("./weights-sst2/layer1_output_weight2.txt", output[0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("./weights-sst2/layer1_output_weight3.txt", output[0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("./weights-sst2/layer1_output_weight4.txt", output[0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("./weights-sst2/layer1_output_bias.txt", output[0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);
    wrappedOutput = controller.wrapUpExpanded(output);

    wrappedOutput = controller.add(wrappedOutput, output_copy);

    precomputed_mean = controller.read_plain_repeated_input("./weights-sst2/layer1_output_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    vy = controller.read_plain_input("./weights-sst2/layer1_output_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    bias = controller.read_plain_expanded_input("./weights-sst2/layer1_output_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    if (verbose) cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("encoder2.intermediate_output", log_start, {{"level", std::to_string(output[0]->GetLevel())}});

    return output[0];
}

vector<Ctxt> encoder1(FHEController& controller, vector<Ctxt> inputs, bool verbose, utils::ServerLog& log) {
    auto start = high_resolution_clock::now();
    auto log_start = log.stage_start("encoder1.self_attention");

    // `inputs` are the already-encrypted embedding ciphertexts handed to the server by the
    // client (see client_main.cpp). Server-side code never reads the client's plaintext
    // embedding files, and never performs the encryption itself -- that boundary used to be
    // blurred: this function previously read `input_folder + "input_N.txt"` (client plaintext)
    // directly off disk and encrypted it here, inside what is nominally the server's circuit.
    if (verbose) cout << inputs.size() << " input ciphertexts received!" << endl << endl;
    log.event("encoder1.inputs_received", {{"count", std::to_string(inputs.size())}});

    Ptxt query_w = controller.read_plain_input("./weights-sst2/layer0_attself_query_weight.txt");
    Ptxt query_b = controller.read_plain_repeated_input("./weights-sst2/layer0_attself_query_bias.txt");
    Ptxt key_w = controller.read_plain_input("./weights-sst2/layer0_attself_key_weight.txt");
    Ptxt key_b = controller.read_plain_repeated_input("./weights-sst2/layer0_attself_key_bias.txt");

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);
    scores = controller.eval_exp(scores, inputs.size());

    Ctxt scores_sum = controller.rotsum(scores, 128, 128);
    Ctxt scores_denominator = controller.eval_inverse_naive(scores_sum, 2, 5000);

    scores = controller.mult(scores, scores_denominator);

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());

    // `value_w`/`value_b` are multiplied against `inputs[i]` (the fresh, level-0 embedding
    // ciphertexts) in matmulRE() below -- NOT against `scores`. Encoding them at
    // `scores->GetLevel() - {2,1}` (levels ~20-21, vs. inputs[i]->GetLevel()==0) was a leftover
    // OpenFHE-CPU idiom: real OpenFHE's EvalMult(ciphertext, plaintext) auto-drops the
    // ciphertext's extra RNS limbs to match a plaintext encoded at a deeper level, so encoding
    // the plaintext "ahead of time" at the level the caller wants worked there. FIDESlib's GPU
    // EvalMult does not replicate that implicit level-adjustment (confirmed via
    // FHEBERT_DEBUG_SCORES instrumentation: inputs[0]->GetLevel()==0 while
    // scores->GetLevel()-2==20), silently multiplying mismatched RNS limb counts and producing a
    // ciphertext whose tracked scale no longer matches its actual numeric scale -- this was the
    // dominant source of the ~2.5-3x undershoot in the final classifier logits relative to the
    // plaintext reference (see notebooks/Precision of FHE Circuit.ipynb for the expected ~1x
    // ratio). Fix: encode at the level the multiplication actually happens at, matching the
    // query/key weights above and layer1's equivalent in encoder2().
    Ptxt value_w = controller.read_plain_input("./weights-sst2/layer0_attself_value_weight.txt", inputs[0]->GetLevel());
    Ptxt value_b = controller.read_plain_repeated_input("./weights-sst2/layer0_attself_value_bias.txt", inputs[0]->GetLevel());

    vector<Ctxt> V = controller.matmulRE(inputs, value_w, value_b);
    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    if (verbose) cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("encoder1.self_attention", log_start, {{"level", std::to_string(output[0]->GetLevel())}});
    // Up to this point I get precision 0.9934

    start = high_resolution_clock::now();
    log_start = log.stage_start("encoder1.self_output");

    Ptxt dense_w = controller.read_plain_input("./weights-sst2/layer0_selfoutput_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("./weights-sst2/layer0_selfoutput_bias.txt", output[0]->GetLevel() + 1); // Bias only do 12 reps

    output = controller.matmulCR(output, dense_w, dense_b);

    for (size_t i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    Ctxt wrappedOutput = controller.wrapUpExpanded(output);

    Ptxt precomputed_mean = controller.read_plain_repeated_input("./weights-sst2/layer0_selfoutput_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    Ptxt vy = controller.read_plain_input("./weights-sst2/layer0_selfoutput_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    Ptxt bias = controller.read_plain_expanded_input("./weights-sst2/layer0_selfoutput_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    wrappedOutput = controller.bootstrap(wrappedOutput, 0, verbose); // Meta-BTS (numIterations=2) for extra precision

    Ctxt output_copy = wrappedOutput->Clone(); // Required at the last layernorm

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    if (verbose) cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("encoder1.self_output", log_start, {{"level", std::to_string(output[0]->GetLevel())}});
    // Up to this point I get 0.9964 precision

    start = high_resolution_clock::now();
    log_start = log.stage_start("encoder1.intermediate_output");

    double GELU_max_abs_value = 1 / 13.5;

    Ptxt intermediate_w_1 = controller.read_plain_input("./weights-sst2/layer0_intermediate_weight1.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("./weights-sst2/layer0_intermediate_weight2.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("./weights-sst2/layer0_intermediate_weight3.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("./weights-sst2/layer0_intermediate_weight4.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    Ptxt intermediate_bias = controller.read_plain_input("./weights-sst2/layer0_intermediate_bias.txt", output[0]->GetLevel() + 1, GELU_max_abs_value);

    output = controller.matmulRElarge(output, dense_weights, intermediate_bias);

    output = controller.generate_containers(output, nullptr);

    for (size_t i = 0; i < output.size(); i++) {
        output[i] = controller.eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 119);
        output[i] = controller.bootstrap(output[i], 0, verbose); // Meta-BTS (numIterations=2) for extra precision
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(output, inputs.size());

    if (verbose) cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    // Up to this point I get 0.9957 precision

    Ptxt output_w_1 = controller.read_plain_input("./weights-sst2/layer0_output_weight1.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("./weights-sst2/layer0_output_weight2.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("./weights-sst2/layer0_output_weight3.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("./weights-sst2/layer0_output_weight4.txt", unwrappedLargeOutput[0][0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("./weights-sst2/layer0_output_bias.txt", unwrappedLargeOutput[0][0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);
    wrappedOutput = controller.wrapUpExpanded(output);

    wrappedOutput = controller.add(wrappedOutput, output_copy);

    precomputed_mean = controller.read_plain_repeated_input("./weights-sst2/layer0_output_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    vy = controller.read_plain_input("./weights-sst2/layer0_output_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    bias = controller.read_plain_expanded_input("./weights-sst2/layer0_output_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    if (verbose) cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    // Removed: this used to call a decrypting debug helper (print()/print_expanded()) here.
    // Server-role code must never decrypt, even for --verbose debugging; per-stage timing
    // and ciphertext level are already captured above and in the ServerLog JSONL output.
    log.stage_end("encoder1.intermediate_output", log_start, {{"level", std::to_string(output[0]->GetLevel())}});
    // Up to this point I get 0.9965 precision

    return output;
}

} // namespace

EvalResult run_server_circuit(FHEController& controller, vector<Ctxt> inputs, bool verbose, utils::ServerLog& log) {
    EvalResult result;

    log.event("request.received", {{"num_input_ciphertexts", std::to_string(inputs.size())}});

    // Retry wrapper: the GPU bootstrap occasionally produces a catastrophically noisy
    // ciphertext (observed via instrumenting OpenFHE's Decode() -- the identical circuit on
    // the identical input decoded fine with ~+23 bits of precision on some runs and failed
    // ("approximation error too high") with ~-4 bits on others, a ~27-bit swing). That points
    // to an intermittent bug (most likely a race condition in FIDESlib's CUDA bootstrap
    // kernels) rather than a per-input or per-parameter precision shortfall, so more
    // circuit_depth headroom doesn't reliably fix it. Retrying the whole circuit re-rolls
    // that kernel's execution and empirically succeeds within a couple of attempts.
    //
    // This retry count, and whether the process ultimately aborts (see main()'s
    // lbcrypto::OpenFHEException rethrow after max_attempts), is itself server-observable
    // metadata -- logged here so the security test harness can check whether it correlates
    // with anything about the (chosen) plaintext input.
    const int max_attempts = 3;
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        result.attempts = attempt;
        auto attempt_start = log.stage_start("circuit.attempt");
        try {
            vector<Ctxt> encoder1output = encoder1(controller, inputs, verbose, log);
            Ctxt encoder2output = encoder2(controller, encoder1output, verbose, log);

            Ctxt pooled = pooler(controller, encoder2output, verbose, log);
            Ctxt classified = classifier(controller, pooled, verbose, log);
            classified = controller.bootstrap(classified, 0, verbose); // Meta-BTS (numIterations=2) for extra precision before the final decode

            result.output = classified;
            result.succeeded = true;
            log.stage_end("circuit.attempt", attempt_start,
                           {{"attempt", std::to_string(attempt)}, {"outcome", "\"success\""}});
            break;
        } catch (const lbcrypto::OpenFHEException& e) {
            result.last_error = e.what();
            log.stage_end("circuit.attempt", attempt_start,
                           {{"attempt", std::to_string(attempt)},
                            {"outcome", "\"openfhe_exception\""},
                            {"error", utils::ServerLog::json_string(e.what())}});
            if (attempt == max_attempts) {
                log.event("request.failed", {{"attempts", std::to_string(attempt)}});
                throw;
            }
        }
    }

    log.event("request.completed",
              {{"attempts", std::to_string(result.attempts)}, {"output_level", std::to_string(result.output->GetLevel())}});

    return result;
}

} // namespace server_circuit
