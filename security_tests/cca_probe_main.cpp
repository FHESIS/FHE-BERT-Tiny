//
// cca_probe_main.cpp -- fast, local probe for CKKS ciphertext-malleability / decryption-oracle
// risk. This is a SEPARATE tool from the production FHE-BERT-Tiny client/server binary: it is
// research/test code that deliberately loads the secret key into the same process as the
// "attacker" logic, in order to simulate what happens if a decryption oracle is exposed
// (directly, or indirectly via an error message, a status code, or any endpoint that returns
// information derived from decrypting attacker-influenced ciphertexts). It does not touch
// BERT-tiny, the tokenizer, or any of the production input/weight files.
//
// Two experiments, run for `trials` iterations each (default 500):
//
//  1. Plaintext-recovery-by-malleability. CKKS ciphertexts are additively homomorphic and carry
//     no integrity tag (no MAC/AEAD): Dec(c + Enc(delta)) == Dec(c) + delta for any
//     attacker-chosen `delta`, using ONLY public-key operations to produce c + Enc(delta). If
//     *any* oracle exists that will decrypt (or leak information derived from decrypting) a
//     ciphertext an attacker submits, the attacker recovers the original plaintext message of a
//     ciphertext they never held the key for, exactly, by choosing delta and subtracting it back
//     out of the oracle's answer. This experiment measures the recovery error empirically.
//
//  2. IND-CCA1-style distinguishing game. A hidden bit b selects one of two fixed, attacker-known
//     messages m0/m1; the attacker is given Enc(m_b) and one query to a decryption oracle against
//     a *derived* ciphertext (never the challenge ciphertext itself, so this also covers services
//     that special-case "don't decrypt exactly what you were handed"), and must guess b. Chance
//     is 50%; this measures the empirical success rate and its statistical significance.
//
// Output: security_tests/cca_probe_results.csv (one row per trial) for the Python statistical
// analysis in security_tests/analyze_cca.py.
//
#include "FHEController.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>

FHEController controller;

int main(int argc, char *argv[]) {
    int trials = argc > 1 ? std::atoi(argv[1]) : 500;
    const int VEC_SLOTS = 16;

    cout << "Loading crypto context (public + evaluation key material only)..." << endl;
    controller.load_context_server(false);
    // context->LoadContext() (pushing the context to the GPU) only happens at the end of this
    // call -- see FHEController::generate_context()'s comment on ordering. We don't need
    // bootstrapping or rotation for this probe (only encrypt/add/decrypt), but there's currently
    // no lighter-weight public entry point that reaches LoadContext() without it.
    controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);

    cout << "Loading secret key -- ONLY to play the role of the decryption oracle being probed. "
         << "See the file header comment: this models what an attacker gains if such an oracle "
         << "is ever exposed, it is not something the attacker itself needs to hold." << endl;
    controller.load_client_secret_key(false);

    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    ofstream csv("security_tests/cca_probe_results.csv");
    csv << "trial,recovery_linf_error,recovery_l2_error,distinguish_hidden_bit,distinguish_guess,distinguish_correct\n";

    int distinguish_correct = 0;
    double max_linf = 0.0;
    double sum_l2 = 0.0;

    auto t_start = std::chrono::steady_clock::now();

    for (int t = 0; t < trials; t++) {
        // ---- Experiment 1: plaintext recovery via additive malleability ----
        vector<double> secret(VEC_SLOTS);
        for (auto &v : secret) v = val_dist(rng);

        Ctxt c = controller.encrypt(secret, 0, VEC_SLOTS); // what an eavesdropper observes on the wire

        vector<double> delta(VEC_SLOTS);
        for (auto &v : delta) v = val_dist(rng) * 5.0; // attacker's own chosen offset, known to them

        Ptxt delta_pt = controller.encode(delta, c->GetLevel(), VEC_SLOTS);
        Ctxt c_tampered = controller.add(c, delta_pt); // pure public-key-derivable operation

        vector<double> oracle_result = controller.decrypt_tovector(c_tampered, VEC_SLOTS);

        double linf = 0.0, l2 = 0.0;
        for (int i = 0; i < VEC_SLOTS; i++) {
            double recovered = oracle_result[i] - delta[i];
            double err = std::fabs(recovered - secret[i]);
            linf = std::max(linf, err);
            l2 += err * err;
        }
        l2 = std::sqrt(l2);
        max_linf = std::max(max_linf, linf);
        sum_l2 += l2;

        // ---- Experiment 2: IND-CCA1-style distinguishing game ----
        vector<double> m0(VEC_SLOTS, 1000.0), m1(VEC_SLOTS, -1000.0);
        int hidden_bit = static_cast<int>(rng() % 2);
        Ctxt challenge = controller.encrypt(hidden_bit == 0 ? m0 : m1, 0, VEC_SLOTS);

        // Never query the oracle on `challenge` itself -- derive a distinct ciphertext first
        // (add a zero plaintext works: still a different Ctxt object/handle, and the point being
        // demonstrated is that ANY derivative suffices, not specifically the identity operation).
        Ptxt zero_pt = controller.encode(vector<double>(VEC_SLOTS, 0.0), challenge->GetLevel(), VEC_SLOTS);
        Ctxt derived = controller.add(challenge, zero_pt);

        double guess_val = controller.decrypt_tovector(derived, VEC_SLOTS)[0];
        int guess = (guess_val > 0) ? 0 : 1;
        if (guess == hidden_bit) distinguish_correct++;

        csv << t << "," << linf << "," << l2 << "," << hidden_bit << "," << guess << "," << (guess == hidden_bit ? 1 : 0) << "\n";

        if ((t + 1) % 50 == 0) {
            cout << "  " << (t + 1) << "/" << trials << " trials done" << endl;
        }
    }

    csv.close();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();

    cout << endl << "=== CCA / malleability probe summary (" << trials << " trials, " << elapsed << " ms) ===" << endl;
    cout << "Experiment 1 (plaintext recovery via malleability): max L-inf error = " << max_linf
         << ", mean L2 error = " << (sum_l2 / trials) << endl;
    cout << "Experiment 2 (IND-CCA1 distinguishing game): accuracy = "
         << (100.0 * distinguish_correct / trials) << "% (chance = 50%)" << endl;
    cout << "Results written to security_tests/cca_probe_results.csv" << endl;

    controller.clear_mask_cache();
    return 0;
}
