//
// Created by Lorenzo on 12/01/24.
//

#ifndef NEWBERT_FHECONTROLLER_H
#define NEWBERT_FHECONTROLLER_H

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include <fideslib.hpp>
#include <thread>
#include <map>
#include <tuple>
#include <functional>
#include "Utils.h"

using namespace std;
using namespace std::chrono;

using namespace utils;

using Ptxt = fideslib::Plaintext;
using Ctxt = fideslib::Ciphertext<fideslib::DCRTPoly>;

class FHEController {
    fideslib::CryptoContext<fideslib::DCRTPoly> context;

public:
    int circuit_depth;
    int num_slots;

    FHEController() {}

    /*
     * Context generating/loading stuff
     */
    void generate_context(bool serialize = false, bool secure = false);
    void generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels, int stc_levels, int relu_deg, bool serialize = false);

    // Client/server key-material boundary.
    //
    // load_context_server() loads the crypto context, the public key and the evaluation
    // (relinearization) key only -- it never opens secret-key.txt, so a controller that has only
    // ever called this is structurally unable to decrypt anything: the secret key bytes are not
    // merely permission-gated, they are not present in process memory at all. This is what the
    // FHE circuit evaluation (encoder1/encoder2/pooler/classifier, see ServerCircuit.cpp) runs
    // against.
    //
    // load_client_secret_key() is the client-only counterpart: it deserializes secret-key.txt
    // into this same controller so decrypt()/decrypt_tovector() become usable. Call it only after
    // the server-side circuit evaluation has already produced its result ciphertext -- so that
    // for the entire duration of circuit evaluation, the process holds no secret-key material at
    // all. See FHEBERT_SECURITY_REPORT.md for why this ordering matters: the previous
    // load_context() loaded the secret key unconditionally, before any evaluation began, into the
    // same object that evaluates the circuit (the local variable was even named
    // "serverSecretKey").
    void load_context_server(bool verbose = true);
    void load_client_secret_key(bool verbose = true);
    bool has_secret_key() const { return secret_key_loaded; }

    void test_context();

    /*
     * Generating bootstrapping and rotation keys stuff
     */
    void generate_bootstrapping_keys(int bootstrap_slots);
    void generate_rotation_keys(vector<int> rotations, bool serialize = false, string filename = "");
    void generate_bootstrapping_and_rotation_keys(vector<int> rotations,
                                                  int bootstrap_slots,
                                                  bool serialize,
                                                  const string& filename);


    void load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose);
    void load_rotation_keys(const string& filename, bool verbose);
    void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);
    void clear_rotation_keys();
    void clear_context(int bootstrapping_key_slots);


    /*
     * CKKS Encoding/Decoding/Encryption/Decryption
     */
    Ptxt encode(const vector<double>& vec, int level, int plaintext_num_slots);
    Ptxt encode(double val, int level, int plaintext_num_slots);
    Ctxt encrypt(const vector<double>& vec, int level = 0, int plaintext_num_slots = 0);
    Ctxt encrypt_ptxt(const Ptxt& p);
    Ptxt decrypt(const Ctxt& c);
    vector<double> decrypt_tovector(const Ctxt& c, int slots);

    /*
     * Homomorphic operations
     */
    Ctxt add(const Ctxt& c1, const Ctxt& c2);
    Ctxt add(const Ctxt& c1, const Ptxt& c2);
    Ctxt add(vector<Ctxt> c);
    Ctxt mult(const Ctxt& c1, const Ctxt& c2);
    Ctxt mult(const Ctxt& c, double d);
    Ctxt mult(const Ctxt& c, const Ptxt& p);
    Ctxt rotate(const Ctxt& c, int index);
    Ctxt bootstrap(const Ctxt& c, bool timing = false);
    Ctxt bootstrap(const Ctxt& c, int precision, bool timing = false);
    Ctxt relu(const Ctxt& c, double scale, bool timing = false);
    Ctxt relu_wide(const Ctxt& c, double a, double b, int degree, double scale, bool timing = false);

    /*
     * I/O
     */
    Ctxt read_input(const string& filename, double scale = 1);
    Ctxt read_repeated_input(const string& filename, double scale = 1);
    Ctxt read_expanded_input(const string& filename, double scale = 1);

    Ptxt read_plain_input(const string& filename, int level = 0, double scale = 1);
    //Ptxt read_plain_512_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_repeated_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_repeated_512_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_expanded_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_expanded_input(const string& filename, int level, double scale, int num_inputs);


    void print(const Ctxt& c, int slots = 0, string prefix = "");
    void print_padded(const Ctxt& c, int slots = 0, int padding = 1, string prefix = "");
    void print_expanded(const Ctxt& c, int slots = 0, int expansion_factor = 1, string prefix = "");
    void print_min_max(const Ctxt& c);

    Ctxt rotsum(const Ctxt &in, int slots, int padding);
    Ctxt rotsum_padded(const Ctxt &in, int slots);

    Ctxt repeat(const Ctxt &in, int slots);
    Ctxt repeat(const Ctxt &in, int slots, int padding);

    vector<Ctxt> matmulRE(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias );
    vector<Ctxt> matmulRE(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias, int row_size, int padding );
    vector<Ctxt> matmulRE(vector<Ctxt> rows, const Ctxt& weight, int row_size, int padding );
    vector<Ctxt> matmulRElarge(vector<Ctxt>& rows, const vector<Ptxt>& weight, const Ptxt& bias, double mask_value = 1);
    vector<Ctxt> matmulCR(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias );
    vector<Ctxt> matmulCR(vector<Ctxt> rows, const Ctxt& matrix);
    vector<Ctxt> matmulCRlarge(vector<vector<Ctxt>> rows, vector<Ptxt> weights, const Ptxt& bias);

    Ctxt matmulScores(vector<Ctxt> queries, const Ctxt& key);

    Ctxt wrapUpRepeated(vector<Ctxt> vectors);
    Ctxt wrapUpExpanded(vector<Ctxt> vectors);
    vector<Ctxt> unwrapExpanded(Ctxt c, int inputs_num);
    vector<vector<Ctxt>> unwrapRepeatedLarge(vector<Ctxt> c, int input_number);
    vector<Ctxt> unwrapScoresExpanded(Ctxt c, int inputs_num);
    vector<Ctxt> unwrap_512_in_4_128(const Ctxt& c, int index);

    vector<Ctxt> generate_containers(vector<Ctxt> inputs, const Ptxt& bias);
    Ctxt wrap_containers(vector<Ctxt> inputs, int inputs_number);

    Ctxt mask_block(const Ctxt& c, int from, int to, double mask_value = 1);
    Ctxt mask_heads(const Ctxt& c, double mask_value = 1);
    Ctxt mask_mod_n(const Ctxt& c, int n);
    Ctxt mask_mod_n(const Ctxt& c, int n, int padding, int max_slots);
    Ctxt mask_first_n(const Ctxt& c, int n, double mask_value = 1);

    Ctxt eval_exp(const Ctxt& c, int inputs_number);
    Ctxt eval_inverse(const Ctxt& c, double min, double max);
    Ctxt eval_inverse_naive(const Ctxt& c, double min, double max);
    Ctxt eval_inverse_naive_2(const Ctxt& c, double min, double max, double mult);
    Ctxt eval_gelu_function(const Ctxt& c, double min, double max, double mult, int degree);
    Ctxt eval_tanh_function(const Ctxt& c, double min, double max, double mult, int degree);

    vector<Ctxt> slicing(vector<Ctxt> &arr, int X, int Y);

    void save(Ctxt v, string filename);
    void save(vector<Ctxt> v, string filename);
    vector<Ctxt> load_vector(string filename);
    Ctxt load_ciphertext(string filename);

    int relu_degree = 119;
    string parameters_folder = "keys";

private:
    fideslib::KeyPair<fideslib::DCRTPoly> key_pair;
    vector<uint32_t> level_budget = {4, 4};
    bool serialize_context_pending = false;

    // True only once secret-key.txt has actually been deserialized into key_pair.secretKey by
    // load_client_secret_key() (or generated by generate_context()). decrypt()/decrypt_tovector()
    // refuse to run while this is false, so a server-role controller can never decrypt even by
    // accident.
    bool secret_key_loaded = false;

    // The mask_* helpers below encode a fresh all-zero-except-a-pattern plaintext of num_slots
    // (16384) values on every call, even though within one circuit evaluation they are invoked
    // repeatedly (once per token/row) with the exact same (kind, params, level) -- e.g.
    // unwrapExpanded() calls mask_mod_n() with identical arguments for every token. The pattern
    // and level fully determine the encoded plaintext, so it's cached instead of being rebuilt
    // and re-NTT-encoded each time. Keyed by (kind, p1, p2, p3, level, mask_value).
    map<tuple<int, int, int, int, int, double>, Ptxt> mask_ptxt_cache;
    Ptxt get_cached_mask(int kind, int p1, int p2, int p3, int level, double mask_value,
                         const std::function<vector<double>()>& build);

public:
    // Must be called before returning from main(), NOT left to the global `controller`'s
    // destructor at program exit: destroying these cached Ptxt objects calls into FIDESlib's
    // GPU-plaintext eviction path, but by the time global destructors run, the CUDA runtime's
    // own atexit-registered stream-pool teardown has already fired (it's registered on first
    // CUDA use inside main(), i.e. *after* this global object was constructed, so it tears down
    // *before* this object's destructor in the LIFO exit sequence). Evicting here, while CUDA is
    // still known-alive, avoids that use-after-teardown segfault.
    void clear_mask_cache() { mask_ptxt_cache.clear(); }


};


#endif //NEWBERT_FHECONTROLLER_H
