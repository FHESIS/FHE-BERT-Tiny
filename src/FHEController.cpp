//
// Created by Lorenzo on 24/10/23.
//

#include "FHEController.h"

void FHEController::generate_context(bool serialize, bool secure) {
    fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters;

    num_slots = 1 << 14;

    parameters.SetSecretKeyDist(fideslib::SPARSE_TERNARY);
    parameters.SetSecurityLevel(fideslib::HEStd_128_classic);
    if (!secure) parameters.SetSecurityLevel(fideslib::HEStd_NotSet);
    parameters.SetNumLargeDigits(4); //d_{num} Se lo riduci, aumenti il logQP, se lo aumenti, aumenti memori
    parameters.SetKeySwitchTechnique(fideslib::HYBRID);
    parameters.SetRingDim(1 << 16);
    if (!secure) parameters.SetRingDim(1 << 15);
    parameters.SetBatchSize(num_slots);
    parameters.SetDevices({0});

    level_budget = {4, 4};

    fideslib::ScalingTechnique rescaleTech = fideslib::FLEXIBLEAUTOEXT; // higher-precision scaling: reserves an
    // extra modulus to preserve precision through the final decode, fixing "approximation error is too high"
    // at the last bootstrap+decrypt step.

    int dcrtBits               = 52;
    int firstMod               = 55;

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    uint32_t approxBootstrapDepth = 4 + 4;

    uint32_t levelsUsedBeforeBootstrap = 15; // +1 vs. the original OpenFHE value: fideslib eval_exp uses a Horner-based
    // polynomial evaluation (no EvalPoly exposed by fideslib) that costs one more
    // multiplicative level than OpenFHE's Paterson-Stockmeyer EvalPoly, so the
    // circuit needs one more level of total budget to avoid a bootstrap call
    // landing exactly at circuit_depth with zero headroom (confirmed via a
    // debug print: crash occurred with c->GetLevel()==circuit_depth==26).
    // +1 again: the final Meta-BTS bootstrap in main() (after classifier()) still landed
    // exactly at circuit_depth with zero headroom, causing Decode() to throw "approximation
    // error is too high" (confirmed via debug print: classified->GetLevel()==circuit_depth==29
    // right before that bootstrap call, decrypting fine at that point).
    // +1 again (14->15): input "This movie was fantastic" still hit the same "approximation
    // error is too high" at that same final bootstrap+decode step, even though other inputs
    // (e.g. the 8-token Dune sentence) decoded fine -- the error is data-dependent, and the
    // prior headroom was still occasionally insufficient. Confirmed FIDESlib's GPU EvalBootstrap
    // ignores numIterations/precision entirely (api/CryptoContext.cpp), so the only available
    // fix is more circuit_depth headroom, requiring a full context/key regeneration.

    circuit_depth = levelsUsedBeforeBootstrap + lbcrypto::FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, lbcrypto::SPARSE_TERNARY);

    cout << endl << "Ciphertexts depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;

    parameters.SetMultiplicativeDepth(circuit_depth);

    context = fideslib::GenCryptoContext(parameters);

    cout << "Context built, generating keys..." << endl;

    context->Enable(fideslib::PKE);
    context->Enable(fideslib::KEYSWITCH);
    context->Enable(fideslib::LEVELEDSHE);
    context->Enable(fideslib::ADVANCEDSHE);
    context->Enable(fideslib::FHE);

    key_pair = context->KeyGen();
    secret_key_loaded = true;

    context->EvalMultKeyGen(key_pair.secretKey);

    // fideslib::CryptoContext::LoadContext() must be the *last* precomputation step -- it pushes
    // everything generated so far to the GPU, and any further EvalRotateKeyGen/EvalBootstrapSetup
    // call after it throws ("Context is already loaded"). Rotation and bootstrap keys are
    // generated later (generate_bootstrapping_and_rotation_keys), so LoadContext is deferred to
    // the end of that function instead of being called here.

    cout << "Generated." << endl;

    // fideslib requires crypto-context.txt to be serialized only after EVERY key (mult, bootstrap,
    // and rotation) has been generated on the context -- a context serialized earlier (as OpenFHE's
    // own factory-keyed CryptoContext would allow) silently forgets which rotation indices are
    // registered once deserialized, even though the automorphism key file itself deserializes fine,
    // causing EvalRotate to fail with "Rotation index N not found" for every index. So this only
    // records the intent to serialize; the actual writing happens at the end of
    // generate_bootstrapping_and_rotation_keys(), once rotation keys exist too.
    serialize_context_pending = serialize;
}

void FHEController::generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels,
                                     int stc_levels, int relu_deg, bool serialize) {

    fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters;

    num_slots = 1 << 14;

    parameters.SetSecretKeyDist(fideslib::SPARSE_TERNARY);
    //parameters.SetSecurityLevel(fideslib::HEStd_128_classic);
    parameters.SetSecurityLevel(fideslib::HEStd_NotSet);
    parameters.SetNumLargeDigits(digits_hks);
    parameters.SetKeySwitchTechnique(fideslib::HYBRID);
    parameters.SetRingDim(1 << log_ring);
    parameters.SetBatchSize(num_slots);
    parameters.SetDevices({0});

    level_budget = vector<uint32_t>();

    level_budget.push_back(cts_levels);
    level_budget.push_back(stc_levels);

    int dcrtBits = log_primes;
    int firstMod = log_scale;

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
    parameters.SetFirstModSize(firstMod);

    uint32_t approxBootstrapDepth = 4 + 4; //During EvalRaise, Chebyshev, DoubleAngle

    uint32_t levelsUsedBeforeBootstrap = 14; // +1 vs. the original OpenFHE value: fideslib eval_exp uses a Horner-based
    // polynomial evaluation (no EvalPoly exposed by fideslib) that costs one more
    // multiplicative level than OpenFHE's Paterson-Stockmeyer EvalPoly, so the
    // circuit needs one more level of total budget to avoid a bootstrap call
    // landing exactly at circuit_depth with zero headroom (confirmed via a
    // debug print: crash occurred with c->GetLevel()==circuit_depth==26).
    // +1 again: the final Meta-BTS bootstrap in main() (after classifier()) still landed
    // exactly at circuit_depth with zero headroom, causing Decode() to throw "approximation
    // error is too high" (confirmed via debug print: classified->GetLevel()==circuit_depth==29
    // right before that bootstrap call, decrypting fine at that point).

    circuit_depth = levelsUsedBeforeBootstrap +
                    lbcrypto::FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, lbcrypto::SPARSE_TERNARY);

    cout << endl << "Ciphertexts depth: " << circuit_depth << ", available multiplications: "
         << levelsUsedBeforeBootstrap - 2 << endl;

    parameters.SetMultiplicativeDepth(circuit_depth);

    context = fideslib::GenCryptoContext(parameters);

    cout << "Context built, generating keys..." << endl;

    context->Enable(fideslib::PKE);
    context->Enable(fideslib::KEYSWITCH);
    context->Enable(fideslib::LEVELEDSHE);
    context->Enable(fideslib::ADVANCEDSHE);
    context->Enable(fideslib::FHE);

    key_pair = context->KeyGen();
    secret_key_loaded = true;

    context->EvalMultKeyGen(key_pair.secretKey);

    // fideslib::CryptoContext::LoadContext() must be the *last* precomputation step -- it pushes
    // everything generated so far to the GPU, and any further EvalRotateKeyGen/EvalBootstrapSetup
    // call after it throws ("Context is already loaded"). Rotation and bootstrap keys are
    // generated later (generate_bootstrapping_and_rotation_keys), so LoadContext is deferred to
    // the end of that function instead of being called here.

    cout << "Generated." << endl;

    // See the comment in the other generate_context() overload: serialization is deferred to
    // generate_bootstrapping_and_rotation_keys(), once rotation keys also exist on the context.
    serialize_context_pending = serialize;
}

void FHEController::load_context_server(bool verbose) {
    if (verbose) cout << "Reading serialized context (server role: public/evaluation material only)..." << endl;

    if (!fideslib::Serial::DeserializeFromFile("../" + parameters_folder + "/crypto-context.txt", context, fideslib::SerType::BINARY)) {
        cerr << "I cannot read serialized data from: " << "../" + parameters_folder + "/crypto-context.txt" << endl;
        exit(1);
    }

    fideslib::PublicKey<fideslib::DCRTPoly> clientPublicKey;
    if (!fideslib::Serial::DeserializeFromFile("../" + parameters_folder + "/public-key.txt", clientPublicKey, fideslib::SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.txt" << endl;
        exit(1);
    }

    key_pair.publicKey = clientPublicKey;
    // secret-key.txt is deliberately never opened here. key_pair.secretKey stays default-
    // constructed and secret_key_loaded stays false, so decrypt()/decrypt_tovector() refuse to
    // run against this controller until (and unless) load_client_secret_key() is called
    // separately, later, on the client side of the boundary.

    std::ifstream multKeyIStream("../" + parameters_folder + "/mult-keys.txt", ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "mult-keys.txt" << endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, fideslib::SerType::BINARY)) {
        cerr << "Could not deserialize eval mult key file" << endl;
        exit(1);
    }

    context->SetDevices({0});
    // LoadContext() is deferred to the end of load_bootstrapping_and_rotation_keys(): it must be
    // the last precomputation step, called only after bootstrap setup and rotation keys are
    // also in place (see the comment in generate_context()).

    level_budget = {4, 4};

    if (verbose) cout << "CtoS: " << level_budget[0] << ", StoC: " << level_budget[1] << endl;

    uint32_t approxBootstrapDepth = 8;

    uint32_t levelsUsedBeforeBootstrap = 15; // Must stay in sync with generate_context()'s value -- see the
    // comments there. This function only recomputes bookkeeping (circuit_depth) for a context that was
    // already built and serialized with this depth; it does not itself change the underlying parameters.

    circuit_depth = levelsUsedBeforeBootstrap + lbcrypto::FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, lbcrypto::SPARSE_TERNARY);

    if (verbose) cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;

    num_slots = 1 << 14;
}

void FHEController::load_client_secret_key(bool verbose) {
    if (verbose) cout << "Reading secret-key.txt (client role)..." << endl;

    fideslib::PrivateKey<fideslib::DCRTPoly> clientSecretKey;
    if (!fideslib::Serial::DeserializeFromFile("../" + parameters_folder + "/secret-key.txt", clientSecretKey, fideslib::SerType::BINARY)) {
        cerr << "I cannot read serialized data from secret-key.txt" << endl;
        exit(1);
    }

    key_pair.secretKey = clientSecretKey;
    secret_key_loaded = true;
}


void FHEController::generate_bootstrapping_keys(int bootstrap_slots) {
    context->EvalBootstrapSetup(level_budget, {0, 0}, bootstrap_slots);
    context->EvalBootstrapKeyGen(key_pair.secretKey, bootstrap_slots);
}

void FHEController::generate_rotation_keys(vector<int> rotations, bool serialize, std::string filename) {
    if (serialize && filename.size() == 0) {
        cout << "Filename cannot be empty when serializing rotation keys." << endl;
        return;
    }

    context->EvalRotateKeyGen(key_pair.secretKey, rotations);

    if (serialize) {
        ofstream rotationKeyFile("../" + parameters_folder + "/rot_" + filename, ios::out | ios::binary);
        if (rotationKeyFile.is_open()) {
            if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, fideslib::SerType::BINARY)) {
                cerr << "Error writing rotation keys" << std::endl;
                exit(1);
            }
            cout << "Rotation keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing Rotation keys" << "../" + parameters_folder + "/rot_" + filename << std::endl;
            exit(1);
        }
    }
}

void FHEController::generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, bool serialize, const string& filename) {
    if (serialize && filename.empty()) {
        cout << "Filename cannot be empty when serializing bootstrapping and rotation keys." << endl;
        return;
    }

    generate_bootstrapping_keys(bootstrap_slots);
    generate_rotation_keys(rotations, serialize, filename);

    if (serialize_context_pending) {
        cout << "Now serializing keys ..." << endl;

        ofstream multKeyFile("../" + parameters_folder + "/mult-keys.txt", ios::out | ios::binary);
        if (multKeyFile.is_open()) {
            if (!context->SerializeEvalMultKey(multKeyFile, fideslib::SerType::BINARY)) {
                cerr << "Error writing eval mult keys" << std::endl;
                exit(1);
            }
            cout << "Relinearization Keys have been serialized" << std::endl;
            multKeyFile.close();
        } else {
            cerr << "Error serializing EvalMult keys in \"" << "../" + parameters_folder + "/mult-keys.txt" << "\"" << endl;
            exit(1);
        }

        // Serialized last, deliberately: fideslib needs every key (mult, bootstrap, rotation)
        // already generated on the context before crypto-context.txt is written, or the
        // deserialized context silently forgets which rotation indices are registered (see the
        // comment in generate_context()).
        if (!fideslib::Serial::SerializeToFile("../" + parameters_folder + "/crypto-context.txt", context, fideslib::SerType::BINARY)) {
            cerr << "Error writing serialization of the crypto context to crypto-context.txt" << endl;
        } else {
            cout << "Crypto Context have been serialized" << std::endl;
        }

        if (!fideslib::Serial::SerializeToFile("../" + parameters_folder + "/public-key.txt", key_pair.publicKey, fideslib::SerType::BINARY)) {
            cerr << "Error writing serialization of public key to public-key.txt" << endl;
        } else {
            cout << "Public Key has been serialized" << std::endl;
        }

        if (!fideslib::Serial::SerializeToFile("../" + parameters_folder + "/secret-key.txt", key_pair.secretKey, fideslib::SerType::BINARY)) {
            cerr << "Error writing serialization of public key to secret-key.txt" << endl;
        } else {
            cout << "Secret Key has been serialized" << std::endl;
        }
    }

    context->LoadContext(key_pair.publicKey);
}

void FHEController::load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose) {
    if (verbose) cout << endl << "Loading bootstrapping and rotations keys from " << filename << "..." << endl;

    auto start = start_time();

    context->EvalBootstrapSetup(level_budget, {0, 0}, bootstrap_slots);

    if (verbose)  cout << "(1/2) Bootstrapping precomputations completed!" << endl;


    ifstream rotKeyIStream("../" + parameters_folder + "/rot_" + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "../" + parameters_folder + "/" << "rot_" << filename << std::endl;
        exit(1);
    }

    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, fideslib::SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) cout << "(2/2) Rotation keys read!" << endl;

    context->LoadContext(key_pair.publicKey);

    if (verbose) print_duration(start, "Loading bootstrapping pre-computations + rotations");

    if (verbose) cout << endl;
}

void FHEController::load_rotation_keys(const string& filename, bool verbose) {
    if (verbose) cout << endl << "Loading rotations keys from " << filename << "..." << endl;

    auto start = start_time();

    ifstream rotKeyIStream("../" + parameters_folder + "/rot_" + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "../" + parameters_folder + "/" << "rot_" << filename << std::endl;
        exit(1);
    }

    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, fideslib::SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) {
        cout << "(1/1) Rotation keys read!" << endl;
        print_duration(start, "Loading rotation keys");
        cout << endl;
    }
}

void FHEController::clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots) {
    // fideslib::CryptoContext instances are plain per-object shared_ptrs, not entries in a
    // global factory-keyed registry, so there is no equivalent of OpenFHE's key-cache clearing
    // to perform here.
    clear_rotation_keys();
}

void FHEController::clear_rotation_keys() {
    // No-op under fideslib: see clear_bootstrapping_and_rotation_keys above.
}

void FHEController::clear_context(int bootstrapping_key_slots) {
    if (bootstrapping_key_slots != 0)
        clear_bootstrapping_and_rotation_keys(bootstrapping_key_slots);
    else
        clear_rotation_keys();
}

/*
 * CKKS Encoding/Decoding/Encryption/Decryption
 */
Ptxt FHEController::encode(const vector<double> &vec, int level, int plaintext_num_slots) {
    if (plaintext_num_slots == 0) {
        plaintext_num_slots = num_slots;
    }

    Ptxt p = context->MakeCKKSPackedPlaintext(vec, 1, level, nullptr, plaintext_num_slots);
    p->SetLength(plaintext_num_slots);
    return p;
}

Ptxt FHEController::encode(double val, int level, int plaintext_num_slots) {
    if (plaintext_num_slots == 0) {
        plaintext_num_slots = num_slots;
    }

    vector<double> vec;
    for (int i = 0; i < plaintext_num_slots; i++) {
        vec.push_back(val);
    }

    Ptxt p = context->MakeCKKSPackedPlaintext(vec, 1, level, nullptr, plaintext_num_slots);
    p->SetLength(plaintext_num_slots);
    return p;
}

Ctxt FHEController::encrypt(const vector<double> &vec, int level, int plaintext_num_slots) {
    if (plaintext_num_slots == 0) {
        plaintext_num_slots = num_slots;
    }

    Ptxt p = encode(vec, level, plaintext_num_slots);

    return context->Encrypt(p, key_pair.publicKey);
}

Ctxt FHEController::encrypt_ptxt(const Ptxt& p) {
    Ptxt p_mut = p;
    return context->Encrypt(p_mut, key_pair.publicKey);
}

Ptxt FHEController::decrypt(const Ctxt &c) {
    if (!secret_key_loaded) {
        throw std::runtime_error(
            "FHEController::decrypt(): no secret key is loaded on this controller. Server-role "
            "controllers (load_context_server()) never load one; call load_client_secret_key() "
            "on the client side first.");
    }
    Ptxt p;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &p);
    return p;
}

vector<double> FHEController::decrypt_tovector(const Ctxt &c, int slots) {
    if (!secret_key_loaded) {
        throw std::runtime_error(
            "FHEController::decrypt_tovector(): no secret key is loaded on this controller. "
            "Server-role controllers (load_context_server()) never load one; call "
            "load_client_secret_key() on the client side first.");
    }
    if (slots == 0) {
        slots = num_slots;
    }

    Ptxt p;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &p);
    p->SetSlots(slots);
    p->SetLength(slots);
    vector<double> vec = p->GetRealPackedValue();
    return vec;
}

/*
 * Homomorphic operations
 */
Ctxt FHEController::add(const Ctxt &c1, const Ctxt &c2) {
    return context->EvalAdd(c1, c2);
}

Ctxt FHEController::add(const Ctxt &c1, const Ptxt &c2) {
    Ptxt c2_mutable = c2;
    return context->EvalAdd(c1, c2_mutable);
}

Ctxt FHEController::add(vector<Ctxt> c) {
    return context->EvalAddMany(c);
}

Ctxt FHEController::mult(const Ctxt &c1, double d) {
    Ptxt p = encode(d, c1->GetLevel(), num_slots);
    return context->EvalMult(c1, p);
}

Ctxt FHEController::mult(const Ctxt &c, const Ptxt& p) {
    Ptxt p_mut = p;
    return context->EvalMult(c, p_mut);
}

Ctxt FHEController::mult(const Ctxt &c1, const Ctxt& c2) {
    return context->EvalMult(c1, c2);
}

Ctxt FHEController::rotate(const Ctxt &c, int index) {
    return context->EvalRotate(c, index);
}

Ctxt FHEController::bootstrap(const Ctxt &c, bool timing) {
    auto start = start_time();

    Ctxt res = context->EvalBootstrap(c);

    if (timing) {
        print_duration(start, "Bootstrapping " + to_string(num_slots) + " slots");
    }

    return res;
}

Ctxt FHEController::bootstrap(const Ctxt &c, int precision, bool timing) {
    if (static_cast<int>(c->GetLevel()) + 2 < circuit_depth) {
        cout << "You are bootstrapping with remaining levels! You are at " << to_string(c->GetLevel()) << "/" << circuit_depth - 2 << endl;
    }

    auto start = start_time();

    Ctxt res = context->EvalBootstrap(c, 2, precision);

    if (timing) {
        print_duration(start, "Double Bootstrapping " + to_string(num_slots) + " slots");
    }


    return res;
}

Ctxt FHEController::relu(const Ctxt &c, double scale, bool timing) {
    auto start = start_time();

    /*
     * Max min
     */
    Ptxt result;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &result);
    vector<double> v = result->GetRealPackedValue();

    //cout << "min: " << *min_element(v.begin(), v.end()) << ", max: " << *max_element(v.begin(), v.end()) << endl;
    /*
     * Max min
     */

    std::function<double(double)> relu_func = [scale](double x) -> double { if (x < 0) return 0; else return (1 / scale) * x; };
    std::vector<double> relu_coeffs = context->GetChebyshevCoefficients(relu_func, -1, 1, relu_degree);
    Ctxt res = context->EvalChebyshevSeries(c, relu_coeffs, -1, 1);

    if (timing) {
        print_duration(start, "ReLU d = " + to_string(relu_degree) + " evaluation");
    }

    return res;
}

Ctxt FHEController::relu_wide(const Ctxt &c, double a, double b, int degree, double scale, bool timing) {
    auto start = start_time();

    /*
     * Max min
     */
    Ptxt result;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &result);
    vector<double> v = result->GetRealPackedValue();

    //cout << "min: " << *min_element(v.begin(), v.end()) << ", max: " << *max_element(v.begin(), v.end()) << endl;
    /*
     * Max min
     */

    std::function<double(double)> relu_wide_func = [scale](double x) -> double { if (x < 0) return 0; else return (1 / scale) * x; };
    std::vector<double> relu_wide_coeffs = context->GetChebyshevCoefficients(relu_wide_func, a, b, degree);
    Ctxt res = context->EvalChebyshevSeries(c, relu_wide_coeffs, a, b);
    if (timing) {
        print_duration(start, "ReLU d = " + to_string(degree) + " evaluation");
    }

    return res;
}


/*
 * I/O
 */

Ctxt FHEController::read_input(const string& filename, double scale) {
    vector<double> input = read_values_from_file(filename);

    int size = static_cast<int>(input.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            input[i] = input[i] * scale;
        }
    }

    Ptxt pt = context->MakeCKKSPackedPlaintext(input, 1, circuit_depth - 10, nullptr, num_slots);
    return context->Encrypt(key_pair.publicKey, pt);
}

Ptxt FHEController::read_plain_input(const string& filename, int level, double scale) {
    vector<double> input = read_values_from_file(filename);

    int size = static_cast<int>(input.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            input[i] = input[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(input, 1, level, nullptr, num_slots);
}

Ctxt FHEController::read_repeated_input(const string& filename, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[i]);
        }
    }

    int size = static_cast<int>(input.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            input[i] = input[i] * scale;
        }
    }

    Ptxt pt = context->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, num_slots);
    return context->Encrypt(key_pair.publicKey, pt);
}

Ptxt FHEController::read_plain_repeated_input(const string& filename, int level, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[i]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

Ptxt FHEController::read_plain_repeated_512_input(const string& filename, int level, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 32; j++) {
        for (int i = 0; i < 512; i++) {
            repeated.push_back(input[i]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

Ctxt FHEController::read_expanded_input(const string& filename, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[j]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    Ptxt pt = context->MakeCKKSPackedPlaintext(repeated, 1, 0, nullptr, num_slots);
    return context->Encrypt(key_pair.publicKey, pt);
}

Ptxt FHEController::read_plain_expanded_input(const string& filename, int level, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[j]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

Ptxt FHEController::read_plain_expanded_input(const string& filename, int level, double scale, int num_inputs) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < num_inputs; i++) {
            repeated.push_back(input[j]);
        }
        for (int i = 0; i < 128 - num_inputs; i++) {
            repeated.push_back(0);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

void FHEController::print(const Ctxt &c, int slots, string prefix) {
    if (!secret_key_loaded) {
        throw std::runtime_error(
            "FHEController::print(): this is a decrypting debug helper and no secret key is "
            "loaded on this controller. Server-role code must not call it -- log ciphertext "
            "level/timing via ServerLog instead.");
    }
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix << " (Lv. " << c->GetLevel() << ") ";

    Ptxt result;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &result);
    result->SetSlots(num_slots);
    vector<double> v = result->GetRealPackedValue();

    cout << setprecision(4) << fixed;
    cout << "[ ";

    for (int i = 0; i < slots; i += 1) {
        string segno = "";
        if (v[i] > 0) {
            segno = " ";
        } else {
            segno = "-";
            v[i] = -v[i];
        }


        if (i == slots - 1) {
            cout << segno << v[i] << " ]";
        } else {
            if (abs(v[i]) < 0.00000001)
                cout << " 0.0000" << ", ";
            else
                cout << segno << v[i] << ", ";
        }
    }

    cout << endl;
}

void FHEController::print_expanded(const Ctxt &c, int slots, int expansion_factor, string prefix) {
    if (!secret_key_loaded) {
        throw std::runtime_error(
            "FHEController::print_expanded(): this is a decrypting debug helper and no secret "
            "key is loaded on this controller. Server-role code must not call it.");
    }
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix << " (Lv. " << c->GetLevel() << ") ";

    Ptxt result;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &result);
    result->SetSlots(num_slots);
    vector<double> v = result->GetRealPackedValue();


    cout << setprecision(4) << fixed;
    cout << "[ ";

    for (int i = 0; i < slots; i += 1) {
        if (i % expansion_factor != 0) {
            continue;
        }
        string segno = "";
        if (v[i] > 0) {
            segno = " ";
        } else {
            segno = "-";
            v[i] = -v[i];
        }


        if (i == slots - 1) {
            cout << segno << v[i] << " ]";
        } else {
            if (abs(v[i]) < 0.00000001)
                cout << " 0.000" << ", ";
            else
                cout << segno << v[i] << ", ";
        }
    }

    cout << " ]";

    cout << endl;
}

void FHEController::print_padded(const Ctxt &c, int slots, int padding, string prefix) {
    if (!secret_key_loaded) {
        throw std::runtime_error(
            "FHEController::print_padded(): this is a decrypting debug helper and no secret key "
            "is loaded on this controller. Server-role code must not call it.");
    }
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix;

    Ptxt result;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &result);
    result->SetSlots(num_slots);
    vector<double> v = result->GetRealPackedValue();

    cout << setprecision(10) << fixed;
    cout << "[ ";

    for (int i = 0; i < slots * padding; i += padding) {
        string segno = "";
        if (v[i] > 0) {
            segno = " ";
        } else {
            segno = "-";
            v[i] = -v[i];
        }


        if (i == slots - 1) {
            cout << segno << v[i] << " ]";
        } else {
            if (abs(v[i]) < 0.00000001)
                cout << " 0.000" << ", ";
            else
                cout << segno << v[i] << ", ";
        }
    }

    cout << endl;
}

void FHEController::print_min_max(const Ctxt &c) {
    if (!secret_key_loaded) {
        throw std::runtime_error(
            "FHEController::print_min_max(): this is a decrypting debug helper and no secret "
            "key is loaded on this controller. Server-role code must not call it.");
    }
    Ptxt result;
    Ctxt c_mut = c;
    context->Decrypt(key_pair.secretKey, c_mut, &result);
    vector<double> v = result->GetRealPackedValue();

    //cout << "min: " << *min_element(v.begin(), v.end()) << ", max: " << *max_element(v.begin(), v.end()) << endl;
}


Ctxt FHEController::rotsum(const Ctxt &in, int slots, int padding) {
    Ctxt result = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        result = add(result, context->EvalRotate(result, padding * pow(2, i)));
    }

    return result;
}

Ctxt FHEController::rotsum_padded(const Ctxt &in, int slots) {
    Ctxt result = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        result = add(result, context->EvalRotate(result, slots * pow(2, i)));
    }

    return result;
}

Ctxt FHEController::repeat(const Ctxt &in, int slots) {
    Ctxt res = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        res = add(res, rotate(res, -pow(2, i)));
    }

    return res;
}

Ctxt FHEController::repeat(const Ctxt &in, int slots, int padding) {
    Ctxt res = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        res = add(res, rotate(res, padding * (-pow(2, i))));
    }

    return res;
}

vector<Ctxt> FHEController::matmulRE(vector<Ctxt> rows, const Ptxt &weight, const Ptxt &bias) {
    vector<Ctxt> columns;

    for (size_t i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, 128, 128);

        if (bias != nullptr) m = add(m, bias);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulRE(vector<Ctxt> rows, const Ptxt &weight, const Ptxt &bias, int row_size, int padding) {
    vector<Ctxt> columns;

    for (size_t i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, row_size, padding);

        if (bias != nullptr) m = add(m, bias);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulRE(vector<Ctxt> rows, const Ctxt &weight, int row_size, int padding) {
    vector<Ctxt> columns;

    for (size_t i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, row_size, padding);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulRElarge(vector<Ctxt>& inputs, const vector<Ptxt> &weights, const Ptxt &bias, double mask_val) {
    vector<Ctxt> densed;

    for (size_t i = 0; i < inputs.size(); i++) {
        Ctxt i_th_result;
        for (int j = weights.size() - 1; j >= 0; j--) {
            Ctxt out = mult(inputs[i], weights[j]);
            out = rotsum(out, 128, 128);

            out = mask_first_n(out, 128, mask_val);

            if (j == static_cast<int>(weights.size()) - 1)
                i_th_result = out;
            else {
                //i_th_result = rotate(i_th_result, -128);
                i_th_result = rotate(i_th_result, -64);
                i_th_result = rotate(i_th_result, -64);

                i_th_result = add(i_th_result, out);
            }

        }

        i_th_result = add(i_th_result, bias);

        densed.push_back(i_th_result);
    }

    return densed;
}

vector<Ctxt> FHEController::matmulCR(vector<Ctxt> rows, const Ctxt& matrix) {
    vector<Ctxt> columns;

    for (size_t i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], matrix);

        m = rotsum(m, 64, 1);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulCR(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias) {
    vector<Ctxt> columns;

    for (size_t i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, 128, 1);

        if (bias != nullptr) m = add(m, bias);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulCRlarge(vector<vector<Ctxt>> rows, vector<Ptxt> weights, const Ptxt &bias) {
    vector<Ctxt> output;

    for (size_t i = 0; i < rows.size(); i++) {
        //Qua sotto posso fare prima add-many e poi un solo rotsum mi sa:)
        /*
        Ctxt p1 = rotsum(mult(rows[i][0], weights[0]), 128, 1);
        Ctxt p2 = rotsum(mult(rows[i][1], weights[1]), 128, 1);
        Ctxt p3 = rotsum(mult(rows[i][2], weights[2]), 128, 1);
        Ctxt p4 = rotsum(mult(rows[i][3], weights[3]), 128, 1);

        Ctxt res = add({p1, p2, p3, p4});
        */

        Ctxt p1 = mult(rows[i][0], weights[0]);
        Ctxt p2 = mult(rows[i][1], weights[1]);
        Ctxt p3 = mult(rows[i][2], weights[2]);
        Ctxt p4 = mult(rows[i][3], weights[3]);

        Ctxt res = add({p1, p2, p3, p4});
        res = rotsum(res, 128, 1);

        if (bias != nullptr) res = add(res, bias);

        output.push_back(res);
    }

    return output;
}

Ctxt FHEController::matmulScores(vector<Ctxt> queries, const Ctxt &key) {
    vector<Ctxt> scores = matmulCR(queries, key);

    double r = 1 / 8.0; //Later corrected with e^(x/r)

    Ctxt scores_wrapped = mask_heads(scores[scores.size() - 1], 1 / 8.0 * r);
    scores_wrapped = rotate(scores_wrapped, -1);

    for (int i = scores.size() - 2; i >= 0; i--) {
        scores_wrapped = add(scores_wrapped,
                             mask_heads(scores[i], 1 / 8.0 * r));

        if (i > 0) scores_wrapped = rotate(scores_wrapped, -1);
    }

    return scores_wrapped;
}

Ctxt FHEController::wrapUpRepeated(vector<Ctxt> vectors) {
    vector<Ctxt> masked;

    for (size_t i = 0; i < vectors.size(); i++) {
        masked.push_back(mask_block(vectors[i], 128 * i, 128 * (i + 1), 1));
    }

    return context->EvalAddMany(masked);
}

Ctxt FHEController::wrapUpExpanded(vector<Ctxt> vectors) {
    //I use a vector to contain all the partial computations so that EvalAddTree adds less error than summing each
    //iteration
    Ctxt masked = mask_mod_n(vectors[vectors.size() - 1], 128);
    if (vectors.size() > 1) masked = rotate(masked, -1);

    for (int i = vectors.size() - 2; i >= 0; i--) {
        masked = add(masked, mask_mod_n(vectors[i], 128));
        if (i > 0) {
            masked = rotate(masked, -1);
        }
    }

    return masked;
}

vector<Ctxt> FHEController::unwrapExpanded(Ctxt c, int inputs_num) {
    vector<Ctxt> result;

    for (int i = 0; i < inputs_num; i++) {
        Ctxt out = mask_mod_n(c, 128, 0,inputs_num * 128);
        out = repeat(out, 128);


        if (i < inputs_num - 1) c = rotate(c, 1);

        result.push_back(out);
    }

    return result;
}

vector<vector<Ctxt>> FHEController::unwrapRepeatedLarge(vector<Ctxt> containers, int input_number) {
    vector<vector<Ctxt>> unwrapped_output;
    vector<int> quantities;

    for (int i = 0; i < input_number / 32.0; i++) {
        int quantity = 32;
        if ((i + 1) * 32 > input_number) {
            quantity = input_number - (i * 32);
        }

        quantities.push_back(quantity);
    }

    for (size_t i = 0; i < containers.size(); i++) {
        for (int j = 0; j < quantities[i]; j++) {
            vector<Ctxt> unwrapped_container = unwrap_512_in_4_128(containers[i], j);
            unwrapped_output.push_back(unwrapped_container);
        }
    }

    return unwrapped_output;
}

vector<Ctxt> FHEController::unwrapScoresExpanded(Ctxt c, int inputs_num) {
    vector<Ctxt> result;

    for (int i = 0; i < inputs_num; i++) {
        Ctxt i_th_1 = mask_mod_n(c, 128, 0,inputs_num * 128);
        Ctxt i_th_2 = mask_mod_n(c, 128, 64, inputs_num * 128);
        i_th_1 = repeat(i_th_1, 64);
        i_th_2 = repeat(i_th_2, 64);

        if (i < inputs_num - 1) c = rotate(c, 1);

        result.push_back(add(i_th_1, i_th_2));
    }

    return result;
}

vector<Ctxt> FHEController::unwrap_512_in_4_128(const Ctxt &c, int index) {
    vector<Ctxt> result;

    int shift = index * 512;

    Ctxt score1 = mask_block(c, shift + 0, shift + 128, 1);
    score1 = repeat(score1, 128, -128);
    Ctxt score2 = mask_block(c, shift + 128, shift + 256, 1);
    score2 = repeat(score2, 128, -128);
    Ctxt score3 = mask_block(c, shift+ 256, shift + 384, 1);
    score3 = repeat(score3, 128, -128);
    Ctxt score4 = mask_block(c, shift + 384, shift + 512, 1);
    score4 = repeat(score4, 128, -128);

    result.push_back(score1);
    result.push_back(score2);
    result.push_back(score3);
    result.push_back(score4);

    return result;
}

vector<Ctxt> FHEController::generate_containers(vector<Ctxt> inputs, const Ptxt& bias) {
    vector<Ctxt> containers;
    vector<int> quantities;

    //This reverse is not fine
    //reverse(inputs.begin(), inputs.end());

    for (int i = 0; i < inputs.size() / 32.0; i++) {
        int quantity = 32;
        if ((i + 1) * 32 > static_cast<int>(inputs.size())) {
            quantity = static_cast<int>(inputs.size()) - (i * 32);
        }

        quantities.push_back(quantity);

        vector<Ctxt> sliced_input = slicing(inputs, (i) * 32, (i + 1) * 32);
        reverse(sliced_input.begin(), sliced_input.end());

        Ctxt partial_container = wrap_containers( sliced_input, quantity);

        if (bias != nullptr) partial_container = add(partial_container, bias);

        containers.push_back(partial_container);
    }

    return containers;
}

Ctxt FHEController::wrap_containers(vector<Ctxt> c, int inputs_number) {
    //Resulting Ctxt will contain all the ciphertexts as follows:
    //c_n | c_n-1 | c_n-2 | ... | c_0

    Ctxt result = c[0];

    for (int i = 1; i < inputs_number; i++) {
        result = rotate(result, -512);
        result = add(result, c[i]);
    }

    return result;
}

Ptxt FHEController::get_cached_mask(int kind, int p1, int p2, int p3, int level, double mask_value,
                                    const std::function<vector<double>()>& build) {
    auto key = std::make_tuple(kind, p1, p2, p3, level, mask_value);
    auto it = mask_ptxt_cache.find(key);
    if (it != mask_ptxt_cache.end()) {
        return it->second;
    }

    Ptxt p = encode(build(), level, num_slots);
    mask_ptxt_cache.emplace(key, p);
    return p;
}

Ctxt FHEController::mask_block(const Ctxt& c, int from, int to, double mask_value) {
    Ptxt p = get_cached_mask(0, from, to, 0, c->GetLevel(), mask_value, [&]() {
        vector<double> mask(num_slots, 0.0);
        for (int i = from; i < to && i < num_slots; i++) mask[i] = mask_value;
        return mask;
    });

    return mult(c, p);
}

Ctxt FHEController::mask_heads(const Ctxt& c, double mask_value) {
    Ptxt p = get_cached_mask(1, 0, 0, 0, c->GetLevel(), mask_value, [&]() {
        vector<double> mask(num_slots, 0.0);
        for (int i = 0; i < num_slots; i += 64) mask[i] = mask_value;
        return mask;
    });

    return mult(c, p);
}

Ctxt FHEController::mask_mod_n(const Ctxt& c, int n) {
    Ptxt p = get_cached_mask(2, n, 0, 0, c->GetLevel(), 1.0, [&]() {
        vector<double> mask(num_slots, 0.0);
        for (int i = 0; i < num_slots; i += n) mask[i] = 1;
        return mask;
    });

    return mult(c, p);
}

Ctxt FHEController::mask_mod_n(const Ctxt& c, int n, int padding, int max_slots) {
    Ptxt p = get_cached_mask(3, n, padding, max_slots, c->GetLevel(), 1.0, [&]() {
        vector<double> mask(num_slots, 0.0);
        for (int i = 0; i < num_slots; i++) {
            if (i % n == padding) mask[i] = 1;
        }
        return mask;
    });

    return mult(c, p);
}

Ctxt FHEController::mask_first_n(const Ctxt &c, int n, double mask_value) {
    Ptxt p = get_cached_mask(4, n, 0, 0, c->GetLevel(), mask_value, [&]() {
        vector<double> mask(num_slots, 0.0);
        for (int i = 0; i < n && i < num_slots; i++) mask[i] = mask_value;
        return mask;
    });

    return mult(c, p);
}


Ctxt FHEController::eval_exp(const Ctxt &c, int inputs_number) {
    // fideslib has no EvalPoly (Paterson-Stockmeyer) helper, so this Taylor series for e^x is
    // evaluated with a plain Horner chain instead: 1 + x(1 + x(1/2 + x(1/6 + x(1/24 + x(1/120 + x/720))))).
    // Horner costs one multiplicative level per term (6 levels here) instead of OpenFHE
    // EvalPoly's ~4 for the same degree, and the 3 EvalSquare calls below (computing res^8) cost
    // 3 more, so we bootstrap up front if there isn't room for all 9 levels -- doing the check
    // after the fact (as the OpenFHE version did) would be too late to prevent underflow mid-chain.
    Ctxt input = c;
    if (static_cast<int>(input->GetLevel()) + 9 > circuit_depth) {
        input = bootstrap(input);
    }

    Ctxt res = context->EvalMult(input, 1.0 / 720.0);
    res = context->EvalAdd(res, 1.0 / 120.0);
    res = context->EvalMult(res, input);
    res = context->EvalAdd(res, 1.0 / 24.0);
    res = context->EvalMult(res, input);
    res = context->EvalAdd(res, 1.0 / 6.0);
    res = context->EvalMult(res, input);
    res = context->EvalAdd(res, 1.0 / 2.0);
    res = context->EvalMult(res, input);
    res = context->EvalAdd(res, 1.0);
    res = context->EvalMult(res, input);
    res = context->EvalAdd(res, 1.0);

    res = context->EvalSquare(res);
    res = context->EvalSquare(res);
    res = context->EvalSquare(res);

    //values must be corrected, slots that were 0 will now be 1, and this will break the following computations
    vector<double> mask;
    for (int i = 0; i < num_slots; i++) {
        //Here 12 è il numero di token, da cambiare
        if (i % 64 < inputs_number && i < (128 * inputs_number)) {
            mask.push_back(0);
        } else {
            mask.push_back(-1);
        }
    }

    return add(res, encode(mask, res->GetLevel(), num_slots));
}

Ctxt FHEController::eval_inverse(const Ctxt &c, double min, double max) {
    double middle = (max - min) / 2; //9995

    Ctxt res = add(c, encode(-middle - min, c->GetLevel(), num_slots)); // lo centro
    res = mult(res, encode(1 / middle, res->GetLevel(), num_slots)); //basta prima mascherare con 1 /9995 e addare -10005/9995 dopopì

    std::function<double(double)> inverse_func = [](double x) -> double { return 1 / ((x * 9895) + 9995); };
    std::vector<double> inverse_coeffs = context->GetChebyshevCoefficients(inverse_func, -1, 1, 200);
    return context->EvalChebyshevSeries(res, inverse_coeffs, -1, 1);
}

Ctxt FHEController::eval_inverse_naive(const Ctxt &c, double min, double max) {
    std::function<double(double)> inverse_naive_func = [](double x) -> double { return 1 / x; };
    std::vector<double> inverse_naive_coeffs = context->GetChebyshevCoefficients(inverse_naive_func, min, max, 119);
    return context->EvalChebyshevSeries(c, inverse_naive_coeffs, min, max);
}

Ctxt FHEController::eval_inverse_naive_2(const Ctxt &c, double min, double max, double mult) {
    std::function<double(double)> inverse_naive_2_func = [mult](double x) -> double { return mult / x; };
    std::vector<double> inverse_naive_2_coeffs = context->GetChebyshevCoefficients(inverse_naive_2_func, min, max, 200);
    return context->EvalChebyshevSeries(c, inverse_naive_2_coeffs, min, max);
}

Ctxt FHEController::eval_gelu_function(const Ctxt &c, double min, double max, double mult, int degree) {
    std::function<double(double)> gelu_func = [mult](double x) -> double { return  (0.5 * (x * (1 / mult)) * (1 + erf((x * (1 / mult)) / 1.41421356237))); };
    std::vector<double> gelu_coeffs = context->GetChebyshevCoefficients(gelu_func, min, max, degree);
    return context->EvalChebyshevSeries(c, gelu_coeffs, min, max);
}

Ctxt FHEController::eval_tanh_function(const Ctxt &c, double min, double max, double mult, int degree) {
    std::function<double(double)> tanh_func = [mult](double x) -> double { return tanh(x * (1 / mult)); };
    std::vector<double> tanh_coeffs = context->GetChebyshevCoefficients(tanh_func, min, max, degree);
    return context->EvalChebyshevSeries(c, tanh_coeffs, min, max);
}

vector<Ctxt> FHEController::slicing(vector<Ctxt> &arr, int X, int Y) {
    if (Y - X >= static_cast<int>(arr.size()))
        return arr;

    if (Y > static_cast<int>(arr.size())) {
        Y = static_cast<int>(arr.size());
    }

    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y;

    // To store the sliced vector
    vector<Ctxt> result(Y - X);

    copy(start, end, result.begin());

    // Return the final sliced vector
    return result;
}


// fideslib's Serial wrapper (FIDESlib/api/Serialize.hpp) only exposes (de)serialization for the
// CryptoContext and the key pair, not for individual ciphertexts -- unlike OpenFHE's Serial, which
// also handles Ciphertext<DCRTPoly>. These three functions aren't called anywhere in this circuit
// (no checkpointing path is currently wired up), so rather than silently no-op they fail loudly if
// something starts relying on them.
void FHEController::save(Ctxt v, std::string filename) {
    cerr << "FHEController::save(Ctxt): ciphertext serialization is not supported by the fideslib backend." << endl;
    exit(1);
}

void FHEController::save(vector<Ctxt> v, std::string filename) {
    cerr << "FHEController::save(vector<Ctxt>): ciphertext serialization is not supported by the fideslib backend." << endl;
    exit(1);
}

vector<Ctxt> FHEController::load_vector(string filename) {
    cerr << "FHEController::load_vector: ciphertext serialization is not supported by the fideslib backend." << endl;
    exit(1);

    vector<Ctxt> result;
    return result;
}

Ctxt FHEController::load_ciphertext(string filename) {
    cerr << "FHEController::load_ciphertext: ciphertext serialization is not supported by the fideslib backend." << endl;
    exit(1);

    Ctxt result;
    return result;
}
