//
// client_main.cpp -- the CLIENT side of FHE-BERT-Tiny.
//
// This process still runs both roles in the same address space (see the note below on why),
// but the code is now split so that:
//   - Only THIS file (and FHEController's public-key-only operations it calls) ever touches the
//     client's plaintext text, its tokenized embeddings, and its secret key.
//   - src/ServerCircuit.cpp -- the "server" -- only ever receives already-encrypted ciphertexts,
//     evaluates the FHE circuit against them and server-owned plaintext model weights, and hands
//     back a result ciphertext. It has no path to the secret key (FHEController::load_context_server
//     never opens secret-key.txt) and never calls decrypt()/decrypt_tovector().
//
// Why one process instead of two, over a real network/IPC boundary: fideslib's GPU-resident
// Ciphertext type does not support serialization (see FHEController::save/load_ciphertext/
// load_vector in FHEController.cpp, which are stubs that exit(1)), so a ciphertext produced here
// cannot currently be written to a file or socket and read back by a separate `server` process.
// Implementing that is the natural next step for genuine network-level separation (it would let
// this file become a thin client that talks to a long-running server binary instead of calling
// server_circuit::run_server_circuit() in-process) -- see FHEBERT_SECURITY_REPORT.md for details
// and for why the in-process boundary enforced here (the server-role controller structurally
// cannot decrypt; the secret key is not even deserialized until after circuit evaluation
// returns) is still a meaningful security improvement over the previous code, which deserialized
// the secret key unconditionally before evaluation began.
//
#include <iostream>
#include "FHEController.h"
#include "ServerCircuit.h"
#include "ServerLog.h"
#include <chrono>
#include <filesystem>
#include <random>
#include <sstream>

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"

using namespace std::chrono;

enum class Parameters { Generate, Load };

void setup_environment(int argc, char *argv[]);
string make_request_id();

void run_command(const string &command) {
    if (system(command.c_str()) != 0) {
        cerr << "Warning: command failed: " << command << endl;
    }
}

FHEController controller;

bool IDE_MODE = false;

string input_folder;

//Argument
string text;

//<OPTIONS>
bool verbose = false;
bool security128bits = false;
Parameters p = Parameters::Load;
bool plain;

int main(int argc, char *argv[]) {
    setup_environment(argc, argv);

    if (p == Parameters::Generate) {
        // Key generation is inherently a client/trusted-party operation: it is the only place
        // in the program that ever creates a secret key.
        run_command("mkdir -p ./keys");
        controller.generate_context(true, security128bits);
        vector<int> rotations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1, -2, -4, -8, -16, -32, -64, -128, -256, -512};
        controller.generate_bootstrapping_and_rotation_keys(rotations, 16384, true, "rotation_keys.txt");
        controller.clear_mask_cache();
        return 0;
    }

    // ---------------------------------------------------------------------------------------
    // Load the crypto context and evaluation (public) key material. This is intentionally
    // load_context_server(): the secret key is NOT read here, and won't be for as long as
    // circuit evaluation is running below. See FHEController.h for the full rationale.
    // ---------------------------------------------------------------------------------------
    controller.load_context_server(false);
    controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);

    cout << "input folder " << input_folder << endl;
    if (input_folder.empty()) {
        cerr << "The input folder \"" << input_folder << "\" is empty!";
        exit(1);
    }

    // =========================================================================================
    // CLIENT-SIDE: tokenize (already done by setup_environment() via ExtractEmbeddings.py) and
    // encrypt the resulting embeddings. Only the client ever reads these plaintext embedding
    // files off disk.
    // =========================================================================================
    if (verbose) cout << "\nCLIENT-SIDE\nEncrypting the tokenized embeddings..." << endl;

    int inputs_count = 0;
    std::filesystem::path input_path{input_folder};
    for (__attribute__((unused)) auto& entry : std::filesystem::directory_iterator(input_path)) {
        ++inputs_count;
    }
    if (verbose) cout << inputs_count << " token embeddings found, encrypting each..." << endl;

    vector<Ctxt> encrypted_inputs;
    encrypted_inputs.reserve(inputs_count);
    for (int i = 0; i < inputs_count; i++) {
        encrypted_inputs.push_back(controller.read_expanded_input(input_folder + "input_" + to_string(i) + ".txt"));
    }

    // =========================================================================================
    // Hand off to the server. In a real deployment `encrypted_inputs` would be serialized and
    // sent over the network here, and this process would exit; what follows would run in a
    // separate server process with no access to the client's disk or secret key. See the file
    // header comment for why that hop is in-process today.
    // =========================================================================================
    run_command("mkdir -p ./logs");
    string request_id = make_request_id();

    // Client-level retry loop, on top of run_server_circuit()'s own internal retries.
    //
    // Why this exists: CKKS's approximate decode only actually detects "the noise blew up too
    // much" (OpenFHEException, "approximation error is too high") at Decode() time -- i.e. when
    // the CLIENT decrypts. A ciphertext can sail through every server-side step, including the
    // final bootstrap, without throwing, and still fail to decode correctly. In the original
    // single-role code this was invisible: encrypt/evaluate/decrypt all shared one try/catch, so
    // a decode failure transparently re-ran the whole circuit. Once decrypt is correctly isolated
    // to the client (server has no secret key to even attempt it), that safety net has to move
    // with it -- the client asks the server to recompute from scratch when its own decode fails.
    // (Discovered empirically while building the CPA test harness: an earlier version of this
    // file let a decode failure here propagate as an uncaught exception -> process abort. See
    // FHEBERT_SECURITY_REPORT.md.)
    const int max_client_attempts = 3;
    server_circuit::EvalResult eval_result;
    vector<double> plain_result;
    int timing = 0;
    int client_attempt = 1;

    for (; client_attempt <= max_client_attempts; client_attempt++) {
        if (verbose) cout << "\nSERVER-SIDE\nThe evaluation of the circuit started (client attempt "
                           << client_attempt << "/" << max_client_attempts << ")." << endl;

        utils::ServerLog server_log;
        server_log.open("./logs/server_processing.jsonl", request_id + "-c" + to_string(client_attempt));

        auto start = high_resolution_clock::now();
        bool server_ok = true;
        try {
            eval_result = server_circuit::run_server_circuit(controller, encrypted_inputs, verbose, server_log);
        } catch (const lbcrypto::OpenFHEException &e) {
            server_ok = false;
            cerr << "Server-side circuit evaluation exhausted its own retries (" << e.what() << ")." << endl;
        }
        timing = (duration_cast<milliseconds>(high_resolution_clock::now() - start)).count() / 1000.0;
        server_log.close();

        if (!server_ok) {
            if (client_attempt == max_client_attempts) {
                controller.clear_mask_cache();
                cerr << "Giving up after " << max_client_attempts << " client-level attempt(s)." << endl;
                exit(1);
            }
            continue;
        }

        if (verbose) cout << endl << "The evaluation of the FHE circuit took: " << timing << " seconds "
                           << "(" << eval_result.attempts << " attempt(s)). [client attempt "
                           << client_attempt << "/" << max_client_attempts << "]" << endl;
        if (verbose) cout << "The circuit has been evaluated, the results are sent back to the client" << endl << endl;

        // =====================================================================================
        // CLIENT-SIDE: only now -- after the server-role controller has finished evaluating the
        // circuit -- does the process load the secret key at all, and only decrypt the result.
        // =====================================================================================
        if (verbose && !controller.has_secret_key()) cout << "CLIENT-SIDE" << endl;

        if (!controller.has_secret_key()) {
            controller.load_client_secret_key(verbose);
        }

        try {
            if (verbose) controller.print(eval_result.output, 2, "Output logits");
            plain_result = controller.decrypt_tovector(eval_result.output, 2);
            break;
        } catch (const lbcrypto::OpenFHEException &e) {
            cerr << "Client-side decode failed on attempt " << client_attempt << "/" << max_client_attempts
                 << " (" << e.what() << "). Asking the server to recompute from scratch..." << endl;
            if (client_attempt == max_client_attempts) {
                controller.clear_mask_cache();
                cerr << "Giving up after " << max_client_attempts << " client-level attempt(s)." << endl;
                throw;
            }
        }
    }

    if (plain) {
        cout << "Outcomes:" << endl << "FHE              : ";
        if (plain_result[0] > plain_result[1]){
            cout << "negative sentiment!" << endl;
        } else {
            cout << "positive sentiment!" << endl;
        }
        run_command("python3 ./src/python/PlainCircuit.py \"" + text + "\"");
        run_command("python3 ./src/python/Precision.py \"" + text + "\" " + "\"[" + to_string(plain_result[0]) + ", " +
                to_string(plain_result[1]) + "\" " + to_string(timing));
    } else {
        cout << "Outcome: ";
        if (plain_result[0] > plain_result[1]){
            cout << GREEN_TEXT << "negative" << RESET_COLOR << " sentiment!" << endl;
        } else {
            cout << GREEN_TEXT << "positive" << RESET_COLOR << " sentiment!" << endl;
        }
    }

    // See the comment on clear_mask_cache(): must run here, before main() returns, not left to
    // the global `controller`'s destructor at process exit.
    controller.clear_mask_cache();
}

string make_request_id() {
    // Deliberately NOT derived from `text` or anything plaintext-derived -- the server log must
    // not be able to leak the client's input even indirectly through the id it's filed under.
    static std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream oss;
    oss << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() << "-" << std::hex << dist(gen);
    return oss.str();
}

void setup_environment(int argc, char *argv[]) {
    string command;

    if (IDE_MODE) {
        filesystem::remove_all("./src/tmp_embeddings");
        run_command("mkdir ./src/tmp_embeddings");

        input_folder = "./src/tmp_embeddings/";

        text = "This is a bad movie.";
        cout << "\nCLIENT-SIDE\nTokenizing the following sentence: '" << text << "'" << endl;
        command = "python3 ./src/python/ExtractEmbeddings.py \"" + text + "\"";

        run_command(command);

        verbose = true;
        return;
    }

    if (argc < 2) {
        cout << "This is FHEBERT-Tiny, an encrypted text classifier based on BERT-tiny. It relies on the CKKS homomorphic encryption scheme.\n\nUsage: ./FHEBERT-tiny <text_input> [OPTIONS]\n\nthe following [OPTIONS] are available:\n--verbose: activates verbose mode\n--secure: creates parameters with 128 bits of security. Use only if necessary, as it adds computational overhead \n\nExample:\n./FHEBERT-tiny \"I wonder if this text will be well classified!\" --verbose\n";
        exit(0);
    } else {
        if (string(argv[1]) == "--generate_keys")
        {
            if (argc > 2 && string(argv[2]) == "--secure") {
                security128bits = true;
            }

            p = Parameters::Generate;
            return;
        }

        text = argv[1];

        // Removing any previous embedding
        filesystem::remove_all("./src/tmp_embeddings/");
        run_command("mkdir ./src/tmp_embeddings");

        input_folder = "./src/tmp_embeddings/";


        for (int i = 2; i < argc; i++) {
            if (string(argv[i]) == "--verbose") {
                verbose = true;
            }

            if (string(argv[i]) == "--plain") {
                plain = true;
            }
        }

        if (verbose) cout << "\nCLIENT-SIDE\nTokenizing the following sentence: '" << text << "'" << endl;
        command = "python3 ./src/python/ExtractEmbeddings.py \"" + text + "\"";
        run_command(command);
    }

}
