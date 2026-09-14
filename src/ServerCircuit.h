//
// Server-side FHE circuit: the honest-but-curious server's half of FHE-BERT-Tiny.
//
// Every function in ServerCircuit.cpp takes an FHEController& that MUST be a server-role
// controller (constructed via FHEController::load_context_server(), never
// FHEController::load_client_secret_key()). None of them call decrypt()/decrypt_tovector() --
// they only ever see and produce ciphertexts, plus the plaintext *model weights* (which are
// server-owned, not client data). This file is the client/server processing boundary: nothing in
// here should ever gain the ability to read the client's plaintext input or secret key.
//

#ifndef NEWBERT_SERVERCIRCUIT_H
#define NEWBERT_SERVERCIRCUIT_H

#include "FHEController.h"
#include "ServerLog.h"

namespace server_circuit {

struct EvalResult {
    Ctxt output;
    int attempts = 0;
    bool succeeded = false;
    std::string last_error;
};

// Runs the full 2-layer BERT-tiny FHE circuit (self-attention, self-output, intermediate/output
// per layer, pooler, classifier) against already-encrypted input ciphertexts, retrying on the
// known intermittent GPU-bootstrap noise blow-up (see the comment at the call site in
// ServerCircuit.cpp). `inputs` must already be encrypted -- this function performs no
// client-side work (no tokenization, no encryption, no file I/O against client data) and no
// decryption.
EvalResult run_server_circuit(FHEController& controller, std::vector<Ctxt> inputs, bool verbose, utils::ServerLog& log);

} // namespace server_circuit

#endif // NEWBERT_SERVERCIRCUIT_H
