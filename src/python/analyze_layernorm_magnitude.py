#!/usr/bin/env python3
"""
Tests the "PrecomputedLayerNorm" hypothesis: the circuit replaces each token's
TRUE per-example mean/std (reduced over the 128 hidden dims) with a FIXED,
position-indexed dataset-average mean/inv-std (mean[i], vy[i] for token
position i), read from weights-sst2/layer{0,1}_{selfoutput,output}_{mean,vy}.txt.
Real per-token stats vary with content; the position-only constant does not.
This checks how much that mismatch shrinks/inflates the LayerNorm output for
real sentences, isolated from all self-attention approximations (this script
uses PyTorch's exact self-attention/FFN, only swapping the LayerNorm itself).
"""
import sys
import numpy as np
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

sentence = sys.argv[1] if len(sys.argv) > 1 else "This was a complete waste of my time and money"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
trained = torch.load("./notebooks/SST-2-BERT-tiny.bin", map_location=torch.device("cpu"))
model.load_state_dict(trained, strict=False)
model.eval()

text = "[CLS] " + sentence + " [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
n_tokens = len(tokenized_text)
print(f"Sentence: {sentence!r} -- {n_tokens} tokens")

x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * n_tokens])).double()


def read_repeated_scalar_per_line(path):
    """weights-sst2 '*_mean.txt'/'*_vy.txt': one line per token position, each
    line a SIMD-repeated encoding of a single scalar (padded with zeros)."""
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(v) for v in line.split(",")]
            nz = [v for v in parts if v != 0.0]
            vals.append(nz[0] if nz else 0.0)
    return np.array(vals)


def precomputed_layernorm(vec_per_token, mean_arr, vy_arr, weight, bias):
    """vec_per_token: [n_tokens, 128]. mean_arr/vy_arr indexed by token position."""
    out = []
    for i in range(vec_per_token.shape[0]):
        v = vec_per_token[i]
        corr = (v - mean_arr[i]) * vy_arr[i]
        corr = corr * weight + bias
        out.append(corr)
    return np.stack(out)


def true_layernorm(vec_per_token, weight, bias, eps=1e-12):
    t = torch.tensor(vec_per_token)
    return torch.layer_norm(t, (128,), weight, bias, eps=eps).detach().numpy()


def analyze_layer(layer_idx, layer_input, mean_path_self, vy_path_self, mean_path_out, vy_path_out):
    L = model.bert.encoder.layer[layer_idx]
    query = L.attention.self.query.weight.detach().double().T
    key = L.attention.self.key.weight.detach().double().T
    value = L.attention.self.value.weight.detach().double().T
    qb = L.attention.self.query.bias.detach().double()
    kb = L.attention.self.key.bias.detach().double()
    vb = L.attention.self.value.bias.detach().double()

    q = (torch.matmul(layer_input, query) + qb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])
    k = (torch.matmul(layer_input, key) + kb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 3, 1])
    v = (torch.matmul(layer_input, value) + vb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])
    qk = torch.matmul(q, k) / 8
    attn = torch.softmax(qk, -1)
    fin = torch.matmul(attn, v).permute([0, 2, 1, 3]).reshape([1, n_tokens, 128])

    fin2 = torch.matmul(fin, L.attention.output.dense.weight.detach().double().T) + L.attention.output.dense.bias.detach().double()
    fin2_backup = (fin2 + layer_input).squeeze(0).detach().numpy()  # [n_tokens, 128], pre-LN residual

    true_mean_tok = fin2_backup.mean(axis=1)
    true_std_tok = fin2_backup.std(axis=1)

    mean_arr = read_repeated_scalar_per_line(mean_path_self)[:n_tokens]
    vy_arr = read_repeated_scalar_per_line(vy_path_self)[:n_tokens]

    print(f"\n--- Layer {layer_idx} attention-output LayerNorm ---")
    print(f"{'pos':>3} {'true_mean':>10} {'precomp_mean':>13} {'true_std':>9} {'precomp_std(1/vy)':>18}")
    for i in range(n_tokens):
        print(f"{i:>3} {true_mean_tok[i]:>10.4f} {mean_arr[i]:>13.4f} {true_std_tok[i]:>9.4f} {1.0/vy_arr[i]:>18.4f}")

    w_ln = L.attention.output.LayerNorm.weight.double()
    b_ln = L.attention.output.LayerNorm.bias.double()

    fin3_precomp = precomputed_layernorm(fin2_backup, mean_arr, vy_arr, w_ln.detach().numpy(), b_ln.detach().numpy())
    fin3_true = true_layernorm(fin2_backup, w_ln, b_ln)

    ratio = np.linalg.norm(fin3_precomp) / np.linalg.norm(fin3_true)
    print(f"norm(precomputed-LN output) / norm(true-LN output) = {ratio:.4f}")

    # continue with FFN using the precomputed-LN path (what the real circuit does)
    fin3_t = torch.tensor(fin3_precomp).unsqueeze(0)
    fin4 = torch.matmul(fin3_t, L.intermediate.dense.weight.detach().double().T) + L.intermediate.dense.bias.detach().double()
    fin5 = torch.nn.functional.gelu(fin4)
    fin6 = torch.matmul(fin5, L.output.dense.weight.detach().double().T) + L.output.dense.bias.detach().double()
    fin6 = (fin6 + fin3_t).squeeze(0).detach().numpy()

    mean_arr2 = read_repeated_scalar_per_line(mean_path_out)[:n_tokens]
    vy_arr2 = read_repeated_scalar_per_line(vy_path_out)[:n_tokens]

    true_mean_tok2 = fin6.mean(axis=1)
    true_std_tok2 = fin6.std(axis=1)
    print(f"\n--- Layer {layer_idx} FFN-output LayerNorm ---")
    print(f"{'pos':>3} {'true_mean':>10} {'precomp_mean':>13} {'true_std':>9} {'precomp_std(1/vy)':>18}")
    for i in range(n_tokens):
        print(f"{i:>3} {true_mean_tok2[i]:>10.4f} {mean_arr2[i]:>13.4f} {true_std_tok2[i]:>9.4f} {1.0/vy_arr2[i]:>18.4f}")

    w_ln2 = L.output.LayerNorm.weight.double()
    b_ln2 = L.output.LayerNorm.bias.double()
    fin7_precomp = precomputed_layernorm(fin6, mean_arr2, vy_arr2, w_ln2.detach().numpy(), b_ln2.detach().numpy())
    fin7_true = true_layernorm(fin6, w_ln2, b_ln2)
    ratio2 = np.linalg.norm(fin7_precomp) / np.linalg.norm(fin7_true)
    print(f"norm(precomputed-LN output) / norm(true-LN output) = {ratio2:.4f}")

    return torch.tensor(fin7_precomp).unsqueeze(0), torch.tensor(fin7_true).unsqueeze(0)


out0_precomp, out0_true = analyze_layer(
    0, x,
    "weights-sst2/layer0_selfoutput_mean.txt", "weights-sst2/layer0_selfoutput_vy.txt",
    "weights-sst2/layer0_output_mean.txt", "weights-sst2/layer0_output_vy.txt",
)

out1_precomp, out1_true = analyze_layer(
    1, out0_precomp,
    "weights-sst2/layer1_selfoutput_mean.txt", "weights-sst2/layer1_selfoutput_vy.txt",
    "weights-sst2/layer1_output_mean.txt", "weights-sst2/layer1_output_vy.txt",
)

print(f"\n=== End-to-end after both layers (precomputed-LN path, all-real self-attn) ===")
print(f"norm ratio (precomputed-LN pipeline / true-LN pipeline): {np.linalg.norm(out1_precomp.numpy())/np.linalg.norm(out1_true.numpy()):.4f}")

# Push both through pooler + classifier to see final logit-level impact
def finish(hidden):
    pooled = torch.tanh(torch.matmul(hidden, model.bert.pooler.dense.weight.detach().double().T) + model.bert.pooler.dense.bias.detach().double())
    cls = pooled[0][0]
    logits = torch.matmul(cls, model.classifier.weight.detach().double().T) + model.classifier.bias.detach().double()
    return logits.detach().numpy()

logits_precomp = finish(out1_precomp)
logits_true = finish(out1_true)
print(f"logits with precomputed-LN (both layers): {logits_precomp}")
print(f"logits with true LayerNorm (both layers):  {logits_true}")
print(f"logit magnitude ratio: {np.linalg.norm(logits_precomp)/np.linalg.norm(logits_true):.4f}")
