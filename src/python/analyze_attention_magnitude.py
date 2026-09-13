#!/usr/bin/env python3
"""
Quantifies, stage by stage, how much magnitude/precision the FHE circuit's
layer-1 self-attention softmax approximation loses relative to true softmax,
using the *actual* BERT-tiny-SST2 weights and a real sentence -- no GPU/FHE
run required, this replicates the exact math each FHEController function
performs (matmulScores' 1/64 pre-scale, eval_exp's degree-6 Taylor + ^8, and
eval_inverse_naive_2's degree-200 Chebyshev fit of mult/x on [3, 145000]).

Run from repo root:
    python3 src/python/analyze_attention_magnitude.py "<sentence>"
"""
import sys
import numpy as np
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging
from numpy.polynomial import chebyshev as C

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

x = model.bert.embeddings(tokens_tensor, torch.tensor([[1] * n_tokens]))


def layer0_output(x):
    """Full layer-0 forward (attention + FFN), replicating PlainCircuit.py lines 1-105."""
    input_tensor = x.double()
    original_input_tensor = input_tensor

    L = model.bert.encoder.layer[0]
    query = L.attention.self.query.weight.detach().double().T
    key = L.attention.self.key.weight.detach().double().T
    value = L.attention.self.value.weight.detach().double().T
    qb = L.attention.self.query.bias.detach().double()
    kb = L.attention.self.key.bias.detach().double()
    vb = L.attention.self.value.bias.detach().double()

    q = (torch.matmul(input_tensor, query) + qb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])
    k = (torch.matmul(input_tensor, key) + kb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 3, 1])
    v = (torch.matmul(input_tensor, value) + vb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])

    qk = torch.matmul(q, k) / 8
    attn = torch.softmax(qk, -1)
    fin = torch.matmul(attn, v).permute([0, 2, 1, 3]).reshape([1, n_tokens, 128])

    fin2 = torch.matmul(fin, L.attention.output.dense.weight.detach().double().T) + L.attention.output.dense.bias.detach().double()
    fin2 = fin2 + original_input_tensor
    fin3 = torch.layer_norm(fin2, (128,), L.attention.output.LayerNorm.weight.double(), L.attention.output.LayerNorm.bias.double(), eps=1e-12)

    fin4 = torch.matmul(fin3, L.intermediate.dense.weight.detach().double().T) + L.intermediate.dense.bias.detach().double()
    fin5 = torch.nn.functional.gelu(fin4)
    fin6 = torch.matmul(fin5, L.output.dense.weight.detach().double().T) + L.output.dense.bias.detach().double()
    fin6 = fin6 + fin3
    fin7 = torch.layer_norm(fin6, (128,), L.output.LayerNorm.weight.double(), L.output.LayerNorm.bias.double(), eps=1e-12)
    return fin7


layer1_input = layer0_output(x)

L1 = model.bert.encoder.layer[1]
query = L1.attention.self.query.weight.detach().double().T
key = L1.attention.self.key.weight.detach().double().T
value = L1.attention.self.value.weight.detach().double().T
qb = L1.attention.self.query.bias.detach().double()
kb = L1.attention.self.key.bias.detach().double()
vb = L1.attention.self.value.bias.detach().double()

q = (torch.matmul(layer1_input, query) + qb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])
k = (torch.matmul(layer1_input, key) + kb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 3, 1])
v = (torch.matmul(layer1_input, value) + vb).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])

qk_raw = torch.matmul(q, k)  # this is exactly matmulCR's raw output, pre any scaling
qk_raw_np = qk_raw.detach().numpy()  # shape [1, 2, n_tokens, n_tokens]

print(f"\nRaw Q.K^T range: min={qk_raw_np.min():.3f} max={qk_raw_np.max():.3f} "
      f"mean_abs={np.abs(qk_raw_np).mean():.3f}")

# ---- True softmax (what PyTorch / the plaintext reference computes) ----
true_scores = qk_raw / 8.0
true_weights = torch.softmax(true_scores, -1)
true_out = torch.matmul(true_weights, v)

# ---- FHE approximation, replicating matmulScores + eval_exp exactly ----
# matmulScores: mask_heads(scores, 1/8 * 1/8) -> raw_qk / 64
x_fhe = qk_raw_np / 64.0

# eval_exp: degree-6 Taylor of e^x via Horner, then cubed 3x (^8) to recover e^(x*8) = e^(raw_qk/8)
def taylor_exp6(x):
    res = x / 720.0
    res = res + 1.0 / 120.0
    res = res * x
    res = res + 1.0 / 24.0
    res = res * x
    res = res + 1.0 / 6.0
    res = res * x
    res = res + 1.0 / 2.0
    res = res * x
    res = res + 1.0
    res = res * x
    res = res + 1.0
    return res

taylor = taylor_exp6(x_fhe)
approx_exp = taylor ** 8  # 3x EvalSquare == ^8

true_exp = np.exp(true_scores.detach().numpy())

rel_err_exp = np.abs(approx_exp - true_exp) / np.abs(true_exp)
print(f"\n[Stage: eval_exp Taylor-6 + ^8]")
print(f"  input to Taylor (raw_qk/64) range: [{x_fhe.min():.4f}, {x_fhe.max():.4f}]")
print(f"  true exp(raw_qk/8) range: [{true_exp.min():.4f}, {true_exp.max():.4f}]")
print(f"  approx exp range:        [{approx_exp.min():.4f}, {approx_exp.max():.4f}]")
print(f"  mean relative error: {rel_err_exp.mean()*100:.2f}%   max relative error: {rel_err_exp.max()*100:.2f}%")

# ---- eval_inverse_naive_2: Chebyshev fit of (mult / x) on [3, 145000], degree 200 ----
true_sum = true_exp.sum(axis=-1, keepdims=True)          # true softmax denominator
approx_sum = approx_exp.sum(axis=-1, keepdims=True)       # what the ciphertext denominator actually holds

lo, hi = 3.0, 145000.0
degree = 200
# Fit degree-200 Chebyshev poly to 1/x on [lo, hi] using Chebyshev-node interpolation,
# same approach OpenFHE's GetChebyshevCoefficients uses internally.
nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
x_nodes = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
y_nodes = 1.0 / x_nodes
cheb_coeffs = C.chebfit((x_nodes - 0.5 * (hi + lo)) / (0.5 * (hi - lo)), y_nodes, degree)


def cheb_inverse(x):
    xn = (np.clip(x, lo, hi) - 0.5 * (hi + lo)) / (0.5 * (hi - lo))
    return C.chebval(xn, cheb_coeffs)


approx_denom = cheb_inverse(approx_sum)
true_denom = 1.0 / true_sum

print(f"\n[Stage: eval_inverse_naive_2 Chebyshev(1/x) on [{lo:.0f}, {hi:.0f}], degree {degree}]")
print(f"  true sum(exp) range:   [{true_sum.min():.4f}, {true_sum.max():.4f}]  (n_tokens={n_tokens} => "
      f"note true_sum is tiny relative to the [3,145000] domain!)")
print(f"  approx sum(exp) range: [{approx_sum.min():.4f}, {approx_sum.max():.4f}]")
print(f"  true 1/sum range:      [{true_denom.min():.6f}, {true_denom.max():.6f}]")
print(f"  Chebyshev-approx 1/sum range: [{approx_denom.min():.6f}, {approx_denom.max():.6f}]")
rel_err_inv = np.abs(approx_denom - true_denom) / np.abs(true_denom)
print(f"  mean relative error: {rel_err_inv.mean()*100:.2f}%   max relative error: {rel_err_inv.max()*100:.2f}%")

# ---- Combine: full approximate softmax weights + attention output ----
approx_weights = approx_exp * approx_denom
true_weights_np = true_weights.detach().numpy()

print(f"\n[Combined softmax weights]")
print(f"  true weights sum per row (should be 1.0): {true_weights_np.sum(-1).mean():.6f}")
print(f"  approx weights sum per row (should be ~1.0): {approx_weights.sum(-1).mean():.6f}")
w_rel_err = np.abs(approx_weights - true_weights_np) / (np.abs(true_weights_np) + 1e-9)
print(f"  mean relative error on weights: {w_rel_err.mean()*100:.2f}%")

approx_out = np.einsum("bhij,bhjd->bhid", approx_weights, v.detach().numpy())
true_out_np = true_out.detach().numpy()

print(f"\n[Attention output ('Self-Attention (Repeated)' equivalent)]")
print(f"  true output norm:   {np.linalg.norm(true_out_np):.4f}")
print(f"  approx output norm: {np.linalg.norm(approx_out):.4f}")
print(f"  norm ratio (approx/true): {np.linalg.norm(approx_out)/np.linalg.norm(true_out_np):.4f}")

# ---- Isolate each stage's individual contribution to the magnitude ratio ----
# (a) exp-only error, using TRUE sum for normalization (isolates Taylor+^8 error alone)
weights_exp_only = approx_exp / true_sum
out_exp_only = np.einsum("bhij,bhjd->bhid", weights_exp_only, v.detach().numpy())
# (b) inverse-only error, using TRUE exp for the numerator (isolates Chebyshev inverse error alone)
weights_inv_only = true_exp * approx_denom
out_inv_only = np.einsum("bhij,bhjd->bhid", weights_inv_only, v.detach().numpy())

print(f"\n[Isolated stage contributions to output norm ratio vs. true]")
print(f"  exp-approx only (Taylor+^8), exact denom:      ratio={np.linalg.norm(out_exp_only)/np.linalg.norm(true_out_np):.4f}")
print(f"  inverse-approx only (Chebyshev), exact numer:  ratio={np.linalg.norm(out_inv_only)/np.linalg.norm(true_out_np):.4f}")
print(f"  both combined (full FHE approximation):        ratio={np.linalg.norm(approx_out)/np.linalg.norm(true_out_np):.4f}")

# ============================================================================
# Now redo the same analysis for encoder1 / BERT layer 0's self-attention,
# which uses a *different* inverse domain: eval_inverse_naive(sum, 2, 5000)
# degree 119, no +500 bootstrap rescale trick.
# ============================================================================
print("\n" + "=" * 70)
print("LAYER 0 (encoder1) self-attention -- eval_inverse_naive(sum, 2, 5000), degree 119")
print("=" * 70)

query0 = model.bert.encoder.layer[0].attention.self.query.weight.detach().double().T
key0 = model.bert.encoder.layer[0].attention.self.key.weight.detach().double().T
value0 = model.bert.encoder.layer[0].attention.self.value.weight.detach().double().T
qb0 = model.bert.encoder.layer[0].attention.self.query.bias.detach().double()
kb0 = model.bert.encoder.layer[0].attention.self.key.bias.detach().double()
vb0 = model.bert.encoder.layer[0].attention.self.value.bias.detach().double()

input_tensor0 = x.double()
q0 = (torch.matmul(input_tensor0, query0) + qb0).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])
k0 = (torch.matmul(input_tensor0, key0) + kb0).reshape([1, n_tokens, 2, 64]).permute([0, 2, 3, 1])
v0 = (torch.matmul(input_tensor0, value0) + vb0).reshape([1, n_tokens, 2, 64]).permute([0, 2, 1, 3])

qk_raw0 = torch.matmul(q0, k0).detach().numpy()
print(f"Raw Q.K^T range (layer 0): min={qk_raw0.min():.3f} max={qk_raw0.max():.3f}")

true_scores0 = qk_raw0 / 8.0
true_exp0 = np.exp(true_scores0)
true_sum0 = true_exp0.sum(axis=-1, keepdims=True)

x_fhe0 = qk_raw0 / 64.0
approx_exp0 = taylor_exp6(x_fhe0) ** 8
approx_sum0 = approx_exp0.sum(axis=-1, keepdims=True)

lo0, hi0 = 2.0, 5000.0
deg0 = 119
nodes0 = np.cos(np.pi * (np.arange(deg0 + 1) + 0.5) / (deg0 + 1))
x_nodes0 = 0.5 * (hi0 - lo0) * nodes0 + 0.5 * (hi0 + lo0)
y_nodes0 = 1.0 / x_nodes0
cheb_coeffs0 = C.chebfit((x_nodes0 - 0.5 * (hi0 + lo0)) / (0.5 * (hi0 - lo0)), y_nodes0, deg0)


def cheb_inverse0(xv):
    xn = (np.clip(xv, lo0, hi0) - 0.5 * (hi0 + lo0)) / (0.5 * (hi0 - lo0))
    return C.chebval(xn, cheb_coeffs0)


approx_denom0 = cheb_inverse0(approx_sum0)
true_denom0 = 1.0 / true_sum0

print(f"true sum(exp) range (layer 0):   [{true_sum0.min():.4f}, {true_sum0.max():.4f}]  vs. domain [{lo0:.0f}, {hi0:.0f}]")
print(f"approx sum(exp) range (layer 0): [{approx_sum0.min():.4f}, {approx_sum0.max():.4f}]")
if true_sum0.max() > hi0 or true_sum0.min() < lo0:
    print("  *** sum falls OUTSIDE the fitted domain -- Chebyshev extrapolation, expect large/unstable error ***")

rel_err_inv0 = np.abs(approx_denom0 - true_denom0) / np.abs(true_denom0)
print(f"inverse relative error (layer 0): mean={rel_err_inv0.mean()*100:.2f}%  max={rel_err_inv0.max()*100:.2f}%")

approx_weights0 = approx_exp0 * approx_denom0
true_weights0 = true_exp0 / true_sum0
approx_out0 = np.einsum("bhij,bhjd->bhid", approx_weights0, v0.detach().numpy())
true_out0 = np.einsum("bhij,bhjd->bhid", true_weights0, v0.detach().numpy())
print(f"output norm ratio (approx/true), layer 0: {np.linalg.norm(approx_out0)/np.linalg.norm(true_out0):.4f}")

print(f"\nFor comparison, layer 1's true sum(exp) range [{true_sum.min():.1f}, {true_sum.max():.1f}] occupies "
      f"{(true_sum.max()-true_sum.min())/(145000-3)*100:.3f}% of its [3, 145000] domain width, vs. layer 0's "
      f"sum range occupying {(true_sum0.max()-true_sum0.min())/(5000-2)*100:.3f}% of its [2, 5000] domain width.")
