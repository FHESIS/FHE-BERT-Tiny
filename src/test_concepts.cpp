// Concept tests for FHE-BERT-Tiny.
//
// This binary does NOT run the BERT-tiny circuit. It isolates the handful of
// homomorphic building blocks the paper defines -- the CKKS primitives, the
// four vector/matrix packing algorithms (Algorithms 1-4), and the Chebyshev-
// approximated non-linearities used throughout the encoder -- and checks each
// one against a hand-computed expected value on a few small inputs.
//
// None of these tests call EvalBootstrap, so they exercise the algorithms
// independently of the GPU bootstrap race documented in
// reports/04-attention-signal-collapse.md: a failure here points at one of
// these primitives directly, rather than at bootstrap sequencing.
//
// Two tests are EXPECTED to fail on this branch: see the "KNOWN BUG" comments
// on eval_inverse_naive and eval_exp below. Writing these tests surfaced a
// reproducible defect distinct from the bootstrap race: FHEController's
// Chebyshev-based evaluators (eval_exp, eval_inverse_naive/_2, and, for an
// asymmetric domain, eval_gelu_function/eval_tanh_function too) silently
// evaluate f(x + min) instead of f(x) whenever the declared domain [min, max]
// isn't symmetric about zero. It's invisible for GELU/tanh because the real
// circuit only ever calls them with the symmetric domain [-1, 1] (main.cpp),
// but eval_exp and eval_inverse_naive/_2 -- the softmax numerator and
// denominator -- are always called with asymmetric domains (e.g. [2, 5000]),
// so the real circuit's softmax is computing the wrong function. That's a
// strong candidate root cause for the near-zero self-attention output in
// reports/04-attention-signal-collapse.md.
//
// Run from build/, same as the main binary (paths are relative to it):
//   ./FHE-BERT-Tiny-Tests

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>
#include "FHEController.h"

using namespace std;

FHEController controller;

static int tests_run = 0;
static int tests_passed = 0;

// Runs `fn`, which returns the observed error against a hand-computed
// expected value; catches any exception (this GPU port is known to throw
// mid-circuit) so one broken primitive doesn't take the rest of the suite
// down with it.
static void run_test(const string &name, double tol, const function<double()> &fn) {
    tests_run++;
    try {
        double err = fn();
        bool ok = err <= tol;
        if (ok) tests_passed++;
        cout << (ok ? "[PASS] " : "[FAIL] ") << left << setw(52) << name
             << "err = " << scientific << setprecision(2) << err
             << "  (tol " << tol << ")" << defaultfloat << endl;
    } catch (const std::exception &e) {
        cout << "[FAIL] " << left << setw(52) << name << "threw: " << e.what() << endl;
    }
}

// "Expanded" packing (read_expanded_input): element i repeated 128 times contiguously.
static vector<double> expand128(const vector<double> &v) {
    vector<double> out(controller.num_slots, 0.0);
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
            out[i * 128 + j] = v[i];
    return out;
}

// "Repeated" packing (read_repeated_input): the whole 128-vector tiled 128 times.
static vector<double> repeat128(const vector<double> &v) {
    vector<double> out(controller.num_slots, 0.0);
    for (int j = 0; j < 128; j++)
        for (int i = 0; i < 128; i++)
            out[j * 128 + i] = v[i];
    return out;
}

static double max_abs_diff(const vector<double> &a, const vector<double> &b, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) m = max(m, fabs(a[i] - b[i]));
    return m;
}

int main() {
    cout << "Loading CKKS context and keys (from ../keys)..." << endl;
    controller.load_context(false);
    controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);
    cout << "Context ready, circuit depth " << controller.circuit_depth << ".\n" << endl;

    cout << "=== FHE-BERT-Tiny concept tests ===\n" << endl;

    // --- 1. CKKS round trip -------------------------------------------------
    run_test("Encrypt/Decrypt round-trip", 1e-6, [] {
        vector<double> v(controller.num_slots, 0.0);
        for (int i = 0; i < 128; i++) v[i] = sin(i * 0.1);
        auto dec = controller.decrypt_tovector(controller.encrypt(v), controller.num_slots);
        return max_abs_diff(v, dec, 128);
    });

    // --- 2. EvalAdd, paper primitive (Section 1.4) --------------------------
    run_test("EvalAdd (Section 1.4)", 1e-6, [] {
        vector<double> a(controller.num_slots, 0.0), b(controller.num_slots, 0.0), expected(controller.num_slots, 0.0);
        for (int i = 0; i < 128; i++) { a[i] = i * 0.01; b[i] = 1.0 - i * 0.01; expected[i] = a[i] + b[i]; }
        auto dec = controller.decrypt_tovector(controller.add(controller.encrypt(a), controller.encrypt(b)), controller.num_slots);
        return max_abs_diff(expected, dec, 128);
    });

    // --- 3. EvalMult, paper primitive (Section 1.4) -------------------------
    run_test("EvalMult (Section 1.4)", 1e-6, [] {
        vector<double> a(controller.num_slots, 0.0), b(controller.num_slots, 0.0), expected(controller.num_slots, 0.0);
        for (int i = 0; i < 128; i++) { a[i] = (i % 7) - 3; b[i] = 0.5 + 0.01 * i; expected[i] = a[i] * b[i]; }
        auto dec = controller.decrypt_tovector(controller.mult(controller.encrypt(a), controller.encrypt(b)), controller.num_slots);
        return max_abs_diff(expected, dec, 128);
    });

    // --- 4. RotSum, the shared primitive behind Algorithms 1-2 --------------
    run_test("RotSum (Section 2.1.1, Algorithms 1-2)", 1e-4, [] {
        vector<double> v128(128, 0.0);
        double expected_sum = 0;
        for (int i = 0; i < 128; i++) { v128[i] = (i % 5) - 2; expected_sum += v128[i]; }
        vector<double> full(controller.num_slots, 0.0);
        for (int i = 0; i < 128; i++) full[i] = v128[i];
        auto dec = controller.decrypt_tovector(controller.rotsum(controller.encrypt(full), 128, 1), controller.num_slots);
        return fabs(dec[0] - expected_sum);
    });

    // --- 5. VecMatER / matmulRE (Algorithm 1, Section 2.1.1) ----------------
    // Identity weight (its own transpose, so it works unambiguously as a
    // Row-major matrix here): output should reproduce the input vector.
    run_test("VecMatER / matmulRE (Algorithm 1)", 3e-3, [] {
        vector<double> identity_flat(controller.num_slots, 0.0);
        for (int i = 0; i < 128; i++) identity_flat[i * 128 + i] = 1.0;
        Ptxt identity_ptxt = controller.encode(identity_flat, 0, controller.num_slots);

        vector<double> a128(128, 0.0);
        a128[0] = 0.37; a128[1] = -0.82; a128[2] = 0.15; a128[127] = 0.5;
        vector<Ctxt> rows = {controller.encrypt(expand128(a128))};
        auto out = controller.matmulRE(rows, identity_ptxt, nullptr);
        auto dec = controller.decrypt_tovector(out[0], controller.num_slots);
        return max_abs_diff(a128, dec, 128);
    });

    // --- 6. VecMatRC / matmulCR (Algorithm 2, Section 2.1.1) ----------------
    // A Column-major matrix with only "row 0" populated (matrix_flat[c*128] =
    // g(c)) times a Repeated input probe with a[0]=s: per the paper, VecMatRC
    // only guarantees a correct value at slots i with i mod 128 == 0, one per
    // output column c (the "Expanded" broadcast is a separate step the paper
    // calls Repeat) -- so that's what's checked, at dec[c*128] for every c.
    run_test("VecMatRC / matmulCR (Algorithm 2)", 3e-3, [] {
        double s = 0.8;
        vector<double> g(128);
        vector<double> matrix_flat(controller.num_slots, 0.0);
        for (int c = 0; c < 128; c++) { g[c] = sin(c * 0.037); matrix_flat[c * 128] = g[c]; }
        Ptxt weight = controller.encode(matrix_flat, 0, controller.num_slots);

        vector<double> a128(128, 0.0);
        a128[0] = s;
        vector<Ctxt> rows = {controller.encrypt(repeat128(a128))};
        auto out = controller.matmulCR(rows, weight, nullptr);
        auto dec = controller.decrypt_tovector(out[0], controller.num_slots);

        double err = 0;
        for (int c = 0; c < 128; c++) err = max(err, fabs(dec[c * 128] - s * g[c]));
        return err;
    });

    // --- 7. WrapUpRepeated (Algorithm 4, Section 2.1.2) ---------------------
    run_test("WrapUpRepeated (Algorithm 4)", 3e-3, [] {
        vector<vector<double>> vs = {vector<double>(128, 1.0), vector<double>(128, 2.0), vector<double>(128, 3.0)};
        vector<Ctxt> cts;
        for (auto &v : vs) cts.push_back(controller.encrypt(repeat128(v)));
        auto dec = controller.decrypt_tovector(controller.wrapUpRepeated(cts), controller.num_slots);
        double err = 0;
        for (size_t r = 0; r < vs.size(); r++)
            for (int c = 0; c < 128; c++)
                err = max(err, fabs(dec[r * 128 + c] - vs[r][c]));
        return err;
    });

    // --- 8. WrapUpExpanded (Algorithm 3, Section 2.1.2) ---------------------
    run_test("WrapUpExpanded (Algorithm 3)", 3e-3, [] {
        vector<vector<double>> vs(3, vector<double>(128));
        for (int e = 0; e < 128; e++) { vs[0][e] = sin(e * 0.05); vs[1][e] = cos(e * 0.05); vs[2][e] = e * 0.001; }
        vector<Ctxt> cts;
        for (auto &v : vs) cts.push_back(controller.encrypt(expand128(v)));
        auto dec = controller.decrypt_tovector(controller.wrapUpExpanded(cts), controller.num_slots);
        // Column-major output: slot e*128 + i holds vs[i][e].
        double err = 0;
        for (int e = 0; e < 128; e++)
            for (size_t i = 0; i < vs.size(); i++)
                err = max(err, fabs(dec[e * 128 + (int) i] - vs[i][e]));
        return err;
    });

    // --- 9. eval_inverse_naive, softmax denominator (Table 1, Encoder 1) ----
    // KNOWN BUG: this uses the exact (asymmetric) domain main.cpp's encoder1()
    // passes at the real call site. eval_inverse_naive silently evaluates
    // 1/(x + min) instead of 1/x for an asymmetric [min, max] domain -- see
    // the file header. Expect this to FAIL, worse at small x (where the min=2
    // offset is proportionally larger): e.g. x=5 returns ~1/7, not 1/5.
    run_test("eval_inverse_naive, 1/x on [2,5000] (Table 1)", 2e-3, [] {
        double lo = 2, hi = 5000;
        vector<double> xs = {5, 50, 500, 4000};
        double err = 0;
        for (double x : xs) {
            auto ct = controller.encrypt(vector<double>(controller.num_slots, x));
            auto dec = controller.decrypt_tovector(controller.eval_inverse_naive(ct, lo, hi), controller.num_slots);
            err = max(err, fabs(dec[0] - 1.0 / x));
        }
        return err;
    });

    // --- 10. eval_gelu_function (Eq. 5, Section 2.3) ------------------------
    // Mirrors the real call shape exactly (main.cpp: domain always [-1,1],
    // asymmetry folded into `mult` instead) -- passes, since the bug above
    // only shows up for an asymmetric [min,max] domain.
    run_test("eval_gelu_function, GELU(x) via [-1,1]+mult (Eq. 5)", 2e-2, [] {
        auto gelu = [](double x) { return 0.5 * x * (1 + erf(x / sqrt(2.0))); };
        double mult = 1 / 13.5; // matches encoder1()'s GELU_max_abs_value
        vector<double> xs_real = {-10.0, -3.0, 3.0, 10.0};
        double err = 0;
        for (double x : xs_real) {
            auto ct = controller.encrypt(vector<double>(controller.num_slots, x * mult));
            auto dec = controller.decrypt_tovector(controller.eval_gelu_function(ct, -1, 1, mult, 119), controller.num_slots);
            err = max(err, fabs(dec[0] - gelu(x)));
        }
        return err;
    });

    // --- 11. eval_tanh_function, Pooler (Table 1) ---------------------------
    // Mirrors pooler()'s real call shape exactly (domain [-1,1], asymmetry
    // folded into `tanhScale`) -- also passes for the same reason as GELU.
    run_test("eval_tanh_function, tanh(x) via [-1,1]+scale (Table 1, Pooler)", 2e-2, [] {
        double tanh_scale = 1 / 30.0; // matches pooler()'s tanhScale
        vector<double> xs_real = {-15.0, -5.0, 5.0, 15.0};
        double err = 0;
        for (double x : xs_real) {
            auto ct = controller.encrypt(vector<double>(controller.num_slots, x * tanh_scale));
            auto dec = controller.decrypt_tovector(controller.eval_tanh_function(ct, -1, 1, tanh_scale, 300), controller.num_slots);
            err = max(err, fabs(dec[0] - tanh(x)));
        }
        return err;
    });

    // --- 12. eval_exp, softmax numerator (Section 2.2.1, Figure 6 S5) ------
    // KNOWN BUG: exact (asymmetric) domain from encoder1()'s real call site.
    // eval_exp fits exp(x) via Chebyshev then cubes-by-squaring three times
    // (EvalSquare x3), so a *correct* implementation would return exp(x)^8 =
    // exp(8x). Instead -- see the file header -- it evaluates at (x + min):
    // for x=-1.0 that shifted argument (-2.1) falls outside the fitted
    // domain entirely, and the Chebyshev polynomial's extrapolation blows up
    // enough to make the ciphertext undecodable (caught below, reported as a
    // thrown exception rather than a numeric error).
    run_test("eval_exp, softmax numerator e^(8x) (Section 2.2.1 S5)", 0.1, [] {
        double lo = -1.1, hi = 1.6;
        vector<double> xs = {-1.0, -0.3, 0.5, 1.4};
        double err = 0;
        for (double x : xs) {
            auto ct = controller.encrypt(vector<double>(controller.num_slots, x));
            auto dec = controller.decrypt_tovector(controller.eval_exp(ct, 1, lo, hi, 30), controller.num_slots);
            double expected = exp(8 * x);
            err = max(err, fabs(dec[0] - expected) / expected);
        }
        return err;
    });

    cout << endl << tests_passed << "/" << tests_run << " tests passed." << endl;
    if (tests_passed != tests_run) {
        cout << "(Failures in eval_inverse_naive / eval_exp are expected on this branch --\n"
                " see the file header and reports/04-attention-signal-collapse.md.)" << endl;
    }
    return tests_passed == tests_run ? 0 : 1;
}
