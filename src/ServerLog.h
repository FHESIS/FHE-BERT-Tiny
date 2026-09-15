//
// Structured, append-only logging for the server-side FHE circuit evaluation.
//
// This is deliberately hand-rolled (no JSON library dependency): every field written here is
// either a number or a short identifier string we control ourselves (stage names, request ids),
// never attacker- or plaintext-derived free text, so naive escaping is fine.
//
// What this logger must NEVER be given: a decrypted value, a plaintext embedding, or anything
// derived from either. The server-role FHEController has no secret key loaded while this logger
// is in use (see FHEController::load_context_server), so that can't happen by construction, but
// keep it in mind if you add call sites.
//

#ifndef NEWBERT_SERVERLOG_H
#define NEWBERT_SERVERLOG_H

#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

namespace utils {

class ServerLog {
public:
    ServerLog() = default;

    void open(const std::string& path, const std::string& request_id) {
        request_id_ = request_id;
        out_.open(path, std::ios::out | std::ios::app);
        wall_start_ = std::chrono::steady_clock::now();
    }

    bool is_open() const { return out_.is_open(); }

    // Field values passed to event()/stage_end() are written verbatim (so callers can pass
    // pre-formatted numbers or literals like "true"). For a free-text string value (e.g. an
    // exception message, which may contain quotes or newlines), build it with this instead of
    // hand-quoting -- see the ServerCircuit.cpp call site this exists for.
    static std::string json_string(const std::string& s) {
        return "\"" + escape(s) + "\"";
    }

    // A single instantaneous event (e.g. "request received", a retry, an exception).
    void event(const std::string& stage, const std::vector<std::pair<std::string, std::string>>& fields = {}) {
        if (!out_.is_open()) return;
        out_ << "{\"request_id\":\"" << escape(request_id_) << "\","
             << "\"t_ms\":" << elapsed_ms() << ","
             << "\"type\":\"event\","
             << "\"stage\":\"" << escape(stage) << "\"";
        for (auto& kv : fields) {
            out_ << ",\"" << escape(kv.first) << "\":" << kv.second;
        }
        out_ << "}\n";
        out_.flush();
    }

    // RAII-ish helper: call stage_start(name) then stage_end(name, extra fields) to log a timed
    // span (a whole encoder stage, a bootstrap call, ...).
    std::chrono::steady_clock::time_point stage_start(const std::string& stage) {
        auto now = std::chrono::steady_clock::now();
        event(stage + ".start");
        return now;
    }

    void stage_end(const std::string& stage, std::chrono::steady_clock::time_point start,
                   const std::vector<std::pair<std::string, std::string>>& fields = {}) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        std::vector<std::pair<std::string, std::string>> all = {{"elapsed_ms", std::to_string(ms)}};
        all.insert(all.end(), fields.begin(), fields.end());
        event(stage + ".end", all);
    }

    long elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - wall_start_)
            .count();
    }

    void close() {
        if (out_.is_open()) out_.close();
    }

    ~ServerLog() { close(); }

private:
    static std::string escape(const std::string& s) {
        // Proper-enough JSON string escaping. This matters in practice: OpenFHE exception
        // messages (logged verbatim on circuit-evaluation failures, see run_server_circuit())
        // contain embedded literal newlines, which without escaping split one JSONL record
        // across multiple physical lines and corrupt the log for any line-oriented reader.
        static const char* hex = "0123456789abcdef";
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        out += "\\u00";
                        out.push_back(hex[(c >> 4) & 0xF]);
                        out.push_back(hex[c & 0xF]);
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
            }
        }
        return out;
    }

    std::ofstream out_;
    std::string request_id_;
    std::chrono::steady_clock::time_point wall_start_;
};

} // namespace utils

#endif // NEWBERT_SERVERLOG_H
