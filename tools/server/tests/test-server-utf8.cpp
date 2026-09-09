// Unit test for the UTF-8 sanitising in task_result_state::update_chat_msg().
//
// The generated text is a raw byte stream: a byte fallback token, or a prompt cut mid
// character, puts undecodable bytes in it. Those bytes are substituted with U+FFFD before the
// chat parsers see them, following the Unicode "maximal subpart of an ill-formed subsequence"
// rule (Unicode 15 core spec, section 3.9, D93b) which is also what the JSON serialiser's
// error_handler_t::replace does on the way to the client.
//
// The oracle here is an independently written implementation of the WHATWG Encoding Standard
// UTF-8 decoder, https://encoding.spec.whatwg.org/#utf-8-decoder, rather than a copy of the
// code under test. Every named case, all 256 single bytes, all 65536 two byte sequences, a
// three and four byte lead sweep and a biased random fuzz are compared against it, and every
// valid Unicode scalar value must round trip byte identically.
//
// Run with no arguments for everything; a single mode name (named, split, roundtrip,
// exhaustive1, exhaustive2, exhaustive3, fuzz) runs just that one.

#include "server-task.h"
#include "server-common.h"
#include "chat.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

// ---------------------------------------------------------------------------
// independent reference: WHATWG Encoding Standard "UTF-8 decoder" (maximal subpart)
// https://encoding.spec.whatwg.org/#utf-8-decoder
// ---------------------------------------------------------------------------
static std::string ref_decode_replace(const std::string & in) {
    std::string out;
    size_t i = 0;
    const size_t n = in.size();
    uint32_t cp = 0;
    int seen = 0, needed = 0;
    uint8_t lo = 0x80, hi = 0xBF;

    auto emit = [&](uint32_t c) {
        if (c < 0x80) { out += (char) c; }
        else if (c < 0x800) { out += (char) (0xC0 | (c >> 6)); out += (char) (0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += (char) (0xE0 | (c >> 12)); out += (char) (0x80 | ((c >> 6) & 0x3F)); out += (char) (0x80 | (c & 0x3F)); }
        else { out += (char) (0xF0 | (c >> 18)); out += (char) (0x80 | ((c >> 12) & 0x3F)); out += (char) (0x80 | ((c >> 6) & 0x3F)); out += (char) (0x80 | (c & 0x3F)); }
    };
    auto err = [&]() { out += "\xEF\xBF\xBD"; };

    while (i <= n) {
        if (i == n) {
            if (needed != 0) { err(); }
            break;
        }
        uint8_t b = (uint8_t) in[i];
        if (needed == 0) {
            if (b <= 0x7F) { emit(b); i++; continue; }
            if (b >= 0xC2 && b <= 0xDF) { needed = 1; cp = b & 0x1F; }
            else if (b >= 0xE0 && b <= 0xEF) {
                if (b == 0xE0) { lo = 0xA0; }
                if (b == 0xED) { hi = 0x9F; }
                needed = 2; cp = b & 0x0F;
            } else if (b >= 0xF0 && b <= 0xF4) {
                if (b == 0xF0) { lo = 0x90; }
                if (b == 0xF4) { hi = 0x8F; }
                needed = 3; cp = b & 0x07;
            } else { err(); i++; continue; }
            i++; seen = 0; continue;
        }
        if (b < lo || b > hi) {
            // reset, do NOT consume: prefix is the maximal subpart
            cp = 0; seen = 0; needed = 0; lo = 0x80; hi = 0xBF;
            err();
            continue;
        }
        lo = 0x80; hi = 0xBF;
        cp = (cp << 6) | (b & 0x3F);
        seen++; i++;
        if (seen == needed) { emit(cp); cp = 0; seen = 0; needed = 0; }
    }
    return out;
}

// ---------------------------------------------------------------------------

struct outcome {
    bool threw = false;
    std::string what;
    std::string content;
};

// non-streaming: one call, is_partial = false
static outcome run_final(const std::string & text) {
    outcome o;
    common_chat_parser_params p; // defaults to COMMON_CHAT_FORMAT_CONTENT_ONLY
    task_result_state st(p);
    std::vector<common_chat_msg_diff> diffs;
    try {
        auto msg = st.update_chat_msg(text, false, diffs);
        o.content = msg.content;
    } catch (const std::exception & e) {
        o.threw = true;
        o.what = e.what();
    }
    return o;
}

// streaming: each chunk with is_partial = true, then an empty final flush
static outcome run_stream(const std::vector<std::string> & chunks) {
    outcome o;
    common_chat_parser_params p;
    task_result_state st(p);
    std::vector<common_chat_msg_diff> diffs;
    try {
        common_chat_msg msg;
        for (const auto & c : chunks) {
            msg = st.update_chat_msg(c, true, diffs);
        }
        msg = st.update_chat_msg("", false, diffs);
        o.content = msg.content;
    } catch (const std::exception & e) {
        o.threw = true;
        o.what = e.what();
    }
    return o;
}

static std::string hex(const std::string & s) {
    static const char * d = "0123456789ABCDEF";
    std::string r;
    for (unsigned char c : s) { r += d[c >> 4]; r += d[c & 15]; r += ' '; }
    if (!r.empty()) { r.pop_back(); }
    return r;
}

static std::string enc(uint32_t c) {
    std::string out;
    if (c < 0x80) { out += (char) c; }
    else if (c < 0x800) { out += (char) (0xC0 | (c >> 6)); out += (char) (0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { out += (char) (0xE0 | (c >> 12)); out += (char) (0x80 | ((c >> 6) & 0x3F)); out += (char) (0x80 | (c & 0x3F)); }
    else { out += (char) (0xF0 | (c >> 18)); out += (char) (0x80 | ((c >> 12) & 0x3F)); out += (char) (0x80 | ((c >> 6) & 0x3F)); out += (char) (0x80 | (c & 0x3F)); }
    return out;
}

// what the client actually receives: the server serialises every response through
// safe_json_to_str(), which is dump(..., error_handler_t::replace)
static std::string as_client_sees(const std::string & content) {
    nlohmann::ordered_json j = nlohmann::ordered_json{{"content", content}};
    return j.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

static int n_threw_final = 0, n_threw_stream = 0, n_cases = 0;
static int n_mismatch_ref = 0;

// named case: print everything, both drive modes
static void named(const char * id, const char * desc, const std::string & text) {
    n_cases++;
    outcome f = run_final(text);
    // stream it one byte at a time: worst case for the hold-back buffer
    std::vector<std::string> bytes;
    for (char c : text) { bytes.push_back(std::string(1, c)); }
    outcome s = run_stream(bytes);
    const std::string ref = ref_decode_replace(text);

    if (f.threw) { n_threw_final++; }
    if (s.threw) { n_threw_stream++; }
    if (!f.threw && f.content != ref) { n_mismatch_ref++; }

    printf("CASE %-10s in=[%-26s] final=%-6s stream=%-6s out=[%-30s] ref=[%-30s] %s%s\n",
           id,
           hex(text).c_str(),
           f.threw ? "THROW" : "ok",
           s.threw ? "THROW" : "ok",
           f.threw ? f.what.substr(0, 30).c_str() : hex(f.content).c_str(),
           hex(ref).c_str(),
           (!f.threw && f.content == ref) ? "REF-MATCH" : (f.threw ? "" : "REF-DIFF"),
           (!f.threw && !s.threw && f.content != s.content) ? " STREAM-DIFF" : "");
    if (!f.threw && !s.threw && f.content != s.content) {
        printf("            stream-out=[%s]\n", hex(s.content).c_str());
    }
    // the bytes the HTTP client ends up with, after the JSON serialiser's own replacement
    printf("WIRE %-10s %s\n", id, f.threw ? "<request-failed>" : as_client_sees(f.content).c_str());
    (void) desc;
}


// ---------------------------------------------------------------------------
// The server token pipeline, not just update_chat_msg().
//
// tools/server/server-context.cpp process_token() appends the token text to the slot's own
// generated_text and then, if validate_utf8() says the tail is a cut-off multi-byte sequence,
// sends nothing at all for that token. So a trailing incomplete sequence is never delivered as
// a partial, and send_final_response() supplies an empty content in stream mode, which means
// update_chat_msg() never gets a chance to substitute for it. Non-streaming hands over the whole
// text and does get substituted.
//
// The data loss in streaming predates this change: the held back bytes were dropped in both
// modes before it. The divergence is new, because the non-streaming side is now substituted and
// the streaming side still is not. Fixing the streaming side means changing what
// process_token() sends, which is a different file and changes what streaming clients receive,
// so it is pinned here rather than folded in. The hard assertion is the one that matters:
// no decodable content may be lost in either mode.
static int n_pipeline_asym = 0, n_pipe_stream_bad = 0, n_pipe_nostream_bad = 0;

static void pipeline_case(const char * id, const std::vector<std::string> & tokens) {
    // ---- streaming, as process_token() drives it ----
    std::string slot_text;
    size_t n_sent = 0;
    common_chat_parser_params p;
    task_result_state st_stream(p);
    std::vector<common_chat_msg_diff> diffs;
    std::string streamed;
    bool threw = false;
    try {
        for (const auto & tok : tokens) {
            slot_text += tok;
            if (validate_utf8(slot_text) < slot_text.size()) {
                continue;   // incomplete tail: process_token() sends nothing for this token
            }
            const std::string to_send = slot_text.substr(n_sent);
            n_sent = slot_text.size();
            st_stream.update_chat_msg(to_send, true, diffs);
        }
        // send_final_response() sets content to "" in stream mode
        streamed = st_stream.update_chat_msg("", false, diffs).content;
    } catch (const std::exception &) {
        threw = true;
    }

    // ---- non-streaming: the whole text in one final call ----
    outcome nostream = run_final(slot_text);

    const std::string held_back = slot_text.substr(n_sent);
    const std::string ref_sent  = ref_decode_replace(slot_text.substr(0, n_sent));
    const std::string ref_all   = ref_decode_replace(slot_text);

    const bool stream_ok   = !threw && streamed == ref_sent;
    const bool nostream_ok = !nostream.threw && nostream.content == ref_all;
    const bool agree       = !threw && !nostream.threw && streamed == nostream.content;

    if (!agree)       { n_pipeline_asym++;    }
    if (!stream_ok)   { n_pipe_stream_bad++;  }
    if (!nostream_ok) { n_pipe_nostream_bad++; }

    printf("PIPE %-14s text=[%-20s] held_back=[%-8s] stream=[%-14s] nostream=[%-14s] %s%s%s\n",
           id, hex(slot_text).c_str(), hex(held_back).c_str(),
           threw ? "THROW" : hex(streamed).c_str(),
           nostream.threw ? "THROW" : hex(nostream.content).c_str(),
           stream_ok ? "stream-ok" : "STREAM-LOST-CONTENT",
           nostream_ok ? " nostream-ok" : " NOSTREAM-BAD",
           agree ? " agree" : " ASYMMETRIC");
}

static int g_fail = 0;

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    const bool all = mode == "all";

    if (all || mode == "named") {
        printf("== named enumeration ==\n");
        // valid controls, must be byte identical between base and head
        named("V-ascii",   "plain ascii",            "Hello, world!");
        named("V-latin",   "2-byte latin",           "caf\xC3\xA9");
        named("V-cjk",     "3-byte CJK",             "\xE4\xB8\xAD\xE6\x96\x87");
        named("V-emoji",   "4-byte emoji",           "\xF0\x9F\x98\x80");
        named("V-combine", "combining marks",        "e\xCC\x81");
        named("V-min2",    "U+0080 min 2-byte",      "\xC2\x80");
        named("V-max3",    "U+FFFF max BMP",         "\xEF\xBF\xBF");
        named("V-min4",    "U+10000 min astral",     "\xF0\x90\x80\x80");
        named("V-max4",    "U+10FFFF max scalar",    "\xF4\x8F\xBF\xBF");
        named("V-e0a0",    "U+0800 E0 A0 80",        "\xE0\xA0\x80");
        named("V-ed9f",    "U+D7FF ED 9F BF",        "\xED\x9F\xBF");
        named("V-bom",     "U+FEFF BOM",             "\xEF\xBB\xBF");
        named("V-fffd",    "literal U+FFFD in text", "a\xEF\xBF\xBDz");
        // truncated multi-byte at the very end, no following token
        named("T-c3",      "trailing lead of 2-byte",   "abc\xC3");
        named("T-e4",      "trailing lead of 3-byte",   "abc\xE4");
        named("T-e4b8",    "trailing 2 of 3 bytes",     "abc\xE4\xB8");
        named("T-f0",      "trailing lead of 4-byte",   "abc\xF0");
        named("T-f09f",    "trailing 2 of 4",           "abc\xF0\x9F");
        named("T-f09f98",  "trailing 3 of 4",           "abc\xF0\x9F\x98");
        named("T-only-c3", "lead byte is whole output", "\xC3");
        // continuation byte with no lead
        named("C-80",      "lone 80",                "abc\x80");
        named("C-bf",      "lone BF",                "abc\xBF");
        named("C-a1",      "lone A1 (byte fallback)","\xA1");
        named("C-lead",    "continuation first",     "\x80\x41");
        named("C-run",     "run of continuations",   "\x80\x80\x80\x80");
        // overlong
        named("O-c080",    "C0 80 overlong NUL",     "\xC0\x80");
        named("O-c1bf",    "C1 BF overlong",         "\xC1\xBF");
        named("O-e080af",  "E0 80 AF overlong /",    "\xE0\x80\xAF");
        named("O-f08080af","F0 80 80 AF overlong",   "\xF0\x80\x80\xAF");
        named("O-e09fbf",  "E0 9F BF overlong",      "\xE0\x9F\xBF");
        named("O-f08fbfbf","F0 8F BF BF overlong",   "\xF0\x8F\xBF\xBF");
        // surrogates
        named("S-d800",    "ED A0 80 = U+D800",      "\xED\xA0\x80");
        named("S-dfff",    "ED BF BF = U+DFFF",      "\xED\xBF\xBF");
        named("S-pair",    "CESU-8 surrogate pair",  "\xED\xA0\xBD\xED\xB8\x80");
        // above U+10FFFF
        named("X-f4908080","F4 90 80 80 = U+110000", "\xF4\x90\x80\x80");
        named("X-f5",      "F5 lead",                "\xF5\x80\x80\x80");
        named("X-f7bfbfbf","F7 BF BF BF",            "\xF7\xBF\xBF\xBF");
        // invalid leads
        named("L-f8",      "F8 lead (5-byte form)",  "\xF8\x88\x80\x80\x80");
        named("L-fc",      "FC lead (6-byte form)",  "\xFC\x84\x80\x80\x80\x80");
        named("L-fe",      "FE never valid",         "\xFE");
        named("L-ff",      "FF never valid",         "\xFF");
        named("L-fefe",    "FE FF",                  "\xFE\xFF");
        // replacement counting, the maximal-subpart cases
        named("M-e280-41", "E2 80 41 -> 1 FFFD + A", "\xE2\x80\x41");
        named("M-c3c3",    "C3 C3 -> 2 FFFD",        "\xC3\xC3");
        named("M-f08080-41","F0 80 80 41",           "\xF0\x80\x80\x41");
        named("M-uni-ex",  "UTS worked example",     "\x61\xF1\x80\x80\xE1\x80\xC2\x62");
        named("M-e1-80-e2","E1 80 E2 F0 91 92 F1 BF 41", "\xE1\x80\xE2\xF0\x91\x92\xF1\xBF\x41");
        // NUL
        named("N-nul",     "embedded NUL",           std::string("a\0b", 3));

        // mixed valid + invalid, the realistic byte-fallback stream
        named("R-mix1",    "valid then invalid",     "hello \xC3\xA9 \x80 world");
        named("R-mix2",    "invalid then valid",     "\xA1 caf\xC3\xA9");
        named("R-long",    "long mixed",             std::string("x") + "\xE4\xB8\xAD" + "\xFF" + "\xF0\x9F\x98\x80" + "\xED\xA0\x80" + "y");

        printf("\nnamed: cases=%d threw_final=%d threw_stream=%d ref_diff=%d\n",
               n_cases, n_threw_final, n_threw_stream, n_mismatch_ref);
        printf("RESULT named threw_final=%d threw_stream=%d ref_diff=%d\n",
               n_threw_final, n_threw_stream, n_mismatch_ref);
        g_fail += n_threw_final + n_threw_stream + n_mismatch_ref;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "pipeline") {
        printf("== server token pipeline, streaming against non streaming ==\n");
        // generation stops on a lead byte with nothing after it: the classic byte fallback tail
        pipeline_case("tail-c3",   {"abc", "\xC3"});
        pipeline_case("tail-e4",   {"abc", "\xE4"});
        pipeline_case("tail-e4b8", {"abc", "\xE4", "\xB8"});
        pipeline_case("tail-f0",   {"abc", "\xF0"});
        pipeline_case("only-c3",   {"\xC3"});
        // a run of lead bytes: validate_utf8() holds the last one back every time
        pipeline_case("run-c3",    {"\xC3", "\xC3", "\xC3", "\xC3"});
        // a complete but ill formed sequence at the end: not held back, so both modes see it
        pipeline_case("surrogate", {"abc", "\xED\xA0\x80"});
        pipeline_case("overlong",  {"abc", "\xC0\x80"});
        pipeline_case("above-max", {"abc", "\xF4\x90\x80\x80"});
        // a stray continuation byte: also not held back
        pipeline_case("stray-80",  {"abc", "\x80"});
        // valid characters split across tokens: nothing may be lost or duplicated
        pipeline_case("split-cjk", {"a", "\xE4", "\xB8", "\xAD", "b"});
        pipeline_case("split-emo", {"a", "\xF0\x9F", "\x98\x80", "b"});
        pipeline_case("valid-mix", {"caf", "\xC3\xA9", " ", "\xE4\xB8\xAD"});
        printf("RESULT pipeline stream_lost_content=%d nostream_bad=%d asymmetric=%d\n",
               n_pipe_stream_bad, n_pipe_nostream_bad, n_pipeline_asym);
        g_fail += n_pipe_stream_bad + n_pipe_nostream_bad;
        printf("note: an asymmetric row means the held back bytes never reached update_chat_msg()\n"
               "      in stream mode, so only the non-streaming side could substitute for them.\n"
               "      Streaming dropped them before this change too, so nothing is lost that was\n"
               "      not lost already; it is not counted as a failure.\n");
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "split") {
        // a valid character split across token / SSE chunk boundaries.
        // Every split point of every valid sample, in 2 and 3 chunk form.
        printf("== split enumeration ==\n");
        const std::vector<std::string> samples = {
            "caf\xC3\xA9 au lait",
            "\xE4\xB8\xAD\xE6\x96\x87\xE3\x83\x86",
            "a\xF0\x9F\x98\x80" "b\xF0\x9F\x91\x8D" "c",
            "\xC2\x80\xDF\xBF\xE0\xA0\x80\xEF\xBF\xBF\xF0\x90\x80\x80\xF4\x8F\xBF\xBF",
        };
        int bad = 0, threw = 0, total = 0;
        for (const auto & s : samples) {
            for (size_t i = 0; i <= s.size(); i++) {
                total++;
                outcome o = run_stream({s.substr(0, i), s.substr(i)});
                if (o.threw) { threw++; printf("  SPLIT2 THROW at %zu of [%s]: %s\n", i, hex(s).c_str(), o.what.c_str()); }
                else if (o.content != s) { bad++; printf("  SPLIT2 CORRUPT at %zu: got [%s] want [%s]\n", i, hex(o.content).c_str(), hex(s).c_str()); }
                for (size_t j = i; j <= s.size(); j++) {
                    total++;
                    outcome o3 = run_stream({s.substr(0, i), s.substr(i, j - i), s.substr(j)});
                    if (o3.threw) { threw++; }
                    else if (o3.content != s) { bad++; }
                }
            }
            // one byte at a time, the pathological SSE chunking
            total++;
            std::vector<std::string> bytes;
            for (char c : s) { bytes.push_back(std::string(1, c)); }
            outcome ob = run_stream(bytes);
            if (ob.threw) { threw++; printf("  SPLIT1 THROW on [%s]\n", hex(s).c_str()); }
            else if (ob.content != s) { bad++; printf("  SPLIT1 CORRUPT: got [%s] want [%s]\n", hex(ob.content).c_str(), hex(s).c_str()); }
        }
        printf("RESULT split total=%d threw=%d corrupted=%d\n", total, threw, bad);
        g_fail += bad + threw;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "roundtrip") {
        // every valid Unicode scalar value must survive byte identical.
        printf("== valid scalar round trip ==\n");
        int bad = 0, threw = 0; long total = 0;
        std::string batch;
        for (uint32_t cp = 0; cp <= 0x10FFFF; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) { continue; }
            if (cp == 0) { continue; } // NUL is tested separately, chat content trims nothing but keep it simple
            batch += enc(cp);
            if (batch.size() >= 4096 || cp == 0x10FFFF) {
                total++;
                outcome o = run_final(batch);
                if (o.threw) { threw++; printf("  THROW on batch ending U+%04X: %s\n", cp, o.what.c_str()); }
                else if (o.content != batch) {
                    bad++;
                    printf("  CORRUPT in batch ending U+%04X\n", cp);
                }
                batch.clear();
            }
        }
        printf("RESULT roundtrip batches=%ld threw=%d corrupted=%d\n", total, threw, bad);
        g_fail += bad + threw;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "exhaustive1") {
        printf("== all 256 single bytes ==\n");
        int threw = 0, refdiff = 0;
        for (int b = 0; b < 256; b++) {
            std::string s(1, (char) b);
            outcome o = run_final(s);
            if (o.threw) { threw++; printf("  THROW %02X: %s\n", b, o.what.substr(0, 60).c_str()); }
            else if (o.content != ref_decode_replace(s)) { refdiff++; printf("  REFDIFF %02X got[%s] ref[%s]\n", b, hex(o.content).c_str(), hex(ref_decode_replace(s)).c_str()); }
        }
        printf("RESULT single threw=%d refdiff=%d\n", threw, refdiff);
        g_fail += threw + refdiff;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "exhaustive2") {
        printf("== all 65536 two-byte sequences ==\n");
        int threw = 0, refdiff = 0;
        std::map<std::string, int> refdiff_examples;
        for (int a = 0; a < 256; a++) {
            for (int b = 0; b < 256; b++) {
                std::string s;
                s += (char) a; s += (char) b;
                outcome o = run_final(s);
                if (o.threw) {
                    if (threw < 5) { printf("  THROW %02X %02X: %s\n", a, b, o.what.substr(0, 60).c_str()); }
                    threw++;
                } else {
                    const std::string r = ref_decode_replace(s);
                    if (o.content != r) {
                        if (refdiff < 20) { printf("  REFDIFF %02X %02X got[%s] ref[%s]\n", a, b, hex(o.content).c_str(), hex(r).c_str()); }
                        refdiff++;
                    }
                }
            }
        }
        printf("RESULT two threw=%d refdiff=%d of 65536\n", threw, refdiff);
        g_fail += threw + refdiff;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "exhaustive3") {
        printf("== 3 and 4 byte lead sweeps ==\n");
        int threw = 0, refdiff = 0; long total = 0;
        const int tail[] = {0x00, 0x41, 0x7F, 0x80, 0x9F, 0xA0, 0xBF, 0xC0, 0xE0, 0xFF};
        for (int a = 0xC0; a <= 0xFF; a++) {
            for (int b = 0; b < 256; b++) {
                for (int t : tail) {
                    std::string s; s += (char) a; s += (char) b; s += (char) t;
                    total++;
                    outcome o = run_final(s);
                    if (o.threw) { if (threw < 5) { printf("  THROW %02X %02X %02X\n", a, b, t); } threw++; }
                    else if (o.content != ref_decode_replace(s)) {
                        if (refdiff < 20) { printf("  REFDIFF %02X %02X %02X got[%s] ref[%s]\n", a, b, t, hex(o.content).c_str(), hex(ref_decode_replace(s)).c_str()); }
                        refdiff++;
                    }
                }
            }
        }
        for (int a = 0xF0; a <= 0xFF; a++) {
            for (int b = 0; b < 256; b++) {
                for (int t : tail) {
                    std::string s; s += (char) a; s += (char) b; s += (char) 0x80; s += (char) t;
                    total++;
                    outcome o = run_final(s);
                    if (o.threw) { threw++; }
                    else if (o.content != ref_decode_replace(s)) { refdiff++; }
                }
            }
        }
        printf("RESULT three_four total=%ld threw=%d refdiff=%d\n", total, threw, refdiff);
        g_fail += threw + refdiff;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (all || mode == "fuzz") {
        // random byte soup, streamed in random chunk sizes: the realistic byte-fallback stream
        printf("== random byte fuzz ==\n");
        uint64_t seed = 0x9E3779B97F4A7C15ull;
        auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
        int threw = 0, streamdiff = 0, refdiff = 0;
        const int N = 20000;
        for (int i = 0; i < N; i++) {
            size_t len = 1 + (rnd() % 24);
            std::string s;
            for (size_t k = 0; k < len; k++) {
                // bias towards lead and continuation bytes so the interesting cases dominate
                uint64_t r = rnd() % 100;
                if (r < 30) { s += (char) (0x80 + (rnd() % 0x40)); }
                else if (r < 60) { s += (char) (0xC0 + (rnd() % 0x40)); }
                else if (r < 80) { s += (char) (0x20 + (rnd() % 0x5F)); }
                else { s += (char) (rnd() % 256); }
            }
            outcome f = run_final(s);
            if (f.threw) { if (threw < 5) { printf("  THROW [%s]: %s\n", hex(s).c_str(), f.what.substr(0, 50).c_str()); } threw++; continue; }
            if (f.content != ref_decode_replace(s)) {
                if (refdiff < 10) { printf("  REFDIFF [%s]\n    got[%s]\n    ref[%s]\n", hex(s).c_str(), hex(f.content).c_str(), hex(ref_decode_replace(s)).c_str()); }
                refdiff++;
            }
            // same bytes, random chunking
            std::vector<std::string> chunks;
            size_t p = 0;
            while (p < s.size()) { size_t take = 1 + (rnd() % 3); chunks.push_back(s.substr(p, take)); p += take; }
            outcome st = run_stream(chunks);
            if (st.threw) { threw++; continue; }
            if (st.content != f.content) {
                if (streamdiff < 5) { printf("  STREAMDIFF [%s] final[%s] stream[%s]\n", hex(s).c_str(), hex(f.content).c_str(), hex(st.content).c_str()); }
                streamdiff++;
            }
        }
        printf("RESULT fuzz n=%d threw=%d refdiff=%d streamdiff=%d\n", N, threw, refdiff, streamdiff);
        g_fail += threw + refdiff + streamdiff;
        if (!all) { return g_fail == 0 ? 0 : 1; }
    }

    if (!all) {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }

    printf("\n%s: %d failure(s)\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
