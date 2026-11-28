#include "../src/unicode.h"

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

// Exact expression from unslothai/unsloth#19376. Keep in sync with the GPT4O /
// Granite expression in llama-vocab.cpp, including the combining marks.
static const std::string gpt4o_regex =
    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

static bool check_split(const char * name, const std::string & text,
                        const std::vector<std::string> & expressions,
                        const std::vector<std::string> & expected) {
    try {
        for (bool byte_encode : { false, true }) {
            auto encoded = expected;
            if (byte_encode) {
                for (auto & piece : encoded) {
                    std::string bytes;
                    for (unsigned char byte : piece) {
                        bytes += unicode_byte_to_utf8(byte);
                    }
                    piece = std::move(bytes);
                }
            }
            const auto actual = unicode_regex_split(text, expressions, byte_encode);
            if (actual != encoded) {
                fprintf(stderr, "%s: unexpected split (%zu bytes, byte_encode=%d)\n",
                        name, text.size(), byte_encode);
                if (text.size() < 128) {
                    for (const auto & piece : actual) {
                        fprintf(stderr, "  actual:");
                        for (unsigned char byte : piece) {
                            fprintf(stderr, " %02x", byte);
                        }
                        fprintf(stderr, "\n");
                    }
                }
                return false;
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s (%zu bytes)\n", name, e.what(), text.size());
        return false;
    }
    return true;
}

static bool test_gpt4o_split() {
    bool success = true;
    const std::vector<std::string> expressions = { gpt4o_regex };
    const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
        { "", {} },
        { "Hello world!", { "Hello", " world", "!" } },
        { "getUserByID", { "get", "User", "By", "ID" } },
        { "ABCéDE", { "ABCé", "DE" } },
        { "éA", { "é", "A" } },
        { "éAa", { "éAa" } },
        { "Ab\u0301C", { "Ab\u0301", "C" } },
        { "\u0301A", { "\u0301", "A" } },
        { "\u0301\u0301A", { "\u0301\u0301", "A" } },
        { "नमस्ते", { "नमस्ते" } },
        { "ΑβΓ", { "ΑβΓ" } },
        { "CAN'T we'll I’M", { "CAN'T", " we'll", " I", "’M" } },
        { "a'S a'T a'rE a'Ve a'M a'lL a'D a'REx", {
            "a'S", " a'T", " a'rE", " a'Ve", " a'M", " a'lL", " a'D", " a'RE", "x",
        } },
        { "1234567 ١٢٣٤", { "123", "456", "7", " ", "١٢٣", "٤" } },
        { "!\r\n/", { "!\r\n/" } },
        { "\r\n/", { "\r\n", "/" } },
        { "  \tfoo", { "  ", "\tfoo" } },
        { "\t\n  foo", { "\t\n", " ", " foo" } },
        { " \t ", { " \t " } },
        { "\u00a0X", { "\u00a0X" } },
        { std::string("\0A", 2), { std::string("\0A", 2) } },
        { std::string(" \0!", 3), { std::string(" \0!", 3) } },
        { "\xc0\xa0", { "\ufffd\ufffd" } },
    };
    for (const auto & test : cases) {
        success = check_split("gpt4o boundary", test.first, expressions, test.second) && success;
    }

    // A preceding regex pass imposes boundaries the scanner must not cross.
    success = check_split("gpt4o multiple passes", "foo1Bar2's",
                          { "\\p{N}", gpt4o_regex }, { "foo", "1", "Bar", "2", "'s" }) && success;

    // The wrapper is semantically identical but bypasses exact-expression dispatch,
    // so short inputs are compared against the original std::regex implementation.
    const std::vector<std::string> reference = { "(?:" + gpt4o_regex + ")" };
    const std::vector<std::string> alphabet = {
        "a", "A", "é", "中", "\u0301", "!", "'", " ", "\t", "\n", "1", "/", "\u0085",
    };
    for (const auto & a : alphabet) {
        for (const auto & b : alphabet) {
            for (const auto & c : alphabet) {
                const auto text = a + b + c;
                success = check_split("gpt4o std::regex parity", text, expressions,
                                      unicode_regex_split(text, reference, false)) && success;
            }
        }
    }
    auto random_alphabet = alphabet;
    random_alphabet.insert(random_alphabet.end(), {
        std::string(1, '\0'), "\r", "\u00a0", "\u2028", "\u200d", "\u0378", "١", "'RE", "'S", "foo", "BAR",
    });
    uint32_t seed = 19376;
    for (size_t i = 0; i < 2048; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const size_t length = (seed >> 24) % 24;
        std::string text;
        for (size_t j = 0; j < length; ++j) {
            seed = seed * 1664525u + 1013904223u;
            text += random_alphabet[(seed >> 16) % random_alphabet.size()];
        }
        success = check_split("gpt4o randomized std::regex parity", text, expressions,
                              unicode_regex_split(text, reference, false)) && success;
    }

    // Long individual matches exhaust recursive std::regex engines (MSVC and
    // older libstdc++). Arbitrary chunking would change BPE token boundaries.
    for (char c : { 'a', 'A', '!', ' ' }) {
        const std::string run(1 << 17, c);
        success = check_split("gpt4o long ASCII run", run, expressions, { run }) && success;
    }
    std::string marks;
    std::string letters;
    for (size_t i = 0; i < (1 << 16); ++i) {
        marks += "\u0301";
        letters += "中";
    }
    success = check_split("gpt4o long combining marks", marks, expressions, { marks }) && success;
    success = check_split("gpt4o long Unicode letters", letters, expressions, { letters }) && success;
    // Both case classes accept non-ASCII letters. The first alternative must
    // backtrack to the final such letter before an ASCII uppercase suffix.
    success = check_split("gpt4o long overlapping case classes", letters + "ABC", expressions,
                          { letters, "ABC" }) && success;
    return success;
}

int main() {
    bool success = true;
    const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> normalization = {
        {{}, {}},
        {{0x65, 0x301}, {0xE9}},
        {{0xE9}, {0xE9}},
        {{0x212B}, {0xC5}},
        {{0x61, 0x315, 0x300}, {0xE0, 0x315}},
        {{0x61, 0x305, 0x301}, {0x61, 0x305, 0x301}},
        {{0x301, 0x327}, {0x327, 0x301}},
        {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
        {{0xAC01}, {0xAC01}},
        {{0xFB01, 0xFF21}, {0xFB01, 0xFF21}},
        {{0x958}, {0x915, 0x93C}},
        {{0x1D15E}, {0x1D157, 0x1D165}},
    };
    for (const auto & test : normalization) {
        if (unicode_cpts_normalize_nfc(test.first) != test.second || unicode_cpts_normalize_nfc(test.second) != test.second) {
            fprintf(stderr, "unexpected NFC normalization\n");
            success = false;
            break;
        }
    }

    const struct {
        std::string text;
        std::vector<uint32_t> expected;
    } utf8_tests[] = {
        { "\x7f",                 { 0x7f } },
        { "\xc2\x80",             { 0x80 } },
        { "\xdf\xbf",             { 0x7ff } },
        { "\xe0\xa0\x80",         { 0x800 } },
        { "\xef\xbf\xbf",         { 0xffff } },
        { "\xf0\x90\x80\x80",     { 0x10000 } },
        { "\xf4\x8f\xbf\xbf",     { 0x10ffff } },
        // Invalid-input cases preserve the existing recovery behavior: one U+FFFD per byte.
        { "\xc0\xa0",             { 0xfffd, 0xfffd } },
        { "\xc1\xbf",             { 0xfffd, 0xfffd } },
        { "\xe0\x80\xa0",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xe0\x9f\xbf",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x80\x80\xa0",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x8f\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf4\x90\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf5\x80\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf7\xbf\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xe2\x80",             { 0xfffd, 0xfffd } },
    };

    for (const auto & test : utf8_tests) {
        auto expected = test.expected;
        expected.insert(expected.begin(), 'a');
        expected.push_back('b');
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        fprintf(stderr, "unexpected split:");
        for (const auto & piece : actual) {
            fprintf(stderr, " [%s]", piece.c_str());
        }
        fprintf(stderr, "\n");
        success = false;
    }

    success = test_gpt4o_split() && success;
    return success ? 0 : 1;
}
