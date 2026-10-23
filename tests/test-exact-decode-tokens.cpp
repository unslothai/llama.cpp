// [TAG_EXACT_CONCURRENCY] llama_set_exact_decode_tokens() checks the explicit column bound before any context exists: a figure past
// it is refused and leaves the stored one, which is never lowered, so nothing later is refused because of it.

#include "llama.h"

#include <cstdio>
#include <cstdlib>

static int check(const char * what, bool ok) {
    printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
#ifdef _WIN32
    _putenv_s("GGML_CUDA_BATCH_INVARIANT_MAX_COLS", "8");
#else
    setenv("GGML_CUDA_BATCH_INVARIANT_MAX_COLS", "8", 1);
#endif

    int bad = 0;

    const uint32_t tokens0 = llama_exact_decode_tokens();
    const uint32_t width0  = llama_exact_decode_width();

    bad += check("9 tokens per sequence, bound 8, no context yet: refused", !llama_set_exact_decode_tokens(9));
    bad += check("  the stored figure is unchanged", llama_exact_decode_tokens() == tokens0);
    bad += check("  the reported width is unchanged", llama_exact_decode_width() == width0);
    bad += check("8 tokens per sequence: accepted", llama_set_exact_decode_tokens(8));
    bad += check("  stored as 8, width 8", llama_exact_decode_tokens() == 8 && llama_exact_decode_width() == 8);
    bad += check("9 after 8: refused, 8 kept", !llama_set_exact_decode_tokens(9) && llama_exact_decode_tokens() == 8);
    bad += check("1 after 8: accepted, never lowered", llama_set_exact_decode_tokens(1) && llama_exact_decode_tokens() == 8);
    bad += check("0 is 1: accepted, 8 kept", llama_set_exact_decode_tokens(0) && llama_exact_decode_tokens() == 8);
    bad += check("a width of 9 past the bound: refused", !llama_set_exact_decode_width(9) && llama_exact_decode_width() == 8);
    bad += check("a width of 8: accepted", llama_set_exact_decode_width(8));

    printf("%s\n", bad == 0 ? "ok" : "FAIL");
    return bad == 0 ? 0 : 1;
}
