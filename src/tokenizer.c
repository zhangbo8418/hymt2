/* 字节级 BPE 分词器 (移植自 HyMT2Sharp 的 BpeTokenizer.cs)。
 * 预切分正则用手写匹配器实现, 语义对应 .NET 正则的各个分支 (见 pretok_hunyuan 注释)。
 * 与 C# 的差异: 这里按 Unicode 码点处理; .NET 按 UTF-16 码元, 因此 C# 会静默丢掉辅助平面字符 (如 emoji)。 */
#include "internal.h"
#include "unicode_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ Unicode 辅助 */

static int uc_class(uint32_t cp) {
    size_t lo = 0, hi = UC_NRANGES;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < UC_RANGES[mid].lo) hi = mid;
        else if (cp > UC_RANGES[mid].hi) lo = mid + 1;
        else return UC_RANGES[mid].cls;
    }
    return 0;
}
static inline bool is_L(uint32_t c) { return uc_class(c) & UC_L; }
static inline bool is_N(uint32_t c) { return uc_class(c) & UC_N; }
static inline bool is_LM(uint32_t c) { return uc_class(c) & (UC_L | UC_M); }
static inline bool is_PS(uint32_t c) { return uc_class(c) & (UC_P | UC_S); }
static inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
/* .NET \s = [\f\n\r\t\v\x85\p{Z}] */
static bool is_space(uint32_t c) {
    if ((c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680) return true;
    if (c >= 0x2000 && c <= 0x200A) return true;
    return c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static int utf8_decode(const uint8_t *s, size_t n, uint32_t *cp) {
    uint8_t c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
    if (len == 0 || (size_t)len > n) { *cp = 0xFFFD; return 1; }
    uint32_t v = c & (0xFF >> (len + 1));
    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (s[i] & 0x3F);
    }
    *cp = v;
    return len;
}
static int utf8_encode(uint32_t cp, uint8_t *o) {
    if (cp < 0x80) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) { o[0] = (uint8_t)(0xC0 | (cp >> 6)); o[1] = (uint8_t)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (uint8_t)(0xE0 | (cp >> 12)); o[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (uint8_t)(0xF0 | (cp >> 18)); o[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (uint8_t)(0x80 | (cp & 0x3F));
    return 4;
}

/* ------------------------------------------------------------------ 哈希表 (key = 字节串, 指向 mmap) */

typedef struct { const char *k; uint32_t len; int32_t val; } hent;
typedef struct { hent *e; size_t cap; } htab;

static uint64_t fnv(const char *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)p[i]; h *= 1099511628211ull; }
    return h;
}
static bool ht_init(htab *t, size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    t->e = (hent *)calloc(cap, sizeof(hent));
    t->cap = cap;
    return t->e != NULL;
}
static void ht_put(htab *t, const char *k, size_t len, int32_t val) { /* 重复键保留第一个 */
    size_t i = fnv(k, len) & (t->cap - 1);
    while (t->e[i].k) {
        if (t->e[i].len == len && memcmp(t->e[i].k, k, len) == 0) return;
        i = (i + 1) & (t->cap - 1);
    }
    t->e[i].k = k; t->e[i].len = (uint32_t)len; t->e[i].val = val;
}
static int32_t ht_get(const htab *t, const char *k, size_t len) {
    size_t i = fnv(k, len) & (t->cap - 1);
    while (t->e[i].k) {
        if (t->e[i].len == len && memcmp(t->e[i].k, k, len) == 0) return t->e[i].val;
        i = (i + 1) & (t->cap - 1);
    }
    return -1;
}

/* ------------------------------------------------------------------ 分词器 */

struct tokenizer {
    gstr *tokens;
    size_t n_tokens;
    bool *is_hy;        /* 包含 "｜hy_" 的 token: 解码时跳过 */
    htab vocab, merges;
    int bos, eos, unk, eog;
    int *specials;      /* 按长度降序的特殊 token id */
    size_t n_specials;
    bool hunyuan_pre;
};

static bool contains(const char *p, size_t n, const char *needle) {
    size_t m = strlen(needle);
    if (m > n) return false;
    for (size_t i = 0; i + m <= n; i++)
        if (memcmp(p + i, needle, m) == 0) return true;
    return false;
}

static struct tokenizer *g_sort_tok; /* qsort 比较函数需要 */
static int cmp_special(const void *a, const void *b) {
    size_t la = g_sort_tok->tokens[*(const int *)a].len, lb = g_sort_tok->tokens[*(const int *)b].len;
    return la < lb ? 1 : la > lb ? -1 : 0;
}

tokenizer *tok_create(const gguf *g, char *err, size_t errcap) {
#define FAIL(...) do { if (err) snprintf(err, errcap, __VA_ARGS__); tok_destroy(t); return NULL; } while (0)
    tokenizer *t = (tokenizer *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    const gguf_kv *tk = gguf_find(g, "tokenizer.ggml.tokens");
    if (!tk || tk->type != 9 || tk->arr_type != 8 || tk->arr_len == 0) FAIL("GGUF missing tokenizer.ggml.tokens");
    t->n_tokens = (size_t)tk->arr_len;
    t->tokens = (gstr *)calloc(t->n_tokens, sizeof(gstr));
    t->is_hy = (bool *)calloc(t->n_tokens, sizeof(bool));
    if (!t->tokens || !t->is_hy || !gguf_array_strings(tk, t->tokens, t->n_tokens)) FAIL("bad token array");
    if (!ht_init(&t->vocab, t->n_tokens)) FAIL("out of memory");
    for (size_t i = 0; i < t->n_tokens; i++) {
        ht_put(&t->vocab, t->tokens[i].p, (size_t)t->tokens[i].len, (int32_t)i);
        t->is_hy[i] = contains(t->tokens[i].p, (size_t)t->tokens[i].len, "\xEF\xBD\x9Chy_"); /* "｜hy_" */
    }
    const gguf_kv *mg = gguf_find(g, "tokenizer.ggml.merges");
    if (mg && mg->type == 9 && mg->arr_type == 8 && mg->arr_len > 0) {
        size_t n = (size_t)mg->arr_len;
        gstr *ms = (gstr *)calloc(n, sizeof(gstr));
        if (!ms || !gguf_array_strings(mg, ms, n) || !ht_init(&t->merges, n)) { free(ms); FAIL("bad merges"); }
        for (size_t i = 0; i < n; i++) ht_put(&t->merges, ms[i].p, (size_t)ms[i].len, (int32_t)i);
        free(ms);
    } else if (!ht_init(&t->merges, 1)) FAIL("out of memory");

    t->bos = (int)gguf_get_int(g, "tokenizer.ggml.bos_token_id", 0);
    t->eos = (int)gguf_get_int(g, "tokenizer.ggml.eos_token_id", 0);
    t->unk = (int)gguf_get_int(g, "tokenizer.ggml.unknown_token_id", 0);
    const char *eog_name = "<\xEF\xBD\x9Chy_place\xE2\x96\x81holder\xE2\x96\x81no\xE2\x96\x81" "2\xEF\xBD\x9C>"; /* <｜hy_place▁holder▁no▁2｜> */
    int eog = ht_get(&t->vocab, eog_name, strlen(eog_name));
    t->eog = eog >= 0 ? eog : t->eos;

    const gguf_kv *tt = gguf_find(g, "tokenizer.ggml.token_type");
    t->specials = (int *)malloc(sizeof(int) * t->n_tokens);
    for (size_t i = 0; i < t->n_tokens; i++) {
        bool typed = false;
        if (tt && tt->type == 9 && i < tt->arr_len && tt->arr_type == 5) {
            int32_t ty;
            memcpy(&ty, tt->arr_data + i * 4, 4);
            typed = (ty == 3 || ty == 4);
        }
        if (typed || t->is_hy[i] || (int)i == t->bos || (int)i == t->eos) t->specials[t->n_specials++] = (int)i;
    }
    g_sort_tok = t;
    qsort(t->specials, t->n_specials, sizeof(int), cmp_special);

    gstr pre = { "gpt-2", 5 };
    gguf_get_str(g, "tokenizer.ggml.pre", &pre);
    t->hunyuan_pre = (pre.len == 13 && !memcmp(pre.p, "hunyuan-dense", 13)) || (pre.len == 11 && !memcmp(pre.p, "deepseek-v3", 11)) ||
                     (pre.len == 9 && !memcmp(pre.p, "joyai-llm", 9)) || (pre.len == 5 && !memcmp(pre.p, "hy_v4", 5));
    return t;
#undef FAIL
}

void tok_destroy(tokenizer *t) {
    if (!t) return;
    free(t->tokens); free(t->is_hy); free(t->vocab.e); free(t->merges.e); free(t->specials);
    free(t);
}
int tok_vocab_size(const tokenizer *t) { return (int)t->n_tokens; }
bool tok_is_stop(const tokenizer *t, int id) { return id == t->eos || id == t->eog; }

/* ------------------------------------------------------------------ 预切分 */

/* 返回从 pos 开始的一个匹配的码点个数; 0 表示没有任何分支匹配 (正则引擎会跳过该字符) */

/* .NET 里的 \p{N}{1,3}|[一-龥぀-ゟ゠-ヿ]+|[ASCII 标点][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|
 *        ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+  (依次尝试, 先匹配到的分支胜出) */
static bool is_cjk(uint32_t c) { return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF); }
static bool is_ascii_punct(uint32_t c) {
    return c < 128 && c != 0 && strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", (int)c) != NULL;
}
static bool is_ascii_alpha(uint32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

static int pretok_hunyuan(const uint32_t *c, int n, int pos) {
    int i;
    /* 1: \p{N}{1,3} */
    if (is_N(c[pos])) {
        for (i = pos; i < n && i < pos + 3 && is_N(c[i]); i++) {}
        return i - pos;
    }
    /* 2: CJK 连续段 */
    if (is_cjk(c[pos])) {
        for (i = pos; i < n && is_cjk(c[i]); i++) {}
        return i - pos;
    }
    /* 3: ASCII 标点 + 英文字母串 */
    if (is_ascii_punct(c[pos]) && pos + 1 < n && is_ascii_alpha(c[pos + 1])) {
        for (i = pos + 1; i < n && is_ascii_alpha(c[i]); i++) {}
        return i - pos;
    }
    /* 4: 可选的一个"非 换行/字母/标点/符号"字符 + 字母或组合符号串 (可选字符不成立时回溯) */
    {
        bool opt_ok = !is_nl(c[pos]) && !is_L(c[pos]) && !is_PS(c[pos]);
        if (opt_ok && pos + 1 < n && is_LM(c[pos + 1])) {
            for (i = pos + 1; i < n && is_LM(c[i]); i++) {}
            return i - pos;
        }
        if (is_LM(c[pos])) {
            for (i = pos; i < n && is_LM(c[i]); i++) {}
            return i - pos;
        }
    }
    /* 5: 可选空格 + 标点/符号串 + 换行串 */
    {
        int j = pos;
        if (c[j] == ' ') j++;
        if (j < n && is_PS(c[j])) {
            while (j < n && is_PS(c[j])) j++;
            while (j < n && is_nl(c[j])) j++;
            return j - pos;
        }
    }
    /* 6/7/8: 空白相关 */
    if (is_space(c[pos])) {
        int w = pos;
        while (w < n && is_space(c[w])) w++;            /* 空白串 [pos, w) */
        /* 6: \s*[\r\n]+ : \s* 贪婪后回溯, 找最大的 L 使 c[pos+L] 是换行 */
        for (int L = w - pos - 1; L >= 0; L--) {
            if (is_nl(c[pos + L])) {
                int e = pos + L;
                while (e < n && is_nl(c[e])) e++;
                return e - pos;
            }
        }
        /* 7: \s+(?!\S) */
        if (w == n) return w - pos;
        if (w - pos - 1 >= 1) return w - pos - 1;
        /* 8: \s+ */
        return w - pos;
    }
    return 0;
}

/* GPT-2 风格后备: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ */
static int pretok_gpt2(const uint32_t *c, int n, int pos) {
    static const char *contr[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    for (size_t k = 0; k < sizeof(contr) / sizeof(contr[0]); k++) {
        size_t m = strlen(contr[k]);
        if (pos + (int)m <= n) {
            bool ok = true;
            for (size_t j = 0; j < m; j++) if (c[pos + j] != (uint32_t)contr[k][j]) { ok = false; break; }
            if (ok) return (int)m;
        }
    }
    int j = pos;
    if (c[j] == ' ' && j + 1 < n) j++;
    if (is_L(c[j])) { while (j < n && is_L(c[j])) j++; return j - pos; }
    if (is_N(c[j])) { while (j < n && is_N(c[j])) j++; return j - pos; }
    if (!is_space(c[j]) && !is_L(c[j]) && !is_N(c[j])) {
        while (j < n && !is_space(c[j]) && !is_L(c[j]) && !is_N(c[j])) j++;
        return j - pos;
    }
    if (is_space(c[pos])) {
        int w = pos;
        while (w < n && is_space(c[w])) w++;
        if (w == n) return w - pos;
        if (w - pos - 1 >= 1) return w - pos - 1;
        return w - pos;
    }
    return 0;
}

/* ------------------------------------------------------------------ BPE */

static uint32_t byte_to_cp(uint8_t b) {
    if (b == 0xAD) return 0x0143;
    if (b <= 0x20) return (uint32_t)b + 0x0100;
    if (b >= 0x7F && b <= 0xA0) return (uint32_t)b + 0x00A2;
    return b;
}

typedef struct { int *v; size_t n, cap; } ivec;
static void iv_push(ivec *a, int x) {
    if (a->n == a->cap) { a->cap = a->cap ? a->cap * 2 : 64; a->v = (int *)realloc(a->v, a->cap * sizeof(int)); }
    a->v[a->n++] = x;
}

/* 对一个预切分片段 (原始 UTF-8 字节) 做字节编码 + BPE */
static void bpe_piece(const tokenizer *t, const uint8_t *src, size_t n, ivec *out) {
    if (n == 0) return;
    uint8_t *buf = (uint8_t *)malloc(n * 2 + 2);
    int *st = (int *)malloc(sizeof(int) * (n + 1)), *ln = (int *)malloc(sizeof(int) * (n + 1));
    size_t blen = 0;
    int ns = 0;
    for (size_t i = 0; i < n; i++) {
        int l = utf8_encode(byte_to_cp(src[i]), buf + blen);
        st[ns] = (int)blen; ln[ns] = l; ns++;
        blen += (size_t)l;
    }
    int32_t direct = ht_get(&t->vocab, (const char *)buf, blen);
    if (direct >= 0) { iv_push(out, direct); goto done; }

    uint8_t key[512];
    while (ns > 1) {
        int32_t best = INT32_MAX;
        int bi = -1;
        for (int i = 0; i + 1 < ns; i++) {
            size_t a = (size_t)ln[i], b = (size_t)ln[i + 1];
            if (a + b + 1 > sizeof(key)) continue;
            memcpy(key, buf + st[i], a);
            key[a] = ' ';
            memcpy(key + a + 1, buf + st[i + 1], b);
            int32_t rank = ht_get(&t->merges, (const char *)key, a + 1 + b);
            if (rank >= 0 && rank < best) { best = rank; bi = i; }
        }
        if (bi < 0) break;
        ln[bi] += ln[bi + 1];  /* 两个符号在 buf 中相邻, 直接合并长度 */
        for (int i = bi + 1; i + 1 < ns; i++) { st[i] = st[i + 1]; ln[i] = ln[i + 1]; }
        ns--;
    }
    for (int i = 0; i < ns; i++) {
        int32_t id = ht_get(&t->vocab, (const char *)buf + st[i], (size_t)ln[i]);
        iv_push(out, id >= 0 ? id : t->unk);
    }
done:
    free(buf); free(st); free(ln);
}

static void encode_plain(const tokenizer *t, const uint8_t *s, size_t n, ivec *out) {
    if (n == 0) return;
    uint32_t *cp = (uint32_t *)malloc(sizeof(uint32_t) * n);
    size_t *off = (size_t *)malloc(sizeof(size_t) * (n + 1));
    int nc = 0;
    for (size_t i = 0; i < n;) {
        int l = utf8_decode(s + i, n - i, &cp[nc]);
        off[nc++] = i;
        i += (size_t)l;
    }
    off[nc] = n;
    int pos = 0;
    while (pos < nc) {
        int m = t->hunyuan_pre ? pretok_hunyuan(cp, nc, pos) : pretok_gpt2(cp, nc, pos);
        if (m <= 0) { pos++; continue; } /* 没有分支匹配: 与正则引擎一样跳过该字符 */
        bpe_piece(t, s + off[pos], off[pos + m] - off[pos], out);
        pos += m;
    }
    free(cp); free(off);
}

typedef struct { const uint8_t *p; size_t len; int id; } frag;

int tok_encode(const tokenizer *t, const char *text, size_t len, int **ids) {
    size_t cap = 8, nf = 1;
    frag *fr = (frag *)malloc(cap * sizeof(frag));
    fr[0] = (frag){ (const uint8_t *)text, len, -1 };
    for (size_t si = 0; si < t->n_specials; si++) {
        const gstr *sp = &t->tokens[t->specials[si]];
        if (sp->len == 0) continue;
        frag *next = (frag *)malloc((nf * 2 + 8) * sizeof(frag));
        size_t nn = 0, ncap = nf * 2 + 8;
        for (size_t f = 0; f < nf; f++) {
            if (fr[f].id >= 0) { if (nn == ncap) { ncap *= 2; next = (frag *)realloc(next, ncap * sizeof(frag)); } next[nn++] = fr[f]; continue; }
            const uint8_t *p = fr[f].p;
            size_t left = fr[f].len, start = 0;
            for (;;) {
                size_t idx = (size_t)-1;
                for (size_t i = start; i + sp->len <= left; i++)
                    if (p[i] == (uint8_t)sp->p[0] && memcmp(p + i, sp->p, (size_t)sp->len) == 0) { idx = i; break; }
                while (nn + 2 >= ncap) { ncap *= 2; next = (frag *)realloc(next, ncap * sizeof(frag)); }
                if (idx == (size_t)-1) {
                    if (start < left) next[nn++] = (frag){ p + start, left - start, -1 };
                    break;
                }
                if (idx > start) next[nn++] = (frag){ p + start, idx - start, -1 };
                next[nn++] = (frag){ NULL, 0, t->specials[si] };
                start = idx + (size_t)sp->len;
            }
        }
        free(fr);
        fr = next; nf = nn; cap = ncap;
    }
    ivec out = { NULL, 0, 0 };
    for (size_t f = 0; f < nf; f++) {
        if (fr[f].id >= 0) iv_push(&out, fr[f].id);
        else encode_plain(t, fr[f].p, fr[f].len, &out);
    }
    free(fr);
    *ids = out.v;
    return (int)out.n;
}

/* ------------------------------------------------------------------ 解码 */

size_t tok_decode_visible(const tokenizer *t, const int *ids, int n, char *out, size_t cap) {
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || (size_t)id >= t->n_tokens || t->is_hy[id]) continue;
        const uint8_t *p = (const uint8_t *)t->tokens[id].p;
        size_t len = (size_t)t->tokens[id].len;
        for (size_t j = 0; j < len;) {
            uint32_t r;
            j += (size_t)utf8_decode(p + j, len - j, &r);
            uint8_t tmp[4];
            int k = 0;
            if (r == 0x0100) continue;
            if (r > 0x0100 && r <= 0x0120) { tmp[0] = (uint8_t)(r - 0x0100); k = 1; }
            else if (r >= 0x0121 && r <= 0x0142) { tmp[0] = (uint8_t)(r - 0x00A2); k = 1; }
            else if (r == 0x0143) { tmp[0] = 0xAD; k = 1; }
            else if (r == 0x2581) { tmp[0] = 0x20; k = 1; }
            else if (r < 0x100) { tmp[0] = (uint8_t)r; k = 1; }
            else k = utf8_encode(r, tmp);
            if (w + (size_t)k > cap) return w;
            memcpy(out + w, tmp, (size_t)k);
            w += (size_t)k;
        }
    }
    return w;
}
