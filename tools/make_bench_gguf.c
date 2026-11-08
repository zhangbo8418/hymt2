/* 生成一个"真实尺寸"的随机权重 GGUF (Hy-MT2-1.8B 的形状, STQ1_0 层 + Q6_K 词嵌入), 只用于测速, 输出无意义。 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#pragma pack(push, 1)
typedef struct { uint8_t qs[32], sign[8]; uint16_t d; } stq_t;
typedef struct { uint8_t ql[128], qh[64]; int8_t sc[16]; uint16_t d; } q6_t;
#pragma pack(pop)
static FILE *F;
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static void w8(uint64_t v) { fwrite(&v, 8, 1, F); }
static void w4(uint32_t v) { fwrite(&v, 4, 1, F); }
static void wstr(const char *s) { w8(strlen(s)); fwrite(s, 1, strlen(s), F); }
static void kv_u32(const char *k, uint32_t v) { wstr(k); w4(4); w4(v); }
static void kv_f32(const char *k, float v) { wstr(k); w4(6); fwrite(&v, 4, 1, F); }
static void kv_str(const char *k, const char *v) { wstr(k); w4(8); wstr(v); }
typedef struct { char name[64]; uint32_t type; uint64_t k, n, off, bytes; } T;
static T ts[2048]; static int nt;
static uint64_t cur;
static void add(const char *name, uint32_t type, uint64_t k, uint64_t n, uint64_t bytes) {
    snprintf(ts[nt].name, 64, "%s", name); ts[nt].type = type; ts[nt].k = k; ts[nt].n = n; ts[nt].bytes = bytes; ts[nt].off = cur;
    cur += (bytes + 31) / 32 * 32; nt++;
}
int main(int argc, char **argv) {
    const int HID = 2048, NL = 32, HEADS = 16, KV = 4, HD = 128, FFN = 6144, V = 120818;
    int layers = argc > 2 ? atoi(argv[2]) : NL;
    char nm[64];
    add("token_embd.weight", 14, HID, V, (uint64_t)V * (HID / 256) * sizeof(q6_t));
    add("output_norm.weight", 0, HID, 0, HID * 4);
    for (int i = 0; i < layers; i++) {
#define S(s, k, n) do { snprintf(nm, 64, "blk.%d." s, i); add(nm, 43, k, n, (uint64_t)(n) * ((k) / 256) * sizeof(stq_t)); } while (0)
#define N(s, n) do { snprintf(nm, 64, "blk.%d." s, i); add(nm, 0, n, 0, (n) * 4); } while (0)
        N("attn_norm.weight", HID); N("ffn_norm.weight", HID); N("attn_q_norm.weight", HD); N("attn_k_norm.weight", HD);
        S("attn_q.weight", HID, HEADS * HD); S("attn_k.weight", HID, KV * HD); S("attn_v.weight", HID, KV * HD);
        S("attn_output.weight", HEADS * HD, HID); S("ffn_gate.weight", HID, FFN); S("ffn_up.weight", HID, FFN); S("ffn_down.weight", FFN, HID);
    }
    F = fopen(argv[1], "wb");
    fwrite("GGUF", 1, 4, F); w4(3); w8(nt); w8(19);
    kv_str("general.architecture", "hunyuan-dense"); kv_str("general.name", "bench");
    kv_u32("hunyuan-dense.embedding_length", HID); kv_u32("hunyuan-dense.block_count", layers);
    kv_u32("hunyuan-dense.feed_forward_length", FFN); kv_u32("hunyuan-dense.attention.head_count", HEADS);
    kv_u32("hunyuan-dense.attention.head_count_kv", KV); kv_u32("hunyuan-dense.attention.key_length", HD);
    kv_u32("hunyuan-dense.rope.dimension_count", HD); kv_f32("hunyuan-dense.rope.freq_base", 10000.f);
    kv_f32("hunyuan-dense.rope.scaling.alpha", 1000.f); kv_f32("hunyuan-dense.attention.layer_norm_rms_epsilon", 1e-5f);
    kv_str("tokenizer.ggml.pre", "hunyuan-dense");
    kv_u32("tokenizer.ggml.bos_token_id", 1); kv_u32("tokenizer.ggml.eos_token_id", 2); kv_u32("tokenizer.ggml.unknown_token_id", 0);
    kv_str("tokenizer.ggml.model", "gpt2"); kv_str("general.file_type", "x"); /* 占位, 不影响 */
    wstr("tokenizer.ggml.tokens"); w4(9); w4(8); w8(V);
    for (int i = 0; i < V; i++) { char b[24]; snprintf(b, 24, "t%d", i); wstr(b); }
    for (int i = 0; i < nt; i++) {
        wstr(ts[i].name);
        int nd = ts[i].n ? 2 : 1; w4(nd); w8(ts[i].k); if (nd == 2) w8(ts[i].n);
        w4(ts[i].type); w8(ts[i].off);
    }
    long pos = ftell(F); while (pos % 32) { fputc(0, F); pos++; }
    for (int i = 0; i < nt; i++) {
        uint64_t written = 0;
        if (ts[i].type == 0) { float one = 1.0f; for (uint64_t j = 0; j < ts[i].bytes / 4; j++) fwrite(&one, 4, 1, F); written = ts[i].bytes; }
        else if (ts[i].type == 43) {
            uint64_t nb = ts[i].bytes / sizeof(stq_t);
            for (uint64_t b = 0; b < nb; b++) { stq_t s; for (int j = 0; j < 32; j++) s.qs[j] = (uint8_t)rnd(); for (int j = 0; j < 8; j++) s.sign[j] = (uint8_t)rnd(); s.d = 0x2000 + (uint16_t)(rnd() % 256); /* ≈0.008~0.016: 让随机网络的激活保持在正常范围 */ fwrite(&s, sizeof s, 1, F); }
            written = ts[i].bytes;
        } else {
            uint64_t nb = ts[i].bytes / sizeof(q6_t);
            for (uint64_t b = 0; b < nb; b++) { q6_t s; for (int j = 0; j < 128; j++) s.ql[j] = (uint8_t)rnd(); for (int j = 0; j < 64; j++) s.qh[j] = (uint8_t)rnd(); for (int j = 0; j < 16; j++) s.sc[j] = (int8_t)((int)(rnd() % 41) - 20); s.d = 0x1400 + (uint16_t)(rnd() % 256); fwrite(&s, sizeof s, 1, F); }
            written = ts[i].bytes;
        }
        for (uint64_t p = written; p % 32; p++) fputc(0, F);
    }
    fclose(F);
    printf("wrote %s, %d tensors, %.1f MB\n", argv[1], nt, (double)(cur) / 1e6);
    return 0;
}
