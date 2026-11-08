/* 加载 GGUF, 对固定文本做一次 prefill, 打印 logits 摘要 (用于跨架构一致性检查, 如 x86 vs aarch64/qemu) */
#include "../include/hymt2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
const float *hymt2_debug_logits(hymt2_model *m, const int *ids, int n, int *vocab_out);
int hymt2_debug_encode(hymt2_model *m, const char *text, int **ids);
int main(int argc, char **argv) {
    char err[512];
    hymt2_model *m = hymt2_model_load(argv[1], argc > 2 ? atoi(argv[2]) : 1, 256, err, sizeof(err));
    if (!m) { fprintf(stderr, "load failed: %s\n", err); return 1; }
    int *ids; int n = hymt2_debug_encode(m, "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into Chinese. Hello world, 今天天气很好。<｜hy_Assistant｜>", &ids);
    int v; const float *lg = hymt2_debug_logits(m, ids, n, &v);
    double sum = 0; int arg = 0;
    for (int i = 0; i < v; i++) { sum += (double)lg[i] * (1 + (i % 7)); if (lg[i] > lg[arg]) arg = i; }
    printf("isa=%s tokens=%d vocab=%d argmax=%d checksum=%.4f first=[%.5f %.5f %.5f %.5f]\n", hymt2_isa(), n, v, arg, sum, lg[0], lg[1], lg[2], lg[3]);
    hymt2_model_free(m);
    return 0;
}
