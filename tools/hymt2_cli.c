/* 命令行翻译 / 测速:
 *   hymt2_cli <model.gguf> <目标语言(英文名)> "<文本>" | @文件.txt  [threads=0(自动)] [max_tokens=128]
 *
 * 文本以 @ 开头表示从 UTF-8 文件读取 (Windows 命令行传中文容易乱码, 建议用这种方式)。
 * 环境变量: HYMT2_OUT=结果文件 (把译文以 UTF-8 写入该文件); HYMT2_CLI_RUNS=2 (运行次数, 第 2 次起命中前缀缓存)
 *           以及库本身的 HYMT2_ISA / HYMT2_REPACK / HYMT2_DEVICE (兼容 HYMT2_VK) / HYMT2_THREADS / HYMT2_BATCH / HYMT2_VERBOSE。
 * 最后一行 "RESULT ..." 是机器可读的汇总, 测试时把它贴回来即可。 */
#include "../include/hymt2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
static double ms(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart * 1000.0 / (double)f.QuadPart; }
#else
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#endif

/* 输出 UTF-8 文本。Windows 控制台用 WriteConsoleW (msvcrt 的 printf 在 chcp 65001 下对多字节字符会乱码); 重定向到文件时原样输出字节 */
static void print_utf8(const char *s) {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(h, &mode)) {
        int wn = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
        wchar_t *w = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)wn);
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w, wn);
        DWORD written;
        WriteConsoleW(h, w, (DWORD)(wn - 1), &written, NULL);
        WriteConsoleW(h, L"\r\n", 2, &written, NULL);
        free(w);
        return;
    }
#endif
    printf("%s\n", s);
    fflush(stdout);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    b[got] = 0;
    if (got >= 3 && (unsigned char)b[0] == 0xEF && (unsigned char)b[1] == 0xBB && (unsigned char)b[2] == 0xBF) memmove(b, b + 3, got - 2); /* 去 BOM */
    while (got > 0 && (b[got - 1] == '\n' || b[got - 1] == '\r')) b[--got] = 0;
    return b;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf TargetLanguage \"text\"|@file.txt [threads] [max_tokens]\n", argv[0]);
        return 2;
    }
    char err[512];
    const int threads = argc > 4 ? atoi(argv[4]) : 0, max_tokens = argc > 5 ? atoi(argv[5]) : 128;
    const char *runs_env = getenv("HYMT2_CLI_RUNS");
    const int runs = runs_env && atoi(runs_env) > 0 ? atoi(runs_env) : 2;
#if defined(_WIN32)
    char *text = argv[3][0] == '@' ? read_file(argv[3] + 1) : _strdup(argv[3]);
#else
    char *text = argv[3][0] == '@' ? read_file(argv[3] + 1) : strdup(argv[3]);
#endif
    if (!text) { fprintf(stderr, "cannot read %s\n", argv[3] + 1); return 2; }

    double t0 = ms();
    hymt2_model *m = hymt2_model_load(argv[1], threads, 1024, err, sizeof err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err); return 1; }
    const double load_ms = ms() - t0;
    char cfg[256];
    hymt2_model_info(m, cfg, sizeof cfg);
    fprintf(stderr, "[%s] loaded in %.0f ms | cpu: %s\n", cfg, load_ms, hymt2_cpu_features());

    size_t cap = strlen(text) + 512;
    char *prompt = (char *)malloc(cap), out[8192];
    snprintf(prompt, cap, "Translate the following text into %s. Note that you should only output the translated result without any additional explanation:\n%s", argv[2], text);
    int pt = 0, ct = 0, dtok = 0;
    double pms = 0, dms = 0, total = 0;
    for (int run = 0; run < runs; run++) { /* 第 2 次起演示前缀缓存 */
        t0 = ms();
        int n = hymt2_model_generate(m, prompt, out, sizeof out - 1, max_tokens, 0);
        total = ms() - t0;
        if (n < 0) { fprintf(stderr, "generate failed (%d): %s\n", n, hymt2_model_last_error(m)); return 1; }
        out[n] = 0;
        print_utf8(out);
        hymt2_model_stats(m, &pt, &ct, &pms, &dtok, &dms);
        fprintf(stderr, "[run %d] %d bytes, total %.0f ms | prefill %d tok (+%d cached) %.0f ms | decode %d tok %.0f ms = %.1f ms/tok (%.1f tok/s)\n",
                run + 1, n, total, pt, ct, pms, dtok, dms, dtok ? dms / dtok : 0.0, dms > 0 ? dtok * 1000.0 / dms : 0.0);
        if (getenv("HYMT2_OUT") && run == runs - 1) {
            FILE *f = fopen(getenv("HYMT2_OUT"), "wb");
            if (f) { fwrite(out, 1, (size_t)n, f); fputc('\n', f); fclose(f); }
        }
    }
    printf("RESULT %s load_ms=%.0f total_ms=%.0f prefill_tok=%d prefill_ms=%.0f decode_tok=%d decode_ms_per_tok=%.1f decode_tok_s=%.2f\n",
           cfg, load_ms, total, pt, pms, dtok, dtok ? dms / dtok : 0.0, dms > 0 ? dtok * 1000.0 / dms : 0.0);
#ifdef HYMT2_PROFILE
    {
        void hymt2_debug_profile_dump(void);
        fprintf(stderr, "[profile]\n");
        hymt2_debug_profile_dump();
    }
#endif
    hymt2_model_free(m);
    return 0;
}
