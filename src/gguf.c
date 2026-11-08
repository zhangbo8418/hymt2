/* GGUF v2/v3 读取 (mmap, 零拷贝)。 */
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct gguf {
    const uint8_t *base;
    uint64_t size;
#ifdef _WIN32
    HANDLE hfile, hmap;
#endif
    gguf_kv *kv;
    size_t n_kv;
    gguf_tensor *tensors;
    size_t n_tensors;
    uint64_t data_off;
};

typedef struct { const uint8_t *p, *end; bool bad; } cur_t;

static bool rd(cur_t *c, void *dst, size_t n) {
    if (c->bad || (size_t)(c->end - c->p) < n) { c->bad = true; return false; }
    memcpy(dst, c->p, n);
    c->p += n;
    return true;
}
static uint32_t rd_u32(cur_t *c) { uint32_t v = 0; rd(c, &v, 4); return v; }
static uint64_t rd_u64(cur_t *c) { uint64_t v = 0; rd(c, &v, 8); return v; }
static gstr rd_str(cur_t *c) {
    gstr s = { NULL, 0 };
    uint64_t n = rd_u64(c);
    if (c->bad || n > (uint64_t)(c->end - c->p)) { c->bad = true; return s; }
    s.p = (const char *)c->p;
    s.len = n;
    c->p += n;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    }
    return 0;
}

static void skip_value(cur_t *c, uint32_t type);
static void read_value(cur_t *c, gguf_kv *kv) {
    kv->type = rd_u32(c);
    switch (kv->type) {
    case 0: { uint8_t v = 0; rd(c, &v, 1); kv->v.u = v; break; }
    case 1: { int8_t v = 0; rd(c, &v, 1); kv->v.i = v; break; }
    case 2: { uint16_t v = 0; rd(c, &v, 2); kv->v.u = v; break; }
    case 3: { int16_t v = 0; rd(c, &v, 2); kv->v.i = v; break; }
    case 4: { uint32_t v = 0; rd(c, &v, 4); kv->v.u = v; break; }
    case 5: { int32_t v = 0; rd(c, &v, 4); kv->v.i = v; break; }
    case 6: { float v = 0; rd(c, &v, 4); kv->v.f = v; break; }
    case 7: { uint8_t v = 0; rd(c, &v, 1); kv->v.u = v; break; }
    case 8: kv->v.s = rd_str(c); break;
    case 10: kv->v.u = rd_u64(c); break;
    case 11: { int64_t v = 0; rd(c, &v, 8); kv->v.i = v; break; }
    case 12: { double v = 0; rd(c, &v, 8); kv->v.f = v; break; }
    case 9: {
        kv->arr_type = rd_u32(c);
        kv->arr_len = rd_u64(c);
        kv->arr_data = c->p;
        for (uint64_t i = 0; i < kv->arr_len && !c->bad; i++) skip_value(c, kv->arr_type);
        break;
    }
    default: c->bad = true;
    }
}
static void skip_value(cur_t *c, uint32_t type) {
    if (type == 8) { (void)rd_str(c); return; }
    size_t n = scalar_size(type);
    if (!n || (size_t)(c->end - c->p) < n) { c->bad = true; return; }
    c->p += n;
}

gguf *gguf_open(const char *path, char *err, size_t errcap) {
#define FAIL(...) do { if (err) snprintf(err, errcap, __VA_ARGS__); gguf_close(g); return NULL; } while (0)
    gguf *g = (gguf *)calloc(1, sizeof(*g));
    if (!g) return NULL;
#ifdef _WIN32
    g->hfile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g->hfile == INVALID_HANDLE_VALUE) { g->hfile = NULL; FAIL("cannot open %s", path); }
    LARGE_INTEGER sz; GetFileSizeEx(g->hfile, &sz);
    g->size = (uint64_t)sz.QuadPart;
    g->hmap = CreateFileMappingA(g->hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!g->hmap) FAIL("CreateFileMapping failed");
    g->base = (const uint8_t *)MapViewOfFile(g->hmap, FILE_MAP_READ, 0, 0, 0);
    if (!g->base) FAIL("MapViewOfFile failed");
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) FAIL("cannot open %s", path);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); FAIL("stat failed"); }
    g->size = (uint64_t)st.st_size;
    void *m = mmap(NULL, (size_t)g->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) FAIL("mmap failed");
    g->base = (const uint8_t *)m;
#endif
    if (g->size < 24 || memcmp(g->base, "GGUF", 4) != 0) FAIL("not a GGUF file");
    cur_t c = { g->base + 4, g->base + g->size, false };
    uint32_t version = rd_u32(&c);
    if (version != 2 && version != 3) FAIL("unsupported GGUF version %u", version);
    uint64_t n_tensors = rd_u64(&c), n_kv = rd_u64(&c);
    if (n_tensors > 1000000 || n_kv > 1000000) FAIL("implausible GGUF header");
    g->kv = (gguf_kv *)calloc((size_t)n_kv + 1, sizeof(gguf_kv));
    g->tensors = (gguf_tensor *)calloc((size_t)n_tensors + 1, sizeof(gguf_tensor));
    if (!g->kv || !g->tensors) FAIL("out of memory");
    for (uint64_t i = 0; i < n_kv; i++) {
        g->kv[i].key = rd_str(&c);
        read_value(&c, &g->kv[i]);
        if (c.bad) FAIL("truncated GGUF metadata");
    }
    g->n_kv = (size_t)n_kv;
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        t->name = rd_str(&c);
        t->n_dims = rd_u32(&c);
        if (t->n_dims > 4) FAIL("bad tensor dims");
        for (uint32_t d = 0; d < t->n_dims; d++) t->dims[d] = rd_u64(&c);
        t->type = rd_u32(&c);
        t->offset = rd_u64(&c);
        if (c.bad) FAIL("truncated GGUF tensor table");
    }
    g->n_tensors = (size_t)n_tensors;
    uint64_t align = (uint64_t)gguf_get_int(g, "general.alignment", 32);
    if (align == 0) align = 32;
    uint64_t pos = (uint64_t)(c.p - g->base);
    g->data_off = (pos + align - 1) / align * align;
    if (g->data_off > g->size) FAIL("data section past end of file");
    return g;
#undef FAIL
}

void gguf_close(gguf *g) {
    if (!g) return;
#ifdef _WIN32
    if (g->base) UnmapViewOfFile(g->base);
    if (g->hmap) CloseHandle(g->hmap);
    if (g->hfile) CloseHandle(g->hfile);
#else
    if (g->base) munmap((void *)g->base, (size_t)g->size);
#endif
    free(g->kv);
    free(g->tensors);
    free(g);
}

const gguf_kv *gguf_find(const gguf *g, const char *key) {
    size_t n = strlen(key);
    for (size_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key.len == n && memcmp(g->kv[i].key.p, key, n) == 0) return &g->kv[i];
    return NULL;
}
int64_t gguf_get_int(const gguf *g, const char *key, int64_t def) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv) return def;
    switch (kv->type) {
    case 0: case 2: case 4: case 7: case 10: return (int64_t)kv->v.u;
    case 1: case 3: case 5: case 11: return kv->v.i;
    }
    return def;
}
double gguf_get_float(const gguf *g, const char *key, double def) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv) return def;
    if (kv->type == 6 || kv->type == 12) return kv->v.f;
    return (double)gguf_get_int(g, key, (int64_t)def);
}
bool gguf_get_str(const gguf *g, const char *key, gstr *out) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type != 8) return false;
    *out = kv->v.s;
    return true;
}
const gguf_tensor *gguf_find_tensor(const gguf *g, const char *name) {
    size_t n = strlen(name);
    for (size_t i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].name.len == n && memcmp(g->tensors[i].name.p, name, n) == 0) return &g->tensors[i];
    return NULL;
}
const uint8_t *gguf_tensor_data(const gguf *g, const gguf_tensor *t) { return g->base + g->data_off + t->offset; }
uint64_t gguf_tensor_bytes_avail(const gguf *g, const gguf_tensor *t) {
    uint64_t start = g->data_off + t->offset;
    return start >= g->size ? 0 : g->size - start;
}
bool gguf_array_strings(const gguf_kv *kv, gstr *out, size_t n) {
    if (!kv || kv->type != 9 || kv->arr_type != 8 || kv->arr_len < n) return false;
    const uint8_t *p = kv->arr_data;
    for (size_t i = 0; i < n; i++) {
        uint64_t len;
        memcpy(&len, p, 8);
        p += 8;
        out[i].p = (const char *)p;
        out[i].len = len;
        p += len;
    }
    return true;
}

/* 预读整个文件: 先提示内核顺序预读, 再每页触碰一次, 把首次推理时的缺页/磁盘读取挪到加载阶段 */
void gguf_prefault(const gguf *g) {
#ifndef _WIN32
    madvise((void *)g->base, (size_t)g->size, MADV_WILLNEED);
#endif
    volatile uint8_t sink = 0;
    for (uint64_t off = 0; off < g->size; off += 4096) sink ^= g->base[off];
    (void)sink;
}
