/* 极简线程池: tp_for 把 [0,n) 切成 grain 大小的块, 动态分给各线程 (含调用线程)。 */
#include "internal.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HANDLE tp_thread;
typedef SRWLOCK tp_mutex;
typedef CONDITION_VARIABLE tp_cond;
#define MUTEX_INIT(m) InitializeSRWLock(m)
#define MUTEX_LOCK(m) AcquireSRWLockExclusive(m)
#define MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(m)
#define COND_INIT(c) InitializeConditionVariable(c)
#define COND_WAIT(c, m) SleepConditionVariableSRW(c, m, INFINITE, 0)
#define COND_BROADCAST(c) WakeAllConditionVariable(c)
#define COND_SIGNAL(c) WakeConditionVariable(c)
#define CPU_RELAX() YieldProcessor()
#else
#include <pthread.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
typedef pthread_t tp_thread;
typedef pthread_mutex_t tp_mutex;
typedef pthread_cond_t tp_cond;
#define MUTEX_INIT(m) pthread_mutex_init(m, NULL)
#define MUTEX_LOCK(m) pthread_mutex_lock(m)
#define MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
#define COND_INIT(c) pthread_cond_init(c, NULL)
#define COND_WAIT(c, m) pthread_cond_wait(c, m)
#define COND_BROADCAST(c) pthread_cond_broadcast(c)
#define COND_SIGNAL(c) pthread_cond_signal(c)
#if defined(__x86_64__) || defined(__i386__)
#define CPU_RELAX() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define CPU_RELAX() __asm__ __volatile__("yield")
#else
#define CPU_RELAX() ((void)0)
#endif
#endif

/* 物理核计数: 每个超线程对只计 1。不臆测「逻辑核/2」。 */
int hy_physical_cpu_count(void) {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len == 0) return 1;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    if (!buf) return 1;
    int cores = 0;
    if (GetLogicalProcessorInformation(buf, &len)) {
        DWORD n = len / sizeof(*buf);
        for (DWORD i = 0; i < n; i++)
            if (buf[i].Relationship == RelationProcessorCore) cores++;
    }
    free(buf);
    return cores > 0 ? cores : 1;
#elif defined(__APPLE__)
    int phys = 0;
    size_t sz = sizeof(phys);
    if (sysctlbyname("hw.physicalcpu", &phys, &sz, NULL, 0) == 0 && phys > 0) return phys;
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#else
    /* Linux: 每个 thread_siblings 组取最小 CPU id，去重后即物理核数。 */
    int seen[512];
    int n_seen = 0;
    for (int cpu = 0; cpu < 512; cpu++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE *f = fopen(path, "r");
        if (!f) {
            if (cpu == 0) break;
            continue;
        }
        char line[128] = "";
        if (!fgets(line, sizeof line, f)) { fclose(f); continue; }
        fclose(f);
        /* 解析列表首个数字作为该物理核代表 id */
        int first = -1;
        char *p = line;
        while (*p) {
            if (*p >= '0' && *p <= '9') {
                int v = (int)strtol(p, &p, 10);
                if (first < 0 || v < first) first = v;
                if (*p == '-') { /* 范围 a-b: 取 a */
                    strtol(p + 1, &p, 10);
                }
            } else {
                p++;
            }
        }
        if (first < 0) continue;
        int dup = 0;
        for (int i = 0; i < n_seen; i++) if (seen[i] == first) { dup = 1; break; }
        if (!dup && n_seen < (int)(sizeof seen / sizeof seen[0])) seen[n_seen++] = first;
    }
    if (n_seen > 0) return n_seen;
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

#define SPIN_ITERS 4000

struct threadpool {
    int n_workers;
    tp_thread *threads;
    tp_mutex mu;
    tp_cond cv_work, cv_done;
    atomic_int gen;
    atomic_int stop;
    atomic_int pending;
    atomic_int next;
    int total, grain;
    tp_fn fn;
    void *ctx;
};

static void run_chunks(struct threadpool *tp) {
    for (;;) {
        int b = atomic_fetch_add(&tp->next, tp->grain);
        if (b >= tp->total) break;
        int e = b + tp->grain;
        if (e > tp->total) e = tp->total;
        tp->fn(tp->ctx, b, e);
    }
}

#ifdef _WIN32
static DWORD WINAPI worker_main(LPVOID arg)
#else
static void *worker_main(void *arg)
#endif
{
    struct threadpool *tp = (struct threadpool *)arg;
    int my_gen = 0;
    for (;;) {
        for (int i = 0; i < SPIN_ITERS; i++) {
            if (atomic_load(&tp->gen) != my_gen || atomic_load(&tp->stop)) break;
            CPU_RELAX();
        }
        MUTEX_LOCK(&tp->mu);
        while (atomic_load(&tp->gen) == my_gen && !atomic_load(&tp->stop)) COND_WAIT(&tp->cv_work, &tp->mu);
        MUTEX_UNLOCK(&tp->mu);
        if (atomic_load(&tp->stop)) break;
        my_gen = atomic_load(&tp->gen);
        run_chunks(tp);
        if (atomic_fetch_sub(&tp->pending, 1) == 1) {
            MUTEX_LOCK(&tp->mu);
            COND_SIGNAL(&tp->cv_done);
            MUTEX_UNLOCK(&tp->mu);
        }
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

threadpool *tp_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    struct threadpool *tp = (struct threadpool *)calloc(1, sizeof(*tp));
    if (!tp) return NULL;
    MUTEX_INIT(&tp->mu);
    COND_INIT(&tp->cv_work);
    COND_INIT(&tp->cv_done);
    atomic_init(&tp->gen, 0);
    atomic_init(&tp->stop, 0);
    atomic_init(&tp->pending, 0);
    atomic_init(&tp->next, 0);
    tp->n_workers = n_threads - 1;
    if (tp->n_workers > 0) {
        tp->threads = (tp_thread *)calloc((size_t)tp->n_workers, sizeof(tp_thread));
        int ok = 0;
        for (int i = 0; i < tp->n_workers; i++) {
#ifdef _WIN32
            tp->threads[i] = CreateThread(NULL, 0, worker_main, tp, 0, NULL);
            if (tp->threads[i]) ok++;
#else
            if (pthread_create(&tp->threads[i], NULL, worker_main, tp) == 0) ok++;
            else break;
#endif
        }
        tp->n_workers = ok;
    }
    return tp;
}

void tp_destroy(threadpool *tp) {
    if (!tp) return;
    atomic_store(&tp->stop, 1);
    MUTEX_LOCK(&tp->mu);
    COND_BROADCAST(&tp->cv_work);
    MUTEX_UNLOCK(&tp->mu);
    for (int i = 0; i < tp->n_workers; i++) {
#ifdef _WIN32
        WaitForSingleObject(tp->threads[i], INFINITE);
        CloseHandle(tp->threads[i]);
#else
        pthread_join(tp->threads[i], NULL);
#endif
    }
    free(tp->threads);
    free(tp);
}

int tp_threads(const threadpool *tp) { return tp ? tp->n_workers + 1 : 1; }

void tp_for(threadpool *tp, int n, int grain, tp_fn fn, void *ctx) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    if (!tp || tp->n_workers == 0 || n <= grain) {
        fn(ctx, 0, n);
        return;
    }
    tp->fn = fn;
    tp->ctx = ctx;
    tp->total = n;
    tp->grain = grain;
    atomic_store(&tp->next, 0);
    atomic_store(&tp->pending, tp->n_workers);
    atomic_fetch_add(&tp->gen, 1);
    MUTEX_LOCK(&tp->mu);
    COND_BROADCAST(&tp->cv_work);
    MUTEX_UNLOCK(&tp->mu);

    run_chunks(tp);

    for (int i = 0; i < SPIN_ITERS && atomic_load(&tp->pending) > 0; i++) CPU_RELAX();
    if (atomic_load(&tp->pending) > 0) {
        MUTEX_LOCK(&tp->mu);
        while (atomic_load(&tp->pending) > 0) COND_WAIT(&tp->cv_done, &tp->mu);
        MUTEX_UNLOCK(&tp->mu);
    }
}
