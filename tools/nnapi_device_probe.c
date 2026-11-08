/* Android 真机 NNAPI 探测: 枚举设备、建空 Model、可选跑一次 FLOAT32 FC。
 * NDK: aarch64-linux-android28-clang nnapi_device_probe.c -lneuralnetworks -llog -o nnapi_device_probe */
#include <android/NeuralNetworks.h>
#include <android/log.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LOG(...) do { printf(__VA_ARGS__); fflush(stdout); } while (0)

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(void) {
    LOG("nnapi_device_probe start\n");

    void *lib = dlopen("libneuralnetworks.so", RTLD_NOW);
    if (!lib) {
        LOG("dlopen libneuralnetworks.so FAIL: %s\n", dlerror());
        /* 仍尝试直接链接符号（若静态依赖成功） */
    } else {
        LOG("dlopen libneuralnetworks.so OK\n");
    }

    uint32_t ndev = 0;
    int rc = ANeuralNetworks_getDeviceCount(&ndev);
    LOG("ANeuralNetworks_getDeviceCount rc=%d count=%u\n", rc, ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        ANeuralNetworksDevice *dev = NULL;
        rc = ANeuralNetworks_getDevice(i, &dev);
        if (rc != ANEURALNETWORKS_NO_ERROR || !dev) {
            LOG("  device[%u] get FAIL rc=%d\n", i, rc);
            continue;
        }
        const char *name = "?";
        const char *ver = "?";
        int64_t flevel = 0;
        ANeuralNetworksDevice_getName(dev, &name);
        ANeuralNetworksDevice_getVersion(dev, &ver);
        ANeuralNetworksDevice_getFeatureLevel(dev, &flevel);
        int32_t dtype = -1;
        ANeuralNetworksDevice_getType(dev, &dtype);
        const char *tn = "unknown";
        switch (dtype) {
        case ANEURALNETWORKS_DEVICE_CPU: tn = "CPU"; break;
        case ANEURALNETWORKS_DEVICE_GPU: tn = "GPU"; break;
        case ANEURALNETWORKS_DEVICE_ACCELERATOR: tn = "ACCELERATOR"; break;
        case ANEURALNETWORKS_DEVICE_OTHER: tn = "OTHER"; break;
        }
        LOG("  device[%u] name=%s type=%s(%d) version=%s featureLevel=%lld\n",
            i, name ? name : "?", tn, dtype, ver ? ver : "?", (long long)flevel);
    }

    ANeuralNetworksModel *model = NULL;
    rc = ANeuralNetworksModel_create(&model);
    LOG("ANeuralNetworksModel_create rc=%d model=%p\n", rc, (void *)model);
    if (rc != ANEURALNETWORKS_NO_ERROR || !model) return 1;

    /* 最小 FLOAT32 全连接: in[2] x W[3,2] + b[3] -> out[3] */
    float w[6] = {1, 0, 0, 1, 0.5f, 0.5f};
    float b[3] = {0, 0, 0};
    float in[2] = {1, 2};
    float out[3] = {0, 0, 0};
    uint32_t dims_in[] = {1, 2};
    uint32_t dims_w[] = {3, 2};
    uint32_t dims_b[] = {3};
    uint32_t dims_out[] = {1, 3};

    ANeuralNetworksOperandType t_in = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dims_in, .scale = 0, .zeroPoint = 0};
    ANeuralNetworksOperandType t_w = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dims_w, .scale = 0, .zeroPoint = 0};
    ANeuralNetworksOperandType t_b = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 1, .dimensions = dims_b, .scale = 0, .zeroPoint = 0};
    ANeuralNetworksOperandType t_out = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dims_out, .scale = 0, .zeroPoint = 0};
    ANeuralNetworksOperandType t_act = {.type = ANEURALNETWORKS_INT32, .dimensionCount = 0, .dimensions = NULL, .scale = 0, .zeroPoint = 0};

    if ((rc = ANeuralNetworksModel_addOperand(model, &t_in)) != ANEURALNETWORKS_NO_ERROR) { LOG("add in %d\n", rc); return 2; }
    if ((rc = ANeuralNetworksModel_addOperand(model, &t_w)) != ANEURALNETWORKS_NO_ERROR) { LOG("add w %d\n", rc); return 2; }
    if ((rc = ANeuralNetworksModel_addOperand(model, &t_b)) != ANEURALNETWORKS_NO_ERROR) { LOG("add b %d\n", rc); return 2; }
    if ((rc = ANeuralNetworksModel_addOperand(model, &t_act)) != ANEURALNETWORKS_NO_ERROR) { LOG("add act %d\n", rc); return 2; }
    if ((rc = ANeuralNetworksModel_addOperand(model, &t_out)) != ANEURALNETWORKS_NO_ERROR) { LOG("add out %d\n", rc); return 2; }

    ANeuralNetworksModel_setOperandValue(model, 1, w, sizeof w);
    ANeuralNetworksModel_setOperandValue(model, 2, b, sizeof b);
    int32_t act = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 3, &act, sizeof act);

    uint32_t inputs[] = {0, 1, 2, 3};
    uint32_t outputs[] = {4};
    rc = ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inputs, 1, outputs);
    LOG("addOperation FC rc=%d\n", rc);
    uint32_t model_ins[] = {0};
    uint32_t model_outs[] = {4};
    rc = ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, model_ins, 1, model_outs);
    LOG("identify I/O rc=%d\n", rc);
    rc = ANeuralNetworksModel_finish(model);
    LOG("Model_finish rc=%d\n", rc);
    if (rc != ANEURALNETWORKS_NO_ERROR) return 3;

    /* 优先 ACCELERATOR，否则全部设备 */
    ANeuralNetworksDevice * Prefer[8];
    uint32_t nprefer = 0;
    for (uint32_t i = 0; i < ndev && nprefer < 8; i++) {
        ANeuralNetworksDevice *dev = NULL;
        if (ANeuralNetworks_getDevice(i, &dev) != ANEURALNETWORKS_NO_ERROR) continue;
        int32_t dtype = -1;
        ANeuralNetworksDevice_getType(dev, &dtype);
        if (dtype == ANEURALNETWORKS_DEVICE_ACCELERATOR) Prefer[nprefer++] = dev;
    }
    ANeuralNetworksCompilation *comp = NULL;
    if (nprefer > 0) {
        rc = ANeuralNetworksCompilation_createForDevices(model, Prefer, nprefer, &comp);
        LOG("Compilation_createForDevices(ACCEL x%u) rc=%d\n", nprefer, rc);
    } else {
        rc = ANeuralNetworksCompilation_create(model, &comp);
        LOG("Compilation_create(all) rc=%d\n", rc);
    }
    if (rc != ANEURALNETWORKS_NO_ERROR || !comp) return 4;
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    double t0 = now_ms();
    rc = ANeuralNetworksCompilation_finish(comp);
    LOG("Compilation_finish rc=%d (%.2f ms)\n", rc, now_ms() - t0);
    if (rc != ANEURALNETWORKS_NO_ERROR) return 5;

    ANeuralNetworksExecution *exec = NULL;
    rc = ANeuralNetworksExecution_create(comp, &exec);
    LOG("Execution_create rc=%d\n", rc);
    if (rc != ANEURALNETWORKS_NO_ERROR || !exec) return 6;
    ANeuralNetworksExecution_setInput(exec, 0, NULL, in, sizeof in);
    ANeuralNetworksExecution_setOutput(exec, 0, NULL, out, sizeof out);
    t0 = now_ms();
    rc = ANeuralNetworksExecution_compute(exec);
    LOG("Execution_compute rc=%d (%.2f ms) out=[%.3f %.3f %.3f] expect~[1,2,1.5]\n",
        rc, now_ms() - t0, out[0], out[1], out[2]);

    ANeuralNetworksExecution_free(exec);
    ANeuralNetworksCompilation_free(comp);
    ANeuralNetworksModel_free(model);
    if (lib) dlclose(lib);
    LOG("nnapi_device_probe done ok=%d\n", rc == ANEURALNETWORKS_NO_ERROR);
    return rc == ANEURALNETWORKS_NO_ERROR ? 0 : 7;
}
