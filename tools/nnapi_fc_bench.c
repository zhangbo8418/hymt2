/* 真机测 NNAPI FC: 常量 W vs 动态 W，以及 256xK 规模耗时。 */
#include <android/NeuralNetworks.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int run_fc(int n, int k, int dynamic_w, int accel_only) {
    uint32_t ndev = 0;
    ANeuralNetworks_getDeviceCount(&ndev);
    ANeuralNetworksDevice *accels[8];
    uint32_t nacc = 0;
    for (uint32_t i = 0; i < ndev && nacc < 8; i++) {
        ANeuralNetworksDevice *d = NULL;
        if (ANeuralNetworks_getDevice(i, &d) != ANEURALNETWORKS_NO_ERROR) continue;
        int32_t t = -1;
        ANeuralNetworksDevice_getType(d, &t);
        if (t == ANEURALNETWORKS_DEVICE_ACCELERATOR) accels[nacc++] = d;
    }

    float *W = (float *)malloc((size_t)n * (size_t)k * sizeof(float));
    float *b = (float *)calloc((size_t)n, sizeof(float));
    float *x = (float *)malloc((size_t)k * sizeof(float));
    float *y = (float *)calloc((size_t)n, sizeof(float));
    if (!W || !b || !x || !y) return 10;
    for (int i = 0; i < n * k; i++) W[i] = 0.001f * (float)((i % 17) - 8);
    for (int i = 0; i < k; i++) x[i] = 0.01f * (float)((i % 13) - 6);

    ANeuralNetworksModel *model = NULL;
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR) return 1;
    uint32_t dx[] = {1u, (uint32_t)k};
    uint32_t dw[] = {(uint32_t)n, (uint32_t)k};
    uint32_t db[] = {(uint32_t)n};
    uint32_t dy[] = {1u, (uint32_t)n};
    ANeuralNetworksOperandType tx = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dx};
    ANeuralNetworksOperandType tw = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dw};
    ANeuralNetworksOperandType tb = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 1, .dimensions = db};
    ANeuralNetworksOperandType ty = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dy};
    ANeuralNetworksOperandType tact = {.type = ANEURALNETWORKS_INT32, .dimensionCount = 0};
    ANeuralNetworksModel_addOperand(model, &tx);
    ANeuralNetworksModel_addOperand(model, &tw);
    ANeuralNetworksModel_addOperand(model, &tb);
    ANeuralNetworksModel_addOperand(model, &tact);
    ANeuralNetworksModel_addOperand(model, &ty);
    int32_t act = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 3, &act, sizeof act);
    if (!dynamic_w) {
        ANeuralNetworksModel_setOperandValue(model, 1, W, (size_t)n * (size_t)k * sizeof(float));
        ANeuralNetworksModel_setOperandValue(model, 2, b, (size_t)n * sizeof(float));
    }
    uint32_t op_ins[] = {0, 1, 2, 3}, op_outs[] = {4};
    int rc = ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, op_ins, 1, op_outs);
    if (rc != ANEURALNETWORKS_NO_ERROR) { printf("addOp %d\n", rc); return 2; }
    if (dynamic_w) {
        uint32_t mins[] = {0, 1, 2}, mouts[] = {4};
        rc = ANeuralNetworksModel_identifyInputsAndOutputs(model, 3, mins, 1, mouts);
    } else {
        uint32_t mins[] = {0}, mouts[] = {4};
        rc = ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, mins, 1, mouts);
    }
    if (rc != ANEURALNETWORKS_NO_ERROR) { printf("identify %d\n", rc); return 3; }
    rc = ANeuralNetworksModel_finish(model);
    if (rc != ANEURALNETWORKS_NO_ERROR) { printf("finish %d\n", rc); return 4; }

    ANeuralNetworksCompilation *comp = NULL;
    double t0 = now_ms();
    if (accel_only && nacc > 0)
        rc = ANeuralNetworksCompilation_createForDevices(model, (const ANeuralNetworksDevice *const *)accels, nacc, &comp);
    else
        rc = ANeuralNetworksCompilation_create(model, &comp);
    if (rc != ANEURALNETWORKS_NO_ERROR || !comp) {
        printf("compCreate rc=%d dynamic=%d accel=%d\n", rc, dynamic_w, accel_only);
        ANeuralNetworksModel_free(model);
        free(W); free(b); free(x); free(y);
        return 5;
    }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    rc = ANeuralNetworksCompilation_finish(comp);
    double tcomp = now_ms() - t0;
    printf("n=%d k=%d dynamic=%d accel=%d Compilation_finish rc=%d (%.1f ms)\n", n, k, dynamic_w, accel_only, rc, tcomp);
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        ANeuralNetworksCompilation_free(comp);
        ANeuralNetworksModel_free(model);
        free(W); free(b); free(x); free(y);
        return 6;
    }

    ANeuralNetworksExecution *exec = NULL;
    ANeuralNetworksExecution_create(comp, &exec);
    ANeuralNetworksExecution_setInput(exec, 0, NULL, x, (size_t)k * sizeof(float));
    if (dynamic_w) {
        ANeuralNetworksExecution_setInput(exec, 1, NULL, W, (size_t)n * (size_t)k * sizeof(float));
        ANeuralNetworksExecution_setInput(exec, 2, NULL, b, (size_t)n * sizeof(float));
    }
    ANeuralNetworksExecution_setOutput(exec, 0, NULL, y, (size_t)n * sizeof(float));
    t0 = now_ms();
    rc = ANeuralNetworksExecution_compute(exec);
    double texec = now_ms() - t0;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += y[i];
    printf("  compute rc=%d (%.1f ms) y0=%.4f sum=%.4f\n", rc, texec, y[0], sum);

    ANeuralNetworksExecution_free(exec);
    ANeuralNetworksCompilation_free(comp);
    ANeuralNetworksModel_free(model);
    free(W); free(b); free(x); free(y);
    return rc == ANEURALNETWORKS_NO_ERROR ? 0 : 7;
}

int main(void) {
    printf("=== const W accel ===\n");
    run_fc(64, 256, 0, 1);
    printf("=== dyn W accel ===\n");
    run_fc(64, 256, 1, 1);
    printf("=== dyn W any ===\n");
    run_fc(64, 256, 1, 0);
    printf("=== const W 256x2048 accel ===\n");
    run_fc(256, 2048, 0, 1);
    printf("=== const W 256x2048 any ===\n");
    run_fc(256, 2048, 0, 0);
    return 0;
}
