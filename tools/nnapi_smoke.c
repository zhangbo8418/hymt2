/* NDK 交叉编译冒烟: 只验证 NeuralNetworks.h 与链接面，不要求本机有 Android 运行时。
 * 用法见 scripts/verify_sdks.sh */
#include <android/NeuralNetworks.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    /* 引用符号，强制与 libneuralnetworks 链接面一致；Android 上才会真正解析。 */
    volatile void *syms[] = {
        (void *)(uintptr_t)&ANeuralNetworksModel_create,
        (void *)(uintptr_t)&ANeuralNetworksModel_free,
        (void *)(uintptr_t)&ANeuralNetworksCompilation_create,
        (void *)(uintptr_t)&ANeuralNetworksCompilation_free,
        (void *)(uintptr_t)&ANeuralNetworksExecution_create,
        (void *)(uintptr_t)&ANeuralNetworksExecution_free,
    };
    printf("nnapi_smoke ok refs=%d ANEURALNETWORKS_NO_ERROR=%d\n",
           (int)(sizeof syms / sizeof syms[0]), ANEURALNETWORKS_NO_ERROR);
    return 0;
}
