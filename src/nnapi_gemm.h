/* Android NNAPI float FC 后端 (STQ1 等先反量化再上 NPU)。 */
#ifndef HYMT2_NNAPI_GEMM_H
#define HYMT2_NNAPI_GEMM_H
#include "device.h"
int hy_nnapi_probe(void);
hy_device *hy_nnapi_create(char *info, size_t cap);
#endif
