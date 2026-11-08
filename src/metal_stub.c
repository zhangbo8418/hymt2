/* 非 Darwin 或 METAL=0: Metal 占位 (与 metal.m 互斥编入)。 */
#include "device.h"
#include <stdio.h>
#include <string.h>

int hy_metal_probe(char *out, int cap) {
    if (out && cap) snprintf(out, (size_t)cap, "0");
    return 0;
}

hy_device *hy_metal_create(const char *pref, char *info, size_t cap) {
    (void)pref;
    if (info && cap) snprintf(info, cap, "本构建不含 Metal");
    return NULL;
}
