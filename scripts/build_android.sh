#!/usr/bin/env bash
# 交叉编译 Android arm64 hymt2_cli (+ lib), 默认 API 29 (NNAPI getDevice/compute)。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${ANDROID_NDK_HOME:-$ROOT/third_party/sdks/ndk/android-ndk-r27c}"
API="${ANDROID_API:-29}"
HOST_TAG="${NDK_HOST_TAG:-darwin-x86_64}"
CC="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android${API}-clang"
OUT="$ROOT/build/android-arm64"
mkdir -p "$OUT"

if [[ ! -x "$CC" ]]; then
  # 尝试其它 host tag
  for t in darwin-x86_64 darwin-arm64 linux-x86_64; do
    c="$NDK/toolchains/llvm/prebuilt/$t/bin/aarch64-linux-android${API}-clang"
    if [[ -x "$c" ]]; then CC="$c"; HOST_TAG="$t"; break; fi
  done
fi
[[ -x "$CC" ]] || { echo "NDK clang 未找到: $CC"; exit 1; }

# 默认开 Vulkan (运行时 dlopen libvulkan.so → Mali/Adreno); METAL=0。
# 设 ANDROID_NO_VULKAN=1 可关回空壳。
# mtune 面向近年 Cortex-A/X (Dimensity 8xxx 等); 不抬高 march, 以免关掉运行时 ISA 回退。
CFLAGS=(-O3 -ffast-math -fno-math-errno -funroll-loops -mtune=cortex-a720
  -Wall -Wextra -Wno-unused-parameter -fPIC -fvisibility=hidden
  -I"$ROOT/third_party/vulkan" -Isrc -Wno-missing-field-initializers
  -DHYMT2_METAL=0)
LDFLAGS=()
# 可选: ANDROID_PROFILE=1 打开 forward 分段计时
if [[ "${ANDROID_PROFILE:-0}" == "1" ]]; then
  CFLAGS+=(-DHYMT2_PROFILE)
fi
if [[ "${ANDROID_NO_VULKAN:-0}" == "1" ]]; then
  CFLAGS+=(-DHYMT2_NO_VULKAN)
fi
SRC=(src/model.c src/kernels.c src/gguf.c src/tokenizer.c src/threadpool.c
  src/device.c src/cuda.c src/npu.c src/nnapi_gemm.c src/metal_stub.c src/vulkan.c)

cd "$ROOT"
echo "[android] CC=$CC"
echo "[android] OUT=$OUT"

$CC "${CFLAGS[@]}" "${LDFLAGS[@]}" -shared -o "$OUT/libhymt2.so" "${SRC[@]}" src/capi.c \
  -lm -ldl -llog -lneuralnetworks -pthread

$CC "${CFLAGS[@]}" "${LDFLAGS[@]}" -fPIE -pie -o "$OUT/hymt2_cli" tools/hymt2_cli.c \
  "${SRC[@]}" \
  -lm -ldl -llog -lneuralnetworks -pthread

$CC -O2 -Wall -fPIE -pie -o "$OUT/nnapi_device_probe" tools/nnapi_device_probe.c \
  -lneuralnetworks -ldl -llog

file "$OUT/hymt2_cli" "$OUT/libhymt2.so" "$OUT/nnapi_device_probe"
echo "ok: $OUT"
