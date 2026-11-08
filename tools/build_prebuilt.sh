#!/usr/bin/env bash
# 重新生成 prebuilt 里的二进制 (在 Linux 上运行): 用法  tools/build_prebuilt.sh <输出目录>
#   <输出目录>/linux/  : 静态 hymt2_cli_j4125 / hymt2_cli_generic / bench_kernels_j4125 (无 Vulkan), 动态 hymt2_cli_vulkan / libhymt2.so (含 Vulkan)
#   <输出目录>/win7/   : 需要 x86_64-w64-mingw32-gcc (msvcrt 版): hymt2_cli.exe / hymt2_cli_avx2.exe / hymt2.dll / bench_kernels.exe
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?用法: $0 <输出目录>}"; mkdir -p "$OUT/linux" "$OUT/win7"
CORE="src/model.c src/kernels.c src/gguf.c src/tokenizer.c src/threadpool.c src/vulkan.c"
LIB="$CORE src/capi.c"
INC="-Ithird_party/vulkan -Isrc -Wno-missing-field-initializers"
J="-msse4.2 -mtune=goldmont-plus"
gcc -O3 -static -DHYMT2_NO_VULKAN $INC $J -o "$OUT/linux/hymt2_cli_j4125" tools/hymt2_cli.c $CORE -lm -lpthread
gcc -O3 -static -DHYMT2_NO_VULKAN $INC -o "$OUT/linux/hymt2_cli_generic" tools/hymt2_cli.c $CORE -lm -lpthread
gcc -O3 -static -DHYMT2_NO_VULKAN $INC $J -o "$OUT/linux/bench_kernels_j4125" tests/bench_kernels.c src/kernels.c -lm
gcc -O3 $INC $J -o "$OUT/linux/hymt2_cli_vulkan" tools/hymt2_cli.c $CORE -lm -lpthread -ldl
gcc -O3 -fPIC -shared -fvisibility=hidden $INC -o "$OUT/linux/libhymt2.so" $LIB -lm -lpthread -ldl
strip "$OUT"/linux/hymt2_cli_* "$OUT"/linux/bench_kernels_j4125 "$OUT/linux/libhymt2.so"
if command -v x86_64-w64-mingw32-gcc >/dev/null; then
  W="x86_64-w64-mingw32-gcc -O3 -Wall -Wno-unused-parameter $INC -D_WIN32_WINNT=0x0601 -DWINVER=0x0601"
  V="-Wl,--major-os-version=6 -Wl,--minor-os-version=1 -Wl,--major-subsystem-version=6 -Wl,--minor-subsystem-version=1"
  $W -static $V -o "$OUT/win7/hymt2_cli.exe" tools/hymt2_cli.c $CORE -lm
  $W -mavx2 -mfma -mf16c -static $V -o "$OUT/win7/hymt2_cli_avx2.exe" tools/hymt2_cli.c $CORE -lm
  $W -fvisibility=hidden -shared -static-libgcc $V -o "$OUT/win7/hymt2.dll" $LIB -lm
  $W -static $V -o "$OUT/win7/bench_kernels.exe" tests/bench_kernels.c src/kernels.c -lm
  x86_64-w64-mingw32-strip "$OUT"/win7/*.exe "$OUT"/win7/*.dll
else
  echo "未找到 x86_64-w64-mingw32-gcc, 跳过 Windows 版本"
fi
echo "done: $OUT"
