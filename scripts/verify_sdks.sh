#!/usr/bin/env bash
# 核对 third_party/sdks 下真 SDK：头文件/库架构/关键符号；并用 NDK 交叉编 NNAPI smoke。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="$ROOT/third_party/sdks"
export HYMT2_SDK_ROOT="$SDK"
export HYMT2_ROOT="$ROOT"
if [[ -f "$SDK/ndk/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$SDK/ndk/env.sh"
fi
if [[ -f "$SDK/hiai/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$SDK/hiai/env.sh"
fi
if [[ -f "$SDK/qnn/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$SDK/qnn/env.sh"
fi

ok=0; fail=0; skip=0
pass() { echo "OK  $*"; ok=$((ok+1)); }
bad()  { echo "FAIL $*"; fail=$((fail+1)); }
note() { echo "SKIP $*"; skip=$((skip+1)); }

echo "==== physical CPU (host) ===="
phys=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo "?")
logi=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo "?")
echo "host physical=$phys logical=$logi"

echo "==== NNAPI / NDK ===="
NDK="${ANDROID_NDK_HOME:-}"
if [[ -z "$NDK" || ! -d "$NDK" ]]; then
  NDK="$(echo "$SDK"/ndk/android-ndk-* | awk '{print $1}')"
fi
NN_H=""
for t in darwin-x86_64 darwin-arm64 linux-x86_64; do
  c="$NDK/toolchains/llvm/prebuilt/$t/sysroot/usr/include/android/NeuralNetworks.h"
  if [[ -f "$c" ]]; then NN_H="$c"; break; fi
done
if [[ -n "$NN_H" ]]; then pass "NeuralNetworks.h → $NN_H"; else bad "NeuralNetworks.h missing"; fi

PRE=""
for t in darwin-x86_64 darwin-arm64 linux-x86_64; do
  if [[ -x "$NDK/toolchains/llvm/prebuilt/$t/bin/aarch64-linux-android28-clang" ]]; then
    PRE="$NDK/toolchains/llvm/prebuilt/$t"
    break
  fi
done
if [[ -n "$PRE" ]]; then
  CC="$PRE/bin/aarch64-linux-android28-clang"
  OUT="$SDK/ndk/nnapi_smoke_aarch64"
  # 允许未解析 NNAPI 符号（系统库在设备上）: 用 --unresolved-symbols=ignore-all 仅验头文件可编译
  if "$CC" -O2 -Wall -Wextra -o "$OUT" "$ROOT/tools/nnapi_smoke.c" \
      -Wl,--unresolved-symbols=ignore-all 2>"$SDK/ndk/nnapi_smoke.log"; then
    file "$OUT" | grep -q aarch64 && pass "nnapi_smoke aarch64 交叉编译" || bad "nnapi_smoke 非 aarch64"
  else
    bad "nnapi_smoke 交叉编译失败 (见 $SDK/ndk/nnapi_smoke.log)"
    cat "$SDK/ndk/nnapi_smoke.log" | tail -20
  fi
else
  bad "NDK clang 未找到"
fi

echo "==== HiAI / MindSpore Lite NPU ===="
MS="${MINDSPORE_LITE_HOME:-$SDK/hiai/mindspore-lite-2.12.0-android-aarch64}"
HIAI="$MS/runtime/third_party/hiai_ddk/lib/libhiai.so"
MSLIB="$MS/runtime/lib/libmindspore-lite.so"
if [[ -f "$HIAI" ]]; then
  file "$HIAI" | tee "$SDK/hiai/libhiai.file"
  file "$HIAI" | grep -qiE 'aarch64|ARM aarch64' && pass "libhiai.so aarch64" || bad "libhiai.so 架构异常"
else
  bad "libhiai.so 缺失 ($HIAI)"
fi
if [[ -f "$MSLIB" ]]; then pass "libmindspore-lite.so"; else bad "libmindspore-lite.so 缺失"; fi
echo "hiai_mslite=1" > "$SDK/hiai.status"

echo "==== QNN / QAIRT ===="
QROOT="${QNN_SDK_ROOT:-}"
if [[ -z "$QROOT" && -f "$SDK/qnn/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$SDK/qnn/env.sh" || true
  QROOT="${QNN_SDK_ROOT:-}"
fi
if [[ -z "$QROOT" ]]; then
  QROOT="$(ls -d "$SDK"/qnn/qairt/*/ 2>/dev/null | head -1 || true)"
fi
if [[ -n "${QROOT:-}" && -d "$QROOT" ]]; then
  export QNN_SDK_ROOT="$QROOT"
  echo "QNN_SDK_ROOT=$QNN_SDK_ROOT" | tee "$SDK/qnn/env.sh"
  ln -sfn "$QNN_SDK_ROOT" "$SDK/qnn/current"
  found=0
  for lib in libQnnHtp.so libQnnCpu.so; do
    p=$(find "$QNN_SDK_ROOT/lib" -name "$lib" 2>/dev/null | head -1 || true)
    if [[ -n "$p" ]]; then
      file "$p" | tee "$SDK/qnn/${lib}.file"
      pass "$lib → $p"
      found=1
    fi
  done
  [[ $found -eq 1 ]] && echo "qnn=1" > "$SDK/qnn.status" || { bad "QNN 库未找到"; echo "qnn=0" > "$SDK/qnn.status"; }
else
  if [[ -f "$SDK/qnn/v"*.zip ]] || ls "$SDK/qnn"/v*.zip >/dev/null 2>&1; then
    note "QNN zip 下载中/未解压"
  else
    bad "QNN SDK 未下载"
  fi
  echo "qnn=0" > "$SDK/qnn.status"
fi

echo "==== RKLLM ===="
RK=$(find "$SDK/rknn" -name 'librkllmrt.so' 2>/dev/null | head -1 || true)
if [[ -n "$RK" ]]; then
  file "$RK" | tee "$SDK/rknn/librkllmrt.file"
  pass "librkllmrt.so → $RK"
  echo "rkllm=1" > "$SDK/rknn.status"
else
  note "librkllmrt.so 未找到 (官方包需 Filez 提取码 rkllm: https://console.zbox.filez.com/l/RJJDmB)"
  echo "rkllm=docs" > "$SDK/rknn.status"
fi

echo "==== hymt2 probe (host) ===="
if [[ -x "$ROOT/libhymt2.dylib" || -x "$ROOT/libhymt2.so" ]]; then
  export DYLD_LIBRARY_PATH="/usr/local/lib:${DYLD_LIBRARY_PATH:-}"
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json}"
  "$ROOT/hymt2_cli" 2>&1 | head -1 || true
  # 用短程序打印 capabilities: 跑一次带假路径触发 caps
  MODEL=$(ls /Users/bobo/Documents/GitHub/livesub/models/Hy-MT2-1.8B-1.25Bit.gguf 2>/dev/null || true)
  if [[ -n "$MODEL" ]]; then
    HYMT2_DEVICE=cpu "$ROOT/hymt2_cli" "$MODEL" Chinese "Hi" 1 1 2>&1 | head -5
  fi
fi

echo "---- summary: ok=$ok fail=$fail skip=$skip ----"
[[ $fail -eq 0 ]]
