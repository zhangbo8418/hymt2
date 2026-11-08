#!/usr/bin/env bash
# 下载/准备 hymt2 加速 SDK 到 third_party/sdks/（不进 git）。
# 用法:
#   ./scripts/fetch_sdks.sh              # 本机可自动项 (Metal toolchain 等)
#   ./scripts/fetch_sdks.sh metal
#   ./scripts/fetch_sdks.sh cuda         # 打印 CUDA 安装指引 (需官网 installer)
#   ./scripts/fetch_sdks.sh ndk          # 下载 Android NDK (NNAPI)
#   ./scripts/fetch_sdks.sh mslite-npu   # 下载 MindSpore Lite Android NPU 包 (华为路径)
#   ./scripts/fetch_sdks.sh rknn-llm     # clone airockchip/rknn-llm 文档与示例
#   ./scripts/fetch_sdks.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="$ROOT/third_party/sdks"
mkdir -p "$SDK"
echo "SDK root: $SDK"

have() { command -v "$1" >/dev/null 2>&1; }

fetch_metal() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[metal] 非 Darwin, 跳过"
    return 0
  fi
  echo "[metal] 尝试下载 Metal Toolchain…"
  if xcodebuild -downloadComponent MetalToolchain 2>&1 | tee "$SDK/metal_toolchain.log"; then
    echo "[metal] ok (见 xcodebuild 输出)"
  else
    echo "[metal] xcodebuild 失败: 请在 Xcode → Settings → Components 安装 Metal Toolchain"
    echo "[metal] 运行时仍可用 Metal.framework + newLibraryWithSource (无需 metallib 预编译)"
  fi
  # 记录框架是否存在
  if [[ -d /System/Library/Frameworks/Metal.framework ]]; then
    echo "metal=1" > "$SDK/metal.status"
  else
    echo "metal=0" > "$SDK/metal.status"
  fi
}

fetch_cuda() {
  mkdir -p "$SDK/cuda"
  cat > "$SDK/cuda/README.txt" <<'EOF'
NVIDIA CUDA Toolkit (轨 A: hymt2 CUDA quant gemm)
-------------------------------------------------
本 Mac 通常无 NVIDIA GPU。请在 Linux/Windows + NVIDIA 上安装:
  https://developer.nvidia.com/cuda-downloads
安装后:
  export CUDA_HOME=/usr/local/cuda
  make -C ../.. CUDA=1
有卡时: HYMT2_DEVICE=cuda HYMT2_VERBOSE=1 ./hymt2_cli <model.gguf> ...
EOF
  if have nvcc; then
    nvcc --version | head -5 | tee "$SDK/cuda/version.txt"
    echo "cuda=1" > "$SDK/cuda.status"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version | head -5 | tee "$SDK/cuda/version.txt"
    echo "cuda=1" > "$SDK/cuda.status"
  else
    echo "cuda=0 (toolkit 未安装; 见 $SDK/cuda/README.txt)" | tee "$SDK/cuda.status"
  fi
}

fetch_ndk() {
  mkdir -p "$SDK/ndk"
  local ver="r27c"
  local url="https://dl.google.com/android/repository/android-ndk-${ver}-darwin.xml"
  # 使用官方 zip (darwin x86_64 / arm64 由 uname -m 选择)
  local arch
  arch="$(uname -m)"
  local zip="android-ndk-${ver}-darwin.zip"
  if [[ "$arch" == "arm64" ]]; then
    zip="android-ndk-${ver}-darwin.dmg" # 新 NDK 多为 zip; 回退通用 zip 名
    zip="android-ndk-${ver}-darwin.zip"
  fi
  # Google 直接 zip:
  local dl="https://dl.google.com/android/repository/android-ndk-${ver}-darwin.zip"
  if [[ "$(uname -s)" == "Linux" ]]; then
    dl="https://dl.google.com/android/repository/android-ndk-${ver}-linux.zip"
    zip="android-ndk-${ver}-linux.zip"
  fi
  if [[ -d "$SDK/ndk/android-ndk-${ver}" ]]; then
    echo "[ndk] 已存在 $SDK/ndk/android-ndk-${ver}"
  else
    echo "[ndk] 下载 $dl …"
    if have curl; then
      curl -L --fail -o "$SDK/ndk/$zip" "$dl" || {
        echo "[ndk] 下载失败。请手动安装 NDK 并: export ANDROID_NDK_HOME=..."
        echo "ndk=0" > "$SDK/ndk.status"
        return 0
      }
      (cd "$SDK/ndk" && unzip -q "$zip" && rm -f "$zip")
    else
      echo "[ndk] 需要 curl"
      echo "ndk=0" > "$SDK/ndk.status"
      return 0
    fi
  fi
  local home
  home="$(echo "$SDK/ndk"/android-ndk-*)"
  echo "ANDROID_NDK_HOME=$home" | tee "$SDK/ndk/env.sh"
  echo "ndk=1" > "$SDK/ndk.status"
  # NNAPI 头在 sysroot
  echo "[ndk] NNAPI: #include <android/NeuralNetworks.h> (API 27+)"
}

fetch_mslite_npu() {
  mkdir -p "$SDK/hiai"
  local ver="2.12.0"
  local name="mindspore-lite-${ver}-android-aarch64.tar.gz"
  local url="https://repo.mindspore.cn/mindspore/mindspore-lite/newest/lite/android_aarch64/npu/${name}"
  local out="$SDK/hiai/${name}"
  echo "[hiai/mslite] 下载 $url …"
  if [[ -d "$SDK/hiai/mindspore-lite-${ver}-android-aarch64" ]]; then
    echo "[hiai/mslite] 已存在"
  elif curl -L --fail -o "$out" "$url"; then
    (cd "$SDK/hiai" && tar -xzf "$name")
  else
    cat > "$SDK/hiai/README.txt" <<'EOF'
华为 HiAI / MindSpore Lite NPU
-----------------------------
1) HiAI DDK: https://developer.huawei.com/consumer/cn/doc/development/hiai-Library/ddk-download-0000001053590180
2) MindSpore Lite android npu:
   https://repo.mindspore.cn/mindspore/mindspore-lite/newest/lite/android_aarch64/npu/
下载后解压到本目录, 设置:
  export HWHIAI_DDK=...
  export MINDSPORE_LITE_HOME=...
EOF
    echo "hiai_mslite=0 (需手动下载, 见 README.txt)" | tee "$SDK/hiai.status"
    return 0
  fi
  local home="$SDK/hiai/mindspore-lite-${ver}-android-aarch64"
  echo "MINDSPORE_LITE_HOME=$home" | tee "$SDK/hiai/env.sh"
  echo "HWHIAI_DDK=$home/runtime/third_party/hiai_ddk" >> "$SDK/hiai/env.sh"
  if [[ -f "$home/runtime/third_party/hiai_ddk/lib/libhiai.so" ]]; then
    echo "hiai_mslite=1" | tee "$SDK/hiai.status"
  else
    echo "hiai_mslite=0 (缺 libhiai.so)" | tee "$SDK/hiai.status"
  fi
}

fetch_rknn_llm() {
  mkdir -p "$SDK/rknn"
  if [[ -d "$SDK/rknn/rknn-llm/.git" ]]; then
    echo "[rkllm] 已 clone"
  else
    echo "[rkllm] clone airockchip/rknn-llm …"
    git clone --depth 1 https://github.com/airockchip/rknn-llm.git "$SDK/rknn/rknn-llm" || {
      echo "[rkllm] clone 失败"
      echo "rkllm=0" > "$SDK/rknn.status"
      return 0
    }
  fi
  local ver="${RKLLM_VERSION:-release-v1.3.1}"
  local base="https://github.com/airockchip/rknn-llm/raw/refs/tags/${ver}/rkllm-runtime"
  mkdir -p "$SDK/rknn/rkllm-runtime/Linux/librkllm_api/aarch64" \
           "$SDK/rknn/rkllm-runtime/Android/librkllm_api/arm64-v8a" \
           "$SDK/rknn/rkllm-runtime/include"
  echo "[rkllm] 下载 ${ver} librkllmrt.so + rkllm.h …"
  curl -L --fail -o "$SDK/rknn/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so" \
    "$base/Linux/librkllm_api/aarch64/librkllmrt.so" || true
  curl -L --fail -o "$SDK/rknn/rkllm-runtime/Android/librkllm_api/arm64-v8a/librkllmrt.so" \
    "$base/Android/librkllm_api/arm64-v8a/librkllmrt.so" || true
  curl -L --fail -o "$SDK/rknn/rkllm-runtime/include/rkllm.h" \
    "$base/Linux/librkllm_api/include/rkllm.h" || true
  cat > "$SDK/rknn/README.txt" <<'EOF'
瑞芯微 RKLLM
------------
runtime 自 GitHub release tag 拉取 (librkllmrt.so + rkllm.h)。
完整工具链 zip 也可从 Filez 提取码 rkllm: https://console.zbox.filez.com/l/RJJDmB
板上推理需要 SoC + .rkllm 模型。GGUF STQ1_0/Q2_0C 不要声称可转。
EOF
  if [[ -f "$SDK/rknn/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so" ]]; then
    echo "RKLLM_LIB=$SDK/rknn/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so" | tee "$SDK/rknn/env.sh"
    echo "rkllm=1" | tee "$SDK/rknn.status"
  else
    echo "rkllm=docs" | tee "$SDK/rknn.status"
  fi
}

fetch_qnn() {
  mkdir -p "$SDK/qnn"
  cat > "$SDK/qnn/README.txt" <<'EOF'
高通 QAIRT / QNN (AI Engine Direct) Community
---------------------------------------------
直链 (无需网页登录, 约 1.3GB+):
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/<ver>/v<ver>.zip
解压后:
  export QNN_SDK_ROOT=.../qairt/<ver>
本仓库 probe 会 dlopen libQnnHtp.so / libQnnCpu.so。
EOF
  local ver="${QNN_SDK_VERSION:-2.47.0.260601}"
  local zip="v${ver}.zip"
  local url="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${ver}/${zip}"
  if [[ -d "$SDK/qnn/qairt/${ver}" ]]; then
    echo "[qnn] 已存在 $SDK/qnn/qairt/${ver}"
  elif [[ -f "$SDK/qnn/${zip}" ]]; then
    echo "[qnn] 解压已有 ${zip}…"
    (cd "$SDK/qnn" && unzip -q -o "$zip")
  else
    echo "[qnn] 下载 $url …"
    if curl -L --fail -C - -o "$SDK/qnn/${zip}" "$url"; then
      (cd "$SDK/qnn" && unzip -q -o "$zip")
    else
      echo "qnn=0 (下载失败)" | tee "$SDK/qnn.status"
      return 0
    fi
  fi
  local home="$SDK/qnn/qairt/${ver}"
  if [[ -d "$home" ]]; then
    echo "QNN_SDK_ROOT=$home" | tee "$SDK/qnn/env.sh"
    ln -sfn "$home" "$SDK/qnn/current"
    echo "qnn=1" | tee "$SDK/qnn.status"
  else
    echo "qnn=0 (解压后无 qairt/${ver})" | tee "$SDK/qnn.status"
  fi
}

summary() {
  echo "---- SDK status ----"
  for f in "$SDK"/*.status "$SDK"/*/*.status; do
    [[ -f "$f" ]] && echo "  $(basename "$(dirname "$f")")/$(basename "$f"): $(cat "$f")"
  done 2>/dev/null || true
}

run_one() {
  case "$1" in
    metal) fetch_metal ;;
    cuda) fetch_cuda ;;
    ndk) fetch_ndk ;;
    mslite-npu|hiai) fetch_mslite_npu ;;
    rknn-llm|rkllm) fetch_rknn_llm ;;
    qnn) fetch_qnn ;;
    all)
      fetch_metal; fetch_cuda; fetch_ndk; fetch_mslite_npu; fetch_rknn_llm; fetch_qnn
      ;;
    *)
      echo "未知目标: $1"; return 2
      ;;
  esac
}
if [[ $# -eq 0 ]]; then
  run_one all
else
  for a in "$@"; do run_one "$a" || exit $?; done
fi
summary
echo "done."
