#!/usr/bin/env bash
# 把 shaders/*.comp 编成 SPIR-V 并生成 src/vk_shaders.h (需要 glslangValidator; 发布包里已带生成好的头文件)
set -euo pipefail
cd "$(dirname "$0")/.."
out=src/vk_shaders.h
echo "/* 自动生成 (tools/gen_shaders.sh), 勿手改 */" > $out
for s in panel q4k q6k; do
  glslangValidator -V --target-env vulkan1.0 shaders/$s.comp --vn spv_$s -o /tmp/spv_$s.h >/dev/null
  cat /tmp/spv_$s.h >> $out
done
echo "generated $out"
