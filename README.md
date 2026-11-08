# hymt2 — 纯 C 的 Hy-MT2 CPU 推理库

移植自 [sdcb/HyMT2Sharp](https://github.com/sdcb/HyMT2Sharp) (Apache-2.0) 的 CPU 路径。无外部依赖 (C 编译器 + libm + pthread),
可作为 Go / Flutter(dart:ffi) / Rust(FFI) 的进程内库使用,不开端口。

## 支持的权重类型

| ggml 类型 | 说明 | 面板重排 | 验证 |
|---|---|---|---|
| STQ1_0 (43, 及旧版 42) | Tencent 1.25-bit | 是 | **真实文件** `Hy-MT2-1.8B-1.25Bit.gguf` |
| Q2_0C (40) | Tencent 2-bit (512 权重共用一个 f16 缩放) | 是 (必须) | **真实文件** `Hy-MT2-1.8B-2Bit.gguf` |
| Q4_K / Q6_K | llama.cpp 标准 K-quants (Q4_K_M = Q4_K + Q6_K 混合) | 否 | 与 gguf-py 的独立反量化逐位一致; 合成 Q4_K_M 模型端到端对拍 numpy。**没有真实 Q4_K_M 文件** |
| Q8_0 / F16 / F32 | 浮点激活路径 (Q8_0 有 SSE4.1/AVX2/NEON 点积) | 否 | 与 gguf-py 一致 |

这就是 Hy-MT2-1.8B 各个发布版本实际用到的类型。其它类型 (Q5_K、Q3_K、IQ 等) 不支持, 会明确报错; 请换支持的量化或用进程外 HTTP (`MT_BACKEND=http`)。**进程内不再调用 llama.cpp。**

## 为什么比第一版快: 加载时重排成"面板" + 现场校准

STQ1_0 的原生格式要在运行时做码本查表、符号选择、掩码展开 (约 80% 的指令花在解码上)。原版 HyMT2Sharp 的做法是加载时把权重预解码成 2-bit 三值码并重排;
本库做同样的事 (格式见 `src/kernels.c` 的 "面板重排"): 每个 256 权重块 64 字节 + 一个 float 缩放, 运行时内层循环只剩 `shift → and → maddubs → add`
(AVX-512 VNNI 上是 `dpbusd`)。同一个内核也服务 Q2_0C (值 = `(2c-3)·d`)。

代价是访存量增加约 1.6 倍 (1.25-bit: 约 1.3 → 2.1 bit/权重), 所以**不盲目启用**:
`HYMT2_REPACK=auto` (默认) 会在加载时用真实线程池, 在最多 8 层的 FFN 上 (原生约 100MB / 面板约 160MB, 超过常见缓存) 实测两种实现, 面板快 3% 以上才用。
内存带宽很低的机器上它可能选"原生"; Q2_0C 的原生路径只是标量参考实现, 所以 Q2_0C 一律重排。

其它加载/运行期优化:
* 合并 q/k/v、gate/up 的矩阵乘为一个并行区 (每层少 2 次线程同步, 负载更均衡)。
* 激活量化块 (Q8_K) 按 SIMD 友好顺序重排并 64 字节对齐 (量化一次, 所有权重行复用)。
* 面板内核向前预取约 4KB 的权重。
* 启动时预读整个模型文件 (`HYMT2_PREFAULT=0` 关闭), 把首次翻译的缺页/磁盘读取挪到加载阶段。
* 前缀 KV 缓存: 相同的提示词前缀 (模板 + 指令) 只算一次。
* prefill 分块批量计算 (权重读一次, 服务多个 token)。

## 指令集: 运行时自动检测, 不需要 `-march`

| 内核 | 需要 | 典型 CPU |
|---|---|---|
| `avx512-vnni` | avx512f/bw/vl/dq + vnni | Ice Lake+ / Zen 4 |
| `avx2` | avx2 + f16c | Haswell+ / Zen+ |
| `sse4.1` | sse4.1 | **Intel J4125 / N5105 等 (有 SSE4.2, 无 AVX)** |
| `ssse3` / `scalar` | ssse3 / — | 更老 / 兜底 |
| `neon` / `neon+dotprod` | aarch64 (+ 运行时探测 ASIMDDP) | 手机 / Apple Silicon；有 dotprod 时用 `vdot`（对齐官方 STQ/llama.cpp） |

* J4125 显示 `sse4.1` 而不是 `sse4.2`: 内核用到 SSSE3 指令加 SSE4.1 的 `pblendvb`/`pmovsxbw`; SSE4.2 新增的字符串/CRC32 指令在这里用不上, 按实际用到的最高指令集命名。
* 没有 VNNI 的 AVX-512 核心默认走 `avx2` (512 位指令可能降频); `HYMT2_ISA=avx512` 可强制。
* `HYMT2_ISA=auto|avx512|avx512-bw|avx2|sse4.2|ssse3|scalar` 强制"不高于该等级的最优实现"; `HYMT2_THREADS`、`HYMT2_BATCH`、`HYMT2_VERBOSE=1` (打印内核/线程/重排决策)。
* K-quants: Q4_K / Q6_K 有 scalar / SSSE3 / AVX2 / **NEON(+dotprod)**; Q8_0 有 scalar / SSE4.1 / AVX2 / NEON。

## 加速设备 (`HYMT2_DEVICE`, 兼容 `HYMT2_VK`)

统一抽象见 `src/device.c` (参考 ggml-cuda / ggml-metal / ggml-vulkan 的设备与调度思路, **不链接 llama**):

| 后端 | 状态 | 说明 |
|---|---|---|
| CPU SIMD | 完整 | 默认回退 |
| Vulkan | 完整 | `vulkan.c` + `shaders/*.comp` |
| Metal | **完整** (macOS) | `metal.m` + 内嵌 MSL; 与 CPU 对拍通过 (`tests/test_metal.py`) |
| CUDA | 完整内核 + 整网 fused (需 GPU) | `cuda.c` + `cuda_kernels.cu` + `cuda_fused.cu`; `make CUDA=1` + nvcc; 无卡时 create=NULL；fused 与 Metal 同接口（激活/KV 常驻，整网 ≤1 sync） |
| NPU | probe + 插件名 | `npu.c`: nnapi/qnn/rkllm/hiai 探测真库; 无 SoC/未接线 matmul 时 create=NULL (防假加速)。SDK: `scripts/fetch_sdks.sh` |

* `HYMT2_DEVICE=auto` (默认): 按 **cuda → metal → vulkan → qnn → nnapi → rkllm → hiai** 试用, 加载期与 CPU 校准, **快 ≥10% 才启用**, 日志打印选型。
* 强制: `cuda|metal|vulkan|nnapi|qnn|rkllm|hiai|cpu|force|<n>`。兼容 `HYMT2_VK`。
* `hymt2_capabilities()`: `isa=… cuda=0|1 metal=0|1 vulkan=0|1 npu=0|nnapi|qnn|…`
* 验证: `python3 tests/test_vulkan.py` / `test_metal.py`; SDK 拉取不进 git: `./scripts/fetch_sdks.sh all`
* **暂不迁入 KOTV**; 本仓库 livesub 内调试。
* 编译: Darwin 默认 Metal; `make CUDA=1` / `VULKAN=0` / `METAL=0`。

## 构建

```bash
make                 # libhymt2.{so,dylib,dll} + hymt2_cli
make J4125=1         # 额外 -mtune=goldmont-plus (只影响指令调度; 仍是通用二进制)
make test            # 内核单元测试 (每个 ISA 路径, 含面板)
make static          # 全静态、无 Vulkan 的 hymt2_cli_static (Linux); 可加 J4125=1
make bench           # 内核吞吐
./hymt2_cli model.gguf English "今天天气真好。" [threads] [max_tokens]   # 输出分阶段耗时
```

**Windows 7 (CPU-only)**: MinGW-w64 (**msvcrt**, 不要用 UCRT):  
`make WIN7=1 CC=x86_64-w64-mingw32-gcc`。

**Windows 通用包 (推荐)**: 一份 `hymt2_cli.exe` 同时编入 **CPU + Vulkan + CUDA**。  
默认 `HYMT2_DEVICE=auto`：加载时探测能力并校准，按 **cuda → vulkan → CPU** 选第一个比 CPU 快 ≥10% 的后端（无 fused 的 GPU 走逐算子 matmul，其余仍在 CPU，相当于混合）。  
- Vulkan：运行时 `LoadLibrary(vulkan-1.dll)`，由显卡驱动提供，无 Vulkan 则跳过。  
- CUDA：链入 fused 内核；`cudart` 为 delay-load，无 NVIDIA 运行库时仍可走 Vulkan/CPU。  
Windows 上 `nvcc` 必须搭配 MSVC；CUDA 10.2（Win7 最后一版）支持 **VS 2015 / 2017 / 2019**。  
GitHub Actions `windows-cuda` 产物：  
- `hymt2-win7-universal`（CUDA 10.2 + Vulkan + CPU，子系统 6.1）  
- `hymt2-win10-universal`（CUDA 12 + Vulkan + CPU）

## 测试 (均可在本目录运行)

```bash
pip install numpy regex gguf
python3 tests/test_quants.py ./libhymt2.so                    # Q4_K/Q6_K/Q8_0 vs gguf-py
python3 tests/test_vulkan.py ./libhymt2.so                    # Vulkan shader vs CPU (默认用 lavapipe 软件设备)
TEST_MODE=stq|q4km|q2 python3 tests/test_model.py ./libhymt2.so   # 合成模型端到端 vs 独立 numpy 实现 (+ 分词器 vs python-regex)
HYMT2_ISA=sse4.1 HYMT2_REPACK=1 python3 tests/test_model.py ...     # 任意 ISA / 重排组合
```

端到端测试把容差设为"参考实现自己的舍入分散度" (把参考里的 float64 矩阵乘换成 float32 看它与自己差多少):
激活量化的舍入边界翻转会被网络放大, 个别合成序列极其敏感, 固定容差会误报。

## 速度 (请读: 这是单线程 + 噪声较大的沙箱 VM)

沙箱: 1 vCPU Xeon (AVX-512 VNNI), 真实模型, 一句约 13 个输出 token:

| 内核 | 原生 (decode) | 面板 (decode) |
|---|---|---|
| avx512-vnni | ~107 ms/tok | ~96 ms/tok |
| avx2 | ~125 ms/tok | ~111 ms/tok |
| sse4.1 | ~180 ms/tok | ~118 ms/tok (**-35%**) |

(2-bit 模型 Q2_0C 面板: 约 90–96 ms/tok。第一版同一路径约 150–170 ms/tok。)
这台 VM 的**单核访存**成为了新的瓶颈 (每个 token 要读约 600MB 权重), 所以不同指令集差距缩小。
**我没有多核实测数据** (沙箱只有 1 个核), 多线程只验证了正确性 (1/4/8 线程输出一致)。
HyMT2Sharp README 的 decode 约 47 tok/s 是 8 线程 Ryzen 7 5800X; 本库单线程约 10 tok/s,
线性外推 8 核大约在同一量级, 但**请以你机器上的 `hymt2_cli` 输出为准**。

## 没做的 (如实说明)

* **CUDA、Metal 和 NPU (ARM NPU / Hexagon / ANE)**: 没有做。沙箱没有 NVIDIA GPU / nvcc / macOS / 厂商 SDK, 无法编译验证, 我不交付没跑过的代码。
  Vulkan 已实现 (见上), 但只在软件设备上验证过。
* ARM 上 K-quants 的 NEON 内核、`dotprod` 的运行时检测 (目前是编译期开关)。
* 真实 Q4_K_M 文件、真实 J4125、macOS / iOS / Android 真机、真实 GPU 上的运行与速度。
* 预编译包里的 Windows 文件没有在 Windows 上运行过 (见 `prebuilt/TEST_GUIDE.md`)。
