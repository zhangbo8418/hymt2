#!/usr/bin/env python3
"""Vulkan shader 对拍: GPU (默认用 lavapipe 软件设备) 与 CPU 内核在随机权重上逐项比较。
覆盖 STQ1_0 / Q2_0C (面板 shader)、Q4_K、Q6_K, 以及权重按行分块 (HYMT2_VK_MAXRANGE 强制很小)。
用法: python3 tests/test_vulkan.py ./libhymt2.so"""
import ctypes, os, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "./libhymt2.so")
sig = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.hymt2_debug_vk_matmul.argtypes = sig
lib.hymt2_debug_matmul.argtypes = sig + [ctypes.c_int]
rng = np.random.default_rng(11)
def f16b(v): return np.float16(v).tobytes()

def raw_blocks(gt, rows, k):
    out = bytearray()
    if gt == 43:     # STQ1_0: qs[32] sign[8] d(f16)
        for _ in range(rows * (k // 256)): out += rng.integers(0, 256, 40, dtype=np.uint8).tobytes() + f16b(0.01 + 0.03 * rng.random())
    elif gt == 40:   # Q2_0C: d(f16) qs[128]
        for _ in range(rows * (k // 512)): out += f16b(0.01 + 0.03 * rng.random()) + rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    elif gt == 12:   # Q4_K: d dmin scales[12] qs[128]
        for _ in range(rows * (k // 256)):
            b = bytearray(rng.integers(0, 256, 144, dtype=np.uint8).tobytes()); b[0:2] = f16b(0.004 + 0.01 * rng.random()); b[2:4] = f16b(0.002 + 0.01 * rng.random()); out += b
    elif gt == 14:   # Q6_K: ql[128] qh[64] scales[16] d(f16)
        for _ in range(rows * (k // 256)):
            b = bytearray(rng.integers(0, 256, 210, dtype=np.uint8).tobytes()); b[208:210] = f16b(0.0008 + 0.001 * rng.random())
            for j in range(192, 208): b[j] = int(rng.integers(-24, 25)) & 0xFF
            out += b
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()

names = {43: "STQ1_0", 40: "Q2_0C", 12: "Q4_K", 14: "Q6_K"}
fails = 0
K = 1536
for maxrange in (None, "40000"):          # None: 设备默认上限; 40000: 强制多分块
    if maxrange: os.environ["HYMT2_VK_MAXRANGE"] = maxrange
    else: os.environ.pop("HYMT2_VK_MAXRANGE", None)
    for gt in (43, 40, 12, 14):
        for rows, nt in ((37, 1), (129, 3), (64, 16)):
            raw = raw_blocks(gt, rows, K)
            x = (rng.standard_normal((nt, K)) * np.where(rng.random((nt, K)) < 0.02, 6.0, 1.0)).astype(np.float32)
            cpu = np.zeros((nt, rows), dtype=np.float32); gpu = np.zeros((nt, rows), dtype=np.float32)
            assert lib.hymt2_debug_matmul(gt, raw.ctypes.data, rows, K, x.ctypes.data, nt, cpu.ctypes.data, 1) == 0
            rc = lib.hymt2_debug_vk_matmul(gt, raw.ctypes.data, rows, K, x.ctypes.data, nt, gpu.ctypes.data)
            if rc != 0:
                print("%-6s rows=%3d nt=%2d maxrange=%s: GPU 调用失败 rc=%d" % (names[gt], rows, nt, maxrange, rc)); fails += 1; continue
            mag = np.abs(x.astype(np.float64)).max() * np.abs(cpu).max() + 1e-6
            err = float(np.abs(gpu - cpu).max() / max(np.abs(cpu).max(), 1e-6))
            ok = err < 2e-4
            if not ok: fails += 1
            print("%-6s rows=%3d nt=%2d maxrange=%-6s max rel diff GPU vs CPU = %.2e  %s" % (names[gt], rows, nt, maxrange, err, "ok" if ok else "FAIL"))
print("ALL VULKAN TESTS PASSED" if not fails else "%d FAILURES" % fails)
sys.exit(1 if fails else 0)
