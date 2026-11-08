#!/usr/bin/env python3
"""Metal 对拍: HYMT2_DEVICE=metal 路径 vs CPU。用法: python3 tests/test_metal.py ./libhymt2.dylib"""
import ctypes, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "./libhymt2.dylib")
sig = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.hymt2_debug_device_matmul.argtypes = sig
lib.hymt2_debug_matmul.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
rng = np.random.default_rng(11)

def f16b(v): return np.float16(v).tobytes()

def raw_blocks(gt, rows, k):
    out = bytearray()
    if gt == 43:
        for _ in range(rows * (k // 256)): out += rng.integers(0, 256, 40, dtype=np.uint8).tobytes() + f16b(0.01 + 0.03 * rng.random())
    elif gt == 40:
        for _ in range(rows * (k // 512)): out += f16b(0.01 + 0.03 * rng.random()) + rng.integers(0, 256, 128, dtype=np.uint8).tobytes()
    elif gt == 12:
        for _ in range(rows * (k // 256)):
            b = bytearray(rng.integers(0, 256, 144, dtype=np.uint8).tobytes()); b[0:2] = f16b(0.004 + 0.01 * rng.random()); b[2:4] = f16b(0.002 + 0.01 * rng.random()); out += b
    elif gt == 14:
        for _ in range(rows * (k // 256)):
            b = bytearray(rng.integers(0, 256, 210, dtype=np.uint8).tobytes()); b[208:210] = f16b(0.0008 + 0.001 * rng.random())
            for j in range(192, 208): b[j] = int(rng.integers(-24, 25)) & 0xFF
            out += b
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()

names = {43: "STQ1_0", 40: "Q2_0C", 12: "Q4_K", 14: "Q6_K"}
fails = 0
K = 1536
for gt in (43, 40, 12, 14):
    for rows, nt in ((37, 1), (64, 3)):
        raw = raw_blocks(gt, rows, K)
        x = (rng.standard_normal((nt, K)) * np.where(rng.random((nt, K)) < 0.02, 6.0, 1.0)).astype(np.float32)
        cpu = np.zeros((nt, rows), dtype=np.float32)
        gpu = np.zeros((nt, rows), dtype=np.float32)
        assert lib.hymt2_debug_matmul(gt, raw.ctypes.data, rows, K, x.ctypes.data, nt, cpu.ctypes.data, 1) == 0
        rc = lib.hymt2_debug_device_matmul(b"metal", gt, raw.ctypes.data, rows, K, x.ctypes.data, nt, gpu.ctypes.data)
        if rc != 0:
            print("%-6s rows=%3d nt=%2d: Metal 失败 rc=%d" % (names[gt], rows, nt, rc)); fails += 1; continue
        err = float(np.abs(gpu - cpu).max() / max(np.abs(cpu).max(), 1e-6))
        ok = err < 2e-4
        if not ok: fails += 1
        print("%-6s rows=%3d nt=%2d max rel diff Metal vs CPU = %.2e  %s" % (names[gt], rows, nt, err, "ok" if ok else "FAIL"))
print("FAILS" if fails else "ALL METAL TESTS PASSED")
sys.exit(1 if fails else 0)
