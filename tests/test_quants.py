#!/usr/bin/env python3
"""量化内核对拍: 用 gguf-py (llama.cpp 官方 Python 包) 的独立反量化实现作参考,
检查 Q4_K / Q6_K / Q8_0 的 (1) 反量化 (2) 与 Q8_K 激活的矩阵乘, 并检查各 ISA 路径。
用法: HYMT2_ISA=avx2 python3 tests/test_quants.py build/libhymt2.so"""
import ctypes, sys, os
import numpy as np
from gguf.quants import dequantize
from gguf import GGMLQuantizationType as T

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "./libhymt2.so")
lib.hymt2_isa.restype = ctypes.c_char_p
lib.hymt2_debug_dequant_row.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.hymt2_debug_matmul.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
rng = np.random.default_rng(3)
ROWS, K, NT = 5, 1536, 3
f32 = np.float32

def f16b(v): return np.float16(v).tobytes()
def make_raw(t):
    nb = K // 256 if t != T.Q8_0 else K // 32
    rows = []
    for r in range(ROWS):
        blocks = []
        for b in range(nb):
            if t == T.Q4_K:
                blk = bytearray(rng.integers(0, 256, 144, dtype=np.uint8).tobytes()); blk[0:2] = f16b(0.004 + 0.01 * rng.random()); blk[2:4] = f16b(0.002 + 0.01 * rng.random())
            elif t == T.Q6_K:
                blk = bytearray(rng.integers(0, 256, 210, dtype=np.uint8).tobytes()); blk[208:210] = f16b(0.0008 + 0.001 * rng.random())
                for j in range(192, 208): blk[j] = int(rng.integers(-24, 25)) & 0xFF
            else:  # Q8_0
                blk = bytearray(rng.integers(0, 256, 34, dtype=np.uint8).tobytes()); blk[0:2] = f16b(0.01 + 0.02 * rng.random())
            blocks.append(bytes(blk))
        rows.append(b"".join(blocks))
    return np.frombuffer(b"".join(rows), dtype=np.uint8).copy()

def q8k_roundtrip(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1, 256); out = np.zeros_like(x)
    for i, blk in enumerate(x):
        j = int(np.argmax(np.abs(blk))); mx = blk[j]
        if np.abs(blk).max() == 0: continue
        iscale = f32(-127.0) / f32(mx)
        q = np.clip(np.rint(blk * iscale), -127, 127).astype(np.float32)
        out[i] = q * (f32(1.0) / iscale)
    return out.reshape(-1)

fails = 0
print("isa:", lib.hymt2_isa().decode())
for t in (T.Q4_K, T.Q6_K, T.Q8_0):
    raw = make_raw(t)
    ref = dequantize(raw.reshape(ROWS, -1), t).astype(np.float32)          # (ROWS, K) 独立参考
    # 1. 反量化逐行对拍
    worst_d = 0.0
    for r in range(ROWS):
        dst = np.zeros(K, dtype=np.float32)
        lib.hymt2_debug_dequant_row(int(t), raw.ctypes.data, K, r, dst.ctypes.data)
        worst_d = max(worst_d, float(np.abs(dst - ref[r]).max()))
    # 2. 矩阵乘对拍 (K-quants 用 Q8_K 激活, Q8_0 用浮点激活)
    x = (rng.standard_normal((NT, K)) * np.where(rng.random((NT, K)) < 0.02, 6.0, 1.0)).astype(np.float32)
    out = np.zeros((NT, ROWS), dtype=np.float32)
    lib.hymt2_debug_matmul(int(t), raw.ctypes.data, ROWS, K, x.ctypes.data, NT, out.ctypes.data, 0)
    xe = x if t == T.Q8_0 else np.stack([q8k_roundtrip(row) for row in x])
    want = xe.astype(np.float64) @ ref.astype(np.float64).T
    mag = np.abs(xe.astype(np.float64)) @ np.abs(ref.astype(np.float64)).T
    worst_m = float((np.abs(out - want) / (mag + 1e-6)).max())
    ok = worst_d < 1e-6 * max(1.0, float(np.abs(ref).max())) and worst_m < 1e-5
    print("%-5s dequant max|C-ref| = %.2e   matmul err/Σ|term| = %.2e   %s" % (t.name, worst_d, worst_m, "ok" if ok else "FAIL"))
    if not ok: fails += 1
print("ALL QUANT TESTS PASSED" if not fails else "%d FAILURES" % fails)
sys.exit(1 if fails else 0)
