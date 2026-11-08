#!/usr/bin/env python3
"""Quick generate throughput bench. Usage: HYMT2_DEVICE=cpu python3 tests/bench_generate.py ./libhymt2.dylib [model]"""
import ctypes, os, sys, time

lib = ctypes.CDLL(sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "../../../models/Hy-MT2-1.8B-1.25Bit.gguf")
lib.hymt2_model_load.restype = ctypes.c_void_p
lib.hymt2_model_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.hymt2_model_generate.restype = ctypes.c_int
lib.hymt2_model_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.hymt2_model_stats.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
lib.hymt2_isa.restype = ctypes.c_char_p
lib.hymt2_model_free.argtypes = [ctypes.c_void_p]

err = (ctypes.c_char * 512)()
m = lib.hymt2_model_load(os.path.abspath(model).encode(), 0, 512, err, 512)
if not m:
    raise SystemExit(err.value.decode())
out = (ctypes.c_char * 4096)()
prompt = "Translate to English: hello world"
lib.hymt2_model_generate(m, prompt.encode(), out, 4096, 4, 0)
t0 = time.perf_counter()
n = lib.hymt2_model_generate(m, prompt.encode(), out, 4096, 32, 0)
wall = time.perf_counter() - t0
pt = ctypes.c_int(); ct = ctypes.c_int(); pm = ctypes.c_double()
dt = ctypes.c_int(); dm = ctypes.c_double()
lib.hymt2_model_stats(m, ctypes.byref(pt), ctypes.byref(ct), ctypes.byref(pm), ctypes.byref(dt), ctypes.byref(dm))
toks, dms = dt.value, dm.value
print(f"isa={lib.hymt2_isa().decode()} device={os.environ.get('HYMT2_DEVICE','auto')} "
      f"decode={toks} tok in {dms:.1f}ms = {1000*toks/dms if dms else 0:.2f} tok/s "
      f"wall={wall:.2f}s bytes={n}")
lib.hymt2_model_free(m)
