#!/usr/bin/env python3
"""打印 GGUF 的元数据和张量类型统计 (不加载权重)。"""
import struct, sys, collections
f = open(sys.argv[1], "rb")
def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
def rstr(): n, = rd("Q"); return f.read(n).decode("utf-8", "replace")
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at, n = rd("IQ")
        if at == 8:
            return ("ARR[str]", n, [rstr() for _ in range(min(n, 6))] + ([None] * 0), [rstr() for _ in range(n - min(n, 6))] and None)
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at]
        f.seek(sz * n, 1); return ("ARR[%d]" % at, n)
    fmt = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"B",10:"Q",11:"q",12:"d"}[t]
    return rd(fmt)[0]
magic = f.read(4); ver, nt, nkv = rd("IQQ")
print("magic", magic, "version", ver, "tensors", nt, "kv", nkv)
for _ in range(nkv):
    k = rstr(); t, = rd("I"); v = rval(t)
    if isinstance(v, tuple) and v[0] == "ARR[str]":
        v = "string array, n=%d, first: %s" % (v[1], v[2])
    print("  %-48s %s" % (k, v if not isinstance(v, str) or len(v) < 120 else v[:117] + "..."))
types = collections.Counter(); ex = {}
for _ in range(nt):
    name = rstr(); nd, = rd("I"); dims = rd("%dQ" % nd); ty, off = rd("IQ")
    types[ty] += 1; ex.setdefault(ty, []).append((name, dims))
print("\ntensor types (ggml type id -> count):")
for ty, c in sorted(types.items()): print("  type %2d : %3d tensors   e.g. %s %s" % (ty, c, ex[ty][0][0], ex[ty][0][1]))
