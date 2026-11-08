#!/usr/bin/env python3
"""端到端自测 (无需真实权重):
  1. 生成一个小型合成 GGUF (hunyuan-dense 结构, STQ1_0 + Q6_K 权重, 字节级 BPE 词表)
  2. 用独立的 numpy 实现算 logits, 与 C 库逐项对拍
  3. 用 python-regex + 独立 BPE 实现对拍 C 的分词器
  4. 批量 prefill 与逐 token 解码结果一致 (验证批量内核 / KV cache)
  5. 前缀缓存: 同一前缀的两次 generate 输出与冷启动一致
"""
import ctypes, os, struct, sys, random, subprocess, tempfile
import numpy as np
import regex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rng = np.random.default_rng(7)
random.seed(7)

# ---------------------------------------------------------------- 常量 / 配置
CB = [0xA9,0x89,0x29,0x09,0xA6,0x86,0x26,0x06,0x9A,0x92,0x1A,0x12,0x6A,0x62,0x4A,0x42,
      0x01,0x21,0x81,0xA1,0x04,0x24,0x84,0xA4,0x10,0x18,0x90,0x98,0x40,0x48,0x60,0x68]
MODE = os.environ.get("TEST_MODE", "stq")   # stq | q4km | q2
if MODE == "q2": HID, HEADS, KVH, HD, FFN, NL = 512, 8, 2, 64, 1024, 2   # Q2_0C 要求输入维是 512 的倍数
else: HID, HEADS, KVH, HD, FFN, NL = 256, 4, 2, 64, 512, 2
QD, KD = HEADS * HD, KVH * HD
EPS, BASE, ALPHA = 1e-5, 10000.0, 3.0
BOS_S = "<｜hy_begin▁of▁sentence｜>"; USER_S = "<｜hy_User｜>"; ASSIST_S = "<｜hy_Assistant｜>"
EOG_S = "<｜hy_place▁holder▁no▁2｜>"; SYS_END_S = "<｜hy_place▁holder▁no▁3｜>"

# ---------------------------------------------------------------- 字节编码 (与 BpeTokenizer.cs 相同)
def enc_byte(b):
    if b == 0xAD: return 0x143
    if b <= 0x20: return b + 0x100
    if 0x7F <= b <= 0xA0: return b + 0xA2
    return b
BYTE_CH = [chr(enc_byte(b)) for b in range(256)]
def byte_encode(s): return "".join(BYTE_CH[b] for b in s.encode("utf-8"))

PRE = (r"\p{N}{1,3}|" r"[一-龥぀-ゟ゠-ヿ]+|" r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|"
       r"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|" r" ?[\p{P}\p{S}]+[\r\n]*|" r"\s*[\r\n]+|\s+(?!\S)|\s+")
PRE_RE = regex.compile(PRE)

# ---------------------------------------------------------------- 训练一个小 BPE
CORPUS = ["Translate the following text into Chinese. Note that you should only output the translated result without any additional explanation:",
          "Hello world, this is a test of the byte level tokenizer. 今天天气很好，我们一起去公园玩。 こんにちは世界。",
          "The quick brown fox jumps over the lazy dog 12345 67890 2026-11-03.", "line one\nline two\n\n  indented  text   with   spaces"]
def train_bpe(nmerges):
    words = {}
    for line in CORPUS:
        for m in PRE_RE.finditer(line):
            w = tuple(byte_encode(m.group()))
            words[w] = words.get(w, 0) + 1
    merges = []
    for _ in range(nmerges):
        cnt = {}
        for w, f in words.items():
            for a, b in zip(w, w[1:]): cnt[(a, b)] = cnt.get((a, b), 0) + f
        if not cnt: break
        best = max(cnt, key=lambda k: (cnt[k], k))
        merges.append(best)
        nw = {}
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best: out.append(w[i] + w[i + 1]); i += 2
                else: out.append(w[i]); i += 1
            nw[tuple(out)] = nw.get(tuple(out), 0) + f
        words = nw
    return merges
MERGES = train_bpe(160)
TOKENS = BYTE_CH[:]            # 0..255: 单字节
for a, b in MERGES: TOKENS.append(a + b)
SPECIALS = [BOS_S, USER_S, ASSIST_S, EOG_S, SYS_END_S]
FIRST_SPECIAL = len(TOKENS)
TOKENS += SPECIALS
BOS_ID = FIRST_SPECIAL; EOG_ID = FIRST_SPECIAL + 3
VOCAB = len(TOKENS)
TOK2ID = {t: i for i, t in enumerate(TOKENS)}
RANK = {(a, b): i for i, (a, b) in enumerate(MERGES)}

def py_encode(text):
    frags = [(text, None)]
    for sp in sorted(SPECIALS, key=len, reverse=True):
        nxt = []
        for s, pre in frags:
            if pre is not None: nxt.append((s, pre)); continue
            parts = s.split(sp)
            for i, p in enumerate(parts):
                if p: nxt.append((p, None))
                if i + 1 < len(parts): nxt.append((sp, TOK2ID[sp]))
        frags = nxt
    ids = []
    for s, pre in frags:
        if pre is not None: ids.append(pre); continue
        for m in PRE_RE.finditer(s):
            w = byte_encode(m.group())
            if w in TOK2ID: ids.append(TOK2ID[w]); continue
            sym = list(w)
            while len(sym) > 1:
                best, bi = None, -1
                for i in range(len(sym) - 1):
                    r = RANK.get((sym[i], sym[i + 1]))
                    if r is not None and (best is None or r < best): best, bi = r, i
                if bi < 0: break
                sym[bi:bi + 2] = [sym[bi] + sym[bi + 1]]
            ids += [TOK2ID.get(x, 0) for x in sym]
    return ids

# ---------------------------------------------------------------- 量化格式 (Python 侧, 与 C 独立实现)
def f16(x): return np.float16(x)
def make_stq(n_rows, k):
    nb = k // 256
    qs = rng.integers(0, 256, (n_rows, nb, 32), dtype=np.uint8)
    sg = rng.integers(0, 256, (n_rows, nb, 8), dtype=np.uint8)
    d = (0.015 + 0.03 * rng.random((n_rows, nb))).astype(np.float16)
    raw = bytearray()
    for r in range(n_rows):
        for b in range(nb):
            raw += qs[r, b].tobytes() + sg[r, b].tobytes() + d[r, b].tobytes()
    return bytes(raw), (qs, sg, d)
def deq_stq(parts):
    qs, sg, d = parts
    n_rows, nb = d.shape
    out = np.zeros((n_rows, nb * 256), dtype=np.float32)
    for r in range(n_rows):
        for b in range(nb):
            for g in range(64):
                slot = (int(qs[r, b, g >> 1]) >> ((g & 1) * 4)) & 15
                sign = (int(sg[r, b, g >> 3]) >> (g & 7)) & 1
                qp = CB[(sign << 4) | slot]
                chunk, gloc = g >> 4, g & 15
                for lane in range(4):
                    out[r, b * 256 + chunk * 64 + gloc + lane * 16] = (((qp >> (lane * 2)) & 3) - 1) * np.float32(d[r, b])
    return out
def make_q6k(n_rows, k, tiny_from=None):
    nb = k // 256
    raw = bytearray(); parts = []
    for r in range(n_rows):
        for b in range(nb):
            ql = rng.integers(0, 256, 128, dtype=np.uint8); qh = rng.integers(0, 256, 64, dtype=np.uint8)
            sc = rng.integers(-24, 25, 16, dtype=np.int8); d = np.float16(0.0006 + 0.0008 * rng.random())
            if tiny_from is not None and r >= tiny_from: d = np.float16(1e-6)   # 特殊 token 的 logit 压到 ~0, 避免合成模型一直生成它们
            raw += ql.tobytes() + qh.tobytes() + sc.tobytes() + d.tobytes(); parts.append((ql, qh, sc, d))
    return bytes(raw), parts
def deq_q6k(parts, n_rows, k):
    nb = k // 256
    out = np.zeros((n_rows, k), dtype=np.float32)
    for r in range(n_rows):
        for b in range(nb):
            ql, qh, sc, d = parts[r * nb + b]
            y = np.zeros(256, dtype=np.float32)
            for n in range(0, 256, 128):
                qlo, qho, sco = ql[n // 2:], qh[n // 4:], sc[n // 16:]
                for l in range(32):
                    is_ = l // 16
                    q1 = (int(qlo[l]) & 15) | (((int(qho[l]) >> 0) & 3) << 4)
                    q2 = (int(qlo[l + 32]) & 15) | (((int(qho[l]) >> 2) & 3) << 4)
                    q3 = (int(qlo[l]) >> 4) | (((int(qho[l]) >> 4) & 3) << 4)
                    q4 = (int(qlo[l + 32]) >> 4) | (((int(qho[l]) >> 6) & 3) << 4)
                    y[n + l] = np.float32(d) * int(sco[is_]) * (q1 - 32)
                    y[n + l + 32] = np.float32(d) * int(sco[is_ + 2]) * (q2 - 32)
                    y[n + l + 64] = np.float32(d) * int(sco[is_ + 4]) * (q3 - 32)
                    y[n + l + 96] = np.float32(d) * int(sco[is_ + 6]) * (q4 - 32)
            out[r, b * 256:(b + 1) * 256] = y
    return out

from gguf.quants import dequantize as gdeq
from gguf import GGMLQuantizationType as GT
def make_kq(gt, n_rows, k, tiny_from=None):   # Q4_K: 随机块 + gguf-py 的独立反量化作参考
    nb = k // 256; bs = {GT.Q4_K: 144}[gt]; raw = bytearray()
    for r in range(n_rows):
        for b in range(nb):
            blk = bytearray(rng.integers(0, 256, bs, dtype=np.uint8).tobytes())
            tiny = tiny_from is not None and r >= tiny_from
            blk[0:2] = np.float16(1e-6 if tiny else 0.003 + 0.004 * rng.random()).tobytes()
            blk[2:4] = np.float16(1e-6 if tiny else 0.001 + 0.004 * rng.random()).tobytes()
            raw += blk
    raw = bytes(raw)
    return raw, gdeq(np.frombuffer(raw, dtype=np.uint8).reshape(n_rows, -1), gt).astype(np.float32)
def make_q2(n_rows, k):   # Q2_0C: 512 权重一个块 [f16 d][128 字节 2-bit 码], 值 = (2c-3)*d
    nb = k // 512; raw = bytearray(); deq = np.zeros((n_rows, k), dtype=np.float32)
    for r in range(n_rows):
        for b in range(nb):
            d = np.float16(0.02 + 0.03 * rng.random()); qs = rng.integers(0, 256, 128, dtype=np.uint8)
            raw += d.tobytes() + qs.tobytes()
            for j in range(512): deq[r, b * 512 + j] = (2 * ((int(qs[j >> 2]) >> ((j & 3) * 2)) & 3) - 3) * np.float32(d)
    return bytes(raw), deq

# ---------------------------------------------------------------- GGUF 写出
def w_str(s): b = s.encode(); return struct.pack("<Q", len(b)) + b
def kv(key, typ, val):
    out = w_str(key) + struct.pack("<I", typ)
    if typ == 8: out += w_str(val)
    elif typ == 4: out += struct.pack("<I", val)
    elif typ == 5: out += struct.pack("<i", val)
    elif typ == 6: out += struct.pack("<f", val)
    return out
def kv_arr_str(key, items):
    return w_str(key) + struct.pack("<IIQ", 9, 8, len(items)) + b"".join(w_str(s) for s in items)
def kv_arr_i32(key, items):
    return w_str(key) + struct.pack("<IIQ", 9, 5, len(items)) + struct.pack("<%di" % len(items), *items)

def build_model(path):
    A = "hunyuan-dense"
    kvs = [kv("general.architecture", 8, A), kv("general.name", 8, "synthetic-test"), kv("general.file_type", 4, 0),
           kv(f"{A}.embedding_length", 4, HID), kv(f"{A}.block_count", 4, NL), kv(f"{A}.feed_forward_length", 4, FFN),
           kv(f"{A}.attention.head_count", 4, HEADS), kv(f"{A}.attention.head_count_kv", 4, KVH),
           kv(f"{A}.attention.key_length", 4, HD), kv(f"{A}.rope.dimension_count", 4, HD),
           kv(f"{A}.rope.freq_base", 6, BASE), kv(f"{A}.rope.scaling.alpha", 6, ALPHA),
           kv(f"{A}.attention.layer_norm_rms_epsilon", 6, EPS),
           kv("tokenizer.ggml.model", 8, "gpt2"), kv("tokenizer.ggml.pre", 8, "hunyuan-dense"),
           kv_arr_str("tokenizer.ggml.tokens", TOKENS), kv_arr_str("tokenizer.ggml.merges", [a + " " + b for a, b in MERGES]),
           kv_arr_i32("tokenizer.ggml.token_type", [1] * FIRST_SPECIAL + [3] * len(SPECIALS)),
           kv("tokenizer.ggml.bos_token_id", 4, BOS_ID), kv("tokenizer.ggml.eos_token_id", 4, EOG_ID + 1),
           kv("tokenizer.ggml.unknown_token_id", 4, 0)]
    tensors = []   # (name, dims, type, bytes)
    ref = {}
    if MODE == "q4km":   # 词嵌入/输出头用 Q4_K, 覆盖 dequant_row 和输出头的 Q4_K 路径
        raw, ref["embd"] = make_kq(GT.Q4_K, VOCAB, HID, tiny_from=FIRST_SPECIAL)
        tensors.append(("token_embd.weight", [HID, VOCAB], 12, raw))
    else:
        raw, parts = make_q6k(VOCAB, HID, tiny_from=FIRST_SPECIAL)
        tensors.append(("token_embd.weight", [HID, VOCAB], 14, raw)); ref["embd"] = deq_q6k(parts, VOCAB, HID)
    def vec(name, n, base=1.0, jitter=0.2):
        v = (base + jitter * rng.standard_normal(n)).astype(np.float32); tensors.append((name, [n], 0, v.tobytes())); return v
    ref["out_norm"] = vec("output_norm.weight", HID)
    ref["layers"] = []
    def stq(name, k, n):
        if MODE == "q2":
            raw, deq = make_q2(n, k); tensors.append((name, [k, n], 40, raw)); return deq
        if MODE == "q4km":   # Q4_K_M: 大多数 Q4_K, attn_v 和 ffn_down 用 Q6_K
            if "attn_v" in name or "ffn_down" in name:
                raw, parts = make_q6k(n, k); tensors.append((name, [k, n], 14, raw)); return deq_q6k(parts, n, k)
            raw, deq = make_kq(GT.Q4_K, n, k); tensors.append((name, [k, n], 12, raw)); return deq
        raw, parts = make_stq(n, k); tensors.append((name, [k, n], 43, raw)); return deq_stq(parts)
    for i in range(NL):
        L = {}
        L["attn_norm"] = vec(f"blk.{i}.attn_norm.weight", HID); L["ffn_norm"] = vec(f"blk.{i}.ffn_norm.weight", HID)
        L["q_norm"] = vec(f"blk.{i}.attn_q_norm.weight", HD); L["k_norm"] = vec(f"blk.{i}.attn_k_norm.weight", HD)
        L["wq"] = stq(f"blk.{i}.attn_q.weight", HID, QD); L["wk"] = stq(f"blk.{i}.attn_k.weight", HID, KD)
        L["wv"] = stq(f"blk.{i}.attn_v.weight", HID, KD); L["wo"] = stq(f"blk.{i}.attn_output.weight", QD, HID)
        L["wg"] = stq(f"blk.{i}.ffn_gate.weight", HID, FFN); L["wu"] = stq(f"blk.{i}.ffn_up.weight", HID, FFN)
        L["wd"] = stq(f"blk.{i}.ffn_down.weight", FFN, HID)
        ref["layers"].append(L)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    infos, off, datas = b"", 0, []
    for name, dims, typ, raw in tensors:
        infos += w_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off)
        pad = (-len(raw)) % 32
        datas.append(raw + b"\0" * pad); off += len(raw) + pad
    blob = head + infos
    blob += b"\0" * ((-len(blob)) % 32)
    with open(path, "wb") as f: f.write(blob + b"".join(datas))
    return ref

# ---------------------------------------------------------------- numpy 参考前向
f32 = np.float32
def bf16_round(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)
def q8k_roundtrip(x):  # 量化再反量化 (与 C 的 Q8_K 激活量化一致)
    x = np.asarray(x, dtype=np.float32).reshape(-1, 256); out = np.zeros_like(x)
    for i, blk in enumerate(x):
        j = int(np.argmax(np.abs(blk))); mx = blk[j]
        if mx == 0 and np.abs(blk).max() == 0: continue
        iscale = f32(-127.0) / f32(mx)
        q = np.clip(np.rint(blk * iscale), -127, 127).astype(np.float32)
        out[i] = q * (f32(1.0) / iscale)
    return out.reshape(-1)
def rms(x, w): return (x * (f32(1) / np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS).astype(np.float32)) * w).astype(np.float32)
def mm(W, x): return (W.astype(np.float64) @ q8k_roundtrip(x).astype(np.float64)).astype(np.float32)
def ref_logits(ref, ids):
    base = f32(BASE * ALPHA ** (HD / (HD - 2)))
    half = HD // 2
    freq = (f32(1) / np.power(base, np.arange(half, dtype=np.float32) / f32(half))).astype(np.float32)
    kc = [[] for _ in range(NL)]; vc = [[] for _ in range(NL)]
    last = None
    for pos, t in enumerate(ids):
        h = ref["embd"][t].copy()
        ang = f32(pos) * freq; c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        for li, L in enumerate(ref["layers"]):
            xn = rms(h, L["attn_norm"]); q = mm(L["wq"], xn); k = mm(L["wk"], xn); v = mm(L["wv"], xn)
            q = q.reshape(HEADS, HD).copy(); k = k.reshape(KVH, HD).copy()
            for arr in (q, k):
                x0, x1 = arr[:, :half].copy(), arr[:, half:].copy()
                arr[:, :half] = x0 * c - x1 * s; arr[:, half:] = x0 * s + x1 * c
            q = np.stack([rms(r, L["q_norm"]) for r in q]); k = np.stack([rms(r, L["k_norm"]) for r in k])
            kc[li].append(bf16_round(k)); vc[li].append(bf16_round(v.reshape(KVH, HD)))
            K = np.stack(kc[li]); V = np.stack(vc[li])  # [T, KVH, HD]
            ao = np.zeros((HEADS, HD), dtype=np.float32)
            for hh in range(HEADS):
                kvh = hh // (HEADS // KVH)
                sc = (K[:, kvh, :].astype(np.float64) @ q[hh].astype(np.float64)) / np.sqrt(HD)
                p = np.exp(sc - sc.max()); p /= p.sum()
                ao[hh] = (p @ V[:, kvh, :].astype(np.float64)).astype(np.float32)
            h = h + mm(L["wo"], ao.reshape(-1))
            xn = rms(h, L["ffn_norm"]); g = mm(L["wg"], xn); u = mm(L["wu"], xn)
            a = (g / (f32(1) + np.exp(-g)) * u).astype(np.float32)
            h = h + mm(L["wd"], a)
        last = h
    xn = rms(last, ref["out_norm"])
    return mm(ref["embd"], xn)   # 输出头与 token_embd 共用 (Q6_K × Q8_K)

# ---------------------------------------------------------------- 主流程
def main():
    lib_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "libhymt2.so")
    lib = ctypes.CDLL(lib_path)
    lib.hymt2_model_load.restype = ctypes.c_void_p
    lib.hymt2_model_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.hymt2_debug_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.hymt2_debug_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.hymt2_debug_logits_stepwise.restype = ctypes.POINTER(ctypes.c_float)
    lib.hymt2_debug_logits_stepwise.argtypes = lib.hymt2_debug_logits.argtypes
    lib.hymt2_debug_encode.restype = ctypes.c_int
    lib.hymt2_debug_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int))]
    lib.hymt2_model_generate.restype = ctypes.c_int
    lib.hymt2_model_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.hymt2_isa.restype = ctypes.c_char_p
    print("isa:", lib.hymt2_isa().decode(), "| mode:", MODE, "| repack:", os.environ.get("HYMT2_REPACK", "auto"))

    tmp = os.path.join(tempfile.gettempdir(), "hymt2_synth.gguf")
    ref = build_model(tmp)
    err = ctypes.create_string_buffer(512)
    m = lib.hymt2_model_load(tmp.encode(), int(os.environ.get("TEST_THREADS", "2")), 256, err, 512)
    assert m, err.value.decode()
    fails = 0

    # ---- 1. 分词器对拍
    texts = ["Hello world", "Translate the following text into Chinese.", "今天天气很好，我们一起去公园玩。", "こんにちは世界。",
             "mixed 中英文 text 123456789 and 2026-11-03!", "line one\nline two\n\n  indented  text   with   spaces  ",
             "  leading and trailing   ", "a.b,c;d:e?f!g'h\"i(j)k", "e.g. don't stop; it's 5.5% 😀 done", "price: $1,234.56\r\nnext\r\n\r\n",
             "x" * 40, "…— “quotes” ‘single’ «guillemets»", "tab\tseparated\tvalues", "<｜hy_User｜>hi<｜hy_Assistant｜>",
             "ÀÉÎõü café naïve", ".hello !world @abc #tag", "١٢٣ ٤٥٦ ²³ ½"]
    for s in texts:
        pi = ctypes.POINTER(ctypes.c_int)()
        n = lib.hymt2_debug_encode(m, s.encode(), ctypes.byref(pi))
        got = [pi[i] for i in range(n)]
        want = py_encode(s)
        if "😀" in s:   # C 按码点处理 emoji (C# 按 UTF-16 会丢弃); python 参考同样按码点
            pass
        if got != want:
            fails += 1; print("TOKENIZER MISMATCH:", repr(s)); print("   C :", got); print("   py:", want)
    print("tokenizer: %d strings compared" % len(texts))

    # ---- 2. 前向对拍 (批量 prefill, 序列长度 > MAX_BATCH 以覆盖分块)
    prompt = BOS_S + USER_S + "Translate the following text into Chinese. Hello world, 今天天气很好。" + ASSIST_S
    ids = py_encode(prompt)
    print("prompt tokens:", len(ids))
    arr = (ctypes.c_int * len(ids))(*ids); vo = ctypes.c_int()
    p = lib.hymt2_debug_logits(m, arr, len(ids), ctypes.byref(vo))
    cl = np.array([p[i] for i in range(vo.value)], dtype=np.float32)
    rl = ref_logits(ref, ids)[:vo.value]
    # 参考自身的"舍入分散度": 把参考里的矩阵乘从 float64 换成 float32, 看它与自己差多少。
    # (激活量化 Q8_K 的舍入边界翻转会被深层网络放大, 个别序列非常敏感; C 库的误差应在这个分散度的量级内)
    global mm
    _mm = mm
    mm = lambda W, x: (W.astype(np.float32) @ q8k_roundtrip(x).astype(np.float32)).astype(np.float32)
    rl32 = ref_logits(ref, ids)[:vo.value]
    mm = _mm
    diff = np.abs(cl - rl).max(); scale = np.abs(rl).max()
    spread = np.abs(rl - rl32).max() / scale
    tol = max(1e-5, 3 * spread)
    print("logits: max|C - numpy| = %.3e  (max|logit| = %.3f, rel %.2e)  参考自身舍入分散度 %.2e -> 容差 %.2e  argmax C=%d ref=%d" % (diff, scale, diff / scale, spread, tol, cl.argmax(), rl.argmax()))
    if diff / scale > tol or cl.argmax() != rl.argmax(): fails += 1; print("LOGITS MISMATCH")
    cos = float(np.dot(cl, rl) / (np.linalg.norm(cl) * np.linalg.norm(rl)))
    print("logits cosine similarity = %.8f" % cos)
    if 1 - cos > max(1e-7, 3 * spread * spread * 10): fails += 1; print("COSINE TOO LOW")

    # ---- 3. 批量 prefill vs 逐 token
    p2 = lib.hymt2_debug_logits_stepwise(m, arr, len(ids), ctypes.byref(vo))
    sl = np.array([p2[i] for i in range(vo.value)], dtype=np.float32)
    d2 = np.abs(cl - sl).max()
    print("batch prefill vs step-by-step: max diff = %.3e" % d2)
    if d2 / scale > 1e-3: fails += 1; print("BATCH/STEP MISMATCH")

    # ---- 4. 贪心生成: C (冷启动 / 前缀缓存) 与 numpy 逐步贪心三方一致
    lib.hymt2_debug_generate_ids.restype = ctypes.c_int
    lib.hymt2_debug_generate_ids.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    q1 = "Translate the following text into Chinese. Note that you should only output the translated result.\nHello there"
    q2 = "Translate the following text into Chinese. Note that you should only output the translated result.\nGood night"
    NG = 5
    def cgen(mm_, q):
        buf = (ctypes.c_int * NG)()
        n = lib.hymt2_debug_generate_ids(mm_, q.encode(), buf, NG)
        return [buf[i] for i in range(n)]
    def ref_greedy(q):
        ids = py_encode(BOS_S + USER_S + q + ASSIST_S); out = []
        for _ in range(NG):
            lg = ref_logits(ref, ids + out)[:vo.value]; nxt = int(lg.argmax())
            if nxt in (EOG_ID, EOG_ID + 1): break
            out.append(nxt)
        return out
    g1 = cgen(m, q1)
    g2_warm = cgen(m, q2)           # 复用 q1 的前缀 KV
    m2 = lib.hymt2_model_load(tmp.encode(), 1, 256, err, 512)
    g2_cold = cgen(m2, q2)
    r2 = ref_greedy(q2)
    print("greedy ids  C warm:", g2_warm, " C cold:", g2_cold, " numpy:", r2)
    if not (g2_warm == g2_cold == r2) or not g1 or not r2: fails += 1; print("GENERATION MISMATCH")
    out = ctypes.create_string_buffer(4096)
    nbytes = lib.hymt2_model_generate(m, q2.encode(), out, 4096, 24, 0)
    print("generate() -> %d bytes" % nbytes)
    if nbytes <= 0: fails += 1; print("generate() produced no output")
    print("\n" + ("ALL MODEL TESTS PASSED" if not fails else "%d FAILURES" % fails))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
