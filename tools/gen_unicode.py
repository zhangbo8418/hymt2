#!/usr/bin/env python3
"""生成 src/unicode_tables.h: 把码点按 L / M / N / P / S 五大类压成区间表 (分词器预切分正则需要)。"""
import sys, unicodedata

CLS = {"L": 1, "M": 2, "N": 4, "P": 8, "S": 16}
ranges = []
cur = None
for cp in range(0x110000):
    c = CLS.get(unicodedata.category(chr(cp))[0], 0)
    if cur and cur[2] == c and cur[1] == cp - 1:
        cur[1] = cp
    else:
        if cur and cur[2]:
            ranges.append(tuple(cur))
        cur = [cp, cp, c]
if cur and cur[2]:
    ranges.append(tuple(cur))

out = ["/* 自动生成 (tools/gen_unicode.py), Unicode %s, 勿手改 */" % unicodedata.unidata_version,
       "#ifndef HYMT2_UNICODE_TABLES_H", "#define HYMT2_UNICODE_TABLES_H", "#include <stdint.h>",
       "#define UC_L 1", "#define UC_M 2", "#define UC_N 4", "#define UC_P 8", "#define UC_S 16",
       "typedef struct { uint32_t lo, hi; uint8_t cls; } uc_range;",
       "static const uc_range UC_RANGES[] = {"]
line = []
for lo, hi, c in ranges:
    line.append("{0x%X,0x%X,%d}," % (lo, hi, c))
    if len(line) == 6:
        out.append("  " + "".join(line)); line = []
if line:
    out.append("  " + "".join(line))
out += ["};", "#define UC_NRANGES (sizeof(UC_RANGES)/sizeof(UC_RANGES[0]))", "#endif", ""]
open(sys.argv[1] if len(sys.argv) > 1 else "src/unicode_tables.h", "w").write("\n".join(out))
print("ranges:", len(ranges), "unicode", unicodedata.unidata_version)
