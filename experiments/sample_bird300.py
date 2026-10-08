"""生成 BIRD dev 300 题分层子集 sample_bird300.json。

协议要求 (z.docx §5.1):
  - 全量 1534 题按难度分层: simple 181 / moderate 91 / challenging 28 (与全量比例一致)
  - 单库题数 ≤10% (cap=30)
  - 随机种子 42, 可复现
  - 原型 61 题子集 (chess_bird61.json) 强制嵌套包含
"""
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DEV = json.load(open(BASE / "data" / "bird" / "dev_20240627" / "dev.json", encoding="utf-8"))
EXIST_61 = {d["question_id"] for d in json.load(open(BASE / "results" / "chess_bird61.json", encoding="utf-8"))}

TARGET = {"simple": 181, "moderate": 91, "challenging": 28}
LIB_CAP = 30

buckets = defaultdict(list)
for d in DEV:
    buckets[d["difficulty"]].append(d)

random.seed(42)
picked_ids = set()
final = []
lib_count = Counter()

# 1) 嵌套 61 题
for d in DEV:
    if d["question_id"] in EXIST_61:
        final.append(d)
        picked_ids.add(d["question_id"])
        lib_count[d["db_id"]] += 1

# 2) 逐层补齐 (跳过超 cap 的库)
for diff in ("simple", "moderate", "challenging"):
    need = TARGET[diff] - sum(1 for d in final if d["difficulty"] == diff)
    if need <= 0:
        continue
    pool = [d for d in buckets[diff] if d["question_id"] not in picked_ids]
    random.shuffle(pool)
    for d in pool:
        if need <= 0:
            break
        if lib_count[d["db_id"]] >= LIB_CAP:
            continue
        final.append(d)
        picked_ids.add(d["question_id"])
        lib_count[d["db_id"]] += 1
        need -= 1

# 3) 若 cap 导致某层缺额, 用该层未用库位放宽补足 (保证总数 300)
for diff in ("simple", "moderate", "challenging"):
    short = TARGET[diff] - sum(1 for d in final if d["difficulty"] == diff)
    if short <= 0:
        continue
    pool = [d for d in buckets[diff] if d["question_id"] not in picked_ids]
    random.shuffle(pool)
    for d in pool[:short]:
        final.append(d)
        picked_ids.add(d["question_id"])
        lib_count[d["db_id"]] += 1

final.sort(key=lambda d: (d["difficulty"], d["question_id"]))
diff_counts = Counter(d["difficulty"] for d in final)
out = BASE / "results" / "sample_bird300.json"
json.dump(final, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"总数: {len(final)}  (目标 300)")
print(f"难度分布: {dict(diff_counts)}  (目标 181/91/28)")
print(f"嵌套 61 题子集: {sum(1 for d in final if d['question_id'] in EXIST_61)}/61")
print(f"库数: {len(lib_count)}, 最大单库: {lib_count.most_common(3)} (cap=30)")
print(f"输出: {out}")
