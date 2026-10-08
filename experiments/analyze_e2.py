"""E2 按错误类型分解 + S3 负贡献根因分析(适配实际数据结构)。"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results"
d = json.loads((BASE / "e2_correction_bird.json").read_text(encoding="utf-8"))

print("E2 按错误类型 EX(5 配置)+ 每类样本数:")
sbc = d["summary_by_corruption"]
cfgs = ["none", "regen", "S1", "S1+S3", "S1+S3+S2"]
print(f"{'corruption':18s}{'n':>4s}" + "".join(f"{c:>10s}" for c in cfgs))
for ctype, m in sbc.items():
    row = f"{ctype:18s}{m.get('n', 0):>4d}"
    for c in cfgs:
        v = m.get(c, {}).get("EX", "-")
        row += f"{v if isinstance(v, str) else round(v, 3):>10}"
    print(row)

print()
print("S1 → S1+S3 转移分析:")
tr = d["traces"]
better = [t for t in tr if t["S1"]["ex"] is True
          and t["S1+S3"]["ex"] is not True]
worse = [t for t in tr if t["S1+S3"]["ex"] is True
         and t["S1"]["ex"] is not True]
print(f"S1 正确→S1+S3 失效: {len(better)}; "
      f"S1+S3 正确→S1 失效: {len(worse)}")
print("失效案例类型:", dict(Counter(t["corruption"] for t in better)))
for t in better[:10]:
    c3 = t["S1+S3"]
    print(f"  {t['corruption']:18s} q{t['question_id']} "
          f"db={t['db_id'][:16]:16s} steps={c3['steps']} "
          f"signals={c3['signals']} desc={t['corruption_desc'][:40]}")

print()
print("S1 → S1+S3+S2(全信号)转移:")
b2 = [t for t in tr if t["S1"]["ex"] is True
      and t["S1+S3+S2"]["ex"] is not True]
w2 = [t for t in tr if t["S1+S3+S2"]["ex"] is True
      and t["S1"]["ex"] is not True]
print(f"S1 正确→全信号失效: {len(b2)}; 全信号正确→S1 失效: {len(w2)}")
print("失效类型:", dict(Counter(t["corruption"] for t in b2)))

print()
print("修复步数/信号触发率(总体):")
for c in cfgs:
    steps = [t[c]["steps"] for t in tr]
    sig = Counter(s for t in tr for s in t[c]["signals"])
    fb = sum(1 for t in tr if t[c]["fallback"])
    print(f"  {c:10s} avg_steps={sum(steps)/len(steps):.2f} "
          f"fallback={fb} signals={dict(sig)}")
