"""E3 V3 异常分析:为何 V1+V2+V3 在 silent 上拦截率低于 V1。"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results"
d3 = json.loads((BASE / "e3_verification_bird.json").read_text(encoding="utf-8"))
tr = d3["traces"]

silent = [t for t in tr if t["kind"] == "silent"]
v1_flag = sum(1 for t in silent if t["V1"]["flagged"])
v123_flag = sum(1 for t in silent if t["V1+V2+V3"]["flagged"])
print(f"silent n={len(silent)}, V1_flag={v1_flag}, V123_flag={v123_flag}")

v1_only = [t for t in silent if t["V1"]["flagged"]
           and not t["V1+V2+V3"]["flagged"]]
v123_only = [t for t in silent if not t["V1"]["flagged"]
             and t["V1+V2+V3"]["flagged"]]
print(f"V1 独有(V3 清除的告警): {len(v1_only)}")
print(f"V123 独有(V3 新增): {len(v123_only)}")
print()

c_only = Counter(t["sample"] for t in v1_only)
c_all = Counter(t["sample"] for t in silent)
print("silent 错误类型总数:", dict(c_all))
print("被 V3 清除告警的类型:", dict(c_only))
print()
trig = sum(1 for t in v1_only if t["V1+V2+V3"]["triggered_v3"])
print(f"被清除的 {len(v1_only)} 例中 V3 触发 {trig} 例")
print()
for t in v1_only[:10]:
    v1a = [str(a)[:40] for a in (t["V1"]["alarms"] or [])[:2]]
    print(f"  {t['sample']:18s} db={t['db_id'][:18]:18s} "
          f"V1_alarms={v1a} v3_trig={t['V1+V2+V3']['triggered_v3']}")
