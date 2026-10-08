"""E4 细节:tokens 分相、验证-修复增益案例。"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results"
d4 = json.loads((BASE / "e4_pipeline_bird.json").read_text(encoding="utf-8"))
tr = d4["traces"]

ph = defaultdict(lambda: defaultdict(list))
for t in tr:
    for k, v in t["tokens_by_phase"].items():
        ph[t["config"]][k].append(v)
print("tokens_by_phase(均值):")
for cfg in ("B1000_full", "B2000_full", "B8000_full", "B8000_open",
            "B8000_rgc"):
    print(f"  {cfg:12s}", {k: round(statistics.mean(v))
                          for k, v in sorted(ph[cfg].items())})

print()
full = {t["question_id"]: t for t in tr if t["config"] == "B8000_full"}
rgc = {t["question_id"]: t for t in tr if t["config"] == "B8000_rgc"}
opn = {t["question_id"]: t for t in tr if t["config"] == "B8000_open"}
gained = [q for q in full if full[q]["ex"] and not rgc[q]["ex"]]
lost = [q for q in full if not full[q]["ex"] and rgc[q]["ex"]]
print(f"full 正确而 rgc 错误: {len(gained)} 题; 反之 {len(lost)} 题")
for q in gained:
    t = full[q]
    print(f"  q{q}: corruption={t['corruption']}, "
          f"verify={t['verify_verdict']}, "
          f"steps rgc={rgc[q]['repair_steps']} full={t['repair_steps']}")
print()
rgc_gain = [q for q in rgc if rgc[q]["ex"] and not opn[q]["ex"]]
print(f"rgc 相对 open 增益: {len(rgc_gain)} 题 (EX {0.067:.3f}->{0.25:.3f})")
exec_fix = [q for q in rgc if rgc[q]["executed_ok"] and not opn[q]["executed_ok"]]
print(f"rgc 修复执行失败: {len(exec_fix)} 题 (exec 0.467->0.80)")
