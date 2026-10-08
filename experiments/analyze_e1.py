"""E1 补充细节:压缩率、召回语义、GRAST 大 schema 表现。"""
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent.parent / "results"
d = json.loads((BASE / "e1_retrieval.json").read_text(encoding="utf-8"))

print("E1 全量指标(含压缩率):")
for ds, s in d["summary"].items():
    print(f"--- {ds} (n={s.get('n')}) ---")
    for cfg, m in s.items():
        if cfg == "n" or not isinstance(m, dict) or "table_F1" not in m:
            continue
        parts = []
        for k in ("table_P", "table_R", "table_F1", "col_P", "col_R",
                  "join_cov"):
            v = m.get(k)
            if isinstance(v, dict) and "mean" in v:
                parts.append(f"{k}={v['mean']:.3f}")
        tok = m.get("tokens", {}).get("mean", 0)
        comp = m.get("compression_vs_full", 0)
        parts.append(f"tokens={tok:.0f} comp={comp}")
        print(f"  {cfg:14s} {' '.join(parts)}")

print()
print("GRAST traces 细节(企业级大 schema):")
tr = [t for t in d["traces"] if t.get("dataset") == "spider2lite_grast"]
if tr:
    ntab = [t["n_db_tables"] for t in tr]
    gtab = [t["gold_n_tables"] for t in tr]
    print(f"  n=233, db 表数中位数={statistics.median(ntab):.0f}, "
          f"最大={max(ntab)}, gold 表数中位数={statistics.median(gtab):.1f}, "
          f"最大={max(gtab)}")
    keys = [k for k in tr[0] if "recall" in k or k.startswith("cfg")
            or k in ("config", "cfg_name")]
    print(f"  trace keys 示例: {list(tr[0].keys())[:15]}")
    if "cfg" in tr[0]:
        for name in ("full", "flat_bm25"):
            rows = [t for t in tr if t["cfg"] == name]
            rr = [r["table_recall"] for r in rows]
            print(f"  {name}: table_R mean={statistics.mean(rr):.3f}, "
                  f"R=1 比例={sum(1 for r in rr if r >= 0.999) / len(rr):.3f}")
