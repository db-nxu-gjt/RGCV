"""E-D 探索脚本:检查 v3llm/t22/eA traces/ec aggregate 的字段结构(只读)。"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def keys_of(obj, depth=0, max_depth=3):
    if depth >= max_depth:
        return "..."
    if isinstance(obj, dict):
        return {k: keys_of(v, depth + 1, max_depth) for k, v in list(obj.items())[:30]}
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return [keys_of(obj[0], depth + 1, max_depth), f"...len={len(obj)}"]
    return type(obj).__name__


def show(name, fn, sample_idx=0, max_depth=3):
    p = RESULTS / fn
    print("=" * 70)
    print(f"### {name}  ({p.stat().st_size/1024:.0f} KB)")
    d = json.loads(p.read_text(encoding="utf-8"))
    print(keys_of(d, max_depth=max_depth))


# 1) eA traces: V1 指纹 reason 是否存在
ea = json.loads((RESULTS / "eA_tier2_real_verification.json").read_text(encoding="utf-8"))
print("=" * 70)
print("### eA traces 逐题字段")
t = ea["traces"][0]
print(json.dumps({k: (str(v)[:120] if not isinstance(v, (dict, list)) else
                      {k2: (str(v2)[:100] if not isinstance(v2, (dict, list)) else keys_of(v2, 0, 1))
                       for k2, v2 in v.items()} if isinstance(v, dict) else
                      f"list[{len(v)}]")
                  for k, v in t.items()}, indent=1, ensure_ascii=False))
# 找一条 flagged 的看 V1 reason
flagged = [x for x in ea["traces"] if x.get("tag") == "rgcv_bird300_deepseek-v4-pro_gated_fullschema"
           and x.get("V1+V2", {}).get("flagged")]
print(f"\nflagged traces (ds): {len(flagged)}")
if flagged:
    print(json.dumps(flagged[0], indent=1, ensure_ascii=False)[:1500])
    kinds = {}
    reasons = {}
    for x in flagged:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        r = x.get("V1+V2", {}).get("reason") or x.get("V1", {}).get("reason")
        reasons[str(r)] = reasons.get(str(r), 0) + 1
    print("kind 分布:", kinds)
    print("reason 分布:", reasons)

# 2) v3llm 逐题结构
v3 = json.loads((RESULTS / "eA_tier2_real_verification_v3llm.json").read_text(encoding="utf-8"))
print("=" * 70)
print("### v3llm 顶层:", keys_of(v3, max_depth=1))
items = v3.get("traces") or v3.get("verdicts") or v3.get("results") or []
print(f"逐题条数: {len(items)}")
if items:
    print(json.dumps(items[0], indent=1, ensure_ascii=False)[:1600])

# 3) t22
show("t22_category_decomposition", "t22_category_decomposition.json", max_depth=3)

# 4) ec aggregate: usage 成本字段
ec = json.loads((RESULTS / "ec_verification_baselines.json").read_text(encoding="utf-8"))
print("=" * 70)
print("### ec aggregate 顶层:", keys_of(ec, max_depth=2))

# 5) e4 nc4 / eF vtr 线索
for fn in ("e4_pipeline_bird.json", "eF_verify_then_replace.json", "eG_edge_ablation.json"):
    p = RESULTS / fn
    d = json.loads(p.read_text(encoding="utf-8"))
    print("=" * 70)
    print(f"### {fn}:", keys_of(d, max_depth=1))
    s = json.dumps(d, ensure_ascii=False)
    for kw in ("nc4", "vtr", "n_candidates"):
        if kw in s:
            print(f"  含关键词 {kw}")
