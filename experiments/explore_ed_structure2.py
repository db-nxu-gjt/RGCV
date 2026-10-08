"""E-D 探索 2:v3llm V3 判定细节 / eA V2-only 告警 / ec traces 逐题 / eF vtr / e4 nc4。"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

ea = json.loads((RESULTS / "eA_tier2_real_verification.json").read_text(encoding="utf-8"))
DS = "rgcv_bird300_deepseek-v4-pro_gated_fullschema"
KM = "rgcv_bird300_kimi-k26_gated_fullschema"

# 1) V2-only 触发(V1 未触发但 V1+V2 触发)的 alarms 文本
print("### eA V2-only 触发样本")
v2only = [t for t in ea["traces"] if t.get("tag") == DS and t.get("V1+V2", {}).get("flagged") and not t.get("V1", {}).get("flagged")]
print(f"V2-only 条数(ds): {len(v2only)}")
for t in v2only[:4]:
    print(f" q{t['question_id']} {t['kind']}: alarms={t['alarms']}")

# 2) V1-only 告警文本类别统计(为 T1-2d conditional-clear 扫描做准备)
print("\n### V1 alarm 文本前缀类别统计(ds flagged 集)")
from collections import Counter
cnt = Counter()
for t in ea["traces"]:
    if t.get("tag") == DS and t.get("V1", {}).get("flagged"):
        for a in t.get("alarms", []):
            cnt[a.split(":")[0].split()[0]] += 1
print(cnt.most_common(20))

# 3) v3llm:V3 triggered 的逐题结构 + config
v3 = json.loads((RESULTS / "eA_tier2_real_verification_v3llm.json").read_text(encoding="utf-8"))
print("\n### v3llm config")
print(json.dumps(v3.get("config", {}), indent=1, ensure_ascii=False)[:1200])
trig = [t for t in v3["traces"] if t.get("V1+V2+V3", {}).get("triggered_v3")]
print(f"v3 triggered 条数: {len(trig)}")
print("样本(前2条):")
for t in trig[:2]:
    print(json.dumps(t, indent=1, ensure_ascii=False)[:1200])
print("v3llm summary:")
print(json.dumps(v3.get("summary", {}), indent=1, ensure_ascii=False)[:1200])

# 4) ec traces 逐题字段
ec = json.loads((RESULTS / "ec_verification_baselines.json").read_text(encoding="utf-8"))
print("\n### ec traces 逐题字段(第一条)")
t0 = ec["traces"][0]
def brief(v, d=0):
    if isinstance(v, dict):
        return {k: brief(x, d+1) if d < 2 else "..." for k, x in list(v.items())[:20]}
    if isinstance(v, list):
        return f"list[{len(v)}]" + (f" first={brief(v[0], d+1)}" if v and d < 2 else "")
    return str(v)[:60]
print(json.dumps(brief(t0), indent=1, ensure_ascii=False)[:1800])
print("\nec summary deepseek:")
print(json.dumps(ec["summary"].get("deepseek-v4-pro", {}), indent=1, ensure_ascii=False)[:2000])

# 5) eF vtr engines 结构
ef = json.loads((RESULTS / "eF_verify_then_replace.json").read_text(encoding="utf-8"))
print("\n### eF vtr")
print(json.dumps({k: (str(v)[:150] if not isinstance(v, (dict, list)) else
                      f"{type(v).__name__}[{len(v)}]") for k, v in ef.items()}, indent=1))
eng = ef.get("engines", {})
for k, v in (eng.items() if isinstance(eng, dict) else enumerate(eng)):
    print(f" engine {k}:", json.dumps({k2: (str(v2)[:100] if not isinstance(v2, (dict, list)) else f"{type(v2).__name__}[{len(v2)}]")
                                       for k2, v2 in (v.items() if isinstance(v, dict) else [])}, ensure_ascii=False)[:400])

# 6) e4 nc4
e4 = json.loads((RESULTS / "e4_pipeline_bird.json").read_text(encoding="utf-8"))
s = json.dumps(e4.get("config", {}), ensure_ascii=False)
print("\n### e4 config 含 nc4?", "nc4" in s or "n_candidates" in s)
print(json.dumps(e4.get("config", {}), ensure_ascii=False)[:800])
s4 = json.dumps(e4.get("summary", {}), ensure_ascii=False)
print("e4 summary 片段:", s4[:600])
