"""E-D (T1-2) 验证模块下游效用 — 离线分析(零 API)。

T1-2b 人审成本-收益:三机制(V1+V2 / SC-vote / CHESS-UT)× 两骨干,
  reviews-per-catch(每拦截一个真静默错误需人工复核题数)+ 每 catch API 成本。
T1-2c V2 真实多候选增量:在 E-C 4 采样候选池上离线运行 V2 candidate-divergence
  (与 src 实现 candidates[:3] 同口径:取 [frozen]+samples 前 3 个),
  对比单候选(frozen 重写一致性 only)vs 多候选增量。
T1-2d V3 conditional-clear 折中曲线:v3llm 逐题裁决 × 告警类别权限子集扫描,
  policy S = V3 pass 且全部告警类别 ∈ S 才清除;枚举子集取 Pareto 前沿。
T1-2e metadata-poor 退化量化:按告警类别统计对 catalog 元数据
  (top_values / n_distinct)的依赖占比。

输出 results/eD_utility_analysis.json。
"""
from __future__ import annotations

import itertools
import json
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from common import RESULTS, bird_executor  # noqa: E402

# BIRD dev_databases 路径回退(同 run_ec_verification_baselines.py)。
import common as _common  # noqa: E402

REPO = BASE.parents[1]                                # RGCV root
if not _common.BIRD_DB.exists():
    for _cand in (REPO / "data" / "bird" / "dev_20240627" / "dev_databases",
                  REPO / "baselines" / "DAIL-SQL" / "dataset" / "bird" / "dev"
                  / "dev_databases"):
        if _cand.exists():
            _common.BIRD_DB = _cand
            _common.BIRD_DIR = _cand.parent
            break
    assert _common.BIRD_DB.exists(), \
        f"BIRD dev_databases not found (tried fallbacks); last={_common.BIRD_DB}"

from exp_tier2_real import boot_ci  # noqa: E402
from rgcv.verifier import _result_hash  # noqa: E402

EA_FILE = RESULTS / "eA_tier2_real_verification.json"
EA_V3LLM = RESULTS / "eA_tier2_real_verification_v3llm.json"
EC_FILE = RESULTS / "ec_verification_baselines.json"

DS = "rgcv_bird300_deepseek-v4-pro_gated_fullschema"
KM = "rgcv_bird300_kimi-k26_gated_fullschema"
TAGS = {"deepseek-v4-pro": DS, "kimi-k2.6": KM}
EC_TAGS = {"deepseek-v4-pro": "ec_baselines_deepseek-v4-pro",
           "kimi-k2.6": "ec_baselines_kimi-k26"}

# V1/V2/V3-proxy 告警类别 → 是否依赖 catalog 元数据
METADATA_DEP = {
    "row-magnitude": True,                 # _estimate_cardinality ← n_distinct
    "literal-not-in-domain": True,         # _literal_audit ← top_values
    "predicate-literal-absent-in-question": True,   # ← top_values
    "empty-result": False, "non-unique": False,
    "ratio-out-of-range": False, "high-null-rate": False,
    "rewrite-inconsistency": False, "candidate-divergence": False,
    "missing-group-by-for-per-question": False,
    "contradictory-predicate": False, "intent-coverage-low": False,
}


def alarm_category(a: str) -> str:
    a = a.strip()
    for k in ("rewrite-inconsistency", "candidate-divergence",
              "missing-group-by-for-per-question",
              "predicate-literal-absent-in-question",
              "literal-not-in-domain", "contradictory-predicate",
              "intent-coverage-low", "row-magnitude", "empty-result",
              "non-unique", "ratio-out-of-range", "high-null-rate"):
        if a.startswith(k):
            return k
    return "llm-clause"          # V3(LLM)回传的自由文本子句,非清除候选


def rate_ci(flags):
    n = len(flags)
    if n == 0:
        return {"n": 0, "rate": 0.0, "ci95": [0.0, 0.0]}
    return {"n": n, "rate": round(sum(flags) / n, 4),
            "ci95": boot_ci(flags)}


def load_ea():
    ea = json.loads(EA_FILE.read_text(encoding="utf-8"))
    traces = {}
    for eng, tag in TAGS.items():
        traces[eng] = [t for t in ea["traces"]
                       if t.get("tag") == tag and t.get("kind") in ("silent", "correct")]
    return traces


# ---------------------------------------------------------------- T1-2b
def t12b(traces) -> dict:
    ec = json.loads(EC_FILE.read_text(encoding="utf-8"))
    out = {}
    for eng, tr in traces.items():
        e = ec["summary"]["deepseek-v4-pro" if eng == "deepseek-v4-pro"
                         else "kimi-k26"]
        silent = [t for t in tr if t["kind"] == "silent"]
        correct = [t for t in tr if t["kind"] == "correct"]
        row = {}
        # RGCV V1+V2(规则式,零 LLM)
        itp = [t["V1+V2"]["flagged"] for t in silent]
        fa = [t["V1+V2"]["flagged"] for t in correct]
        n_alarm = sum(itp) + sum(fa)
        row["rgcv_v1v2"] = {
            **{"silent_interception": rate_ci(itp),
               "false_alarm": rate_ci(fa)},
            "alarms_per_100q": round(100 * n_alarm / len(tr), 1),
            "reviews_per_catch": round(n_alarm / sum(itp), 2) if sum(itp) else None,
            "api_calls_per_catch": 0, "api_tokens_in_per_catch": 0,
        }
        # SC-vote / CHESS-UT(E-C aggregate 成本口径)
        for mech, key in (("sc_vote", "sc"), ("chess_ut", "ut")):
            s = e[mech]
            catches = s["silent_interception"]["alarm_n"]
            fas = s["correct_false_alarm"]["alarm_n"]
            reviews = catches + fas
            tot = s["cost_total"]
            row[mech] = {
                "silent_interception": {
                    "n": s["silent_interception"]["n"],
                    "rate": s["silent_interception"]["rate"],
                    "ci95": s["silent_interception"]["ci95"]},
                "false_alarm": {
                    "n": s["correct_false_alarm"]["n"],
                    "rate": s["correct_false_alarm"]["rate"],
                    "ci95": s["correct_false_alarm"]["ci95"]},
                "corrections_silent": s["silent_interception"]["corrections_n"],
                "alarms_per_100q": round(
                    100 * reviews / (s["silent_interception"]["n"]
                                     + s["correct_false_alarm"]["n"]), 1),
                "reviews_per_catch": round(reviews / catches, 2),
                "api_calls_per_catch": round(tot["calls"] / catches, 1),
                "api_tokens_in_per_catch": int(tot["tokens_in"] / catches),
            }
        out[eng] = row
    return out


# ---------------------------------------------------------------- T1-2c
_schema_cache = {}
_c_cache = {}
_lock = threading.Lock()


def exec_hash(ex, sql: str):
    try:
        cols, rows, _ = ex.execute(sql)
        return _result_hash(cols, rows)
    except Exception:
        return "EXEC_ERR"


def t12c(traces) -> dict:
    """V2 多候选增量:implementation-faithful = divergence among
    candidates[:3] of [frozen]+4 samples → {frozen, s0, s1}。"""
    out = {}
    for eng, tr in traces.items():
        resp_dir = RESULTS / EC_TAGS[eng] / "responses"
        samples = {}
        frozen = {}
        for t in tr:
            qid = int(t["question_id"])
            p = resp_dir / f"q{qid}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") != "done":
                continue
            frozen[qid] = r["pred_sql"]
            samples[qid] = r["sampled_sqls"]
        div_flags = {}     # qid → bool(top-3 口径 divergence)
        div5_flags = {}    # qid → bool(all-5 口径)
        db_of = {int(t["question_id"]): t["db_id"] for t in tr}
        ex_cache = {}

        def ex_for(qid):
            d = db_of[qid]
            with _lock:
                if d not in ex_cache:
                    ex_cache[d] = bird_executor(d)
                return ex_cache[d]

        for qid, pred in frozen.items():
            s = samples.get(qid) or []
            if len(s) < 4:
                continue
            ex = ex_for(qid)
            hs = [exec_hash(ex, sql) for sql in [pred] + s]
            div_flags[qid] = len(set(hs[:3]) - {"EXEC_ERR"}) > 1
            div5_flags[qid] = len(set(hs) - {"EXEC_ERR"}) > 1
        arms = {}
        for name, fl in (("div_top3", div_flags), ("div_all5", div5_flags)):
            itp = [fl.get(int(t["question_id"]), False)
                   for t in tr if t["kind"] == "silent"]
            fa = [fl.get(int(t["question_id"]), False)
                  for t in tr if t["kind"] == "correct"]
            arms[name] = {"silent_interception": rate_ci(itp),
                          "false_alarm": rate_ci(fa)}
        # V2-single 基线(eA replay 中的 V2-only 告警 = rewrite 类)
        v2only_itp = [bool(t["V1+V2"]["flagged"]) and not t["V1"]["flagged"]
                      for t in tr if t["kind"] == "silent"]
        v2only_fa = [bool(t["V1+V2"]["flagged"]) and not t["V1"]["flagged"]
                     for t in tr if t["kind"] == "correct"]
        arms["v2_rewrite_only_eA"] = {
            "silent_interception": rate_ci(v2only_itp),
            "false_alarm": rate_ci(v2only_fa)}
        # 联合臂:V1+V2 ∪ divergence(top-3)
        u_itp, u_fa = [], []
        for t in tr:
            f = bool(t["V1+V2"]["flagged"]) or div_flags.get(
                int(t["question_id"]), False)
            (u_itp if t["kind"] == "silent" else u_fa).append(f)
        arms["v12_union_div_top3"] = {
            "silent_interception": rate_ci(u_itp), "false_alarm": rate_ci(u_fa)}
        out[eng] = arms
    return out


# ---------------------------------------------------------------- T1-2d
def t12d(v3_traces) -> dict:
    """conditional-clear 折中曲线。policy S:V1+V2 告警题被 V3(LLM)清除,
    当且仅当 triggered_v3 且 verdict==pass 且全部告警类别 ∈ S。
    拦截率/误报率按全量 silent/correct 人口折算(锚点 = V1+V2 / V1+V2+V3)。"""
    out = {}
    for eng, tag in TAGS.items():
        all_rows = [t for t in v3_traces
                    if t.get("tag") == tag
                    and t.get("kind") in ("silent", "correct")]
        rows = [t for t in all_rows if t.get("V1+V2", {}).get("flagged")]
        n_s = sum(1 for t in all_rows if t["kind"] == "silent")
        n_c = sum(1 for t in all_rows if t["kind"] == "correct")
        cats = sorted({alarm_category(a) for t in rows
                       for a in t.get("alarms", [])})
        prep = []
        for t in rows:
            A = frozenset(alarm_category(a) for a in t.get("alarms", []))
            clearable = (t.get("V1+V2+V3", {}).get("triggered_v3")
                         and t.get("final_verdict") == "pass")
            prep.append((A, clearable, t["kind"]))
        # 枚举权限子集 → (FA, interception) 全量口径
        points = []
        for r in range(len(cats) + 1):
            for S in itertools.combinations(cats, r):
                Sset = frozenset(S)
                itp = sum(1 for A, clr, k in prep
                          if k == "silent" and not (clr and A.issubset(Sset)))
                fa = sum(1 for A, clr, k in prep
                         if k == "correct" and not (clr and A.issubset(Sset)))
                points.append({
                    "S": sorted(Sset),
                    "interception": round(itp / n_s, 4),
                    "false_alarm": round(fa / n_c, 4)})
        uniq = {(p["false_alarm"], p["interception"]): p for p in points}
        pts = sorted(uniq.values(),
                     key=lambda p: (p["false_alarm"], -p["interception"]))
        frontier, best = [], -1.0
        for p in pts:                      # FA 升序;取 interception 递增链
            if p["interception"] > best:
                frontier.append(p)
                best = p["interception"]
        out[eng] = {
            "n_silent": n_s, "n_correct": n_c, "categories": cats,
            "n_flagged": len(rows),
            "n_clearable": sum(1 for _, clr, _ in prep if clr),
            "frontier": frontier,
        }
    return out


# ---------------------------------------------------------------- T1-2e
def t12e(traces) -> dict:
    out = {}
    for eng, tr in traces.items():
        flagged = [t for t in tr if t["V1+V2"]["flagged"]]
        all_alarms = [a for t in flagged for a in t.get("alarms", [])]
        cnt = Counter(alarm_category(a) for a in all_alarms)
        md_alarms = sum(v for k, v in cnt.items() if METADATA_DEP.get(k))
        # 仅依赖 metadata 告警才被触发的 flagged 题(失去 top_values/n_distinct
        # 后 flag 完全消失;其余题仍有非 metadata 告警兜底)
        md_only_q = [
            {"question_id": t["question_id"], "kind": t["kind"]}
            for t in flagged
            if all(METADATA_DEP.get(alarm_category(a), False)
                   for a in t.get("alarms", []))]
        md_caught_silent = sum(1 for t in flagged
                               if t["kind"] == "silent"
                               and not all(METADATA_DEP.get(
                                   alarm_category(a), False)
                                   for a in t.get("alarms", [])))
        out[eng] = {
            "n_flagged": len(flagged),
            "n_alarms": len(all_alarms),
            "alarm_categories": dict(cnt),
            "metadata_dep_alarms": md_alarms,
            "metadata_dep_share": round(md_alarms / len(all_alarms), 3)
            if all_alarms else 0,
            "flagged_via_metadata_only": md_only_q,
            "silent_caught_without_metadata_dep": md_caught_silent,
        }
    return out


def main():
    traces = load_ea()
    v3 = json.loads(EA_V3LLM.read_text(encoding="utf-8"))
    result = {
        "config": {
            "question_set": "eA replay silent(132/128)+correct(165/169)",
            "t12c_candidates": "E-C 4 samples (T=0.7) + frozen pred; "
                               "divergence per verifier.py candidates[:3] = {frozen,s0,s1}",
            "t12d_policy": "cleared iff triggered_v3 AND verdict==pass "
                           "AND all alarm categories in S",
            "t12d_judge": v3.get("config", {}).get("v3_judge", "cross-backbone LLM"),
        },
        "t12b_reviews_per_catch": t12b(traces),
        "t12c_v2_multicandidate": t12c(traces),
        "t12d_conditional_clear": t12d(v3["traces"]),
        "t12e_metadata_degradation": t12e(traces),
    }
    out = RESULTS / "eD_utility_analysis.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"written {out}")
    # 摘要打印
    print(json.dumps(result["t12b_reviews_per_catch"], indent=1)[:2000])
    print(json.dumps(result["t12c_v2_multicandidate"], indent=1)[:1500])
    for eng, d in result["t12d_conditional_clear"].items():
        print(eng, f"flagged={d['n_flagged']} clearable={d['n_clearable']} "
              f"cats={d['categories']}")
        for p in d["frontier"][:14]:
            print("  FA", p["false_alarm"], "ITP", p["interception"], p["S"])
    print(json.dumps(result["t12e_metadata_degradation"], indent=1))


if __name__ == "__main__":
    main()
