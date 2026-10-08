"""E-A:Tier-2 真实 LLM 层静默错误拦截率 / 误报率(revision_plan P0-2)。

动机:原 E3 的拦截率/误报率来自"合成腐蚀 + 规则代理"口径(R1-M2/R4-M2),
本实验在真实 BIRD-300 端到端预测上重放 V1/V2/V3,给出真实错误分布下的
分层拦截率与 FPR,附 bootstrap 95% CI,打破口径循环。

协议:
  输入   results/<tag>/q_*.json(含 question / gold_sql / pred_sql /
         candidates / executed_ok;默认主配置 gated_fullschema 双骨干);
  真值   BIRD gold 执行结果(evalx.EXEvaluator,多重集 + 浮点容差口径);
  分类   gold 可执行且 pred 执行成功:
           EX=False → silent(拦截目标) / EX=True → correct(FPR 对照);
         pred 执行失败 → explicit(显式错误,验证器 reject,不进主指标);
  重放   Verifier(layers=v1,v2,v3)单次全层运行,推导三层配置:
           V1 = v1.alarms 非空;
           V1+V2 = v1 或 v2 alarms 非空(V3 触发前);
           V1+V2+V3 = 最终 verdict ∈ {alarm, reject};
  V3     默认规则代理(与 E3 一致,离线零成本,输出中如实标注);
         --v3_llm 可切换跨骨干 LLM 裁决(kimi 预测 ↔ deepseek 裁决,
         反向同理,缓解自评偏置);
  指标   拦截率/FPR(分层级)+ bootstrap 10^4 95% CI + McNemar(级联增量);
  输出   results/eA_tier2_real_verification.json(traces 供标注抽样)。

运行:python experiments/exp_tier2_real.py [--tags tag1 tag2] [--v3_llm]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, bird_catalog, bird_executor)  # noqa: E402

from rgcv.dbs import DBError  # noqa: E402
from rgcv.evalx import EXEvaluator, mcnemar_test  # noqa: E402
from rgcv.verifier import Verifier  # noqa: E402

DEFAULT_TAGS = ["rgcv_bird300_deepseek-v4-pro_gated_fullschema",
                "rgcv_bird300_kimi-k26_gated_fullschema"]

LAYERS = ("V1", "V1+V2", "V1+V2+V3")


def boot_ci(flags: list, n_boot: int = 10_000, seed: int = 0):
    """Bernoulli 比例的 percentile bootstrap 95% CI。"""
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return [0.0, 0.0]
    hits = [1.0 if f else 0.0 for f in flags]
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += hits[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4),
            round(means[int(0.975 * n_boot)], 4)]


def load_questions(tag: str) -> list:
    outdir = RESULTS / tag
    recs = []
    for p in sorted(outdir.glob("q_*.json"),
                    key=lambda x: int(x.stem.split("_")[1])):
        try:
            d = __import__("json").loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error") or not d.get("pred_sql"):
            continue
        recs.append(d)
    return recs


def classify(rec: dict, ex: EXEvaluator):
    """返回 kind ∈ {silent, correct, explicit, skip_gold, skip_exec}。"""
    gold = ex.gold_result(rec["gold_sql"])
    if gold is None:
        return "skip_gold", None
    try:
        pred = ex.executor.execute(rec["pred_sql"])
    except DBError:
        return "explicit", None
    verdict = ex.ex(rec["pred_sql"], rec["gold_sql"])
    if verdict is True:
        return "correct", pred
    return "silent", pred


def replay(rec: dict, kind: str, ex: EXEvaluator, cat,
           llm_judge=None) -> dict:
    """单次全层 V1+V2+V3 运行,推导三层 flagged。"""
    v = Verifier(ex.executor, cat, layers=("v1", "v2", "v3"),
                 llm_judge=llm_judge)
    rep = v.verify(rec["question"], rec["pred_sql"],
                   rec.get("candidates") or [rec["pred_sql"]])
    lr = rep.layer_reports
    v1_flag = bool(lr.get("v1", {}).get("alarms"))
    v12_flag = v1_flag or bool(lr.get("v2", {}).get("alarms"))
    full_flag = rep.verdict in ("alarm", "reject")
    return {
        "question_id": rec["question_id"], "db_id": rec["db_id"],
        "kind": kind, "tag": rec.get("_tag", ""),
        "V1": {"flagged": v1_flag},
        "V1+V2": {"flagged": v12_flag},
        "V1+V2+V3": {"flagged": full_flag,
                     "triggered_v3": rep.triggered_v3},
        "final_verdict": rep.verdict,
        "alarms": rep.alarmed_clauses,
        "latency_s": rep.latency_s,
    }


def summarize(traces: list) -> dict:
    out = {}
    for kind in ("silent", "correct", "explicit"):
        rows = [t for t in traces if t["kind"] == kind]
        if not rows:
            continue
        s = {"n": len(rows)}
        for lay in LAYERS:
            flags = [t[lay]["flagged"] for t in rows]
            s[lay] = {"rate": round(sum(flags) / len(flags), 4),
                      "ci95": boot_ci(flags, seed=0)}
        v3s = [t["V1+V2+V3"].get("triggered_v3") for t in rows]
        s["v3_trigger_rate"] = round(
            sum(1 for x in v3s if x) / len(v3s), 4) if rows else 0.0
        out[kind] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--v3_llm", action="store_true",
                    help="V3 用跨骨干 LLM 裁决(kimi↔deepseek 互查),"
                         "默认规则代理")
    a = ap.parse_args()

    # 跨骨干裁决:与生成器不同源的骨干(R4-M2 自评偏置缓解)
    judge_cache = {}
    if a.v3_llm:
        from rgcv.llm import LLMClient, LLMGenerator
        cross = {"deepseek": "kimi-k2.6", "kimi": "deepseek-v4-pro"}
        for key, eng in cross.items():
            cli = LLMClient(eng)
            judge_cache[key] = LLMGenerator(cli).judge

    all_traces, summary = [], {}
    for tag in a.tags:
        recs = load_questions(tag)
        for r in recs:
            r["_tag"] = tag
        stats = {"n_loaded": len(recs)}
        traces = []
        for i, rec in enumerate(recs):
            db_id = rec["db_id"]
            try:
                ex = EXEvaluator(bird_executor(db_id))
                kind, _ = classify(rec, ex)
                if kind in ("skip_gold", "explicit"):
                    traces.append({"question_id": rec["question_id"],
                                   "db_id": db_id, "kind": kind,
                                   "tag": tag, "skipped": True})
                    continue
                judge = None
                if a.v3_llm:
                    judge = judge_cache["kimi" if "kimi" in tag
                                        else "deepseek"]
                traces.append(replay(rec, kind, ex, bird_catalog(db_id),
                                     llm_judge=judge))
            except Exception as e:
                print(f"[eA] {tag} q{rec['question_id']} {db_id} ERROR "
                      f"{type(e).__name__}: {e}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"[eA] {tag}: {i + 1}/{len(recs)}", flush=True)

        ok = [t for t in traces if not t.get("skipped")]
        stats["n_replayed"] = len(ok)
        stats["breakdown"] = summarize(ok)
        # 级联增量显著性(silent 上 V1+V2 vs 全层)
        sil = [t for t in ok if t["kind"] == "silent"]
        if sil:
            stats["mcnemar_silent_V1+V2_vs_full"] = mcnemar_test(
                [t["V1+V2"]["flagged"] for t in sil],
                [t["V1+V2+V3"]["flagged"] for t in sil])
        # 报警原因分布(定性)
        alarms = {}
        for t in ok:
            if t["kind"] == "silent" and t["V1+V2+V3"]["flagged"]:
                for al in t["alarms"]:
                    key = str(al).split(":")[0].split("[")[0].strip()[:40]
                    alarms[key] = alarms.get(key, 0) + 1
        stats["alarm_reasons_silent"] = dict(sorted(alarms.items(),
                                                    key=lambda kv: -kv[1]))
        summary[tag] = stats
        all_traces.extend(traces)
        nsil = stats["breakdown"].get("silent", {}).get("n", 0)
        print(f"[eA] {tag}: silent n={nsil}", flush=True)

    import json
    out = {
        "config": {
            "tags": a.tags,
            "v3_judge": "cross-backbone LLM" if a.v3_llm else
                        "rule proxy (no LLM channel; 与 E3 口径一致,如实标注)",
            "ex_criteria": "gold 执行结果多重集比较 + 1e-6 相对浮点容差"
                           "(evalx.exec_equal);行序无关(多重集);"
                           "列名不比对(执行结果列序)",
            "ci": "bootstrap 10^4, percentile 95%",
            "note": "真实静默错误 = 最终预测执行成功但 EX=False"
                    "(R4-M2 口径:以 gold 执行结果为真值,非注入腐蚀)",
        },
        "summary": summary,
        "traces": all_traces,
    }
    (RESULTS / "eA_tier2_real_verification.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    # 控制台摘要
    print("\n===== E-A Tier-2 real silent-error interception =====")
    for tag, st in summary.items():
        print(f"\n[{tag}] loaded={st['n_loaded']} replayed={st['n_replayed']}")
        for kind in ("silent", "correct", "explicit"):
            b = st["breakdown"].get(kind)
            if not b:
                continue
            line = f"  {kind:<9} n={b['n']:<4}"
            for lay in LAYERS:
                line += (f"  {lay}={b[lay]['rate'] * 100:.1f}%"
                         f"[{b[lay]['ci95'][0] * 100:.1f},"
                         f"{b[lay]['ci95'][1] * 100:.1f}]")
            print(line)
        if "mcnemar_silent_V1+V2_vs_full" in st:
            m = st["mcnemar_silent_V1+V2_vs_full"]
            print(f"  McNemar V1+V2 vs full(silent): b={m['b']} c={m['c']} "
                  f"p={m['p_value']}")
    print(f"\n[saved] results/eA_tier2_real_verification.json")


if __name__ == "__main__":
    main()
