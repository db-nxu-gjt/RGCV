"""E3 / RQ3:语义级验证器实验(y.docx 5.6-E3)。

协议:
  数据:BIRD dev 分层子集(60 题);
  SQL 来源:
    (a) gold SQL(正确 → 测误报率 FP);
    (b) 受控注入错误且仍可执行的 SQL(静默错误 → 测拦截率);
    (c) 受控注入错误且执行失败的 SQL(硬错误 → 验证器应直接 reject);
  配置:V1(结果指纹)/ V1+V2(差分执行)/ V1+V2+V3(触发式语义核对,
  LLM 以规则代理,报告中标注);
  指标:静默错误拦截率、误报率、分层时延增量、V3 触发率与 token 代理;
  统计:配对 McNemar + bootstrap(V1 vs 全层,静默错误子集)。

运行:python experiments/exp_verification.py [--n 60]
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (bird_catalog, bird_executor, load_bird_questions,  # noqa: E402
                    save_json, stratified_subset)

from rgcv.dbs import DBError  # noqa: E402
from rgcv.evalx import EXEvaluator, mcnemar_test, paired_bootstrap  # noqa: E402
from rgcv.generator import CORRUPTION_TYPES, corrupt_sql  # noqa: E402
from rgcv.verifier import Verifier  # noqa: E402

LAYER_CONFIGS = {
    "V1": ("v1",),
    "V1+V2": ("v1", "v2"),
    "V1+V2+V3": ("v1", "v2", "v3"),
}


def run(n: int, seed: int = 0):
    questions = load_bird_questions()
    subset = stratified_subset(questions, n=n, seed=seed)
    traces: List[Dict] = []
    n_q = 0
    for qi, q in enumerate(subset):
        db_id = q["db_id"]
        try:
            ex = EXEvaluator(bird_executor(db_id))
            if ex.gold_result(q["SQL"]) is None:
                continue
            cat = bird_catalog(db_id)
            rng = random.Random(seed * 10000 + qi)

            # ---- 构造 SQL 样本:gold + 注入错误(静默/硬)
            samples = [("gold", q["SQL"], "correct")]
            for ctype in CORRUPTION_TYPES:
                bad, desc = corrupt_sql(q["SQL"], cat, ctype, rng)
                if desc == "noop":
                    continue
                try:
                    ex.executor.execute(bad)
                    verdict = ex.ex(bad, q["SQL"])
                    if verdict is False:
                        samples.append((ctype, bad, "silent"))
                    # verdict True → 注入未改变结果,跳过
                except DBError:
                    samples.append((ctype, bad, "hard"))

            for sname, sql, kind in samples:
                rec = {"question_id": q["question_id"], "db_id": db_id,
                       "difficulty": q["difficulty"], "sample": sname,
                       "kind": kind, "seed": seed}
                for cname, layers in LAYER_CONFIGS.items():
                    v = Verifier(bird_executor(db_id), cat, layers=layers)
                    rep = v.verify(q["question"], sql)
                    flagged = rep.verdict in ("alarm", "reject")
                    rec[cname] = {
                        "verdict": rep.verdict,
                        "flagged": flagged,
                        "alarms": rep.alarmed_clauses,
                        "latency_s": rep.latency_s,
                        "triggered_v3": rep.triggered_v3,
                        "tokens": rep.tokens,
                    }
                traces.append(rec)
            n_q += 1
        except Exception as e:
            print(f"[v] {db_id} q{q['question_id']} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (qi + 1) % 10 == 0:
            print(f"[v] {qi + 1}/{len(subset)} questions ({n_q} ok)",
                  flush=True)

    # ---- 汇总
    summary = {}
    for kind in ("correct", "silent", "hard"):
        rows = [t for t in traces if t["kind"] == kind]
        if not rows:
            continue
        s = {"n": len(rows)}
        for cname in LAYER_CONFIGS:
            flags = [t[cname]["flagged"] for t in rows]
            lats = [t[cname]["latency_s"] for t in rows]
            v3s = [t[cname]["triggered_v3"] for t in rows]
            toks = [t[cname]["tokens"] for t in rows]
            s[cname] = {
                "flag_rate": round(sum(1 for f in flags if f) /
                                   len(flags), 4),
                "mean_latency_s": round(statistics.mean(lats), 4),
                "v3_trigger_rate": round(sum(1 for v in v3s if v) /
                                         len(v3s), 4),
                "tokens_mean": round(statistics.mean(toks), 1),
            }
        summary[kind] = s

    # 拦截率 = silent 子集 flag_rate;误报率 = correct 子集 flag_rate
    stats = {}
    silent = [t for t in traces if t["kind"] == "silent"]
    correct = [t for t in traces if t["kind"] == "correct"]
    if silent:
        for a, b in (("V1", "V1+V2"), ("V1", "V1+V2+V3"),
                     ("V1+V2", "V1+V2+V3")):
            stats[f"silent_{a}_vs_{b}"] = mcnemar_test(
                [t[a]["flagged"] for t in silent],
                [t[b]["flagged"] for t in silent])
            stats[f"boot_silent_{a}_vs_{b}"] = paired_bootstrap(
                [t[a]["flagged"] for t in silent],
                [t[b]["flagged"] for t in silent], seed=0)
    if correct:
        for a, b in (("V1", "V1+V2+V3"),):
            stats[f"fp_{a}_vs_{b}"] = mcnemar_test(
                [t[a]["flagged"] for t in correct],
                [t[b]["flagged"] for t in correct])
    # 报警原因分布(定性)
    alarm_counts: Dict[str, int] = {}
    for t in traces:
        if t["kind"] == "silent":
            for a in t["V1+V2+V3"]["alarms"]:
                for part in str(a).split(":"):
                    key = part.split("[")[0].strip()
                    if key and len(key) < 60:
                        alarm_counts[key] = alarm_counts.get(key, 0) + 1

    save_json("e3_verification_bird.json", {
        "config": {"n_questions": n, "seed": seed,
                   "layers": {k: list(v) for k, v in LAYER_CONFIGS.items()},
                   "note": "V3 为规则代理(无 LLM 通道);硬错误=注入后执行失败"},
        "summary": summary,
        "significance": stats,
        "alarm_reasons_silent": dict(sorted(alarm_counts.items(),
                                            key=lambda kv: -kv[1])),
        "traces": traces,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    t0 = time.perf_counter()
    run(a.n)
    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)
