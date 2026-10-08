"""E2 / RQ2 / A2:修复对比实验(y.docx 5.6-E2、5.7-A2)。

协议(受控错误注入,SafeQL 式修复评测的标准做法):
  数据:BIRD dev 分层子集(按 difficulty 配额,3 种子);
  注入:8 类单点受控错误(对应 SafeQL 五类原子动作 + 本文扩展动作);
  对比:(a) 不修复(注入即终点)
        (b) 整体重生成基线(放弃损坏 SQL,由生成器从头生成)
        (c) SafeQL 式(S1 only:错误消息 + AST 定位)
        (d) S1+S3(约束校验)
        (e) S1+S3+S2(执行计划/谓词松弛)
  指标:EX、执行错误消除率、每题步数、token 代理成本、重生成触发率;
  SQL 长度分桶(<30 / 30-60 / >60 行)验证 CTE 长查询可扩展性(另见
  exp_correction_spider2.py)。

运行:python experiments/exp_correction.py [--n 60] [--seeds 0]
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (bird_catalog, bird_executor, bird_graph,  # noqa: E402
                    default_budget, load_bird_questions, save_json,
                    stratified_subset)

from rgcv.corrector import Corrector  # noqa: E402
from rgcv.dbs import DBError  # noqa: E402
from rgcv.evalx import EXEvaluator  # noqa: E402
from rgcv.generator import (CORRUPTION_TYPES, RuleGenerator,  # noqa: E402
                            corrupt_sql)
from rgcv.schema_graph import SchemaGraphRAG  # noqa: E402

CONFIGS = {
    "none": None,                      # 不修复
    "regen": "regen",                  # 整体重生成基线
    "S1": ("s1",),                     # SafeQL 式
    "S1+S3": ("s1", "s3"),
    "S1+S3+S2": ("s1", "s3", "s2"),
}

CORRUPTIONS = ["attribute", "relation", "value", "function", "join",
               "predicate_extra", "predicate_missing", "group_missing"]


def run(n: int, seeds: list, corruptions: list):
    questions_all = load_bird_questions()
    results = {}
    traces = []
    for seed in seeds:
        subset = stratified_subset(questions_all, n=n, seed=seed)
        for qi, q in enumerate(subset):
            db_id = q["db_id"]
            cat = bird_catalog(db_id)
            ex = EXEvaluator(bird_executor(db_id))
            if ex.gold_result(q["SQL"]) is None:
                continue
            graph = bird_graph(db_id)
            rag = SchemaGraphRAG(graph, k1=6, k2_per_table=12)
            gen = RuleGenerator(cat, seed=seed)
            rng = random.Random(seed * 10000 + qi)
            retrieval = rag.retrieve(q["question"])

            for ctype in corruptions:
                bad_sql, desc = corrupt_sql(q["SQL"], cat, ctype, rng)
                if desc == "noop":
                    continue
                rec = {
                    "question_id": q["question_id"], "db_id": db_id,
                    "difficulty": q["difficulty"], "corruption": ctype,
                    "corruption_desc": desc, "seed": seed,
                }
                # 各配置
                for cname, cfg in CONFIGS.items():
                    t0 = time.perf_counter()
                    if cfg is None:
                        sql = bad_sql
                        steps, fallback, signals = 0, False, []
                    elif cfg == "regen":
                        go = gen.generate(q["question"], retrieval,
                                          n_candidates=3)
                        sql = next((c for c in go.candidates if _ok(
                            ex, c)), go.candidates[0])
                        steps, fallback, signals = 0, True, ["regen"]
                    else:
                        cor = Corrector(bird_executor(db_id), cat,
                                        signals=cfg, graph=graph)
                        out = cor.repair(bad_sql, q["question"])
                        sql, steps, fallback = out.sql, out.n_steps, \
                            out.fallback_used
                        signals = list(dict.fromkeys(out.signals_used))
                    latency = time.perf_counter() - t0
                    verdict = ex.ex(sql, q["SQL"])
                    exec_ok = _ok(ex, sql)
                    rec[cname] = {
                        "ex": verdict, "exec_ok": exec_ok, "steps": steps,
                        "fallback": fallback, "signals": signals,
                        "latency_s": round(latency, 3),
                        "tokens_proxy": len(sql) // 4 + steps * 30,
                    }
                traces.append(rec)
            if qi % 10 == 0:
                print(f"[seed{seed}] {qi}/{len(subset)} questions done",
                      flush=True)

    # ---- 汇总
    for ctype in corruptions:
        rows = [t for t in traces if t["corruption"] == ctype]
        if not rows:
            continue
        summary = {"n": len(rows)}
        for cname in CONFIGS:
            exs = [t[cname]["ex"] for t in rows]
            known = [e for e in exs if e is not None]
            exec_ok_before = _ok_rate([t["none"]["exec_ok"] for t in rows])
            summary[cname] = {
                "EX": round(sum(1 for e in known if e) / max(len(rows), 1), 4),
                "exec_ok_rate": round(_ok_rate(
                    [t[cname]["exec_ok"] for t in rows]), 4),
                "avg_steps": round(statistics.mean(
                    [t[cname]["steps"] for t in rows]), 2),
                "fallback_rate": round(sum(
                    1 for t in rows if t[cname]["fallback"]) /
                    max(len(rows), 1), 4),
                "avg_latency_s": round(statistics.mean(
                    [t[cname]["latency_s"] for t in rows]), 3),
                "tokens_proxy_mean": round(statistics.mean(
                    [t[cname]["tokens_proxy"] for t in rows]), 1),
            }
            summary[cname]["exec_error_elimination"] = round(
                summary[cname]["exec_ok_rate"] - exec_ok_before, 4)
        results[ctype] = summary

    # 总汇总(全 corruption 合并)与配对检验
    all_pairs = defaultdict(lambda: defaultdict(list))
    for t in traces:
        for cname in CONFIGS:
            all_pairs["all"][cname].append(t[cname]["ex"])
    total = {"n": len(traces)}
    for cname in CONFIGS:
        exs = all_pairs["all"][cname]
        known = [e for e in exs if e is not None]
        ok = sum(1 for t in traces if t[cname]["exec_ok"])
        total[cname] = {
            "EX_strict": round(sum(1 for e in known if e) /
                               max(len(exs), 1), 4),
            "EX_known": round(sum(1 for e in known if e) /
                              max(len(known), 1), 4),
            "n_known": len(known),
            "exec_ok_rate": round(ok / max(len(traces), 1), 4),
        }
    from rgcv.evalx import mcnemar_test, paired_bootstrap
    stats = {}
    for other in ("regen", "S1", "S1+S3", "S1+S3+S2"):
        stats[f"none_vs_{other}"] = mcnemar_test(
            all_pairs["all"]["none"], all_pairs["all"][other])
        stats[f"bootstrap_none_vs_{other}"] = paired_bootstrap(
            all_pairs["all"]["none"], all_pairs["all"][other], seed=0)
    stats["S1_vs_full"] = mcnemar_test(
        all_pairs["all"]["S1"], all_pairs["all"]["S1+S3+S2"])
    stats["bootstrap_S1_vs_full"] = paired_bootstrap(
        all_pairs["all"]["S1"], all_pairs["all"]["S1+S3+S2"], seed=0)

    save_json("e2_correction_bird.json", {
        "config": {"n_questions": n, "seeds": seeds,
                   "corruptions": corruptions,
                   "note": "受控错误注入协议;LLM 代理为规则生成器"},
        "summary_by_corruption": results,
        "summary_total": total,
        "significance": stats,
        "traces": traces,
    })


def _ok(ex: EXEvaluator, sql: str) -> bool:
    try:
        ex.executor.execute(sql)
        return True
    except DBError:
        return False


def _ok_rate(flags: list) -> float:
    return sum(1 for x in flags if x) / max(len(flags), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    a = ap.parse_args()
    run(a.n, a.seeds, a.corruptions)
