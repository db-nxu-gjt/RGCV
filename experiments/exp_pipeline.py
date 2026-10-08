"""E4 / RQ4:端到端闭环流水线 + 成本感知预算实验(y.docx 5.6-E4)。

协议:
  数据:BIRD dev 分层子集(60 题,seed 0,与 E2/E3 同子集);
  场景:闭环起点 = 受控注入损坏的 gold SQL(轮转 8 类错误),
        模拟"LLM 首次生成含单点错误"的在线闭环;
  配置:
    (a) 预算扫描 B ∈ {1000, 2000, 4000, 32000}(全闭环 R-G-C-V);
        B 按式(3) 6:2.5:1.5 分相,主要调制修复搜索深度
        (检索注入预算同比例收缩;规则代理生成质量不受注入截断影响,
        报告中标注该局限);
    (b) 定预算 B=8000 消融:open(R-G)/ rgc(R-G-C)/ full(R-G-C-V+差分回传);
  指标:EX、执行率、每题 token(分相)、延迟、修复步数、验证触发;
  统计:full vs rgc vs open 的配对 bootstrap。

运行:python experiments/exp_pipeline.py [--n 60]
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
from common import (bird_catalog, bird_executor, bird_graph,  # noqa: E402
                    default_budget, load_bird_questions, save_json,
                    stratified_subset)

from rgcv.budget import BudgetController  # noqa: E402
from rgcv.evalx import EXEvaluator, paired_bootstrap  # noqa: E402
from rgcv.generator import CORRUPTION_TYPES, RuleGenerator, corrupt_sql  # noqa: E402
from rgcv.pipeline import RGCVPipeline  # noqa: E402

# (名称, 预算, use_repair, use_verifier)
CONFIGS = {
    "B1000_full": (1000, True, True),
    "B2000_full": (2000, True, True),
    "B4000_full": (4000, True, True),
    "B32000_full": (32000, True, True),
    "B8000_open": (8000, False, False),
    "B8000_rgc": (8000, True, False),
    "B8000_full": (8000, True, True),
}


def run(n: int, seed: int = 0):
    questions = load_bird_questions()
    subset = stratified_subset(questions, n=n, seed=seed)
    traces: List[Dict] = []
    pipe_cache: Dict[str, Dict] = {}
    n_q = 0
    t_start = time.perf_counter()
    for qi, q in enumerate(subset):
        db_id = q["db_id"]
        try:
            ex = EXEvaluator(bird_executor(db_id))
            if ex.gold_result(q["SQL"]) is None:
                continue
            cat = bird_catalog(db_id)
            graph = bird_graph(db_id)
            gen = RuleGenerator(cat, seed=seed)
            rng = random.Random(seed * 10000 + qi)
            ctype = CORRUPTION_TYPES[qi % len(CORRUPTION_TYPES)]
            bad_sql, desc = corrupt_sql(q["SQL"], cat, ctype, rng)
            if desc == "noop":
                ctype = "relation"   # 保底:换一类
                bad_sql, desc = corrupt_sql(q["SQL"], cat, ctype, rng)
                if desc == "noop":
                    continue

            for cname, (B, use_repair, use_verifier) in CONFIGS.items():
                key = f"{db_id}:{cname}"
                if key not in pipe_cache:
                    pipe = RGCVPipeline(
                        graph, bird_executor(db_id), cat, gen,
                        budget_factory=lambda B=B: BudgetController(total=B),
                        cfg={"use_repair": use_repair,
                             "use_verifier": use_verifier,
                             "retrieval_budget_tokens": int(B * 0.6),
                             "diff_repair_rounds": 1 if use_verifier else 0})
                    if graph.table_embs is not None:
                        pipe.retriever._table_idx._emb = graph.table_embs
                    pipe_cache[key] = pipe
                pipe = pipe_cache[key]
                rec = pipe.run(q["question"], gold_sql=q["SQL"],
                               injected_sql=bad_sql)
                verdict = ex.ex(rec.final_sql, q["SQL"])
                tok_total = sum(v["tokens"] for v in rec.cost.values())
                traces.append({
                    "question_id": q["question_id"], "db_id": db_id,
                    "difficulty": q["difficulty"], "corruption": ctype,
                    "corruption_desc": desc, "config": cname,
                    "ex": verdict, "executed_ok": rec.executed_ok,
                    "tokens": tok_total,
                    "tokens_by_phase": {k: v["tokens"]
                                        for k, v in rec.cost.items()},
                    "latency_s": rec.latency_s,
                    "repair_steps": rec.repair.get("n_steps", 0),
                    "repair_fallback": rec.repair.get("fallback", False),
                    "verify_verdict": rec.verification.get("verdict", "-"),
                    "events": rec.events,
                })
            n_q += 1
        except Exception as e:
            print(f"[e4] {db_id} q{q['question_id']} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (qi + 1) % 10 == 0:
            print(f"[e4] {qi + 1}/{len(subset)} questions ({n_q} ok) "
                  f"{time.perf_counter() - t_start:.1f}s", flush=True)

    # ---- 汇总
    summary = {}
    for cname in CONFIGS:
        rows = [t for t in traces if t["config"] == cname]
        if not rows:
            continue
        exs = [t["ex"] for t in rows]
        known = [e for e in exs if e is not None]
        toks = [t["tokens"] for t in rows]
        lats = [t["latency_s"] for t in rows]
        steps = [t["repair_steps"] for t in rows]
        summary[cname] = {
            "n": len(rows),
            "EX_strict": round(sum(1 for e in known if e) / max(len(rows), 1),
                               4),
            "exec_ok_rate": round(
                sum(1 for t in rows if t["executed_ok"]) / len(rows), 4),
            "tokens_mean": round(statistics.mean(toks), 1),
            "latency_mean_s": round(statistics.mean(lats), 3),
            "repair_steps_mean": round(statistics.mean(steps), 2),
            "verify_alarm_rate": round(
                sum(1 for t in rows
                    if t["verify_verdict"] in ("alarm", "reject")) /
                len(rows), 4),
        }

    # 配对 bootstrap:同题不同配置
    by_q: Dict = {}
    for t in traces:
        by_q.setdefault(t["question_id"], {})[t["config"]] = t["ex"]
    stats = {}
    for a, b in (("B8000_full", "B8000_open"), ("B8000_full", "B8000_rgc"),
                 ("B32000_full", "B1000_full")):
        pairs = [(by_q[i].get(a), by_q[i].get(b)) for i in by_q
                 if a in by_q[i] and b in by_q[i]]
        pa = [x for x, _ in pairs]
        pb = [y for _, y in pairs]
        stats[f"{a}_vs_{b}"] = paired_bootstrap(pa, pb, seed=0)

    save_json("e4_pipeline_bird.json", {
        "config": {"n_questions": n, "seed": seed,
                   "scenario": "受控注入损坏 SQL 作为闭环起点(轮转 8 类)",
                   "configs": {k: {"budget": v[0], "repair": v[1],
                                   "verify": v[2]}
                               for k, v in CONFIGS.items()}},
        "summary": summary,
        "significance": stats,
        "traces": traces,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    run(a.n)
