"""E-H 曲线第 1 点:规则代理(弱生成器) {ungated, gated} × BIRD-300。

RuleGenerator(确定性模板合成,零 LLM 成本)+ 完整 RGCV 闭环;
gated = repair_gate=conservative(执行成功不变异)。V3 用规则裁决
(无 LLM judge)。逐题记录 ex_orig(候选起点)与 ex_final,供强度-收益
曲线与 p* 实测(Prop.4 的 α/β/η/e/p 估计)直接使用。

输出:results/rgcv_bird300_ruleproxy{,_gated}/q_<qid>.json + merged.json
用法:python run_eh_ruleproxy.py [--segment N --n_segments 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from common import bird_catalog, bird_executor, bird_graph  # noqa: E402
from eval_rgcv_e1 import run_sql  # noqa: E402

from rgcv.budget import BudgetController  # noqa: E402
from rgcv.generator import RuleGenerator  # noqa: E402
from rgcv.pipeline import RGCVPipeline  # noqa: E402

DAIL_DEV300 = (BASE.parents[1] / "baselines" / "DAIL-SQL" / "dataset"
               / "bird" / "dev" / "dev_300.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", type=int, default=-1,
                    help="-1 = 全量一次跑完")
    ap.add_argument("--n_segments", type=int, default=5)
    a = ap.parse_args()

    dev = json.loads(DAIL_DEV300.read_text(encoding="utf-8"))
    assert len(dev) == 300
    if a.segment < 0:
        idxs = list(range(len(dev)))
    else:
        per = (len(dev) + a.n_segments - 1) // a.n_segments
        idxs = list(range(a.segment * per,
                          min((a.segment + 1) * per, len(dev))))

    # gold 缓存(300 题执行一次)
    DBROOT = DAIL_DEV300.parents[1] / "database"      # .../bird/database/
    gold_cache = {}
    for i, item in enumerate(dev):
        db_id = item["db_id"]
        db_path = str(DBROOT / db_id / f"{db_id}.sqlite")
        gold_cache[i] = run_sql(db_path, item.get("SQL")
                                or item.get("query", ""))

    for gated in (False, True):
        tag = "rgcv_bird300_ruleproxy" + ("_gated" if gated else "")
        outdir = BASE / "results" / tag
        outdir.mkdir(parents=True, exist_ok=True)
        cfg = {"repair_gate": "conservative"} if gated else {}
        pipe_cache: dict = {}
        t0 = time.perf_counter()
        n_done = 0
        for k, idx in enumerate(idxs):
            item = dev[idx]
            qid = str(item["question_id"])
            fp = outdir / f"q_{qid}.json"
            if fp.exists():
                continue
            db_id = item["db_id"]
            try:
                if db_id not in pipe_cache:
                    cat = bird_catalog(db_id)
                    gen = RuleGenerator(cat, seed=0)
                    pipe = RGCVPipeline(
                        bird_graph(db_id), bird_executor(db_id), cat, gen,
                        budget_factory=lambda: BudgetController(total=32_000),
                        cfg=cfg)
                    pipe_cache[db_id] = pipe
                pipe = pipe_cache[db_id]
                question = item["question"]
                if item.get("evidence"):
                    question += f" [Hint] {item['evidence']}"
                rec = pipe.run(question, gold_sql=item.get("SQL"))
                db_path = str(DBROOT / db_id / f"{db_id}.sqlite")
                gold_res = gold_cache[idx]
                ex_final = int(run_sql(db_path, rec.final_sql) == gold_res
                               if gold_res is not None else False)
                ex_orig = int(run_sql(db_path, rec.candidates[0]) == gold_res
                              if gold_res is not None else False)
                out = {
                    "question_id": item["question_id"], "db_id": db_id,
                    "difficulty": item.get("difficulty", ""),
                    "question": question,
                    "gold_sql": item.get("SQL", ""),
                    "pred_sql": rec.final_sql or "",
                    "executed_ok": rec.executed_ok,
                    "ex_orig": ex_orig, "ex_final": ex_final,
                    "candidates": rec.candidates,
                    "repair": rec.repair,
                    "verification": rec.verification,
                    "events": rec.events,
                }
            except Exception as e:
                print(f"[{tag}] q{qid} {db_id} ERROR "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            fp.write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                     default=str), encoding="utf-8")
            n_done += 1
            if (k + 1) % 20 == 0:
                print(f"[{tag}] {k + 1}/{len(idxs)} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
        ndone = len(list(outdir.glob('q_*.json')))
        print(f"[{tag}] done: +{n_done} this run, {ndone}/300 total, "
              f"wall={time.perf_counter() - t0:.0f}s", flush=True)

        # merged.json(与 eval_rgcv_e1.load_preds 兼容)
        merged = {}
        for item in dev:
            fp = outdir / f"q_{item['question_id']}.json"
            if fp.exists():
                d = json.loads(fp.read_text(encoding="utf-8"))
                pred = "" if d.get("error") else d.get("pred_sql", "")
                merged[str(item["question_id"])] = \
                    f"{pred}\t----- bird -----\t{item['db_id']}"
        (outdir / "merged.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=1),
            encoding="utf-8")


if __name__ == "__main__":
    main()
