"""E-D-a:改写集 RGCV 主配置重跑(双骨干,150 题)。

复用 run_rgcv_e1.py 的管线与主配置(fullschema + conservative gate),
仅两点不同:
  - 题集:results/paraphrase_150.json 中 status=auto_pass 的题(150 抽样);
  - 问题文本:question_para(+ paraphrase 后的 evidence_para),gold 不变
    —— 语义保持改写,EX 仍对原 gold 计算。

协议:与主配置同种子同参数(deepseek t=0 / kimi thinking-disabled t=0.6),
tag 加后缀 _para 与原跑隔离;断点续跑(q_<qid>.json 跳过)。
用法:
  python experiments/run_rgcv_paraphrase.py --engine deepseek-v4-pro
  python experiments/run_rgcv_paraphrase.py --engine kimi-k2.6
输出:results/rgcv_bird300_<engine>_gated_fullschema_para/{q_*.json, merged.json}
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

from run_rgcv_e1 import ABLATIONS, DAIL_DEV300, load_done, make_tag, write_merged  # noqa: E402

from rgcv.budget import BudgetController  # noqa: E402
from rgcv.llm import LLMClient, LLMGenerator  # noqa: E402
from rgcv.pipeline import RGCVPipeline  # noqa: E402
from common import bird_catalog, bird_graph, bird_executor  # noqa: E402

RES = BASE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="deepseek-v4-pro",
                    choices=["deepseek-v4-pro", "kimi-k2.6"])
    a = ap.parse_args()

    dev = json.loads(DAIL_DEV300.read_text(encoding="utf-8"))
    assert len(dev) == 300
    manifest = json.load(open(RES / "paraphrase_150.json", encoding="utf-8"))
    todo_set = {q: m for q, m in manifest.items()
                if m.get("status") == "auto_pass"}
    print(f"改写集 auto_pass: {len(todo_set)}/150")

    tag = make_tag(a.engine, "fullschema", 0, gated=True, tag_suffix="_para")
    outdir = RES / tag
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = outdir / "api_trace.jsonl"
    client = LLMClient(a.engine, trace_path=str(trace_path))
    gen = LLMGenerator(client)
    done = load_done(outdir)
    cfg = dict(ABLATIONS["fullschema"])
    cfg["repair_gate"] = "conservative"

    pipe_cache: dict = {}
    t_start = time.perf_counter()
    n_done = 0
    items = [d for d in dev if str(d["question_id"]) in todo_set]
    for k, item in enumerate(items):
        qid = str(item["question_id"])
        fname = f"q_{qid}.json"
        if fname in done:
            continue
        m = todo_set[qid]
        db_id = item["db_id"]
        try:
            if db_id not in pipe_cache:
                cat = bird_catalog(db_id)
                graph = bird_graph(db_id)
                pipe = RGCVPipeline(
                    graph, bird_executor(db_id), cat, gen,
                    budget_factory=lambda: BudgetController(total=32_000),
                    cfg=cfg)
                if graph.table_embs is not None:
                    pipe.retriever._table_idx._emb = graph.table_embs
                pipe_cache[db_id] = pipe
            pipe = pipe_cache[db_id]
            client.set_qid(qid)
            question = m["question_para"]
            ev = m.get("evidence_para") or ""
            if ev:
                question += f" [Hint] {ev}"
            t0 = time.perf_counter()
            rec = pipe.run(question, gold_sql=item.get("SQL"))
            rec.latency_s = round(time.perf_counter() - t0, 3)
            out = {
                "question_id": item["question_id"], "db_id": db_id,
                "difficulty": item.get("difficulty", ""),
                "question": question,
                "question_orig": m["question_orig"],
                "gold_sql": item.get("SQL", ""),
                "pred_sql": rec.final_sql or "",
                "executed_ok": rec.executed_ok,
                "candidates": rec.candidates,
                "retrieved": rec.retrieved,
                "repair": rec.repair,
                "verification": rec.verification,
                "cost": rec.cost,
                "latency_s": rec.latency_s,
                "events": rec.events,
                "api_usage": client.usage_for_qid(qid),
            }
        except Exception as e:
            err = {"question_id": item["question_id"], "db_id": db_id,
                   "error": f"{type(e).__name__}: {e}"[:300]}
            (outdir / f"q_{qid}.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"[{tag}] q{qid} {db_id} ERROR {err['error']}", flush=True)
            (outdir / f"q_{qid}.json").unlink(missing_ok=True)
            continue
        (outdir / fname).write_text(
            json.dumps(out, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        n_done += 1
        calls = out["api_usage"]["calls"]
        print(f"[{tag}] q{qid} {db_id} "
              f"exec={rec.executed_ok} verify={rec.verification.get('verdict', '-') if rec.verification else '-'} "
              f"calls={calls} {rec.latency_s}s "
              f"({k + 1}/{len(items)}, {time.perf_counter() - t_start:.0f}s)",
              flush=True)

    write_merged(outdir, dev)
    ndone = len(load_done(outdir))
    u = client.total_usage
    print(f"[{tag}] done: +{n_done} this run, {ndone}/{len(items)} total, "
          f"api calls={u['calls']} tok_in={u['tokens_in']} "
          f"tok_out={u['tokens_out']}, wall={time.perf_counter() - t_start:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
