"""E5:RGCV 真实 LLM 接入驱动(BIRD 300 题 × 双骨干)。

协议(y.docx E5 / E1 口径对齐):
  数据:dev_300.json(与 eval_chess_e1 同一 300 题分层子集,行序=idx);
  骨干:deepseek-v4-pro(t=0) / kimi-k2.6(thinking-disabled, t=0.6);
  全部调用关闭 reasoning;usage 实测记账(api_trace.jsonl 行级)。
分段运行(用户协议:300 题分 5 段,每段 60 题):
  python experiments/run_rgcv_e1.py --engine deepseek-v4-pro --segment 0
  ... --segment 4      (段可重复调用,断点续跑跳过已完成题)
冒烟:--smoke N 跑前 N 题(独立 tag,不进正式结果)。
消融:--ablation nc4|norepair|noverify|fullschema(可叠加 --gated,deepseek 侧)。
输出:results/rgcv_bird300_<tag>/{q_<qid>.json, merged.json, api_trace.jsonl}
merged.json 格式与 eval_chess_e1.load_preds 兼容:
  {qid: "sql\t----- bird -----\tdb_id"}
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

from rgcv.budget import BudgetController  # noqa: E402
from rgcv.llm import LLMClient, LLMGenerator  # noqa: E402
from rgcv.pipeline import RGCVPipeline  # noqa: E402

DAIL_DEV300 = (BASE.parents[1] / "baselines" / "DAIL-SQL" / "dataset"
               / "bird" / "dev" / "dev_300.json")

ABLATIONS = {
    "none": {},
    "nc4": {"n_candidates": 4},
    "norepair": {"use_repair": False, "diff_repair_rounds": 0},
    "noverify": {"use_verifier": False, "diff_repair_rounds": 0},
    "fullschema": {"full_schema": True},
}


def make_tag(engine: str, ablation: str, smoke: int, gated: bool = False,
             tag_suffix: str = "") -> str:
    tag = f"rgcv_bird300_{engine.replace('.', '')}"
    if gated:
        tag += "_gated"
    if ablation != "none":
        tag += f"_{ablation}"
    if smoke:
        tag += f"_smoke{smoke}"
    if tag_suffix:
        tag += tag_suffix
    return tag


def load_done(outdir: Path) -> set:
    return {p.name for p in outdir.glob("q_*.json")}


def write_merged(outdir: Path, dev) -> None:
    merged = {}
    for item in dev:
        p = outdir / f"q_{item['question_id']}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            pred = d.get("pred_sql", "")
            if d.get("error"):
                pred = ""
            merged[str(item["question_id"])] = \
                f"{pred}\t----- bird -----\t{item['db_id']}"
    (outdir / "merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")


def run(engine: str, segment: int, n_segments: int, smoke: int,
        ablation: str, gated: bool = False, tag_suffix: str = "",
        seed: int = None) -> None:
    dev = json.loads(DAIL_DEV300.read_text(encoding="utf-8"))
    assert len(dev) == 300
    if smoke:
        idxs = list(range(smoke))
    else:
        per = (len(dev) + n_segments - 1) // n_segments      # 60
        idxs = list(range(segment * per, min((segment + 1) * per, len(dev))))

    tag = make_tag(engine, ablation, smoke, gated, tag_suffix)
    outdir = BASE / "results" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = outdir / "api_trace.jsonl"
    client = LLMClient(engine, trace_path=str(trace_path))
    gen = LLMGenerator(client, repair_seed=seed)
    if seed is not None:
        print(f"[{tag}] repair-search seed={seed} "
              f"(regeneration as explicit stochastic source, t=0.6)",
              flush=True)
    done = load_done(outdir)
    cfg = dict(ABLATIONS.get(ablation, {}))
    if gated:
        # E5 结论:强 LLM 生成器下修复变异需保守门控
        # (执行成功只读,仅执行失败启动修复搜索)
        cfg["repair_gate"] = "conservative"

    # 每库缓存 pipeline(检索图/目录/执行器复用;预算每题新建)
    pipe_cache: dict = {}
    t_start = time.perf_counter()
    n_done = 0
    for k, idx in enumerate(idxs):
        item = dev[idx]
        qid = str(item["question_id"])
        fname = f"q_{qid}.json"
        if fname in done:
            continue
        db_id = item["db_id"]
        try:
            if db_id not in pipe_cache:
                cat = bird_catalog(db_id)
                graph = bird_graph(db_id)
                pipe = RGCVPipeline(
                    graph, bird_executor(db_id), cat, gen,
                    budget_factory=lambda: BudgetController(total=32_000),
                    cfg=cfg)
                if graph.table_embs is not None:   # 跳过重复嵌入(同 E4)
                    pipe.retriever._table_idx._emb = graph.table_embs
                pipe_cache[db_id] = pipe
            pipe = pipe_cache[db_id]
            client.set_qid(qid)
            question = item["question"]
            if item.get("evidence"):
                question += f" [Hint] {item['evidence']}"
            t0 = time.perf_counter()
            rec = pipe.run(question, gold_sql=item.get("SQL"))
            rec.latency_s = round(time.perf_counter() - t0, 3)
            out = {
                "question_id": item["question_id"], "db_id": db_id,
                "difficulty": item.get("difficulty", ""),
                "question": question,
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
        except Exception as e:            # 单题失败不阻塞分段;不写入 done
            err = {"question_id": item["question_id"], "db_id": db_id,
                   "error": f"{type(e).__name__}: {e}"[:300]}
            (outdir / f"q_{qid}.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"[{tag}] idx{idx} q{qid} {db_id} ERROR {err['error']}",
                  flush=True)
            # 失败题删掉错误 json,下次重跑该段时自动重试
            (outdir / f"q_{qid}.json").unlink(missing_ok=True)
            continue
        (outdir / fname).write_text(
            json.dumps(out, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        n_done += 1
        calls = out["api_usage"]["calls"]
        print(f"[{tag}] idx{idx} q{qid} {db_id} "
              f"exec={rec.executed_ok} verify={rec.verification.get('verdict', '-') if rec.verification else '-'} "
              f"calls={calls} {rec.latency_s}s "
              f"({k + 1}/{len(idxs)} seg, "
              f"{time.perf_counter() - t_start:.0f}s)", flush=True)

    write_merged(outdir, dev)
    ndone = len(load_done(outdir))
    u = client.total_usage
    print(f"[{tag}] segment {segment} done: +{n_done} this run, "
          f"{ndone}/300 total, api calls={u['calls']} "
          f"tok_in={u['tokens_in']} tok_out={u['tokens_out']}, "
          f"wall={time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="deepseek-v4-pro",
                    choices=["deepseek-v4-pro", "deepseek-v4-flash",
                             "kimi-k2.6"])
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--n_segments", type=int, default=5)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run first N questions only (separate tag)")
    ap.add_argument("--ablation", default="none", choices=list(ABLATIONS))
    ap.add_argument("--gated", action="store_true",
                    help="conservative repair gate (E5 finding: mutate only "
                         "on execution failure)")
    ap.add_argument("--tag_suffix", default="",
                    help="append suffix to output tag (E-C variance reruns: "
                         "_r1/_r2/_s1/_s2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="repair-search sampling seed (E-C: regeneration "
                         "becomes explicit stochastic source, t=0.6+seed)")
    a = ap.parse_args()
    run(a.engine, a.segment, a.n_segments, a.smoke, a.ablation, a.gated,
        a.tag_suffix, a.seed)
