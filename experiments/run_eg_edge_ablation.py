"""E-G:检索层边类型消融(P1-3 步骤 1)。

配置(均基于同一目录构造,只变图的边):
  full        全 schema 注入(无检索;token 参照)
  flat_bm25   完整退化:平面 BM25(无稠密/重排/两阶段/值签名)
  no_hetero   去异构图:图中无任何扩展边(use_semantic=F, use_implicit=F),
              两阶段流程保留——"退化两阶段 flat"
  no_cooccur  去共现边:隐式连接(同名列+类型桥接)边关闭
  no_sim      去相似度边:BGE-M3 kNN 语义相似边关闭
  sg_rag      完整 SG-RAG(k1=6, k2=12)

指标:表 P@k/R@k(F1)、列 P/R、join 覆盖率、注入 token、压缩比、时延。
数据集:Spider2-lite 本地 SQLite 子集 + GRAST(千列级云目录)。
显著性:sg_rag vs no_hetero / no_hetero vs flat_bm25(table_R, join_cov)配对
bootstrap。

输出:results/eG_edge_ablation.json
用法:python run_eg_edge_ablation.py [--budget 2000] [--skip-grast]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from exp_retrieval import (DATA, GRAST, GoldResolver, SPIDER2_GOLD,  # noqa
                           SPIDER2_QS, SPIDER2_SQLITE, eval_question,
                           gold_triples, induced_pairs, load_bird_questions,
                           stratified_subset)
from common import bird_catalog  # noqa
from rgcv.dbs import Catalog, ColumnMeta, load_sqlite_catalog  # noqa
from rgcv.evalx import mean_ci95  # noqa
from rgcv.schema_graph import SchemaGraphRAG, build_graph  # noqa

CONFIGS = ["full", "flat_bm25", "no_hetero", "no_cooccur", "no_sim",
           "sg_rag"]


def make_edge_rags(cat, k1: int, k2: int):
    """每配置独立 graph(边开关不同),表文档嵌入编码一次共享。"""
    g_full = build_graph(cat, use_semantic=True, use_implicit_joins=True)
    g_noc = build_graph(cat, use_semantic=True, use_implicit_joins=False)
    g_nos = build_graph(cat, use_semantic=False, use_implicit_joins=True)
    g_noh = build_graph(cat, use_semantic=False, use_implicit_joins=False)
    for g in (g_noc, g_nos, g_noh):
        g.table_embs = g_full.table_embs
    rags = {
        "full": None,
        "flat_bm25": SchemaGraphRAG(
            g_noh, k1=k1, k2_per_table=k2, use_dense=False, use_rerank=False,
            use_two_stage=False, use_value_sig=False),
        "no_hetero": SchemaGraphRAG(g_noh, k1=k1, k2_per_table=k2),
        "no_cooccur": SchemaGraphRAG(g_noc, k1=k1, k2_per_table=k2),
        "no_sim": SchemaGraphRAG(g_nos, k1=k1, k2_per_table=k2),
        "sg_rag": SchemaGraphRAG(g_full, k1=k1, k2_per_table=k2),
    }
    shared = rags["sg_rag"]._table_idx
    if g_full.table_embs is not None and shared._emb is None:
        shared._emb = g_full.table_embs
    for name in ("flat_bm25", "no_hetero", "no_cooccur", "no_sim"):
        rags[name]._table_idx = shared
    return rags


def boot_diff(va, vb, n_boot: int = 5000, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    n = len(va)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistics.mean(va[i] for i in idx) -
                     statistics.mean(vb[i] for i in idx))
    diffs.sort()
    return {"diff_mean": round(statistics.mean(va) - statistics.mean(vb), 4),
            "ci95": [round(diffs[int(0.025 * n_boot)], 4),
                     round(diffs[int(0.975 * n_boot)], 4)]}


def summarize(traces: List) -> Tuple[Dict, Dict]:
    by_ds: Dict[str, List] = {}
    for t in traces:
        by_ds.setdefault(t["dataset"], []).append(t)
    keys = ["table_P", "table_R", "table_F1", "col_P", "col_R", "join_cov",
            "tokens", "n_join_paths", "latency_s"]
    summaries, stats = {}, {}
    for ds, rows in by_ds.items():
        s = {"n": len(rows)}
        for cfg in CONFIGS:
            m: Dict[str, Dict] = {}
            for key in keys:
                vals = [r[cfg][key] for r in rows if key in r.get(cfg, {})]
                if vals:
                    mean, (lo, hi) = mean_ci95(vals)
                    m[key] = {"mean": round(mean, 4),
                              "ci95": [round(lo, 4), round(hi, 4)]}
            if "full" in rows[0] and "tokens" in m:
                ft = statistics.mean(r["full"]["tokens"] for r in rows)
                ct = statistics.mean(r[cfg]["tokens"] for r in rows)
                m["compression_vs_full"] = round(ft / max(ct, 1), 2)
            s[cfg] = m
        summaries[ds] = s
        for key in ("table_R", "join_cov"):
            for a, b in (("sg_rag", "no_hetero"), ("no_hetero", "flat_bm25"),
                         ("sg_rag", "flat_bm25"), ("no_sim", "no_hetero"),
                         ("no_cooccur", "no_hetero")):
                va = [r[a][key] for r in rows if key in r.get(a, {})]
                vb = [r[b][key] for r in rows if key in r.get(b, {})]
                if va and vb:
                    stats[f"{ds}_{a}_vs_{b}_{key}"] = boot_diff(va, vb)
    return summaries, stats


def eval_q(question: str, gold_tables: Set[str],
           gold_cols, gold_joins, catalog, rags, budget: int) -> Dict:
    """eval_question 包装:追加 n_join_paths(注入的 join 提示行数)。"""
    rec = eval_question(question, gold_tables, gold_cols, gold_joins,
                        catalog, rags, None, budget)
    for name, rag in rags.items():
        if rag is None:
            continue
        res = rag.retrieve(question)          # 复取只为计数(便宜,BM25)
        rec[name]["n_join_paths"] = len(res.join_paths)
    return rec


def run_bird(n: int, budget: int, traces: List, k1: int = 6, k2: int = 12):
    """BIRD dev 分层子集:gold SQL 可解析 join 对 → join_cov 可评。"""
    questions = load_bird_questions()
    subset = stratified_subset(questions, n=n, seed=0)
    cache: Dict[str, Dict] = {}
    n_done = 0
    for qi, q in enumerate(subset):
        db_id = q["db_id"]
        try:
            if db_id not in cache:
                cat = bird_catalog(db_id)
                cache[db_id] = {"catalog": cat,
                                "resolver": GoldResolver(cat),
                                "rags": make_edge_rags(cat, k1, k2)}
            ctx = cache[db_id]
            gt = gold_triples(q["SQL"], ctx["resolver"])
            if gt is None or not gt[0]:
                continue
            tables, cols, joins = gt
            rec = eval_q(q["question"], tables, cols, joins,
                         ctx["catalog"], ctx["rags"], budget)
            rec.update({"dataset": "bird", "db_id": db_id,
                        "question_id": q["question_id"],
                        "difficulty": q.get("difficulty", ""),
                        "gold_n_tables": len(tables)})
            traces.append(rec)
            n_done += 1
        except Exception as e:
            print(f"[bird] {db_id} q{q.get('question_id')} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (qi + 1) % 10 == 0:
            print(f"[bird] {qi + 1}/{len(subset)} {n_done} ok", flush=True)


def run_spider2(budget: int, traces: List, k1: int = 6, k2: int = 12):
    qs = [json.loads(l) for l in
          SPIDER2_QS.read_text(encoding="utf-8").strip().splitlines()]
    gold = {}
    for l in SPIDER2_GOLD.read_text(encoding="utf-8").strip().splitlines():
        d = json.loads(l)
        gt = d["gold_tables"]
        if isinstance(gt, str):
            try:
                gt = ast.literal_eval(gt)
            except (ValueError, SyntaxError):
                gt = []
        gold[d["instance_id"]] = {str(t) for t in (gt or [])}
    locals_ok = {f.stem for f in SPIDER2_SQLITE.glob("*.sqlite")}
    todo = [q for q in qs if q["db"] in locals_ok]
    print(f"[sp2] 本地库 {len(locals_ok)} 个,题 {len(todo)} 条")
    cache: Dict[str, Dict] = {}
    for qi, q in enumerate(todo):
        db = q["db"]
        try:
            if db not in cache:
                cat = load_sqlite_catalog(
                    str(SPIDER2_SQLITE / f"{db}.sqlite"))
                cache[db] = {"catalog": cat,
                             "resolver": GoldResolver(cat),
                             "rags": make_edge_rags(cat, 6, 12)}
            ctx = cache[db]
            gt = gold.get(q["instance_id"], set())
            if not gt:
                continue
            rec = eval_q(q["question"], gt, None, set(),
                         ctx["catalog"], ctx["rags"], budget)
            rec.update({"dataset": "spider2lite", "db_id": db,
                        "instance_id": q["instance_id"],
                        "gold_n_tables": len(gt)})
            traces.append(rec)
        except Exception as e:
            print(f"[sp2] {db} {q['instance_id']} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (qi + 1) % 10 == 0:
            print(f"[sp2] {qi + 1}/{len(todo)} "
                  f"{len(traces)} traces", flush=True)


def run_grast(budget: int, traces: List, k1: int = 6, k2: int = 12):
    items = json.loads(GRAST.read_text(encoding="utf-8"))
    print(f"[grast] {len(items)} 题")
    cache: Dict[str, Dict] = {}
    for ii, item in enumerate(items):
        db = item["db_id"]
        try:
            if db not in cache:
                schema = item["schema"]
                if isinstance(schema, str):
                    schema = ast.literal_eval(schema)
                tables: Dict[str, List] = {}
                for tc in schema:
                    t, c = tc.split(".", 1)
                    tables.setdefault(t, []).append(ColumnMeta(t, c, "TEXT"))
                cat = Catalog("cloud", tables, [], str(GRAST))
                cache[db] = {"catalog": cat,
                             "rags": make_edge_rags(cat, 6, 12)}
            ctx = cache[db]
            used = item["used_columns"]
            if isinstance(used, str):
                used = ast.literal_eval(used)
            gold_cols = set()
            for tc in used:
                t, c = tc.split(".", 1)
                gold_cols.add((t.lower(), c.lower()))
            gold_tabs = {t for t, _ in gold_cols}
            rec = eval_q(item["question"], gold_tabs, gold_cols,
                         set(), ctx["catalog"], ctx["rags"], budget)
            rec.update({"dataset": "spider2lite_grast", "db_id": db,
                        "instance_id": item["instance_id"],
                        "n_db_tables": ctx["catalog"].n_tables})
            traces.append(rec)
        except Exception as e:
            print(f"[grast] {db} {item['instance_id']} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (ii + 1) % 10 == 0:
            print(f"[grast] {ii + 1}/{len(items)} "
                  f"{len(traces)} traces", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--n-bird", type=int, default=80)
    ap.add_argument("--skip-grast", action="store_true")
    a = ap.parse_args()
    traces: List = []
    print("=== E-G: BIRD dev retrieval (join_cov evaluable) ===", flush=True)
    run_bird(a.n_bird, a.budget, traces)
    print("=== E-G: Spider2-lite local ===", flush=True)
    run_spider2(a.budget, traces)
    if not a.skip_grast:
        print("=== E-G: GRAST cloud schemas ===", flush=True)
        run_grast(a.budget, traces)
    summaries, stats = summarize(traces)
    out = BASE / "results" / "eG_edge_ablation.json"
    out.write_text(json.dumps(
        {"config": {"budget_tokens": a.budget, "k1": 6, "k2_per_table": 12,
                    "configs": CONFIGS},
         "summary": summaries, "significance": stats, "traces": traces},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {out}")
    for ds, s in summaries.items():
        print(f"\n--- {ds} (n={s['n']}) ---")
        for cfg in CONFIGS:
            m = s[cfg]
            tr = m.get("table_R", {}).get("mean")
            jc = m.get("join_cov", {}).get("mean")
            tk = m.get("tokens", {}).get("mean")
            print(f"  {cfg:<11} table_R={tr} join_cov={jc} tokens={tk}")
