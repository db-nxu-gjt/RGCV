"""E1 / RQ1 / A1:模式检索实验(y.docx 5.6-E1、5.7-A1)。

数据:
  (a) BIRD dev 分层子集:gold 表/列/join 从 gold SQL 解析(别名感知);
  (b) Spider2.0-lite 本地 SQLite 子集:gold 表来自官方 gold tables 文件。

配置(A1 消融,均在同一 SchemaGraph 上):
  full        全 schema 注入(无检索;token 上界、召回下界参照)
  flat_bm25   平面 BM25(无稠密召回/重排/两阶段/值签名)
  no_rerank   混合召回但不重排
  no_2stage   表级同 sg_rag,列级退化为全库平面检索
  no_vsig     关闭值签名匹配
  sg_rag      完整两阶段 SchemaGraph-RAG(默认 k1=6, k2=12)

指标:表 P/R/F1;列 P/R(仅 BIRD,别名解析 gold);join 对覆盖率;
     注入 token(预算 2000)与相对全 schema 压缩比;检索时延。

运行:python experiments/exp_retrieval.py [--n-bird 80] [--budget 2000]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BIRD_DB, DATA, RESULTS, bird_catalog, bird_graph,  # noqa: E402
                    load_bird_questions, save_json, stratified_subset)

from rgcv.budget import count_tokens  # noqa: E402
from rgcv.dbs import Catalog, load_sqlite_catalog  # noqa: E402
from rgcv.evalx import mean_ci95  # noqa: E402
from rgcv.schema_graph import SchemaGraphRAG, build_graph  # noqa: E402
from rgcv import embedding  # noqa: E402

CONFIGS: Dict[str, Optional[dict]] = {
    "full": None,
    "flat_bm25": dict(use_dense=False, use_rerank=False,
                      use_two_stage=False, use_value_sig=False),
    "no_rerank": dict(use_rerank=False),
    "no_2stage": dict(use_two_stage=False),
    "no_vsig": dict(use_value_sig=False),
    "sg_rag": dict(),
}

SPIDER2_SQLITE = DATA / "spider2_sqlite"
SPIDER2_QS = DATA / "spider2-lite-official.jsonl"
SPIDER2_GOLD = DATA / "spider2-lite-gold-tables.jsonl"
GRAST = DATA / "grast" / "spider2_lite_samples.json"


# ---------------------------------------------------------------- gold 解析
class GoldResolver:
    """大小写归一的目录查找。"""

    def __init__(self, catalog: Catalog):
        self.tmap = {t.lower(): t for t in catalog.tables}
        self.cols = {tl: {c.name.lower() for c in catalog.tables[t]}
                     for tl, t in self.tmap.items()}

    def has_col(self, t_lower: str, c_lower: str) -> bool:
        return c_lower in self.cols.get(t_lower, set())


def gold_triples(sql: str, res: GoldResolver
                 ) -> Optional[Tuple[Set[str], Set[Tuple[str, str]],
                                     Set[Tuple[str, str]]]]:
    """gold SQL → (表集, 列集, join 对),别名感知、全小写。"""
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    amap: Dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        amap[(t.alias or t.name).lower()] = t.name.lower()
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    cols: Set[Tuple[str, str]] = set()
    for c in tree.find_all(exp.Column):
        name = c.name.lower()
        if c.table:
            real = amap.get(c.table.lower())
            if real and real in tables:
                cols.add((real, name))
        else:
            matched = [t for t in tables if res.has_col(t, name)]
            if len(matched) == 1:
                cols.add((matched[0], name))
    joins: Set[Tuple[str, str]] = set()
    for eq in tree.find_all(exp.EQ):
        l, r = eq.this, eq.expression
        if isinstance(l, exp.Column) and isinstance(r, exp.Column):
            tl = amap.get(l.table.lower()) if l.table else None
            tr = amap.get(r.table.lower()) if r.table else None
            if tl and tr and tl != tr:
                joins.add((tl, tr))
                joins.add((tr, tl))
    return tables, cols, joins


def induced_pairs(catalog: Catalog, tables_lower: Set[str]) -> Set[Tuple[str, str]]:
    """检索表子集诱导的连接边(FK + 隐式连接),双向、小写。"""
    tl = tables_lower
    pairs: Set[Tuple[str, str]] = set()
    for fk in catalog.fks:
        a, b = fk.src[0].lower(), fk.dst[0].lower()
        if a in tl and b in tl and a != b:
            pairs.add((a, b))
            pairs.add((b, a))
    return pairs


def implicit_pairs(catalog: Catalog) -> Set[Tuple[str, str]]:
    """全库隐式连接对(含 FK,用于 full 配置)。"""
    pairs: Set[Tuple[str, str]] = set()
    for fk in catalog.fks:
        a, b = fk.src[0].lower(), fk.dst[0].lower()
        if a != b:
            pairs.add((a, b))
            pairs.add((b, a))
    return pairs


def full_schema_tokens(catalog: Catalog) -> int:
    lines = []
    for t in catalog.tables:
        cols = catalog.columns_of(t)
        cmt = "; ".join(f"{c.name}: {c.comment}" for c in cols if c.comment)
        head = f"-- table {t} (score 1.00)"
        if cmt:
            head += f" -- {cmt[:400]}"
        lines.append(head)
        lines.append("   columns: " + ", ".join(c.name for c in cols))
    return count_tokens("\n".join(lines))


def _prf(pred: Set, gold: Set) -> Tuple[float, float, float]:
    if not pred or not gold:
        p = 1.0 if not pred and not gold else 0.0
        return p, (1.0 if not gold else 0.0), 0.0
    inter = pred & gold
    p = len(inter) / len(pred)
    r = len(inter) / len(gold)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def boot_diff(vals_a: List[float], vals_b: List[float], n_boot: int = 5000,
              seed: int = 0) -> Dict:
    """配对 bootstrap:mean(a) - mean(b) 与 95% CI(连续指标)。"""
    n = len(vals_a)
    if n == 0:
        return {"diff_mean": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistics.mean(vals_a[i] for i in idx) -
                     statistics.mean(vals_b[i] for i in idx))
    diffs.sort()
    d = statistics.mean(vals_a) - statistics.mean(vals_b)
    return {"diff_mean": round(d, 4),
            "ci95": [round(diffs[int(0.025 * n_boot)], 4),
                     round(diffs[int(0.975 * n_boot)], 4)]}


# ---------------------------------------------------------------- 运行
def make_rags(graph, k1: int, k2: int) -> Dict[str, SchemaGraphRAG]:
    """各配置实例;共享一个 HybridIndex(含预计算表嵌入)避免重复编码。"""
    rags: Dict[str, SchemaGraphRAG] = {}
    shared: Optional[embedding.HybridIndex] = None
    for name, cfg in CONFIGS.items():
        if cfg is None:
            rags[name] = None
            continue
        r = SchemaGraphRAG(graph, k1=k1, k2_per_table=k2, **cfg)
        if shared is None:
            shared = r._table_idx
            if graph.table_embs is not None and shared._emb is None:
                shared._emb = graph.table_embs   # 复用 build_graph 的编码
        else:
            r._table_idx = shared  # 只读共享
        rags[name] = r
    return rags


def eval_question(question: str, gold_tables: Set[str],
                  gold_cols: Optional[Set[Tuple[str, str]]],
                  gold_joins: Set[Tuple[str, str]], catalog: Catalog,
                  rags: Dict[str, SchemaGraphRAG], graph,
                  budget: int) -> Dict:
    rec: Dict = {}
    all_lower = {t.lower() for t in catalog.tables}
    full_pairs = induced_pairs(catalog, all_lower) | implicit_pairs(catalog)
    all_cols_lower = {(t.lower(), c.name.lower())
                      for t in catalog.tables
                      for c in catalog.columns_of(t)}
    full_tok = full_schema_tokens(catalog)
    gold_tabs_norm = {t.lower() for t in gold_tables}

    for name, rag in rags.items():
        t0 = time.perf_counter()
        if name == "full":
            tabs = all_lower
            cols = all_cols_lower
            joins = full_pairs
            tokens, latency = full_tok, 0.0
            overflow, stages = 0, {}
        else:
            res = rag.retrieve(question)
            rag.inject(res, budget_tokens=budget)
            tabs = {t.lower() for t, _ in res.tables}
            cols = {(h.table.lower(), h.column.lower())
                    for h in res.columns}
            joins = induced_pairs(catalog, tabs)
            tokens = res.prompt_tokens
            overflow = len(res.overflow_tables)
            latency = res.latency_s
            stages = res.stage_latency
        tp, tr, tf = _prf(tabs, gold_tabs_norm)
        m = {"table_P": round(tp, 4), "table_R": round(tr, 4),
             "table_F1": round(tf, 4), "tokens": tokens,
             "latency_s": round(latency, 4)}
        if gold_cols is not None:
            cp, cr, _ = _prf(cols, gold_cols)
            m["col_P"], m["col_R"] = round(cp, 4), round(cr, 4)
        if gold_joins:
            cov = len(joins & gold_joins) / len(gold_joins)
            m["join_cov"] = round(cov, 4)
        if name != "full":
            m["overflow_tables"] = overflow
            m["n_tables_retrieved"] = len(tabs)
        m["wall_s"] = round(time.perf_counter() - t0, 4)
        rec[name] = m
    return rec


def run_bird(n: int, budget: int, traces: List, k1: int = 6, k2: int = 12):
    questions = load_bird_questions()
    subset = stratified_subset(questions, n=n, seed=0)
    _t0 = time.perf_counter()
    rags_cache: Dict[str, Dict] = {}
    n_done = 0
    for qi, q in enumerate(subset):
        db_id = q["db_id"]
        try:
            if db_id not in rags_cache:
                cat = bird_catalog(db_id)
                graph = bird_graph(db_id)
                rags_cache[db_id] = {
                    "catalog": cat, "graph": graph,
                    "resolver": GoldResolver(cat),
                    "rags": make_rags(graph, k1, k2),
                }
            ctx = rags_cache[db_id]
            gt = gold_triples(q["SQL"], ctx["resolver"])
            if gt is None or not gt[0]:
                continue
            tables, cols, joins = gt
            rec = eval_question(q["question"], tables, cols, joins,
                                ctx["catalog"], ctx["rags"], ctx["graph"],
                                budget)
            rec.update({"dataset": "bird", "db_id": db_id,
                        "question_id": q["question_id"],
                        "difficulty": q["difficulty"],
                        "gold_n_tables": len(tables)})
            traces.append(rec)
            n_done += 1
        except Exception as e:  # 单题失败不中断
            print(f"[bird] {db_id} q{q['question_id']} ERROR {type(e).__name__}"
                  f": {e}", flush=True)
        if (qi + 1) % 5 == 0 or qi < 3:
            print(f"[bird] {qi + 1}/{len(subset)} ({n_done} ok) "
                  f"{time.perf_counter() - _t0:.1f}s", flush=True)


def run_spider2(budget: int, traces: List, k1: int = 6, k2: int = 12,
                n_max: int = 999):
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
    rags_cache: Dict[str, Dict] = {}
    n_done = 0
    todo = [q for q in qs if q["db"] in locals_ok][:n_max]
    _t0 = time.perf_counter()
    for qi, q in enumerate(todo):
        db = q["db"]
        try:
            if db not in rags_cache:
                cat = load_sqlite_catalog(
                    str(SPIDER2_SQLITE / f"{db}.sqlite"))
                graph = build_graph(cat, use_semantic=True)
                rags_cache[db] = {
                    "catalog": cat, "graph": graph,
                    "resolver": GoldResolver(cat),
                    "rags": make_rags(graph, k1, k2),
                }
            ctx = rags_cache[db]
            gt = gold.get(q["instance_id"], set())
            if not gt:
                continue
            rec = eval_question(q["question"], gt, None, set(),
                                ctx["catalog"], ctx["rags"], ctx["graph"],
                                budget)
            rec.update({"dataset": "spider2lite", "db_id": db,
                        "instance_id": q["instance_id"],
                        "gold_n_tables": len(gt)})
            traces.append(rec)
            n_done += 1
        except Exception as e:
            print(f"[sp2] {db} {q['instance_id']} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (qi + 1) % 5 == 0 or qi < 3:
            print(f"[sp2] {qi + 1}/{len(todo)} ({n_done} ok) "
                  f"{time.perf_counter() - _t0:.1f}s", flush=True)


def run_grast(budget: int, traces: List, k1: int = 6, k2: int = 12,
              n_max: int = 999):
    """GRAST 处理版 Spider2-lite(233 题,含 gold SQL/used_columns/schema)。

    云端(bigquery/snowflake)大 schema 企业级场景:schema 字符串构造合成
    目录(仅名称,无注释/值签名);gold 列 = used_columns;gold 表 =
    used_columns 的表前缀。
    """
    from rgcv.dbs import ColumnMeta
    items = json.loads(GRAST.read_text(encoding="utf-8"))
    cache: Dict[str, Dict] = {}
    n_done = 0
    _t0 = time.perf_counter()
    for ii, item in enumerate(items[:n_max]):
        db = item["db_id"]
        try:
            if db not in cache:
                schema = item["schema"]
                if isinstance(schema, str):
                    schema = ast.literal_eval(schema)
                tables: Dict[str, List] = {}
                for tc in schema:
                    t, c = tc.split(".", 1)
                    tables.setdefault(t, []).append(
                        ColumnMeta(t, c, "TEXT"))
                cat = Catalog("cloud", tables, [], str(GRAST))
                graph = build_graph(cat, use_semantic=True)
                cache[db] = {"catalog": cat, "graph": graph,
                             "rags": make_rags(graph, k1, k2)}
            ctx = cache[db]
            used = item["used_columns"]
            if isinstance(used, str):
                used = ast.literal_eval(used)
            gold_cols = set()
            for tc in used:
                t, c = tc.split(".", 1)
                gold_cols.add((t.lower(), c.lower()))
            gold_tabs = {t for t, _ in gold_cols}
            rec = eval_question(item["question"], gold_tabs, gold_cols,
                                set(), ctx["catalog"], ctx["rags"],
                                ctx["graph"], budget)
            rec.update({"dataset": "spider2lite_grast", "db_id": db,
                        "instance_id": item["instance_id"],
                        "db_type": item.get("db_type", "?"),
                        "gold_n_tables": len(gold_tabs),
                        "n_db_tables": ctx["catalog"].n_tables})
            traces.append(rec)
            n_done += 1
        except Exception as e:
            print(f"[grast] {db} {item['instance_id']} ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
        if (ii + 1) % 10 == 0 or ii < 3:
            print(f"[grast] {ii + 1}/{min(len(items), n_max)} ({n_done} ok) "
                  f"{time.perf_counter() - _t0:.1f}s", flush=True)


def summarize(traces: List, budget: int) -> Tuple[Dict, Dict]:
    by_ds: Dict[str, List] = {}
    for t in traces:
        by_ds.setdefault(t["dataset"], []).append(t)
    summaries, stats = {}, {}
    metric_keys = ["table_P", "table_R", "table_F1", "col_P", "col_R",
                   "join_cov", "tokens", "latency_s"]
    for ds, rows in by_ds.items():
        s = {"n": len(rows)}
        for cfg in CONFIGS:
            m: Dict[str, Dict] = {}
            for key in metric_keys:
                vals = [r[cfg][key] for r in rows
                        if key in r.get(cfg, {})]
                if vals:
                    mean, (lo, hi) = mean_ci95(vals)
                    m[key] = {"mean": round(mean, 4),
                              "ci95": [round(lo, 4), round(hi, 4)]}
            if "full" in rows[0] and "tokens" in m:
                full_tok = statistics.mean(
                    [r["full"]["tokens"] for r in rows])
                cfg_tok = statistics.mean(
                    [r[cfg]["tokens"] for r in rows])
                m["compression_vs_full"] = round(
                    full_tok / max(cfg_tok, 1), 2)
            s[cfg] = m
        summaries[ds] = s
        # 显著性:sg_rag vs flat_bm25(表召回、join 覆盖)
        for key in ("table_R", "join_cov", "tokens"):
            va = [r["sg_rag"].get(key) for r in rows if key in r["sg_rag"]]
            vb = [r["flat_bm25"].get(key) for r in rows
                  if key in r["flat_bm25"]]
            if va and vb:
                stats[f"{ds}_sg_rag_vs_flat_bm25_{key}"] = boot_diff(va, vb)
        va = [r["sg_rag"]["table_R"] for r in rows]
        vb = [r["no_rerank"]["table_R"] for r in rows]
        stats[f"{ds}_sg_rag_vs_no_rerank_table_R"] = boot_diff(va, vb)
    return summaries, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bird", type=int, default=80)
    ap.add_argument("--n-sp2", type=int, default=999)
    ap.add_argument("--n-grast", type=int, default=999)
    ap.add_argument("--budget", type=int, default=2000)
    a = ap.parse_args()
    traces: List = []
    print("=== BIRD dev retrieval ===", flush=True)
    run_bird(a.n_bird, a.budget, traces)
    print("=== Spider2-lite local retrieval ===", flush=True)
    run_spider2(a.budget, traces, n_max=a.n_sp2)
    print("=== Spider2-lite GRAST (cloud schemas) ===", flush=True)
    run_grast(a.budget, traces, n_max=a.n_grast)
    summaries, stats = summarize(traces, a.budget)
    save_json("e1_retrieval.json", {
        "config": {"n_bird": a.n_bird, "budget_tokens": a.budget,
                   "k1": 6, "k2_per_table": 12,
                   "configs": {k: v for k, v in CONFIGS.items()}},
        "summary": summaries,
        "significance": stats,
        "traces": traces,
    })
