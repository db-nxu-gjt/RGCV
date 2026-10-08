"""实验共享工具:BIRD 数据加载、Catalog 缓存、流水线构建。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgcv import embedding  # noqa: E402
from rgcv.budget import BudgetController  # noqa: E402
from rgcv.dbs import (Catalog, DBExecutor,  # noqa: E402
                      attach_bird_descriptions, load_duckdb_catalog,
                      load_sqlite_catalog)
from rgcv.generator import RuleGenerator  # noqa: E402
from rgcv.schema_graph import SchemaGraph, build_graph  # noqa: E402

DATA = ROOT / "data"            # BIRD 数据集与 BGE 模型放此处(见 data/README.md)
RESULTS = ROOT / "results"
BIRD_DIR = DATA / "bird" / "dev_20240627"
BIRD_DB = BIRD_DIR / "dev_databases"

embedding.set_model_dirs(str(DATA / "bge-m3"),
                         str(DATA / "bge-reranker-v2-m3"))

_cat_cache: Dict[str, Catalog] = {}
_graph_cache: Dict[str, SchemaGraph] = {}


def load_bird_questions() -> List[dict]:
    return json.loads((BIRD_DIR / "dev.json").read_text(encoding="utf-8"))


def bird_catalog(db_id: str, with_comments: bool = True) -> Catalog:
    key = f"bird:{db_id}:{with_comments}"
    if key not in _cat_cache:
        cat = load_sqlite_catalog(str(BIRD_DB / db_id / f"{db_id}.sqlite"))
        if with_comments:
            attach_bird_descriptions(
                cat, str(BIRD_DB / db_id / "database_description"))
        _cat_cache[key] = cat
    return _cat_cache[key]


def bird_graph(db_id: str, with_comments: bool = True,
               rebuild: bool = False) -> SchemaGraph:
    key = f"bird:{db_id}:{with_comments}"
    if key not in _graph_cache or rebuild:
        _graph_cache[key] = build_graph(
            bird_catalog(db_id, with_comments), use_semantic=True)
    return _graph_cache[key]


def bird_executor(db_id: str) -> DBExecutor:
    return DBExecutor("sqlite", str(BIRD_DB / db_id / f"{db_id}.sqlite"))


def stratified_subset(questions: List[dict], n: int = 150,
                      seed: int = 0) -> List[dict]:
    """分层子集:按 difficulty 配额抽样(y.docx 5.9 统计效度)。"""
    rng = random.Random(seed)
    by_diff: Dict[str, List[dict]] = {}
    for q in questions:
        by_diff.setdefault(q["difficulty"], []).append(q)
    total = len(questions)
    out: List[dict] = []
    for d, items in sorted(by_diff.items()):
        k = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        out.extend(items[:k])
    return out[:n]


def default_budget(total: int = 32_000) -> BudgetController:
    return BudgetController(total=total)


def save_json(name: str, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[saved] results/{name}")
