"""模块一:结构感知的分层模式检索器 SchemaGraph-RAG(y.docx 4.1)。

- 模式异构图(定义 2):表/列/值签名/注释节点;属于/引用/共现/语义相似边。
- 两阶段粗细检索(4.1.2):表级 BM25+BGE-M3 混合召回 + BGE-reranker 重排,
  列级细检 + 值签名匹配;隐式连接发现(列名/类型对齐挖掘候选 join 路径)。
- 成本感知注入(4.1.3):预算内按得分降序装填,注释优先,超出部分生成
  "索引化摘要"供修复阶段懒加载。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .budget import count_tokens
from .dbs import Catalog, ColumnMeta
from . import embedding


# ------------------------------------------------------------------ 图构建
@dataclass
class SchemaGraph:
    """轻量异构图:邻接表表示。"""
    catalog: Catalog
    # 语义相似边:table -> [similar tables]  (BGE-M3 嵌入 kNN, top-3)
    semantic_edges: Dict[str, List[str]] = field(default_factory=dict)
    # 隐式连接边(列名+类型对齐):((t1,c1),(t2,c2))
    implicit_joins: List[Tuple[Tuple[str, str], Tuple[str, str]]] = field(
        default_factory=list)
    table_docs: Dict[str, str] = field(default_factory=dict)
    col_docs: Dict[Tuple[str, str], str] = field(default_factory=dict)
    table_embs: object = None          # BGE-M3 表文档嵌入(供索引复用)
    _built: bool = False


def build_table_doc(catalog: Catalog, table: str) -> str:
    cols = catalog.columns_of(table)
    names = ", ".join(c.name for c in cols[:60])
    doc = f"Table {table} columns: {names}."
    comments = [c.comment for c in cols if c.comment]
    if comments:
        doc += " Comments: " + "; ".join(comments[:40])
    return doc


def build_col_doc(col: ColumnMeta) -> str:
    doc = f"{col.table}.{col.name} type={col.dtype}"
    if col.comment:
        doc += f" comment={col.comment}"
    if col.top_values:
        doc += f" values={col.top_values[:8]}"
    if col.n_distinct is not None:
        doc += f" ndistinct~{col.n_distinct}"
    return doc


def discover_implicit_joins(catalog: Catalog,
                            name_match: bool = True) -> List[
        Tuple[Tuple[str, str], Tuple[str, str]]]:
    """隐式连接发现(4.1.1):按列名与类型挖掘候选 join 路径。

    规则:t1.a 与 t2.b 同名(或 t2 的 <单数化表名>_id 形态)且类型相容。
    表对级去重:同一表对仅保留一条桥接列(连接路径搜索只需一条桥,
    千列级云库中同名列组会产生百万级列对,如 google_dei 23k 列 → 134 万)。
    """
    out: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    explicit = {fk.src for fk in catalog.fks} | {fk.dst for fk in catalog.fks}
    col_index: Dict[str, List[ColumnMeta]] = {}
    for t, cols in catalog.tables.items():
        for c in cols:
            col_index.setdefault(c.name.lower(), []).append(c)
    seen_table_pairs: Set[Tuple[str, str]] = set()
    for name, group in col_index.items():
        if len(group) < 2 or name in ("id", "name", "type", "code", "status",
                                      "date", "value", "description"):
            continue
        seen_pairs: Set[Tuple[str, str]] = set()
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a.table == b.table:
                    continue
                if a.dtype.lower() != b.dtype.lower():
                    continue
                key = tuple(sorted((a.table, b.table)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if key in seen_table_pairs:
                    continue
                if (a.table, a.name) in explicit or (b.table, b.name) in explicit:
                    continue
                seen_table_pairs.add(key)
                out.append(((a.table, a.name), (b.table, b.name)))
    return out


def build_graph(catalog: Catalog, use_semantic: bool = True,
                use_implicit_joins: bool = True) -> SchemaGraph:
    g = SchemaGraph(catalog=catalog)
    g.table_docs = {t: build_table_doc(catalog, t) for t in catalog.tables}
    g.col_docs = {(c.table, c.name): build_col_doc(c)
                  for cols in catalog.tables.values() for c in cols}
    if use_implicit_joins:
        g.implicit_joins = discover_implicit_joins(catalog)
    if use_semantic:
        # BGE-M3 表嵌入 kNN 建语义相似边(嵌入留存供检索索引复用)
        try:
            names = list(g.table_docs)
            embs = embedding._dense_model().encode(
                [g.table_docs[n] for n in names],
                normalize_embeddings=True, show_progress_bar=False)
            import numpy as np
            g.table_embs = embs
            sim = embs @ embs.T
            for i, n in enumerate(names):
                order = np.argsort(-sim[i])[1:4]
                g.semantic_edges[n] = [names[j] for j in order
                                       if sim[i, j] > 0.55]
        except Exception:
            g.semantic_edges = {}
            g.table_embs = None
    g._built = True
    return g


# ------------------------------------------------------------------ 检索结果
@dataclass
class ColumnHit:
    table: str
    column: str
    score: float
    value_matched: bool = False


@dataclass
class RetrievalResult:
    question: str
    tables: List[Tuple[str, float]] = field(default_factory=list)
    columns: List[ColumnHit] = field(default_factory=list)
    join_paths: List[str] = field(default_factory=list)
    prompt: str = ""
    prompt_tokens: int = 0
    overflow_tables: List[str] = field(default_factory=list)
    latency_s: float = 0.0
    stage_latency: Dict[str, float] = field(default_factory=dict)

    @property
    def table_set(self) -> Set[str]:
        return {t for t, _ in self.tables}

    @property
    def column_set(self) -> Set[Tuple[str, str]]:
        return {(h.table, h.column) for h in self.columns}


# ------------------------------------------------------------------ 检索器
def full_schema_result(graph: SchemaGraph, question: str) -> RetrievalResult:
    """全 schema 注入对照(消融):免检索压缩,所有表全列 + 全部 FK 连接。

    格式镜像 inject()(表注释行 + columns 行 + join 行)但不设 token 预算、
    不裁剪列——即"全 schema 直出"控制组,用于隔离检索压缩注入的增益。
    """
    cat = graph.catalog
    lines: List[str] = []
    for tname in cat.tables:
        cols = cat.columns_of(tname)
        ccols = [c for c in cols if c.comment]
        cmt = (" -- " + "; ".join(f"{c.name}: {c.comment}"
                                  for c in ccols[:10])) if ccols else ""
        lines.append(f"-- table {tname} (full schema){cmt}")
        lines.append("   columns: " + ", ".join(c.name for c in cols))
    joins: List[str] = []
    seen: Set[str] = set()
    for fk in cat.fks:
        jp = f"{fk.src[0]}.{fk.src[1]} = {fk.dst[0]}.{fk.dst[1]}"
        if jp not in seen:
            seen.add(jp)
            joins.append(jp)
            lines.append(f"-- join: {jp}")
    res = RetrievalResult(
        question=question,
        tables=[(t, 1.0) for t in cat.tables],
        columns=[ColumnHit(c.table, c.name, 1.0)
                 for cols in cat.tables.values() for c in cols],
        join_paths=joins)
    res.prompt = "\n".join(lines)
    res.prompt_tokens = count_tokens(res.prompt)
    return res


class SchemaGraphRAG:
    """两阶段粗细检索器。config 支持消融开关(A1)。"""

    def __init__(self, graph: SchemaGraph, k1: int = 6, k2_per_table: int = 12,
                 use_bm25: bool = True, use_dense: bool = True,
                 use_rerank: bool = True, use_two_stage: bool = True,
                 use_value_sig: bool = True, use_comments: bool = True,
                 value_ngram: Tuple[int, int] = (2, 4)):
        self.g = graph
        self.k1 = k1
        self.k2_per_table = k2_per_table
        self.cfg = dict(use_bm25=use_bm25, use_dense=use_dense,
                        use_rerank=use_rerank, use_two_stage=use_two_stage,
                        use_value_sig=use_value_sig, use_comments=use_comments)
        self.value_ngram = value_ngram
        self._table_names = list(graph.table_docs)
        self._table_idx = embedding.HybridIndex(
            self._table_names, list(graph.table_docs.values()))
        self._col_idx = None  # 懒建(列级)

    def _col_index(self):
        if self._col_idx is None:
            keys = [f"{t}.{c}" for (t, c) in self.g.col_docs]
            self._col_idx = embedding.HybridIndex(
                keys, list(self.g.col_docs.values()))
        return self._col_idx

    # ---- 值签名匹配:问题 n-gram 与采样值对齐
    def _value_hits(self, question: str) -> Dict[Tuple[str, str], float]:
        hits: Dict[Tuple[str, str], float] = {}
        if not self.cfg["use_value_sig"]:
            return hits
        import re
        toks = re.findall(r"[A-Za-z]+", question)
        ngrams: Set[str] = set()
        lo, hi = self.value_ngram
        for n in range(lo, hi + 1):
            for i in range(len(toks) - n + 1):
                ngrams.add(" ".join(toks[i:i + n]).lower())
        q_lower = question.lower()
        for (t, c), doc in self.g.col_docs.items():
            for v in self.g.catalog.column(t, c).top_values or []:
                vl = str(v).strip().lower()
                if not vl:
                    continue
                if vl in q_lower or vl in ngrams:
                    hits[(t, c)] = max(hits.get((t, c), 0.0), 1.0 +
                                       min(len(vl), 20) / 20)
                    break
        return hits

    def retrieve(self, question: str) -> RetrievalResult:
        t0 = time.perf_counter()
        res = RetrievalResult(question=question)

        # ---------- 阶段一:表级粗检(混合召回 + 重排)
        cand = self._table_idx.search(
            question, use_bm25=self.cfg["use_bm25"],
            use_dense=self.cfg["use_dense"], topk=max(self.k1 * 3, 15))
        t1 = time.perf_counter()
        if self.cfg["use_rerank"] and len(cand) > self.k1:
            # 仅重排前 12 候选(CPU 重排开销 ∝ 对数×长度)
            cand_rr = cand[:12]
            idxs = [self._table_names.index(k) for k, _ in cand_rr]
            rr = embedding.rerank(
                question, [self.g.table_docs[self._table_names[i]]
                           for i in idxs], topk=self.k1)
            tables = [(cand_rr[i][0], s) for i, s in rr]
        else:
            tables = cand[:self.k1]
        t2 = time.perf_counter()
        res.tables = tables[: self.k1]
        res.stage_latency["stage1_search"] = round(t1 - t0, 4)
        res.stage_latency["stage1_rerank"] = round(t2 - t1, 4)

        # 值签名命中先行(可为表级提供额外证据)
        vhits = self._value_hits(question)

        # ---------- 阶段二:列级细检(仅候选表内)
        t3 = time.perf_counter()
        col_hits: List[ColumnHit] = []
        if self.cfg["use_two_stage"]:
            for tname, tscore in res.tables:
                cols = self.g.catalog.columns_of(tname)
                docs = [self.g.col_docs[(tname, c.name)] for c in cols]
                from rank_bm25 import BM25Okapi
                toks = [embedding.tokenize(d) for d in docs]
                bm = BM25Okapi(toks) if any(toks) else None
                qtoks = embedding.tokenize(question)
                base = np_array(bm.get_scores(qtoks)) if bm else None
                for i, c in enumerate(cols):
                    s = float(base[i]) if base is not None else 0.0
                    s = s / (base.max() + 1e-9) if base is not None and base.max() > 0 else 0.0
                    vm = vhits.get((tname, c.name), 0.0)
                    col_hits.append(ColumnHit(
                        tname, c.name, 0.5 * s + 0.5 * tscore + vm,
                        value_matched=bool(vm)))
                col_hits.sort(key=lambda h: -h.score)
        else:
            # 消融:退化为无两阶段的平面列检索(全库范围)。
            # 注:列级稠密检索需编码全库列文档,大 schema(european_football_2
            # 数百列/Spider2 云端千级列)CPU 上单库即分钟级——不可行本身
            # 即两阶段设计(先表后列,列级仅在候选表内做轻量 BM25)的动机
            # 证据;故本消融的平面列检索采用 BM25。
            flat = self._col_index().search(
                question, use_bm25=self.cfg["use_bm25"],
                use_dense=False, topk=self.k1 * self.k2_per_table)
            for k, s in flat:
                t, c = k.split(".", 1)
                col_hits.append(ColumnHit(t, c, s))
        t4 = time.perf_counter()
        res.stage_latency["stage2"] = round(t4 - t3, 4)

        # 列裁剪:每表保留 top-k2,但值命中列强制保留
        keep: List[ColumnHit] = []
        per_table: Dict[str, int] = {}
        for h in col_hits:
            n = per_table.get(h.table, 0)
            if h.value_matched or n < self.k2_per_table:
                keep.append(h)
                per_table[h.table] = n + 1
        res.columns = keep

        # ---------- join 路径:显式 FK + 隐式连接 + 语义边
        res.join_paths = self._join_paths(res.table_set)
        res.latency_s = round(time.perf_counter() - t0, 4)
        return res

    def _join_paths(self, tables: Set[str]) -> List[str]:
        import networkx as nx
        edges: List[Tuple[str, str, str]] = []
        for fk in self.g.catalog.fks:
            if fk.src[0] in tables and fk.dst[0] in tables:
                edges.append((fk.src[0], fk.dst[0],
                              f"{fk.src[0]}.{fk.src[1]} = "
                              f"{fk.dst[0]}.{fk.dst[1]}"))
        for (a, b) in self.g.implicit_joins:
            if a[0] in tables and b[0] in tables:
                edges.append((a[0], b[0], f"{a[0]}.{a[1]} = {b[0]}.{b[1]} "
                                          "(implicit)"))
        if not edges:
            return []
        G = nx.Graph()
        for u, v, label in edges:
            G.add_edge(u, v, label=label)
        paths = []
        for u in tables:
            for v in tables:
                if u < v and G.has_node(u) and G.has_node(v):
                    try:
                        p = nx.shortest_path(G, u, v)
                        if len(p) > 1:
                            labels = [G[p[i]][p[i + 1]]["label"]
                                      for i in range(len(p) - 1)]
                            paths.append(" -- ".join(labels))
                    except nx.NetworkXNoPath:
                        pass
        return sorted(set(paths))

    # ---------------------------------------------------------------- 注入
    def inject(self, res: RetrievalResult, budget_tokens: int,
               include_values: bool = True) -> str:
        """成本感知注入(4.1.3):得分降序装填,注释优先,溢出转索引化摘要。"""
        lines: List[str] = []
        used = 0
        col_by_table: Dict[str, List[ColumnHit]] = {}
        for h in res.columns:
            col_by_table.setdefault(h.table, []).append(h)

        def add(line: str) -> bool:
            nonlocal used
            t = count_tokens(line)
            if used + t > budget_tokens:
                return False
            lines.append(line)
            used += t
            return True

        for tname, tscore in res.tables:
            cat = self.g.catalog
            cmt = ""
            # 注释优先(文献[3]:注释是性价比最高增益项)
            if self.cfg["use_comments"]:
                ccols = [c for c in cat.columns_of(tname) if c.comment]
                cmt = " -- " + "; ".join(
                    f"{c.name}: {c.comment}" for c in ccols[:10])
            if not add(f"-- table {tname} (score {tscore:.2f}){cmt}"):
                res.overflow_tables.append(tname)
                continue
            hits = col_by_table.get(tname, [])
            if hits:
                collist = ", ".join(
                    f"{h.column}" + (f"/*{h.score:.2f}*/" if False else "")
                    for h in hits)
                ok = add(f"   columns: {collist}")
                if not ok:
                    # 索引化摘要:表名 + 一行摘要
                    add(f"   [summary] {tname}(" +
                        ", ".join(h.column for h in hits[:20]) + ")")
                    res.overflow_tables.append(tname)
            if include_values:
                for h in hits:
                    if h.value_matched:
                        vals = (self.g.catalog.column(h.table, h.column)
                                .top_values or [])[:6]
                        if vals:
                            add(f"   values {h.table}.{h.column} in "
                                f"{vals}")
        for jp in res.join_paths[:8]:
            add(f"-- join: {jp}")
        res.prompt = "\n".join(lines)
        res.prompt_tokens = count_tokens(res.prompt)
        return res.prompt


def np_array(x):
    import numpy as np
    return np.asarray(x)
