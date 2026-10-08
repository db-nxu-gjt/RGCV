"""LLM 代理生成器与受控错误注入器。

【复现约束声明】y.docx 的生成器骨干为 TRAE 平台内置国产 LLM
(DeepSeek-V4-Pro / Kimi-K2.6 / DeepSeek-V4-Flash);本环境无脚本化
LLM API 通道(详见 REPORT 环境审计),故按两种可复现代理实现:

1. RuleGenerator:检索条件化的规则式合成器(确定性,种子可控),
   消费模块一输出(表/列/值命中/join 路径),产出候选 SQL;
2. corrupt_sql:对 gold SQL 施加受控错误(对应 y.docx 错误分布:
   模式链接 27.6% / 数据分析逻辑 35.5% / JOIN 8.3%),用于修复
   循环(RQ2)的受控评测——该方法独立于生成器质量,是 SafeQL 式
   修复评测的标准做法。

两者的组合使 R-G-C-V 全链路可在零 LLM 成本下离线运行;LLM 位点
以可插拔钩子保留(set_regenerator / llm_judge)。
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .dbs import Catalog
from .schema_graph import RetrievalResult

AGG_PATTERNS = [
    (r"\bhow many\b|\bnumber of\b|\bcount\b", "COUNT"),
    (r"\baverage\b|\bavg\b|\bmean\b", "AVG"),
    (r"\btotal\b|\bsum\b|\bcombined\b", "SUM"),
    (r"\bhighest\b|\bmaximum\b|\bmax\b|\blargest\b|\bmost\b", "MAX"),
    (r"\blowest\b|\bminimum\b|\bmin\b|\bsmallest\b|\bleast\b", "MIN"),
]

ORDER_WORDS = {
    "highest": "DESC", "most": "DESC", "largest": "DESC", "maximum": "DESC",
    "top": "DESC", "lowest": "ASC", "least": "ASC", "fewest": "ASC",
    "minimum": "ASC", "smallest": "ASC",
}


@dataclass
class GenerationOutput:
    sql: str
    n_candidates: int
    tokens_prompt: int = 0
    tokens_out: int = 0
    candidates: List[str] = None

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = [self.sql]


class RuleGenerator:
    """检索条件化的规则式 SQL 合成器(确定性)。"""

    def __init__(self, catalog: Catalog, seed: int = 0):
        self.cat = catalog
        self.rng = random.Random(seed)

    @staticmethod
    def _q(name: str) -> str:
        """SQLite 标识符引号(BIRD 列名含空格/括号)。"""
        if re.fullmatch(r"[A-Za-z_]\w*", name or ""):
            return name
        return '"' + (name or "").replace('"', '""') + '"'

    def generate(self, question: str, retrieval: RetrievalResult,
                 n_candidates: int = 1) -> GenerationOutput:
        q = question.lower()
        cands = []
        for i in range(n_candidates):
            cands.append(self._one(q, question, retrieval, variant=i))
        return GenerationOutput(
            sql=cands[0], n_candidates=n_candidates, candidates=cands,
            tokens_prompt=retrieval.prompt_tokens,
            tokens_out=sum(len(c) // 8 for c in cands))  # 粗记 token

    def _one(self, q: str, question: str, r: RetrievalResult,
             variant: int = 0) -> str:
        # ---- 聚合意图
        agg = None
        for pat, fn in AGG_PATTERNS:
            if re.search(pat, q):
                agg = fn
                break
        # ---- "each/per X" → GROUP BY
        m = re.search(r"\b(each|per|every)\s+([a-z]+)", q)
        group_col = None
        if m:
            entity = m.group(2)
            group_col = self._find_col(r, entity)

        # ---- 主表 + join(FROM 作用域内的表集合)
        main = r.tables[0][0] if r.tables else list(self.cat.tables)[0]
        scope = {main}
        join_clause = ""
        if len(r.tables) > 1 and r.join_paths:
            first = r.join_paths[0].replace("(implicit)", "").strip()
            m2 = re.match(r"([\w.]+) = ([\w.]+)", first)
            if m2:
                a, b = m2.group(1), m2.group(2)
                t2 = b.split(".")[0] if a.split(".")[0] == main else \
                    a.split(".")[0]
                if t2 != main and t2 in {t for t, _ in r.tables}:
                    join_clause = f"\nJOIN {self._q(t2)} ON " \
                        f"{self._q(a.split('.')[0])}." \
                        f"{self._q(a.split('.')[1])} = " \
                        f"{self._q(b.split('.')[0])}." \
                        f"{self._q(b.split('.')[1])}"
                    scope.add(t2)

        # ---- 作用域过滤后的列
        in_scope = [h for h in r.columns if h.table in scope]

        # ---- 实体值命中 → WHERE(仅作用域内表)
        conds: List[str] = []
        for h in in_scope:
            if h.value_matched:
                c = self.cat.column(h.table, h.column)
                if c and c.top_values:
                    v = c.top_values[variant % len(c.top_values)]
                    conds.append(f"{self._q(h.table)}.{self._q(h.column)} "
                                 f"= '{v}'")
        # ---- 度量列:数值型或聚合列(仅作用域内)
        metric = self._find_metric(in_scope, q)

        sel_parts = []
        if agg:
            target = metric or "*"
            sel_parts.append(f"{agg}({target})")
        else:
            cols = [f"{self._q(h.table)}.{self._q(h.column)}"
                    for h in in_scope[:4]] or ["*"]
            sel_parts.extend(cols)
        if group_col and agg:
            sel_parts.insert(0, group_col)

        sql = f"SELECT {', '.join(sel_parts)}\nFROM {self._q(main)}" + \
            join_clause
        if conds:
            sql += "\nWHERE " + " AND ".join(conds[:2])
        if group_col and agg:
            sql += f"\nGROUP BY {group_col}"
        # 排序 / LIMIT
        om = re.search(r"\b(highest|lowest|most|least|top|largest|smallest)\b",
                       q)
        if om and not agg:
            sql += f"\nORDER BY {metric or 1} " \
                f"{ORDER_WORDS.get(om.group(1), 'DESC')}"
        lm = re.search(r"\btop\s+(\d+)|\bfirst\s+(\d+)", q)
        if lm:
            k = lm.group(1) or lm.group(2) or "5"
            sql += f"\nLIMIT {k}"
        elif om and not agg:
            sql += "\nLIMIT 1"
        return sql + ";"

    def _find_col(self, r: RetrievalResult, word: str) -> Optional[str]:
        for h in r.columns:
            if word in h.column.lower() or h.column.lower().startswith(word):
                return f"{self._q(h.table)}.{self._q(h.column)}"
        return None

    def _find_metric(self, columns, q: str) -> Optional[str]:
        qtoks = set(re.findall(r"[a-z]+", q))
        best, score = None, 0
        for h in columns:
            c = self.cat.column(h.table, h.column)
            if not c:
                continue
            if c.dtype and ("int" in c.dtype.lower() or "real" in
                            c.dtype.lower() or "num" in c.dtype.lower()
                            or "float" in c.dtype.lower() or "double" in
                            c.dtype.lower()):
                s = sum(1 for t in qtoks if t in h.column.lower())
                s += 0.5 if h.value_matched else 0
                if s > score:
                    best, score = f"{self._q(h.table)}." \
                        f"{self._q(h.column)}", s
        return best


# ------------------------------------------------------------------ 受控错误注入
CORRUPTION_TYPES = [
    "attribute",      # attribute 错误:列替换为同表相似列
    "relation",       # relation 错误:表替换为相似表
    "value",          # value 错误:字面量替换为不存在值
    "function",       # function 错误:函数替换为方言误用
    "join",           # join 错误:ON 条件列错位
    "predicate_extra",   # 谓词多余(空结果类)
    "predicate_missing", # 谓词缺失(静默错误类)
    "group_missing",     # 聚合口径错误(静默错误类)
]


def _alias_map(tree) -> Dict[str, str]:
    """FROM/JOIN 别名 → 真实表名(BIRD gold SQL 大量使用 T1/T2 别名)。"""
    from sqlglot import exp
    amap: Dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        alias = t.alias or t.name
        amap[alias] = t.name
    return amap


def corrupt_sql(gold: str, catalog: Catalog, kind: str,
                rng: random.Random) -> Tuple[str, str]:
    """对 gold SQL 施加单点受控错误,返回 (corrupted, 描述)。"""
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(gold, read="sqlite")
    except Exception:
        # 解析失败时做文本级注入
        if kind == "value":
            return _text_corrupt_value(gold, rng), "value@text"
        if kind == "function":
            return _text_corrupt_function(gold), "function@text"
        return gold, "noop"

    amap = _alias_map(tree)
    desc = "noop"
    if kind == "attribute":
        # 仅选可解析到真实表的列(避免别名退化 noop)
        real_cols = []
        for c in tree.find_all(exp.Column):
            tname = amap.get(c.table, c.table) if c.table else None
            if tname and tname in catalog.tables and catalog.column(
                    tname, c.name):
                real_cols.append((c, tname))
        if real_cols:
            c, tname = rng.choice(real_cols)
            sim = [x for x in catalog.columns_of(tname) if x.name != c.name]
            if sim:
                nc = rng.choice(sim)
                c.set("this", sqlglot.exp.to_identifier(nc.name))
                desc = f"attr {tname}.{c.name}->{nc.name}"
    elif kind == "relation":
        tabs = [t for t in tree.find_all(exp.Table)]
        if tabs:
            t = rng.choice(tabs)
            others = [x for x in catalog.tables if x != t.name]
            if others:
                nt = rng.choice(others[:20])
                t.set("this", sqlglot.exp.to_identifier(nt))
                desc = f"rel {t.name}->{nt}"
    elif kind == "value":
        for lit in tree.find_all(exp.Literal):
            if lit.is_string:
                old = str(lit.this)
                lit.set("this", old + "_XYZ")
                desc = f"value {old!r}->{old + '_XYZ'!r}"
                break
    elif kind == "function":
        for fn in tree.find_all(exp.Func):
            name = fn.sql_name().upper()
            mapping = {"COUNT": "TOTAL", "STRFTIME": "JULIANDAY",
                       "AVG": "MEDIAN", "SUM": "TOTAL"}
            if name in mapping:
                # 文本级替换更稳
                sql = tree.sql(dialect="sqlite")
                out = re.sub(rf"\b{name}\s*\(", f"{mapping[name]}(",
                             sql, count=1, flags=re.I)
                return out, f"func {name}->{mapping[name]}"
    elif kind == "join":
        for eq in tree.find_all(exp.EQ):
            cs = list(eq.find_all(exp.Column))
            if len(cs) == 2 and cs[0].table and cs[1].table and \
                    cs[0].table != cs[1].table:
                # 交换 ON 两侧的列(错位 join)
                a, b = cs[0].name, cs[1].name
                cs[0].set("this", sqlglot.exp.to_identifier(b))
                cs[1].set("this", sqlglot.exp.to_identifier(a))
                desc = f"join-swap {a}<->{b}"
                break
    elif kind == "predicate_extra":
        wh = tree.find(exp.Where)
        if wh is not None:
            extra = exp.condition("1 = 2")
            wh.set("this", exp.And(this=wh.this.copy(), expression=extra))
            desc = "predicate+1=2 (empty-result)"
    elif kind == "predicate_missing":
        wh = tree.find(exp.Where)
        if wh is not None and isinstance(wh.this, exp.And):
            wh.set("this", wh.this.args.get("this"))
            desc = "predicate-dropped"
    elif kind == "group_missing":
        gb = tree.find(exp.Group)
        if gb is not None:
            gb.pop()
            desc = "group-by-dropped (silent)"
    return tree.sql(dialect="sqlite"), desc


def _tables_of(tree) -> List[str]:
    from sqlglot import exp
    return [t.name for t in tree.find_all(exp.Table)]


def _text_corrupt_value(sql: str, rng) -> str:
    m = list(re.finditer(r"'([^']*)'", sql))
    if m:
        i = rng.randrange(len(m))
        return sql[:m[i].start()] + "'" + m[i].group(1) + "_XYZ'" + \
            sql[m[i].end():]
    return sql


def _text_corrupt_function(sql: str) -> str:
    for a, b in [("COUNT", "TOTAL"), ("SUM", "TOTAL"), ("AVG", "MEDIAN")]:
        if re.search(rf"\b{a}\b", sql, re.I):
            return re.sub(rf"\b{a}\b", b, sql, count=1, flags=re.I)
    return sql
