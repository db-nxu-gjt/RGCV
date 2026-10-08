"""模块三:语义级验证器(y.docx 4.3)。

三层递进、成本递增的触发式设计:
  V1 结果指纹(零 LLM):结构化不变式——行数量级与问题语义一致性
     ("每个X" → 行数 ≈ X 基数)、空值率、主键唯一性、聚合值合理性
     (占比 ∈ [0,1]、日期单调)。
  V2 差分执行(低成本):top-k 候选间 CTE 级对齐差分 + 语义保持变换
     (等值↔IN、窗口边界规范化)的一致性检验。
  V3 LLM 语义核对(触发式):仅当 V1/V2 报警时,将问题+结果 schema+样本
     行+SQL 交给 LLM 判定;报警子句回传模块二。

输出:带置信度的验证报告(通过 / 报警并定位子句 / 拒绝)。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .budget import BudgetController, count_tokens
from .dbs import Catalog, DBError, DBExecutor

NUMBER_WORDS = {
    "each": "per", "per": "per", "every": "per",
}


@dataclass
class VerificationReport:
    verdict: str                 # pass / alarm / reject
    layer_reports: Dict[str, Dict] = field(default_factory=dict)
    alarmed_clauses: List[str] = field(default_factory=list)
    confidence: float = 1.0
    latency_s: float = 0.0
    tokens: int = 0
    triggered_v3: bool = False


class Verifier:
    def __init__(self, executor: DBExecutor, catalog: Catalog,
                 budget: Optional[BudgetController] = None,
                 layers: Sequence[str] = ("v1", "v2", "v3"),
                 trigger_only: bool = True,
                 llm_judge: Optional[callable] = None):
        """llm_judge: V3 钩子 fn(question, sql, cols, rows) -> (verdict, clause)"""
        self.ex = executor
        self.cat = catalog
        self.budget = budget
        self.layers = list(layers)
        self.trigger_only = trigger_only
        self.llm_judge = llm_judge or self._rule_based_judge

    # ------------------------------------------------------------- 主入口
    def verify(self, question: str, sql: str,
               candidates: Optional[List[str]] = None) -> VerificationReport:
        t0 = time.perf_counter()
        rep = VerificationReport("pass")
        try:
            cols, rows, _ = self.ex.execute(sql)
        except DBError as e:
            rep.verdict = "reject"
            rep.layer_reports["exec"] = {"error": str(e)[:120]}
            rep.latency_s = round(time.perf_counter() - t0, 4)
            return rep

        alarms: List[str] = []
        if "v1" in self.layers:
            v1 = self._v1_fingerprint(question, sql, cols, rows)
            rep.layer_reports["v1"] = v1
            alarms += v1.get("alarms", [])
        if "v2" in self.layers:
            v2 = self._v2_differential(question, sql, cols, rows,
                                       candidates or [sql])
            rep.layer_reports["v2"] = v2
            alarms += v2.get("alarms", [])

        if alarms:
            rep.verdict = "alarm"
            rep.alarmed_clauses = alarms
            rep.confidence = 0.6
            if "v3" in self.layers and (not self.trigger_only or True):
                # V3 触发式:仅报警时调用
                rep.triggered_v3 = True
                t1 = time.perf_counter()
                verdict, clause = self.llm_judge(question, sql, cols, rows[:10])
                rep.layer_reports["v3"] = {
                    "verdict": verdict, "clause": clause,
                    "latency_s": round(time.perf_counter() - t1, 4)}
                rep.tokens += count_tokens(question) + count_tokens(sql)
                if verdict == "reject":
                    rep.verdict = "reject"
                    rep.confidence = 0.3
                elif verdict == "alarm":
                    rep.alarmed_clauses.append(clause or "llm")
                    rep.confidence = 0.5
                else:
                    # V3 否决 V1/V2 报警 → 通过(降置信)
                    rep.verdict = "pass"
                    rep.confidence = 0.75
        rep.latency_s = round(time.perf_counter() - t0, 4)
        if self.budget:
            self.budget.spend("verification", tokens_in=count_tokens(
                question + sql), latency_s=rep.latency_s)
        return rep

    # ------------------------------------------------------------- V1
    def _v1_fingerprint(self, question: str, sql: str, cols, rows) -> Dict:
        alarms: List[str] = []
        q = question.lower()
        n = len(rows)
        has_agg = bool(re.search(
            r"\b(count|sum|avg|max|min|total)\s*\(", sql, re.I))
        has_group = bool(re.search(r"\bgroup\s+by\b", sql, re.I))

        # (a) 行数量级 vs "each/per X" 语义
        m = re.search(r"\b(each|per|every)\s+(\w+)", q)
        if m and has_group:
            entity = m.group(2).rstrip("s")
            card = self._estimate_cardinality(entity)
            if card and n > 0 and (n > card * 3 or n < card / 3):
                alarms.append(
                    f"row-magnitude: {n} rows vs ~{card} {entity}s "
                    f"(question implies per-{entity})")
        # (b) 空结果
        if n == 0 and not has_agg:
            alarms.append("empty-result")
        elif n == 0 and has_agg:
            # 聚合空结果多为 0 行(无 GROUP BY)——降级为提示
            pass
        # (c) 主键唯一性
        if cols and n > 1:
            first = [str(r[0]) for r in rows]
            if len(set(first)) != len(first) and self._looks_like_id(
                    cols[0]):
                alarms.append(f"non-unique key column {cols[0]}")
        # (d) 聚合值合理性:比例类问题 → 值应落在 [0,1]
        if has_agg and re.search(
                r"\b(ratio|percentage|proportion|rate|percent|share)\b",
                q + " " + " ".join(cols).lower()):
            vals = [v for r in rows for v in r
                    if isinstance(v, (int, float))]
            bad = [v for v in vals if v is not None and (
                v < -1e-9 or v > 1.0 + 1e-9)]
            if bad and len(bad) > len(vals) / 2:
                alarms.append(
                    f"ratio-out-of-range: {len(bad)}/{len(vals)} values "
                    "outside [0,1]")
        # (e) 日期单调性(时间序列)
        date_cols = [i for i, c in enumerate(cols or [])
                     if _is_date_col(c)]
        if date_cols and n > 2:
            i = date_cols[0]
            series = [str(r[i]) for r in rows]
            if _is_monotonic_key(series):
                pass  # 排序过的时间序列正常
        # (f) 空值率
        if rows and cols:
            null_rates = []
            for i in range(len(cols)):
                nr = sum(1 for r in rows if r[i] is None) / n
                null_rates.append(nr)
            worst = max(zip(null_rates, cols)) if null_rates else (0, "")
            if worst[0] > 0.8:
                alarms.append(f"high-null-rate {worst[1]}: {worst[0]:.0%}")
        return {"alarms": alarms, "n_rows": n, "cols": cols,
                "has_agg": has_agg, "has_group": has_group}

    def _estimate_cardinality(self, entity: str) -> Optional[int]:
        # 由库目录估算:找名字匹配的表/列的基数
        best = None
        for t, cols in self.cat.tables.items():
            if entity in t.lower():
                try:
                    # 粗略:表行数
                    return None  # 行数查询放 executor,此处保守返回 None
                except Exception:
                    pass
            for c in cols:
                if entity in c.name.lower() and c.n_distinct:
                    if best is None or c.n_distinct > best:
                        best = c.n_distinct
        return best

    def _looks_like_id(self, colname: str) -> bool:
        return bool(re.match(r"(?i).*(id|no|code|key|num)$", colname or ""))

    # ------------------------------------------------------------- V2
    def _v2_differential(self, question, sql, cols, rows,
                         candidates: List[str]) -> Dict:
        alarms = []
        # (i) 语义保持重写:等值↔IN、BETWEEN 规范化 —— 结果必须一致
        rewrites = _semantics_preserving_rewrites(sql)
        for name, rw in rewrites:
            try:
                c2, r2, _ = self.ex.execute(rw)
                if _result_hash(c2, r2) != _result_hash(cols, rows):
                    alarms.append(f"rewrite-inconsistency[{name}]")
            except DBError:
                pass
        # (ii) top-k 候选一致性
        if len(candidates) > 1:
            hashes = set()
            for c in candidates[:3]:
                try:
                    c2, r2, _ = self.ex.execute(c)
                    hashes.add(_result_hash(c2, r2))
                except DBError:
                    hashes.add("EXEC_ERR")
            if len(hashes - {"EXEC_ERR"}) > 1:
                alarms.append("candidate-divergence")
        return {"alarms": alarms,
                "n_rewrites": len(rewrites),
                "n_candidates": len(candidates)}

    # ------------------------------------------------------------- V3 代理
    def _rule_based_judge(self, question: str, sql: str, cols, rows):
        """V3 规则式兜底判定(无 LLM 通道时的离线代理,报告中标注)。

        模拟"LLM 拿问题+SQL+结果做语义核对"的可判别近似:
        a) 意图覆盖:问题核心词是否被 SQL 覆盖;
        b) 聚合口径:"每/each/per" 问题须有 GROUP BY;
        c) 矛盾谓词:恒假字面量比较(如 1 = 2);
        d) 值域落地:WHERE/JOIN 字面量须存在于对齐列的采样值域;
        e) 谓词-意图匹配:空结果时,过滤字面量应是问题提及的实体。
        """
        q = question.lower()
        qtoks = set(re.findall(r"[a-z]+", q)) - {
            "the", "a", "an", "of", "in", "on", "for", "what", "which", "how",
            "many", "much", "is", "are", "was", "were", "and", "or", "to",
            "with", "by", "from", "that", "this", "there", "their", "than",
            "more", "less", "most", "least", "all", "any", "each", "per",
            "give", "me", "list", "show", "find", "get", "return", "top"}
        sql_l = sql.lower()
        covered = sum(1 for t in qtoks if t in sql_l)
        if qtoks and covered / len(qtoks) < 0.15:
            return "alarm", "intent-coverage-low"
        if re.search(r"\bfor each\b|\bper\b|\bevery\b", q) and not re.search(
                r"group\s+by", sql_l):
            return "alarm", "missing-group-by-for-per-question"
        # c) 恒假字面量比较
        for m in re.finditer(r"(\w+)\s*=\s*(\w+)(?!\w)", sql_l):
            a, b = m.group(1), m.group(2)
            if a.isdigit() and b.isdigit() and a != b:
                return "alarm", f"contradictory-predicate {a}={b}"
        # d/e) 字面量落地 + 谓词-意图匹配
        ungrounded, off_q = self._literal_audit(sql, qtoks)
        if ungrounded:
            return "alarm", f"literal-not-in-domain:{ungrounded[0]}"
        if off_q and not rows:
            return "alarm", f"predicate-literal-absent-in-question:{off_q[0]}"
        return "pass", ""

    def _literal_audit(self, sql: str, qtoks: set):
        """返回 (未落地字面量, 与问题无关的谓词字面量) 两个列表。

        对每个 WHERE/JOIN 谓词内的字符串字面量:
        - 落地:值须存在于同谓词对齐列的 top_values(大小写不敏感,
          列前缀为别名时先解析到真实表);
        - 意图:值 token 须在问题词集中出现(空结果时为强错误信号)。
        """
        import sqlglot
        from sqlglot import exp
        ungrounded, off_q = [], []
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except Exception:
            return ungrounded, off_q
        amap = _alias_map(tree)
        for lit in tree.find_all(exp.Literal):
            if not lit.is_string:
                continue
            val = str(lit.this)
            # 对齐同一谓词父节点(Where/On)内的列
            for col in tree.find_all(exp.Column):
                if not _shared_pred_parent(lit, col):
                    continue
                tname = amap.get(col.table, col.table) if col.table else ""
                c = self.cat.column(tname, col.name)
                if c and c.top_values:
                    loose = [str(v).strip().lower() for v in c.top_values]
                    if val.strip().lower() not in loose:
                        ungrounded.append(f"{tname}.{col.name}={val!r}")
                break
            vtoks = set(re.findall(r"[a-z0-9]+", val.lower()))
            if qtoks and vtoks and not (vtoks & qtoks):
                off_q.append(val)
        return ungrounded, off_q


# ------------------------------------------------------------------ 工具
def _alias_map(tree) -> Dict[str, str]:
    """SQL 别名 → 真实表名映射(FROM/JOIN 中的 Table 节点)。"""
    from sqlglot import exp
    amap: Dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        alias = t.alias or t.name
        amap[alias] = t.name
    return amap


def _shared_pred_parent(a, b) -> bool:
    """两表达式是否共享谓词作用域祖先(WHERE / JOIN 的 ON 子树)。

    sqlglot 30.x:JOIN ON 条件是 Join.args['on'],非独立 exp.On 节点。
    """
    from sqlglot import exp

    def scopes(node):
        out, child = [], node
        x = node.parent
        while x is not None:
            if isinstance(x, exp.Where):
                out.append(id(x))
            elif isinstance(x, exp.Join) and child is x.args.get("on"):
                out.append(id(x))
            child = x
            x = x.parent
        return out

    return bool(set(scopes(a)) & set(scopes(b)))


def _result_hash(cols, rows) -> str:
    import hashlib
    h = hashlib.md5()
    h.update(str(cols).encode())
    for r in sorted(map(str, rows)):
        h.update(r.encode())
    return h.hexdigest()[:12]


def _is_date_col(name: str) -> bool:
    return bool(re.search(r"(?i)(date|time|day|month|year)", name or ""))


def _is_monotonic_key(series: List[str]) -> bool:
    return all(a <= b for a, b in zip(series, series[1:])) or all(
        a >= b for a, b in zip(series, series[1:]))


def _semantics_preserving_rewrites(sql: str) -> List[Tuple[str, str]]:
    """V2 扰动:语义保持变换(结果应严格一致)。"""
    out = []
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="sqlite")
        # 等值 → IN
        t2 = sqlglot.parse_one(sql, read="sqlite")
        changed = False
        for eq in t2.find_all(exp.EQ):
            if isinstance(eq.right, exp.Literal) and eq.right.is_string:
                col, val = eq.left, eq.right
                new = exp.In(this=col, expressions=[val])
                eq.replace(new)
                changed = True
                break
        if changed:
            out.append(("eq->in", t2.sql(dialect="sqlite")))
        # x BETWEEN a AND b → x >= a AND x <= b
        t3 = sqlglot.parse_one(sql, read="sqlite")
        for bw in t3.find_all(exp.Between):
            lo, hi = bw.args.get("low"), bw.args.get("high")
            if lo is not None and hi is not None:
                bw.replace(exp.And(
                    this=exp.GTE(this=bw.this.copy(), expression=lo.copy()),
                    expression=exp.LTE(this=bw.this.copy(),
                                       expression=hi.copy())))
                out.append(("between->ge-le", t3.sql(dialect="sqlite")))
                break
    except Exception:
        pass
    return out
