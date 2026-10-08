"""评测指标(y.docx 5.5)。

主指标:EX(执行准确率);检索专项(基于 gold tables/used_columns):
表/列 P@k、R@k、join 命中率、注入 token 数;修复专项:EX 增量、执行
错误消除率、步数、重生成触发率;验证专项:静默错误拦截率、误报率、
分层增量延迟;统计:配对 McNemar(α=0.05)+ 配对 bootstrap(10^4)。
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .dbs import DBError, DBExecutor


# ------------------------------------------------------------------ EX
def _norm_val(v, tol: float = 1e-4):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return round(float(v), 6)
    if isinstance(v, bytes):
        return v[:64]
    s = str(v)
    return s.strip()


def _res_multiset(cols, rows) -> List[Tuple]:
    return sorted((tuple(_norm_val(x) for x in r) for r in rows),
                  key=lambda t: tuple(str(x) for x in t))


def exec_equal(res_a, res_b, float_tol: float = 1e-6) -> bool:
    """结果多重集比较(浮点容差)。res = (cols, rows, n)"""
    if res_a is None or res_b is None:
        return False
    a, b = _res_multiset(*res_a[:2]), _res_multiset(*res_b[:2])
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if isinstance(x, float) and isinstance(y, float):
                if abs(x - y) > float_tol * max(1.0, abs(x), abs(y)):
                    return False
            elif x != y:
                return False
    return True


@dataclass
class EXEvaluator:
    executor: DBExecutor
    _cache: Dict[str, tuple] = None

    def __post_init__(self):
        self._cache = {}

    def gold_result(self, gold_sql: str):
        if gold_sql not in self._cache:
            try:
                self._cache[gold_sql] = self.executor.execute(gold_sql)
            except DBError:
                self._cache[gold_sql] = None
        return self._cache[gold_sql]

    def ex(self, pred_sql: str, gold_sql: str) -> Optional[bool]:
        """True/False = 判对/判错;None = 预测执行失败(显式错误)。"""
        gold = self.gold_result(gold_sql)
        if gold is None:
            return None  # gold 无法执行:跳过该题
        try:
            pred = self.executor.execute(pred_sql)
        except DBError:
            return None
        return exec_equal(pred, gold)


# ------------------------------------------------------------------ 检索指标
def precision_at_k(pred: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    pk = set(pred[:k])
    return len(pk & gold) / len(pk) if pk else 0.0


def recall_at_k(pred: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold or k <= 0:
        return 0.0
    pk = set(pred[:k])
    return len(pk & gold) / len(gold)


def join_hit_rate(pred_joins: List[str], gold_join_pairs: Set[Tuple]) -> float:
    """pred_joins: 'a.x = b.y' 字符串;gold_join_pairs: {(a,b), ...} 表对。"""
    if not gold_join_pairs:
        return 0.0
    hit = set()
    import re
    for j in pred_joins:
        m = re.findall(r"(\w+)\.", j)
        if len(m) >= 2:
            pair = (m[0], m[1])
            hit.add(pair)
            hit.add((pair[1], pair[0]))
    return len(hit & gold_join_pairs) / len(gold_join_pairs)


def gold_tables_from_sql(sql: str) -> Set[str]:
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return {t.name for t in tree.find_all(exp.Table)}
    except Exception:
        import re
        return set(re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)", sql,
                              re.I))


def gold_columns_from_sql(sql: str) -> Set[Tuple[str, str]]:
    import sqlglot
    from sqlglot import exp
    out: Set[Tuple[str, str]] = set()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        for c in tree.find_all(exp.Column):
            if c.table:
                out.add((c.table, c.name))
            else:
                tables = {t.name for t in tree.find_all(exp.Table)}
                out.add(("*", c.name))
                out.update((t, c.name) for t in tables)
    except Exception:
        pass
    return out


def gold_joins_from_sql(sql: str) -> Set[Tuple[str, str]]:
    import sqlglot
    from sqlglot import exp
    pairs: Set[Tuple[str, str]] = set()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        for eq in tree.find_all(exp.EQ):
            cols = [c for c in eq.find_all(exp.Column) if c.table]
            if len(cols) == 2 and cols[0].table != cols[1].table:
                pairs.add((cols[0].table, cols[1].table))
                pairs.add((cols[1].table, cols[0].table))
    except Exception:
        pass
    return pairs


# ------------------------------------------------------------------ 统计
def mcnemar_test(pred_a: Sequence[Optional[bool]],
                 pred_b: Sequence[Optional[bool]]) -> Dict:
    """配对 McNemar:仅统计两者均可判定的题(非 None)。

    b = a对b错, c = a错b对;统计量 (b-c)^2/(b+c), 或精确二项检验。
    """
    pairs = [(a, b) for a, b in zip(pred_a, pred_b)
             if a is not None and b is not None]
    b = sum(1 for a, x in pairs if a and not x)
    c = sum(1 for a, x in pairs if x and not a)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "n_discordant": 0}
    from scipy.stats import binomtest
    # 精确二项检验(McNemar exact):b ~ Bin(n, 0.5) under H0
    p = binomtest(min(b, c), n, 0.5).pvalue * 1.0
    return {"b": b, "c": c, "p_value": round(p, 6), "n_discordant": n}


def paired_bootstrap(pred_a: Sequence[Optional[bool]],
                     pred_b: Sequence[Optional[bool]],
                     n_boot: int = 10_000, seed: int = 0) -> Dict:
    import random
    rng = random.Random(seed)
    pairs = [(a, b) for a, b in zip(pred_a, pred_b)
             if a is not None and b is not None]
    if not pairs:
        return {"diff_mean": 0.0, "ci95": [0.0, 0.0]}
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        ea = sum(1 for a, _ in sample if a) / n
        eb = sum(1 for _, b in sample if b) / n
        diffs.append(ea - eb)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    dmean = sum(1 for a, _ in pairs if a) / n - \
        sum(1 for _, b in pairs if b) / n
    return {"diff_mean": round(dmean, 4), "ci95": [round(lo, 4),
                                                   round(hi, 4)]}


def mean_ci95(values: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
    import statistics
    if not values:
        return 0.0, (0.0, 0.0)
    m = statistics.mean(values)
    if len(values) < 2:
        return m, (m, m)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return m, (m - 1.96 * se, m + 1.96 * se)
