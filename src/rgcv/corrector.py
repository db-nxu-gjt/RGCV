"""模块二:DBMS 深度协同修复循环(y.docx 4.2)。

四类反馈信号:
  S1 执行错误 + AST 定位(SafeQL 原有)
  S2 执行计划(新):EXPLAIN 谓词选择率反馈,驱动谓词松弛/收紧
  S3 约束校验(新):NOT NULL/UNIQUE/FK/值域(字面量是否存在于列中)
  S4 中间结果差分(新):CTE 分块中间结果与期望指纹差分

动作空间(继承 SafeQL 五类 + 扩展两类):
  relation 替换/新增、join 修正、attribute 替换、value 替换、
  function 替换 + 谓词松弛/收紧 + CTE 分块重写

搜索:best-first(语义距离 + α 结构变换惩罚),K>=3 保底候选,
深度上限 d、步数预算,超阈值回退 LLM 重生成(混合策略);
定理 1 终止条件:动作空间有限、深度受限、每步至少消除一个已定位
错误节点(语义距离单调不增)。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp

from .budget import BudgetController, count_tokens
from .dbs import Catalog, DBError, DBExecutor

ALPHA_STRUCT = 0.35          # 结构变换惩罚(y.docx 4.2.3: α ≈ 0.3–0.4)
K_FALLBACK = 3              # 保底候选数(K ≥ 3)
MAX_DEPTH = 8               # 搜索深度上限 d(定理 1 条件)
MAX_STEPS = 40              # 每题修复步数预算(计入总预算)


# ------------------------------------------------------------------ 信号
@dataclass
class Signal:
    kind: str                 # s1_error / s2_plan / s3_constraint / s4_diff
    located: List[str]        # 定位到的错误元素(表.列 / 字面量 / 函数名)
    detail: str = ""
    hint: str = ""            # 建议动作提示


def signal_s1(error: DBError, sql: str, catalog: Catalog) -> Signal:
    """S1:错误消息 + AST 定位。"""
    msg = str(error)
    located: List[str] = []
    m = re.search(r"no such column: (.+)$", msg, re.I)
    if m:
        located.append(m.group(1).strip().strip('"`'))
    m = re.search(r"no such table: (.+)$", msg, re.I)
    if m:
        located.append(m.group(1).strip().strip('"`'))
    m = re.search(r"ambiguous column name: (.+)$", msg, re.I)
    if m:
        located.append(m.group(1).strip())
    for mm in re.finditer(r'near "([^"]+)": syntax error', msg, re.I):
        located.append(f"syntax:{mm.group(1)}")
    m = re.search(r"no such function: (\w+)", msg, re.I)
    if m:
        located.append(f"func:{m.group(1)}")
    if "datatype mismatch" in msg.lower():
        located.append("type:mismatch")
    return Signal("s1_error", located, detail=msg)


def signal_s3(sql: str, catalog: Catalog, executor: DBExecutor) -> Signal:
    """S3:约束校验——字面量值是否存在于对应列的值域。"""
    located, detail = [], []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return Signal("s3_constraint", [], "parse-failed")
    amap = {t.alias or t.name: t.name for t in tree.find_all(exp.Table)}
    tables_in_sql = [t.name for t in tree.find_all(exp.Table)]
    for lit in tree.find_all(exp.Literal):
        if lit.is_string:
            for col in tree.find_all(exp.Column):
                # 粗对齐:同一谓词内的字符串字面量 vs 列值签名
                if _common_pred_parent(lit, col) is None:
                    continue
                # 列前缀可能是别名或省略(单表查询),逐级解析到真实表
                cands = []
                if col.table:
                    tname = amap.get(col.table, col.table)
                    cands.append(tname)
                else:
                    cands.extend(tables_in_sql)
                for tname in cands:
                    c = catalog.column(tname, col.name)
                    if c is not None:
                        break
                else:
                    c = None
                if c and c.top_values and str(lit.this) not in c.top_values:
                    # 尝试大小写/空格不敏感匹配
                    loose = [v.strip().lower() for v in c.top_values]
                    if str(lit.this).strip().lower() not in loose:
                        # DBMS 确认(采样签名有假阳性:正确字面量可能
                        # 恰不在 top-k 采样内):EXISTS 查询确证不存在
                        # 才定位为 S3 错误——"被动校验者 → 主动协同者"
                        if not _value_exists(executor, c.table, col.name,
                                             str(lit.this)):
                            located.append(f"{c.table}.{col.name}"
                                           f"={str(lit.this)!r}")
                            detail.append(
                                f"value {str(lit.this)!r} not in column "
                                f"{c.table}.{col.name} (DBMS-confirmed; "
                                f"sample e.g. {c.top_values[:3]})")
                break  # 每个字面量只对齐第一个候选列
    return Signal("s3_constraint", located, "; ".join(detail[:5]))


def _value_exists(executor: DBExecutor, table: str, column: str,
                  value: str) -> bool:
    """DBMS 存在性确认:SELECT 1 FROM t WHERE c = 'v' LIMIT 1。

    采样值签名(top-k of n 行)对"正确但未采样"的字面量有假阳性;
    仅当 DBMS 确证不存在时才作为 S3 错误定位。
    查询自身失败(表/列不可用)时按"存在"处理(宁缺勿滥)。
    """
    q = (f'SELECT 1 FROM "{table}" WHERE "{column}" = '
         f"'{value.replace(chr(39), chr(39) * 2)}' LIMIT 1")
    try:
        _, rows, _ = executor.execute(q)
        return bool(rows)
    except DBError:
        return True


def _pred_scopes(node: exp.Expression) -> List[exp.Expression]:
    """节点所属谓词作用域祖先列表(WHERE / JOIN 的 ON 子树)。

    sqlglot 30.x:JOIN ON 条件不包装为 exp.On 节点,而是 Join.args['on'];
    节点位于 ON 子树 ⟺ 上溯路径进入 Join 的那一步的子节点正是 on 参数。
    """
    scopes: List[exp.Expression] = []
    child = node
    x = node.parent
    while x is not None:
        if isinstance(x, exp.Where):
            scopes.append(x)
        elif isinstance(x, exp.Join) and child is x.args.get("on"):
            scopes.append(x)
        child = x
        x = x.parent
    return scopes


def _common_pred_parent(a: exp.Expression, b: exp.Expression):
    ids = {id(s) for s in _pred_scopes(a)}
    for s in _pred_scopes(b):
        if id(s) in ids:
            return s
    return None


def signal_s2(sql: str, executor: DBExecutor) -> Signal:
    """S2:执行计划 + 谓词选择率(行数),用于空结果/过量行诊断。"""
    try:
        cols, rows, _ = executor.execute(sql)
        n = len(rows)
    except DBError as e:
        return Signal("s2_plan", [], "exec-failed:" + str(e)[:100])
    if n == 0:
        return Signal("s2_plan", [], hint="EMPTY_RESULT",
                      detail=f"0 rows; consider predicate relaxation")
    return Signal("s2_plan", [], detail=f"{n} rows")


def signal_s4(candidate_sqls: List[str],
              reference_sql: Optional[str],
              executor: DBExecutor) -> Signal:
    """S4:CTE 级中间结果差分(候选间对齐)。"""
    if len(candidate_sqls) < 2:
        return Signal("s4_diff", [], "single-candidate")
    results = []
    for s in candidate_sqls[:3]:
        try:
            cols, rows, _ = executor.execute(s)
            results.append(_fingerprint(cols, rows))
        except DBError:
            results.append("EXEC_ERROR")
    agree = all(r == results[0] for r in results)
    return Signal("s4_diff", [],
                  detail="agree" if agree else "disagree",
                  hint="" if agree else "CANDIDATES_DIVERGE")


def _fingerprint(cols, rows) -> str:
    import hashlib
    h = hashlib.md5()
    h.update(str(cols).encode())
    for r in sorted(map(str, rows))[:50]:
        h.update(r.encode())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ 距离
def semantic_distance(a: str, b: str) -> float:
    """轻量语义距离:token 级 Jaccard + 结构哈希差异。"""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    jac = 1.0 - len(ta & tb) / max(len(ta | tb), 1)
    try:
        ha = sqlglot.parse_one(a, read="sqlite").sql()
        hb = sqlglot.parse_one(b, read="sqlite").sql()
        struct = 0.0 if ha == hb else 0.15
    except Exception:
        struct = 0.2
    return jac + struct


def _edit_distance(a: str, b: str) -> float:
    """best-first 优先级:与原始查询的 token 集合距离(小改动优先)。"""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    return 1.0 - len(ta & tb) / max(len(ta | tb), 1)


# ------------------------------------------------------------------ 动作
@dataclass
class RepairAction:
    name: str
    structural: bool
    apply: Callable[[str, Signal, Catalog, DBExecutor], List[str]]
    triggers: Set[str] = field(default_factory=set)


def _tables_in(sql: str) -> List[str]:
    try:
        return [t.name for t in sqlglot.parse_one(
            sql, read="sqlite").find_all(exp.Table)]
    except Exception:
        return []


def _swap_identifier(sql: str, old: str, new: str) -> str:
    """AST 定位的标识符替换。old 形如 'table' / 'table.col' / 'col'。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql.replace(old, new, 1)
    parts = old.split(".")
    done = False
    if len(parts) == 2:
        tname, cname = parts
        for col in tree.find_all(exp.Column):
            if (col.table == tname or not col.table) and col.name == cname:
                col.set("this", exp.to_identifier(new.split(".")[-1]))
                if "." in new:
                    col.set("table", exp.to_identifier(new.split(".")[0]))
                done = True
                break
    else:
        for tab in tree.find_all(exp.Table):
            if tab.name == old:
                tab.set("this", exp.to_identifier(new))
                done = True
                break
        if not done:
            for col in tree.find_all(exp.Column):
                if col.name == old:
                    col.set("this", exp.to_identifier(new))
                    done = True
                    break
    return tree.sql(dialect="sqlite") if done else sql


def _similar_columns(catalog: Catalog, table: str, col: str,
                     limit: int = 4) -> List[str]:
    out = []
    c0 = catalog.column(table, col)
    for c in catalog.columns_of(table):
        if c.name == col:
            continue
        s = 0.0
        if c0 and c.dtype == c0.dtype:
            s += 0.4
        if col.lower() in c.name.lower() or c.name.lower() in col.lower():
            s += 0.6
        import difflib
        s += difflib.SequenceMatcher(None, col, c.name).ratio() * 0.5
        if s > 0.4:
            out.append((s, f"{table}.{c.name}"))
    return [x for _, x in sorted(out, reverse=True)[:limit]]


def _similar_tables(catalog: Catalog, table: str,
                    graph=None, limit: int = 3) -> List[str]:
    import difflib
    scored = []
    for t in catalog.tables:
        if t == table:
            continue
        s = difflib.SequenceMatcher(None, table, t).ratio()
        if graph and table in graph.semantic_edges and t in \
                graph.semantic_edges.get(table, []):
            s += 0.5
        scored.append((s, t))
    return [t for s, t in sorted(scored, reverse=True)[:limit] if s > 0.3]


def _stringify(node) -> str:
    return str(node.this) if node else ""


ACTIONS: List[RepairAction] = [
    # ---- SafeQL 五类原子动作 ----
    RepairAction(
        "relation_replace",
        structural=True,
        triggers={"s1"},
        apply=lambda sql, sig, cat, ex: _relation_replace(sql, sig, cat)),
    RepairAction(
        "relation_add",
        structural=True,
        triggers={"s1"},
        apply=lambda sql, sig, cat, ex: _add_relation(sql, sig, cat)),
    RepairAction(
        "join_fix",
        structural=True,
        triggers={"s1"},
        apply=lambda sql, sig, cat, ex: _fix_join(sql, sig, cat)),
    RepairAction(
        "attribute_replace",
        structural=False,
        triggers={"s1", "s3"},
        apply=lambda sql, sig, cat, ex: _attribute_replace(sql, sig, cat)),
    RepairAction(
        "value_replace",
        structural=False,
        triggers={"s3"},
        apply=lambda sql, sig, cat, ex: _value_replace(sql, sig, cat)),
    RepairAction(
        "function_replace",
        structural=False,
        triggers={"s1"},
        apply=lambda sql, sig, cat, ex: _function_replace(sql, sig)),
    # ---- 扩展动作(本文新增) ----
    RepairAction(
        "predicate_relax",
        structural=True,
        triggers={"s2"},
        apply=lambda sql, sig, cat, ex: _predicate_relax(sql, sig, ex)),
    RepairAction(
        "predicate_tighten",
        structural=True,
        triggers={"s2"},
        apply=lambda sql, sig, cat, ex: _predicate_tighten(sql, sig, ex)),
]

# 方言函数同义映射(4.4)
FUNC_MAP = {
    "STRFTIME": ["DATE", "DATETIME", "strftime"],
    "DATE": ["STRFTIME", "DATETIME"],
    "JULIANDAY": ["DATE", "STRFTIME"],
    "IFNULL": ["COALESCE"],
    "COALESCE": ["IFNULL"],
    "LENGTH": ["LENGTH", "CHAR_LENGTH"],
    "SUBSTR": ["SUBSTRING", "SUBSTR"],
    "REGEXP_LIKE": ["REGEXP", "LIKE"],
}


def _function_replace(sql: str, sig: Signal) -> List[str]:
    out = []
    for loc in sig.located:
        if loc.startswith("func:"):
            f = loc[5:].upper()
            for cand in FUNC_MAP.get(f, []):
                out.append(re.sub(rf"\b{f}\b", cand, sql, count=1,
                                  flags=re.I))
    return out


def _value_replace(sql: str, sig: Signal, catalog: Catalog) -> List[str]:
    """S3 引导的 value 替换:用值签名中采样值替换不存在字面量。"""
    out = []
    for loc in sig.located:
        m = re.match(r"(.+?)='(.*)'$", loc)
        if not m:
            continue
        col_full, val = m.group(1), m.group(2)
        parts = col_full.split(".")
        table, col = (parts[-2], parts[-1]) if len(parts) >= 2 else ("", parts[0])
        c = catalog.column(table, col) if table else None
        if not c:
            # 全库搜列名
            for t, cols in catalog.tables.items():
                cc = next((x for x in cols if x.name == col), None)
                if cc and cc.top_values:
                    c = cc
                    break
        if c and c.top_values:
            for v in c.top_values[:2]:
                if str(v).lower() != val.lower():
                    out.append(sql.replace(f"'{val}'", f"'{v}'", 1))
    return out[:4]


def _alias_map_sql(sql: str) -> Dict[str, str]:
    """SQL 内 FROM/JOIN 别名 → 真实表名(BIRD/Spider 风格 T1/T2 别名)。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        from sqlglot import exp
        amap: Dict[str, str] = {}
        for t in tree.find_all(exp.Table):
            amap[t.alias or t.name] = t.name
        return amap
    except Exception:
        return {}


def _tables_with_column(catalog: Catalog, col: str,
                        limit: int = 3) -> List[str]:
    """含指定列名的表(列存在性引导的 relation 替换)。"""
    out = []
    for t, cols in catalog.tables.items():
        if any(c.name == col for c in cols):
            out.append(t)
    return out[:limit]


def _relation_replace(sql: str, sig: Signal, catalog: Catalog) -> List[str]:
    """relation 替换(别名感知):
    定位 'T1.col'(T1 为别名)→ 解析真实表;若列不在该表 →
    换成含该列的表(别名保持);定位 'table' → 相似表替换。
    """
    amap = _alias_map_sql(sql)
    out = []
    for loc in sig.located:
        if loc.startswith(("func:", "syntax:", "type:")):
            continue
        parts = loc.split(".")
        if len(parts) >= 2:
            raw_t, c = parts[0], parts[1]
            real_t = amap.get(raw_t, raw_t)
            if catalog.tables.get(real_t) and not catalog.column(real_t, c):
                # 表存在但列不存在 → 换成含该列的表(保持别名/限定符)
                for nt in _tables_with_column(catalog, c):
                    if nt != real_t:
                        out.append(_swap_table(sql, real_t, nt))
        else:
            t = amap.get(parts[0], parts[0])
            for nt in _similar_tables(catalog, t)[:2]:
                if catalog.tables.get(nt):
                    out.append(_swap_table(sql, t, nt))
    return out[:4]


def _swap_table(sql: str, old: str, new: str) -> str:
    """AST 表名替换(别名保持:FROM x AS T1 → FROM new AS T1)。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        from sqlglot import exp
        for tab in tree.find_all(exp.Table):
            if tab.name == old:
                tab.set("this", exp.to_identifier(new))
                return tree.sql(dialect="sqlite")
    except Exception:
        pass
    return sql.replace(old, new, 1)


def _swap_column_name(sql: str, qualifier: str, old_col: str,
                      new_col: str) -> str:
    """仅替换列名,保持限定符(别名)不变:T1.old → T1.new。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        from sqlglot import exp
        for col in tree.find_all(exp.Column):
            if (col.table or "") == qualifier and col.name == old_col:
                col.set("this", exp.to_identifier(new_col))
                return tree.sql(dialect="sqlite")
    except Exception:
        pass
    return sql


def _attribute_replace(sql: str, sig: Signal, catalog: Catalog) -> List[str]:
    """attribute 替换(别名感知):
    'T1.col':T1 → 真实表;col 缺失 → 同表相似列(保持别名限定符);
    限定符本身非法(非别名非表)→ 重限定到含该列的表。
    """
    amap = _alias_map_sql(sql)
    out = []
    for loc in sig.located:
        if loc.startswith(("func:", "syntax:", "type:")):
            continue
        parts = loc.split(".")
        if len(parts) >= 2:
            raw_t, c = parts[0], parts[1]
            real_t = amap.get(raw_t, raw_t)
            if catalog.tables.get(real_t):
                for cand in _similar_columns(catalog, real_t, c)[:3]:
                    new_col = cand.split(".", 1)[-1]
                    out.append(_swap_column_name(sql, raw_t, c, new_col))
            elif raw_t in amap:      # 别名存在但表无该列(已被 relation 处理)
                continue
            else:
                # 限定符非法 → 重限定到含该列的表
                for nt in _tables_with_column(catalog, c, limit=2):
                    out.append(_swap_identifier(sql, loc, f"{nt}.{c}"))
        elif len(parts) == 1:
            c = parts[0]
            for nt in _tables_with_column(catalog, c, limit=2):
                out.append(_swap_identifier(sql, c, c))
    return out[:4]


def _add_relation(sql: str, sig: Signal, catalog: Catalog) -> List[str]:
    """relation 新增:沿 FK/隐式 join 为已有表添加可达表。"""
    out = []
    tabs = _tables_in(sql)
    fk_map = catalog.fk_map()
    for (t, c), neighbors in fk_map.items():
        if t in tabs:
            for (nt, nc) in neighbors:
                if nt not in tabs:
                    out.append(
                        f"{sql.rstrip(';')}\n-- +relation {nt} via "
                        f"{t}.{c} = {nt}.{nc}")
    return out[:3]


def _fix_join(sql: str, sig: Signal, catalog: Catalog) -> List[str]:
    """join 修正:按 FK 调整 ON 条件。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    tabs = _tables_in(sql)
    out = []
    for join in list(tree.find_all(exp.Join))[:2]:
        # 若 join 无 ON 条件或条件列不在 FK 上,按 FK 重建
        conds = list(join.find_all(exp.EQ))
        need_fix = not conds or any(
            not _is_fk_pair(c, catalog, tabs) for c in conds)
        if need_fix:
            fixed = _rewrite_join_on(join, catalog, tabs)
            if fixed:
                new_sql = tree.sql(dialect="sqlite")
                out.append(new_sql)
                break
    return out[:2]


def _is_fk_pair(eq: exp.EQ, catalog: Catalog, tabs) -> bool:
    cols = list(eq.find_all(exp.Column))
    if len(cols) != 2:
        return True
    a = (cols[0].table, cols[0].name)
    b = (cols[1].table, cols[1].name)
    pairs = {(fk.src, fk.dst) for fk in catalog.fks} | {
        (fk.dst, fk.src) for fk in catalog.fks}
    return (a, b) in pairs or (b, a) in pairs


def _rewrite_join_on(join: exp.Join, catalog: Catalog, tabs) -> bool:
    # 找 join 表与已出现表之间的 FK
    for (t, c), neighbors in catalog.fk_map().items():
        for (nt, nc) in neighbors:
            if t in tabs and nt in tabs and t != nt:
                cond = exp.condition(
                    exp.column(c, table=t).eq(exp.column(nc, table=nt)))
                if join.args.get("on") is not None:
                    join.set("on", cond)
                    return True
    return False


def _predicate_relax(sql: str, sig: Signal, executor: DBExecutor) -> List[str]:
    """谓词松弛:空结果时依次松化最紧的谓词(去掉/弱化一个条件)。"""
    if sig.hint != "EMPTY_RESULT":
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    out = []
    wheres = list(tree.find_all(exp.Where))
    if wheres:
        w = wheres[0]
        conds = list(w.find_all(exp.And)) or [w.this]
        # 去掉最后一个 AND 条件(最可能是最紧的)
        if isinstance(w.this, exp.And):
            w.set("this", w.this.args.get("this") or w.this.args.get("then"))
            out.append(tree.sql(dialect="sqlite"))
            tree2 = sqlglot.parse_one(sql, read="sqlite")
            w2 = tree2.find(exp.Where)
            if w2 and isinstance(w2.this, exp.And):
                w2.set("this", w2.this.args.get("then") or w2.this.args.get("this"))
                out.append(tree2.sql(dialect="sqlite"))
        elif isinstance(w.this, exp.EQ):
            tree3 = sqlglot.parse_one(sql, read="sqlite")
            w3 = tree3.find(exp.Where)
            w3.pop()
            out.append(tree3.sql(dialect="sqlite"))
    return out[:2]


def _predicate_tighten(sql: str, sig: Signal, executor: DBExecutor) -> List[str]:
    """谓词收紧:行数过多时增加主键/时间排序限制。"""
    return [sql.rstrip(";") + "\nORDER BY 1 LIMIT 100"]


# ------------------------------------------------------------------ 修复器
@dataclass
class RepairOutcome:
    sql: str
    executed: bool
    n_steps: int
    signals_used: List[str]
    actions_applied: List[str]
    fallback_used: bool
    trace: List[Dict] = field(default_factory=list)


class Corrector:
    """best-first 修复搜索(y.docx 4.2.3 + 定理 1)。"""

    def __init__(self, executor: DBExecutor, catalog: Catalog,
                 budget: Optional[BudgetController] = None,
                 signals: Sequence[str] = ("s1", "s3", "s2"),
                 use_cte_blocking: bool = True,
                 graph=None,
                 gate: str = "off"):
        self.ex = executor
        self.cat = catalog
        self.budget = budget
        self.signals = set(signals)      # A2 消融:S1 → +S2 → +S3 → +S4
        self.use_cte_blocking = use_cte_blocking
        self.graph = graph
        # 保守门控(强生成器形态,E5 实验结论):conservative = 输入 SQL
        # 执行成功时只读(S2/S3 信号仅记录报告,不产生变异候选),
        # 修复搜索仅在执行失败时启动。off = 旧语义(E4 原型口径)。
        self.gate = gate
        self._regen_fn: Optional[Callable[[str, str], str]] = None

    def set_regenerator(self, fn: Callable[[str, str], str]):
        """回退 LLM 重生成钩子(混合策略)。"""
        self._regen_fn = fn

    def repair(self, sql: str, question: str,
               reference_sql: Optional[str] = None,
               max_steps: int = MAX_STEPS) -> RepairOutcome:
        """best-first 搜索。

        定理 1 终止条件的实现:动作空间有限、深度 ≤ MAX_DEPTH、
        visited 集保证无环(每步产生新查询状态)、步数预算兜底;
        优先级 = 编辑距离(至当前)+ α·结构惩罚(结构变换降权)。
        """
        t0 = time.perf_counter()
        outcome = RepairOutcome(sql, False, 0, [], [], False, [])

        # ---- 保守门控:输入 SQL 执行成功 → 只读模式(E5 结论)
        # S2/S3 信号仅记录报告,不变异;修复搜索仅在执行失败时启动。
        if self.gate == "conservative":
            input_ok = False
            input_rows: List = []
            try:
                _, input_rows, _ = self.ex.execute(sql)
                input_ok = True
            except DBError:
                input_ok = False
            if input_ok:
                outcome.executed = True
                if "s2" in self.signals and not input_rows:
                    outcome.signals_used.append("s2")
                    outcome.trace.append(
                        {"step": 0, "event": "s2_empty_result",
                         "gated": True})
                if "s3" in self.signals:
                    try:
                        s3g = signal_s3(sql, self.cat, self.ex)
                        if s3g.located:
                            outcome.signals_used.append("s3")
                            outcome.trace.append(
                                {"step": 0, "event": "s3_constraint",
                                 "gated": True, "detail": s3g.detail[:120]})
                    except Exception:
                        pass
                outcome.sql = sql              # 原始 SQL 无变异覆盖
                outcome.n_steps = 0
                if self.budget:
                    self.budget.spend(
                        "repair", tokens_in=count_tokens(sql), tokens_out=0,
                        latency_s=time.perf_counter() - t0)
                return outcome

        best = sql
        frontier: List[Tuple[float, int, str, List[str], List[str]]] = [
            (0.0, 0, sql, [], [])]          # (优先级, 深度, sql, 动作史, 信号史)
        visited = {sql}
        n_steps = 0
        K_kept = 0

        while frontier and n_steps < max_steps:
            frontier.sort(key=lambda x: x[0])
            prio, depth, cur, history, sig_hist = frontier.pop(0)
            n_steps += 1
            if self.budget and self.budget.exceeded("repair"):
                outcome.trace.append({"step": n_steps, "event": "budget_stop"})
                break

            # ---- 执行并收集信号
            try:
                cols, rows, _ = self.ex.execute(cur)
                executed = True
                err = None
            except DBError as e:
                executed, err = False, e
            outcome.executed = executed

            if executed:
                best = cur
                K_kept += 1
                if K_kept >= K_FALLBACK and depth >= 1:
                    break   # 满足保底候选数
                # S2 检查空结果(执行成功但 0 行 → 可能谓词过紧)
                if "s2" in self.signals and not rows:
                    sig = signal_s2(cur, self.ex)
                    if sig.hint == "EMPTY_RESULT":
                        outcome.signals_used.append("s2")
                        outcome.trace.append(
                            {"step": n_steps, "event": "s2_empty_result"})
                        for act in _acts_for(sig):
                            for cand in act.apply(cur, sig, self.cat, self.ex):
                                self._push(frontier, visited, cand, cur,
                                           act, depth, history, sig_hist)
                # S3 约束校验(执行成功也做:字面量值域落地等)
                if "s3" in self.signals:
                    try:
                        sqlglot.parse_one(cur, read="sqlite")
                        s3 = signal_s3(cur, self.cat, self.ex)
                        if s3.located:
                            outcome.signals_used.append("s3")
                            outcome.trace.append(
                                {"step": n_steps, "event": "s3_constraint",
                                 "detail": s3.detail[:120]})
                            for act in _acts_for(s3):
                                for cand in act.apply(
                                        cur, s3, self.cat, self.ex):
                                    self._push(frontier, visited, cand, cur,
                                               act, depth, history, sig_hist)
                    except Exception:
                        pass
                continue

            # ---- 执行失败:S1 定位
            sig = signal_s1(err, cur, self.cat)
            outcome.signals_used.append("s1")
            candidates: List[Tuple[str, str]] = []
            for act in _acts_for(sig):
                for cand in act.apply(cur, sig, self.cat, self.ex):
                    cand = _strip_comment(cand)
                    candidates.append((act.name, cand))

            # S3 约束校验(解析成功才做)
            if "s3" in self.signals:
                try:
                    sqlglot.parse_one(cur, read="sqlite")
                    s3 = signal_s3(cur, self.cat, self.ex)
                    if s3.located:
                        outcome.signals_used.append("s3")
                        for act in _acts_for(s3):
                            for cand in act.apply(cur, s3, self.cat, self.ex):
                                candidates.append((act.name, cand))
                except Exception:
                    pass

            if not candidates:
                # 无法定位修复 → 回退重生成
                outcome.trace.append(
                    {"step": n_steps, "event": "fallback_regen",
                     "reason": "no candidates from signals"})
                outcome.fallback_used = True
                if self._regen_fn:
                    regen = self._regen_fn(question, cur)
                    if regen and regen not in visited:
                        frontier.append((0.5, 0, regen, history + ["regen"],
                                         sig_hist))
                continue

            for aname, cand in candidates[:8]:
                cand = cand.strip()
                if cand in visited or not cand:
                    continue
                visited.add(cand)
                prio = _edit_distance(cand, sql) + (
                    ALPHA_STRUCT if _act_by_name(aname).structural else 0.0)
                if depth + 1 <= MAX_DEPTH:
                    frontier.append((prio, depth + 1, cand,
                                     history + [aname], sig_hist + [sig.kind]))
            outcome.trace.append({
                "step": n_steps, "event": "expand", "error": str(err)[:120],
                "located": sig.located, "n_cands": len(candidates)})

        outcome.sql = best
        outcome.n_steps = n_steps
        # 记账(修复的 token 以 SQL 文本 + 轨迹近似)
        if self.budget:
            self.budget.spend("repair",
                              tokens_in=count_tokens(sql) * max(n_steps, 1),
                              tokens_out=count_tokens(best),
                              latency_s=time.perf_counter() - t0)
        return outcome

    def _push(self, frontier, visited, cand, cur, act, depth, history,
              sig_hist):
        cand = _strip_comment(cand)
        if cand in visited:
            return
        visited.add(cand)
        d = semantic_distance(cand, cur)
        prio = d + (ALPHA_STRUCT if act.structural else 0.0)
        if depth + 1 <= MAX_DEPTH:
            frontier.append((prio, depth + 1, cand, history + [act.name],
                             sig_hist + [act.name]))


def _strip_comment(sql: str) -> str:
    return "\n".join(l for l in sql.splitlines()
                     if not l.strip().startswith("--"))


def _acts_for(sig: Signal) -> List[RepairAction]:
    k = sig.kind.split("_")[0]     # s1 / s2 / s3 / s4
    return [a for a in ACTIONS if k in a.triggers]


def _act_by_name(name: str) -> RepairAction:
    for a in ACTIONS:
        if a.name == name:
            return a
    return ACTIONS[0]
