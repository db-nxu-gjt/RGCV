"""T2-2c 真实静默错误按类别分解:错误类别标注 × (V1+V2) 拦截率。

背景:验证器 (V1 指纹 + V2 差分执行) 对真实静默错误的端到端拦截率仅
16.7% (DeepSeek, 22/132) 与 21.9% (Kimi, 28/128)。本脚本对每个静默错误题
做五类错误标注 (aggregate-scope / join / filter / predicate-missing / value),
按 (backbone × 类别) 统计拦截率,并估计"对含 GROUP BY 查询无条件触发 V3"
的覆盖潜力 (T2-2c 试点,不调用任何 LLM)。

标注方法 (论文披露口径,automated + deterministic):
  1. 用 sqlglot (read="sqlite") 解析 gold 与预测 SQL;任一方解析失败 →
     unparseable (如实报告数量)。
  2. 按 SQL 执行顺序 (FROM/JOIN → WHERE → GROUP BY/聚合 → 其余子句) 逐阶段
     比较结构特征,以"首个出现结构性差异的阶段"定类:
     a. join:表集合、join 边集合 (无序表对 × join 类型,纯 JOIN ≡ INNER)、
        ON 等值列对 (别名解析到真实表名 + 操作数交换律) 任一不同;
     b. WHERE 谓词按顶层 AND 拆分为合取项,按共享列做一对一贪心匹配:
        - gold 合取项在预测中无任何列交集对应 → predicate-missing
          (仅当 WHERE 差异全为缺失合取项时才归此类);
        - 对应合取项算子族不同 (EQ/GT/IN/BETWEEN/LIKE/IS NULL 等) 或
          比较列集合不同 → filter;
        - 同列同算子、仅字面量不同 → value;
     c. aggregate-scope:GROUP BY 键多重集不同,或聚合表达式多重集不同
        (函数名/DISTINCT/聚合列;MAX/MIN(x) ≡ ORDER BY x DESC/ASC LIMIT 1
        的 argmax 等价形先归一化,避免把等价改写误标为聚合差异);
     d. 其余子句:两 SQL 经别名消解与字面量掩码后仍不同时,仅字面量差异
        (HAVING/LIMIT 等) → value;仅 DISTINCT 增减(去重粒度)→
        aggregate-scope;仅 ASC/DESC 方向或非聚合投影列等其他结构残差 →
        other (如实报告数量);
     e. 比较后无任何结构差异但 EX=False → other (NULL 排序/并列打破等)。
  3. 跨多个阶段均有差异的题仍按 (2) 的首个阶段定类,其数量在
     multi_stage_diff_n 中如实披露。

数据源:
  - results/eA_tier2_real_verification.json (Tier-2 冻结运行上的验证器
    replay;kind=silent 即真实静默错误题,alarm 位为 V1/V2/V3 明细)
  - results/rgcv_bird300_<backbone>_gated_fullschema/merged.json (预测 SQL)
  - baselines/DAIL-SQL/dataset/bird/dev.json (gold SQL,按 question_id 对齐)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

BASE = Path(__file__).resolve().parent.parent          # paper/github
RESULTS = BASE / "results"

TAGS = {
    "deepseek-v4-pro": "rgcv_bird300_deepseek-v4-pro_gated_fullschema",
    "kimi-k26": "rgcv_bird300_kimi-k26_gated_fullschema",
}
CATEGORIES = ["aggregate-scope", "join", "filter", "predicate-missing",
              "value", "other", "unparseable"]

GOLD_CANDIDATES = [
    BASE.parent.parent / "baselines" / "DAIL-SQL" / "dataset" / "bird" / "dev.json",
    BASE.parent.parent / "data" / "bird" / "dev_20240627" / "dev.json",
]

# ------------------------------------------------------------------ 解析工具
def clean_sql(s: str) -> str:
    """去掉 merged.json 尾注、markdown 围栏与尾部分号。"""
    s = s.split("\t-----")[0].strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip().rstrip(";").strip()


def norm_ident(s: str) -> str:
    return s.strip("`\"[]").lower()


def mask(s: str) -> str:
    """字面量掩码:字符串与数字字面量 → ?,小写化(大小写不敏感比较)。"""
    s = re.sub(r"'(?:[^']|'')*'", "?", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "?", s)
    return s.lower()


def alias_map(tree) -> dict:
    """别名 → 真实表名(复用 verifier._alias_map 思路,覆盖全部作用域)。"""
    amap = {}
    for t in tree.find_all(exp.Table):
        amap[t.alias or t.name] = t.name
    return amap


def resolve_col(col: exp.Column, amap: dict) -> str:
    t = col.table or ""
    t = amap.get(t, t)
    return f"{norm_ident(t)}.{norm_ident(col.name)}" if t else \
        f"*.{norm_ident(col.name)}"


def _children(node):
    for v in node.args.values():
        if isinstance(v, exp.Expression):
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, exp.Expression):
                    yield x
        elif isinstance(v, dict):
            for x in v.values():
                if isinstance(x, exp.Expression):
                    yield x


def walk_scope(scope):
    """遍历单个 Select 作用域内的节点,不进入嵌套子查询/CTE。"""
    stack = [scope]
    while stack:
        n = stack.pop()
        yield n
        for c in _children(n):
            if isinstance(c, exp.Select):   # 嵌套作用域由 find_all 另行遍历
                continue
            stack.append(c)


# ------------------------------------------------------------------ 阶段 a: join
def on_shape(on, amap):
    """ON 条件结构:等值列对解析到真实表名并按操作数排序(交换律),
    其余合取项以字面量掩码串表示。"""
    parts = []
    for c in split_conj(on):
        if isinstance(c, exp.EQ):
            l, r = c.this, c.expression
            if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                pair = sorted((resolve_col(l, amap), resolve_col(r, amap)))
                parts.append("eq:" + "<>".join(pair))
                continue
        parts.append(mask(c.sql()))
    return tuple(sorted(parts))


def join_feats(tree, amap):
    """表集合 + join 边(表对×join 类型,表对无序) + ON 等值列对结构。
    FROM 根表顺序不敏感(内连接下语义等价)。"""
    tables, edges, on_shapes = set(), [], []
    for scope in tree.find_all(exp.Select):
        # sqlglot 30.x: FROM 的 args 键为 "from_"
        frm = scope.args.get("from_") or scope.args.get("from")
        root = "<none>"
        if frm is not None:
            ft = frm.this
            root = ft.name if isinstance(ft, exp.Table) else "<sub>"
            tables.add(root)
        for j in scope.args.get("joins") or []:
            jt = j.this
            name = jt.name if isinstance(jt, exp.Table) else "<sub>"
            tables.add(name)
            side = (j.args.get("side") or "").upper()
            kind = (j.args.get("kind") or "").upper()
            # 纯 JOIN ≡ INNER JOIN,归一化
            side = side or ("INNER" if kind in ("", "INNER") else kind)
            edges.append((tuple(sorted({root, name})), side))
            on = j.args.get("on")
            if on is not None:
                on_shapes.append(on_shape(on, amap))
    return {"tables": tuple(sorted(tables)),
            "edges": tuple(sorted(edges)),
            "on": tuple(sorted(on_shapes))}


# ------------------------------------------------------------------ 阶段 b: WHERE
OP_FAMILIES = {exp.EQ: "eq", exp.NEQ: "neq", exp.GT: "gt", exp.LT: "lt",
               exp.GTE: "gte", exp.LTE: "lte", exp.In: "in",
               exp.Between: "between", exp.Like: "like", exp.ILike: "like",
               exp.Is: "isnull"}


def split_conj(node):
    """按顶层 AND 拆合取项(OR 组视为一个整体)。"""
    if isinstance(node, exp.Paren):
        return split_conj(node.this)
    if isinstance(node, exp.And):
        return split_conj(node.this) + split_conj(node.expression)
    return [node]


def conj_info(c, amap):
    cols = frozenset(resolve_col(x, amap) for x in c.find_all(exp.Column))
    n = c
    neg = False
    while isinstance(n, (exp.Not, exp.Paren)):
        neg = neg or isinstance(n, exp.Not)
        n = n.this
    fam = None
    for cls, f in OP_FAMILIES.items():
        if isinstance(n, cls):
            fam = f
            break
    if fam is None:
        fam = "subq" if n.find(exp.Select) is not None else "other"
    if neg:
        fam = f"not-{fam}"
    lits = tuple(sorted(str(l.this) for l in c.find_all(exp.Literal)))
    return {"cols": cols or {"<const>"}, "fam": fam, "lits": lits}


def where_feats(tree, amap):
    out = []
    for w in tree.find_all(exp.Where):
        out.extend(conj_info(c, amap) for c in split_conj(w.this))
    return out


def where_diff(gcs, pcs):
    """→ (missing, extra, opmm, valmm) 计数。贪心一对一匹配(列交集最大优先)。"""
    missing = extra = opmm = valmm = 0
    used = [False] * len(pcs)
    for g in gcs:
        best, bi = 0, -1
        for i, p in enumerate(pcs):
            if used[i]:
                continue
            if g["cols"] == {"<const>"} or p["cols"] == {"<const>"}:
                inter = 1 if (g["cols"] == p["cols"]
                              and g["fam"] == p["fam"]) else 0
            else:
                inter = len(g["cols"] & p["cols"])
            if inter > best:
                best, bi = inter, i
        if best == 0:
            missing += 1
            continue
        used[bi] = True
        p = pcs[bi]
        if g["fam"] != p["fam"] or g["cols"] != p["cols"]:
            opmm += 1
        elif g["lits"] != p["lits"]:
            valmm += 1
    extra = sum(1 for u in used if not u)
    return missing, extra, opmm, valmm


# ------------------------------------------------------------------ 阶段 c: 聚合
def agg_arg(node, amap):
    sub = [x for x in node.find_all(exp.Expression) if x is not node]
    nested = any(isinstance(x, (exp.AggFunc, exp.Select)) for x in sub)
    cols = [resolve_col(c, amap) for c in sub if isinstance(c, exp.Column)]
    if not nested and len(cols) == 1:
        return cols[0]
    return "sql:" + mask(node.sql())


def agg_feats(tree, amap):
    groups, aggs = [], []
    for scope in tree.find_all(exp.Select):
        g = scope.args.get("group")
        if g is not None:
            for e in g.expressions:
                if isinstance(e, exp.Column):
                    groups.append(resolve_col(e, amap))
                else:
                    groups.append("sql:" + mask(e.sql()))
        for node in walk_scope(scope):
            if isinstance(node, exp.AggFunc):
                fname = type(node).__name__.upper()
                # COUNT(DISTINCT x):Distinct 为 node.this(sqlglot 30.x)
                distinct = (node.args.get("distinct") is not None
                            or isinstance(node.this, exp.Distinct))
                aggs.append((fname, distinct, agg_arg(node, amap)))
    return Counter(groups), Counter(aggs)


def order_limit_cols(tree, amap):
    cols = set()
    for scope in tree.find_all(exp.Select):
        lim = scope.args.get("limit")
        l = lim.expression if lim is not None else None
        if not (isinstance(l, exp.Literal) and str(l.this) == "1"):
            continue
        o = scope.args.get("order")
        if o is None:
            continue
        for ordered in o.expressions:
            for c in ordered.find_all(exp.Column):
                cols.add(resolve_col(c, amap))
    return cols


def argmax_normalize(tg, tp, gtree, ptree, amapg, amapp):
    """MAX/MIN(x) ≡ ORDER BY x ... LIMIT 1 等价形归一化(仅当一侧全为
    MAX/MIN 聚合且另一侧无任何聚合、有对应 ORDER BY ... LIMIT 1 列时)。"""
    gg, ga = tg
    pg, pa = tp

    def try_drop(agg_counter, other_aggs, tree, amap):
        if other_aggs:                      # 另一侧已有聚合 → 不做等价归一
            return agg_counter
        mm = [a for a in agg_counter if a[0] in ("MAX", "MIN")]
        if not mm or len(mm) != len(agg_counter):
            return agg_counter              # 需全部为 MAX/MIN 才归一
        olc = order_limit_cols(tree, amap)
        if olc and all(m[2] in olc for m in mm):
            return Counter({k: v for k, v in agg_counter.items()
                            if k[0] not in ("MAX", "MIN")})
        return agg_counter

    return (gg, try_drop(ga, pa, ptree, amapp)), \
        (pg, try_drop(pa, ga, gtree, amapg))


# ------------------------------------------------------------------ 残差
def norm_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    # INNER JOIN ≡ JOIN,LEFT/RIGHT OUTER JOIN ≡ LEFT/RIGHT JOIN(语义等价)
    s = re.sub(r"\binner\s+join\b", "join", s)
    s = re.sub(r"\bouter\s+join\b", "join", s)
    return s


def canonicalize_aliases(tree):
    """别名消解:列限定符与表别名统一替换为真实表名(文本残差比较用)。"""
    amap = {t.alias: t.name for t in tree.find_all(exp.Table)
            if t.alias and t.alias != t.name}
    for col in list(tree.find_all(exp.Column)):
        if col.table and col.table in amap:
            col.set("table", exp.to_identifier(amap[col.table]))
    for t in list(tree.find_all(exp.Table)):
        if t.alias:
            t.set("alias", None)


def residual_classify(gtree, ptree):
    """别名规范化渲染后的残差比较 → None / (category, detail)。

    仅 DISTINCT 增减(去重粒度)→ aggregate-scope;仅 ASC/DESC → other;
    仅字面量(HAVING/LIMIT 等)→ value;其余结构残差 → other。"""
    canonicalize_aliases(gtree)
    canonicalize_aliases(ptree)
    c1 = norm_text(gtree.sql(dialect="sqlite"))
    c2 = norm_text(ptree.sql(dialect="sqlite"))
    if c1 == c2:
        return None
    m1, m2 = mask(c1), mask(c2)
    if m1 == m2:
        return "value", "residual:literal"

    def strip(s, pat):
        return re.sub(r"\s+", " ", re.sub(pat, "", s))

    if strip(m1, r"\b(asc|desc)\b") == strip(m2, r"\b(asc|desc)\b"):
        return "other", "residual:asc/desc-only"
    if strip(m1, r"\bdistinct\b") == strip(m2, r"\bdistinct\b"):
        return "aggregate-scope", "residual:distinct-only"
    return "other", "residual:structural"


# ------------------------------------------------------------------ 分类主入口
def classify(gold_sql: str, pred_sql: str):
    """返回 (category, diff_detail, multi_stage)。多阶段差异仍按执行顺序
    首个阶段定类,multi_stage 如实标记。"""
    try:
        gtree = sqlglot.parse_one(gold_sql, read="sqlite")
        ptree = sqlglot.parse_one(pred_sql, read="sqlite")
    except Exception:
        return "unparseable", "parse-failed", False

    amapg, amapp = alias_map(gtree), alias_map(ptree)
    stages = []                                   # 结构差异阶段(执行序)

    # a. join:表集合 / join 边(表对×类型) / ON 等值列对
    fj, pj = join_feats(gtree, amapg), join_feats(ptree, amapp)
    jdiff = fj != pj
    jdet = ""
    if jdiff:
        stages.append("join")
        jdet = "join:" + ";".join(k for k in ("tables", "edges", "on")
                                  if fj[k] != pj[k])

    # b. WHERE 谓词(全部作用域的合取项池)
    gcs, pcs = where_feats(gtree, amapg), where_feats(ptree, amapp)
    missing, extra, opmm, valmm = where_diff(gcs, pcs)
    wdiff = bool(missing or extra or opmm or valmm)
    wdet = ""
    if wdiff:
        stages.append("where")
        parts = []
        if missing:
            parts.append(f"missing({missing})")
        if extra:
            parts.append(f"extra({extra})")
        if opmm:
            parts.append(f"op-mismatch({opmm})")
        if valmm:
            parts.append(f"lit-mismatch({valmm})")
        wdet = "where:" + ",".join(parts)

    # c. 聚合范围(GROUP BY 键 / 聚合表达式,argmax 等价形先归一化)
    tg, tp = agg_feats(gtree, amapg), agg_feats(ptree, amapp)
    tg, tp = argmax_normalize(tg, tp, gtree, ptree, amapg, amapp)
    adiff = tg != tp
    adet = ""
    if adiff:
        stages.append("agg")
        parts = []
        if tg[0] != tp[0]:
            parts.append("group_by")
        if tg[1] != tp[1]:
            parts.append("aggs")
        adet = "agg:" + ";".join(parts)

    # d. 其余子句(残差:别名规范化后,仅后置子句的投影/LIMIT/字面量等)
    res = residual_classify(gtree, ptree)

    multi = len(stages) >= 2                     # 仅统计结构阶段(join/where/agg)

    # ---- 以首个出现结构性差异的阶段定类(执行顺序)
    if jdiff:
        return "join", jdet, multi
    if wdiff:
        if missing and not (extra or opmm):
            return "predicate-missing", wdet, multi
        if extra or opmm:
            return "filter", wdet, multi
        return "value", wdet, multi               # 仅字面量不同
    if adiff:
        return "aggregate-scope", adet, multi
    if res is not None:
        return res[0], res[1], multi
    return "other", "no-structural-diff", multi


# ------------------------------------------------------------------ 主流程
def load_gold():
    for p in GOLD_CANDIDATES:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            gold = {int(q["question_id"]): q["SQL"] for q in data}
            return gold, p
    raise FileNotFoundError("gold dev.json not found")


def main():
    ea = json.loads((RESULTS / "eA_tier2_real_verification.json")
                    .read_text(encoding="utf-8"))
    gold, gold_path = load_gold()

    traces = ea["traces"]
    per_question, out_by_engine = [], {}

    for engine, tag in TAGS.items():
        merged = json.loads(
            (RESULTS / tag / "merged.json").read_text(encoding="utf-8"))
        preds = {}
        for k, v in merged.items():
            try:
                preds[int(k.split("\t")[0])] = v
            except ValueError:
                continue

        rows = [t for t in traces if t["tag"] == tag and t["kind"] == "silent"]
        n_silent = len(rows)
        n_v1v2 = sum(1 for t in rows if t["V1+V2"]["flagged"])
        assert n_silent == ea["summary"][tag]["breakdown"]["silent"]["n"], tag

        cat_stats = {c: {"n": 0, "intercepted": 0} for c in CATEGORIES}
        n_multi_stage = 0
        groupby_rows = []
        for t in rows:
            qid = t["question_id"]
            gsql, psql = gold.get(qid), preds.get(qid)
            if gsql is None or psql is None:
                cat, det, multi = "unparseable", "missing-sql", False
            else:
                cat, det, multi = classify(clean_sql(gsql), clean_sql(psql))
            if multi:
                n_multi_stage += 1
            flagged = bool(t["V1+V2"]["flagged"])
            cat_stats[cat]["n"] += 1
            cat_stats[cat]["intercepted"] += int(flagged)
            has_gb = bool(re.search(r"\bgroup\s+by\b", psql or "", re.I))
            has_gb_g = bool(re.search(r"\bgroup\s+by\b", gsql or "", re.I))
            if has_gb or has_gb_g:
                groupby_rows.append(t)
            per_question.append({
                "engine": engine, "tag": tag, "question_id": qid,
                "db_id": t["db_id"], "category": cat, "diff_detail": det,
                "v1v2_flagged": flagged, "v1_flagged": bool(t["V1"]["flagged"]),
                "triggered_v3": bool(t["V1+V2+V3"]["triggered_v3"]),
                "final_flagged_v123": bool(t["V1+V2+V3"]["flagged"]),
                "pred_has_group_by": has_gb,
                "gold_has_group_by": has_gb_g,
            })

        # ---- T2-2c 试点:含 GROUP BY 查询无条件触发 V3 的覆盖潜力
        n_gb = sum(1 for q in per_question if q["engine"] == engine
                   and q["pred_has_group_by"])
        n_gb_either = len(groupby_rows)
        already_trig = sum(1 for t in groupby_rows
                           if t["V1+V2+V3"]["triggered_v3"])
        already_final = sum(1 for t in groupby_rows
                            if t["V1+V2+V3"]["flagged"])
        t22c = {
            "rule": "若对含 GROUP BY 的查询无条件触发 V3 判定 "
                    "(规则代理 replay,未调用 LLM);覆盖口径主用预测 SQL "
                    "含 GROUP BY(与 V1 检测器一致),另报 pred∪gold 并集",
            "n_silent": n_silent,
            "n_pred_has_group_by": n_gb,
            "share_of_silent": round(n_gb / n_silent, 4) if n_silent else 0.0,
            "n_group_by_pred_or_gold": n_gb_either,
            "share_of_silent_pred_or_gold": round(
                n_gb_either / n_silent, 4) if n_silent else 0.0,
            "already_triggered_v3_under_current_policy": already_trig,
            "newly_routed_if_unconditional": n_gb_either - already_trig,
            "already_final_flagged_v123": already_final,
        }

        cats_out = {}
        for c in CATEGORIES:
            n = cat_stats[c]["n"]
            k = cat_stats[c]["intercepted"]
            cats_out[c] = {"n": n, "intercepted_v1v2": k,
                           "rate": round(k / n, 4) if n else None}

        out_by_engine[engine] = {
            "tag": tag,
            "n_silent": n_silent,
            "intercepted_v1v2_total": n_v1v2,
            "categories": cats_out,
            "multi_stage_diff_n": n_multi_stage,
            "t22c_groupby_unconditional_v3": t22c,
        }
        print(f"[{engine}] silent={n_silent} V1+V2={n_v1v2} "
              f"multi_stage={n_multi_stage}")
        for c in CATEGORIES:
            s = cats_out[c]
            if s["n"]:
                print(f"  {c:18s} n={s['n']:3d} "
                      f"int={s['intercepted_v1v2']:2d} "
                      f"rate={'' if s['rate'] is None else format(s['rate'], '.1%')}")
        print(f"  T2-2c: pred GROUP BY={n_gb}/{n_silent} "
              f"({t22c['share_of_silent']:.1%}), "
              f"pred∪gold={n_gb_either} "
              f"({t22c['share_of_silent_pred_or_gold']:.1%}), "
              f"already_triggered={already_trig}, "
              f"newly_routed={t22c['newly_routed_if_unconditional']}")

    # 合并两 backbone 的类别合计(便于论文单表)
    combined = {}
    for c in CATEGORIES:
        n = sum(out_by_engine[e]["categories"][c]["n"] for e in TAGS)
        k = sum(out_by_engine[e]["categories"][c]["intercepted_v1v2"]
                for e in TAGS)
        combined[c] = {"n": n, "intercepted_v1v2": k,
                       "rate": round(k / n, 4) if n else None}
    n_tot = sum(v["n"] for v in combined.values())
    k_tot = sum(v["intercepted_v1v2"] for v in combined.values())
    gb_n = sum(out_by_engine[e]["t22c_groupby_unconditional_v3"]
               ["n_pred_has_group_by"] for e in TAGS)
    gb_either = sum(out_by_engine[e]["t22c_groupby_unconditional_v3"]
                    ["n_group_by_pred_or_gold"] for e in TAGS)
    gb_trig = sum(out_by_engine[e]["t22c_groupby_unconditional_v3"]
                  ["already_triggered_v3_under_current_policy"] for e in TAGS)
    out_by_engine["both"] = {
        "n_silent": n_tot, "intercepted_v1v2_total": k_tot,
        "categories": combined,
        "t22c_groupby_unconditional_v3": {
            "n_silent": n_tot, "n_pred_has_group_by": gb_n,
            "share_of_silent": round(gb_n / n_tot, 4) if n_tot else 0.0,
            "n_group_by_pred_or_gold": gb_either,
            "share_of_silent_pred_or_gold": round(
                gb_either / n_tot, 4) if n_tot else 0.0,
            "already_triggered_v3_under_current_policy": gb_trig,
            "newly_routed_if_unconditional": gb_either - gb_trig,
        },
    }

    result = {
        "config": {
            "purpose": "T2-2c: 真实静默错误按类别分解 + V1+V2 拦截率 + "
                       "GROUP BY 无条件 V3 覆盖试点(规则 replay,无 LLM)",
            "silent_def": ea["config"]["note"],
            "intercepted_def": "eA_tier2_real_verification.json 中 V1+V2 层 "
                               "flagged=true(冻结预测上的验证器 replay)",
            "annotation_method": ("1. 用 sqlglot" +
                                  __doc__.split("1. 用 sqlglot")[1]
                                  .split("数据源")[0].strip()),
            "gold_source": str(gold_path),
            "sqlglot_version": sqlglot.__version__,
            "caveats": "自动标注存在已知局限:OR 组/多列谓词按整体匹配;"
                       "argmax 等价形已归一化;跨阶段多处差异按执行顺序首个"
                       "阶段定类,数量见 multi_stage_diff_n。",
        },
        "by_engine": out_by_engine,
        "per_question": per_question,
    }
    out = RESULTS / "t22_category_decomposition.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
