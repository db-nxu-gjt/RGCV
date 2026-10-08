"""SafeQL BIRD EX 评估 (跨引擎结果集比对: gold/pred 均在 PG17 执行)。

输入: results/safeql_bird61.json
      {qid: {init_sql, pg_sql, pred_rows (safeql() 修正后结果集), gold_rows (gold 方言转换后 PG 结果集), status, gold_status}}
评估:
  - EX_strict: pred_rows == gold_rows (multiset, 数值感知归一化: 可解析 float 的值统一格式,
    规避 SQLite/PG 数值文本差异; 其余 str.strip() 比较)
  - 执行率: status==ok (safeql() 返回结果) 的题数比例
输出: EX_strict / 执行率 + 六基线横向对比
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
PRED = BASE / "results" / "safeql_bird61.json"


def norm_cell(v: str) -> str:
    try:
        f = float(v)
        return f"{f:.6g}"
    except (ValueError, TypeError):
        return (v or "").strip()


def norm_rows(rows):
    if rows is None:
        return None
    return sorted(tuple(norm_cell(c) for c in r) for r in rows)


def main():
    data = json.load(open(PRED, encoding="utf-8"))
    dev = json.load(open(BASE / "results" / "chess_bird61.json", encoding="utf-8"))
    n = len(dev)

    ex_strict = 0
    pred_exec = 0
    gold_exec = 0
    wrong_or_fail = []
    for d in dev:
        qid = str(d["question_id"])
        r = data.get(qid)
        if not r:
            wrong_or_fail.append((qid, d["db_id"], "MISSING"))
            continue
        p, g = norm_rows(r.get("pred_rows")), norm_rows(r.get("gold_rows"))
        if r.get("gold_status") == "ok":
            gold_exec += 1
        if r.get("status") == "ok" and p is not None:
            pred_exec += 1
            if g is not None and p == g:
                ex_strict += 1
            else:
                wrong_or_fail.append((qid, d["db_id"], "WRONG" if r.get("gold_status") == "ok" else "GOLD_FAIL"))
        else:
            wrong_or_fail.append((qid, d["db_id"], (r.get("status") or "ERR")[:70]))

    print(f"{'='*55}")
    print(f"  SafeQL (deepseek-chat) EX — BIRD 61 题子集 (PG17)")
    print(f"{'='*55}")
    print(f"  gold PG 可执行:  {gold_exec}/{n}")
    print(f"  Pred 执行成功:   {pred_exec}/{n} ({pred_exec/n*100:.1f}%)")
    print(f"  EX_strict:       {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    print(f"\n  失败/错误题 ({len(wrong_or_fail)}):")
    for qid, db, info in wrong_or_fail[:20]:
        print(f"    q{qid} ({db}): {info}")

    print(f"\n{'='*55}")
    print(f"  横向对比 (同 61 题子集)")
    print(f"{'='*55}")
    print(f"  {'基线':<22} {'EX_strict':>14} {'执行率':>10} {'骨干':<18}")
    print(f"  {'-'*70}")
    print(f"  {'DAIL-SQL (Pass2)':<22} {'29/61 = 47.5%':>14} {'90.2%':>10} {'deepseek-chat':<18}")
    print(f"  {'ReFoRCE':<22} {'32/61 = 52.5%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'CHESS':<22} {'32/61 = 52.5%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'DIN-SQL':<22} {'30/61 = 49.2%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'MAC-SQL':<22} {'35/61 = 57.4%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'SQLCoder-7B-2':<22} {'11/61 = 18.0%':>14} {'55.7%':>10} {'本地 7B Q4':<18}")
    print(f"  {'SafeQL':<22} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>14} {f'{pred_exec/n*100:.1f}%':>10} {'deepseek-chat':<18}")


if __name__ == "__main__":
    main()
