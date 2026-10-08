"""ReFoRCE BIRD EX 评估。

输入: reforce_bird61/<local_BIRD_xxxx>/result.sql + reforce_bird61_omnisql.json (gold)
输出: EX_strict / 执行率,并与 DAIL-SQL、MAC-SQL 对比
"""
import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RES = BASE / "results"


def run_sql(db_path: str, sql: str, timeout: int = 10):
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted([tuple(r) for r in rows])
    except Exception:
        return None


def main():
    omni = json.load(open(RES / "reforce_bird61_omnisql.json", encoding="utf-8"))

    n = len(omni)
    ex_strict = 0
    pred_exec = 0
    missing = []
    errors = []
    results = []

    for item in omni:
        inst = item["instance_id"]
        sql_path = RES / "reforce_bird61" / inst / "result.sql"
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_res = run_sql(db_path, item["SQL"])

        if not sql_path.exists():
            missing.append(inst)
            results.append((inst, db_id, None, False))
            continue

        pred_sql = sql_path.read_text(encoding="utf-8").strip()
        pred_res = run_sql(db_path, pred_sql)

        ok = pred_res is not None
        if ok:
            pred_exec += 1
        else:
            errors.append(inst)
        if gold_res is not None and ok and gold_res == pred_res:
            ex_strict += 1
        results.append((inst, db_id, pred_res, ok))

    done = n - len(missing)
    print(f"{'='*55}")
    print(f"  ReFoRCE (deepseek-chat) EX — BIRD 61 题子集")
    print(f"{'='*55}")
    print(f"  完成题数:        {done}/{n}")
    if missing:
        print(f"  未完成:          {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"  Pred 执行成功:   {pred_exec}/{done} ({pred_exec/max(done,1)*100:.1f}%)")
    print(f"  EX_strict:       {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    if errors:
        print(f"  执行错误题目:    {errors[:8]}{'...' if len(errors) > 8 else ''}")

    print(f"\n{'='*55}")
    print(f"  横向对比 (同 61 题子集, deepseek-chat)")
    print(f"{'='*55}")
    print(f"  {'基线':<18} {'EX_strict':>14} {'执行率':>10}")
    print(f"  {'-'*46}")
    print(f"  {'DAIL-SQL (Pass2)':<18} {'29/61 = 47.5%':>14} {'90.2%':>10}")
    print(f"  {'MAC-SQL':<18} {'35/61 = 57.4%':>14} {'100.0%':>10}")
    print(f"  {'ReFoRCE':<18} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>14} {f'{pred_exec/max(done,1)*100:.1f}%':>10}")


if __name__ == "__main__":
    main()
