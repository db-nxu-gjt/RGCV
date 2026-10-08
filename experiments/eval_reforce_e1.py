"""ReFoRCE E1 300 题 EX 评估 (双骨干: deepseek-v4-pro / kimi-k2.6)。

输入: results/reforce_bird300_omnisql.json (300 题含 gold SQL)
      results/reforce_bird300_<tag>/<local_BIRD_xxxx>/result.sql
输出: 各骨干 EX_strict / 执行率; gold 结果集执行一次缓存复用
"""
import json
import sqlite3
import concurrent.futures as cf
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RES = BASE / "results"
TAGS = ["deepseek-v4-pro", "kimi-k2.6"]
Q_TIMEOUT = 60


def _exec(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted(tuple(r) for r in rows)
    except Exception:
        return None


def run_sql(db_path, sql):
    """带线程超时的执行 (Windows 无 signal.alarm)。超时返回 None。"""
    if not sql or not sql.strip():
        return None
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(_exec, db_path, sql).result(timeout=Q_TIMEOUT)
        except cf.TimeoutError:
            return None


def main():
    omni = json.load(open(RES / "reforce_bird300_omnisql.json", encoding="utf-8"))
    n = len(omni)
    print(f"BIRD 300 题 × {len(TAGS)} 骨干 — ReFoRCE EX 评估")

    # gold 结果集缓存 (只执行一次)
    gold_cache = {}
    for i, item in enumerate(omni):
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_cache[item["instance_id"]] = (db_path, run_sql(db_path, item["SQL"]))
        if (i + 1) % 100 == 0:
            print(f"  gold 执行 {i+1}/{n}")
    gold_fail = sum(1 for v in gold_cache.values() if v[1] is None)
    print(f"gold 可执行: {n - gold_fail}/{n}")

    summary = {}
    for tag in TAGS:
        ex_strict = pred_exec = comparable = 0
        errors, mismatch, missing = [], [], []
        for item in omni:
            inst = item["instance_id"]
            db_path, gold_res = gold_cache[inst]
            sql_path = RES / f"reforce_bird300_{tag}" / inst / "result.sql"
            if not sql_path.exists():
                missing.append(inst)
                continue
            pred_res = run_sql(db_path, sql_path.read_text(encoding="utf-8").strip())
            if pred_res is None:
                errors.append(inst)
                continue
            pred_exec += 1
            if gold_res is None:
                continue  # gold 不可执行, 不计分
            comparable += 1
            if pred_res == gold_res:
                ex_strict += 1
            else:
                mismatch.append(inst)
        summary[tag] = (ex_strict, pred_exec, comparable, errors, mismatch, missing)

    print(f"\n{'='*60}")
    print(f"  {'骨干':<20} {'EX_strict':>16} {'执行率':>10} {'可比':>8}")
    print(f"  {'-'*58}")
    for tag in TAGS:
        ex, pe, comp, errs, mm, miss = summary[tag]
        print(f"  {tag:<20} {f'{ex}/{n} = {ex/n*100:.1f}%':>16} {f'{pe}/{n-pe-len(errs)-len(miss)}' if True else '':>10} {comp:>8}")
        if errs:
            print(f"    执行错误({len(errs)}): {errs[:6]}")
        if mm:
            print(f"    结果不匹配({len(mm)}): {mm[:6]}")
        if miss:
            print(f"    缺失({len(miss)}): {miss[:6]}")


if __name__ == "__main__":
    main()
