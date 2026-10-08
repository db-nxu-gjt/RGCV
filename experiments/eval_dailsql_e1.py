"""DAIL-SQL E1 300 题 EX 评估 (pass1/pass2 × 双骨干)。

对齐: RESULTS 行序 = questions.json 序 = dev_300.json 题序 (resplit 时固定)
输入: baselines/DAIL-SQL/dataset/bird/dev/dev_300.json (300 题含 gold SQL)
      process 目录下 4 个 RESULTS_MODEL-*.txt
"""
import json
import sqlite3
import concurrent.futures as cf
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RES = BASE / "results"
PROC = DAIL / "dataset" / "process"
P1 = PROC / "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096"
P2 = PROC / "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096"
P2K = PROC / "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096_KIMI"

RUNS = [
    ("pass1 deepseek-v4-pro", P1, "deepseek-v4-pro"),
    ("pass2 deepseek-v4-pro", P2, "deepseek-v4-pro"),
    ("pass1 kimi-k2.6", P1, "kimi-k2.6"),
    ("pass2 kimi-k2.6", P2K, "kimi-k2.6"),
]
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
    if not sql or not sql.strip():
        return None
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(_exec, db_path, sql).result(timeout=Q_TIMEOUT)
        except cf.TimeoutError:
            return None


def main():
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json", encoding="utf-8"))
    n = len(dev)
    assert all(len((p / f"RESULTS_MODEL-{m}.txt").read_text(encoding="utf-8").splitlines()) == n
               for _, p, m in RUNS), "RESULTS 行数 != 300"

    # gold 结果集缓存
    gold_cache = {}
    for i, item in enumerate(dev):
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_cache[i] = (db_path, run_sql(db_path, item.get("SQL") or item.get("query", "")))
        if (i + 1) % 100 == 0:
            print(f"  gold 执行 {i+1}/{n}")
    gold_fail = sum(1 for v in gold_cache.values() if v[1] is None)
    print(f"gold 可执行: {n - gold_fail}/{n}")

    print(f"\n{'='*62}")
    print(f"  {'配置':<24} {'EX_strict':>16} {'执行成功':>9}")
    print(f"  {'-'*60}")
    for name, pdir, model in RUNS:
        lines = (pdir / f"RESULTS_MODEL-{model}.txt").read_text(encoding="utf-8").splitlines()
        ex_strict = pred_exec = comparable = 0
        errs = 0
        for i, sql in enumerate(lines):
            db_path, gold_res = gold_cache[i]
            pred_res = run_sql(db_path, sql)
            if pred_res is None:
                errs += 1
                continue
            pred_exec += 1
            if gold_res is None:
                continue
            comparable += 1
            if pred_res == gold_res:
                ex_strict += 1
        print(f"  {name:<24} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>16} {pred_exec}/{n}")
    print(f"  gold 不可执行 {gold_fail} 题不计分")


if __name__ == "__main__":
    main()
