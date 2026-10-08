"""SQLCoder-7B-2 E1 300 题 EX 评估 (本地单模型, 无双骨干维度)。

输入: results/SQLCODER_results_300.jsonl (idx/question_id/db_id/pred)
评估口径与 eval_chess_e1 / eval_macsql_e1 一致:
  gold 结果集缓存执行一次 + 60s/查询线程超时, EX_strict (结果集精确匹配)
"""
import json
import sqlite3
import concurrent.futures as cf
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
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
    assert n == 300

    recs = [json.loads(l) for l in open(BASE / "results" / "SQLCODER_results_300.jsonl", encoding="utf-8") if l.strip()]
    pred_by_qid = {r["question_id"]: r["pred"] for r in recs}
    preds = [pred_by_qid.get(d["question_id"], "") for d in dev]
    ptok = sum(r.get("prompt_tokens") or 0 for r in recs)
    ctok = sum(r.get("completion_tokens") or 0 for r in recs)
    errs = sum(1 for r in recs if r["error"])
    print(f"records={len(recs)} api_errors={errs} prompt_tok={ptok:,} comp_tok={ctok:,}")

    gold_cache = {}
    for i, item in enumerate(dev):
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_cache[i] = (db_path, run_sql(db_path, item.get("SQL") or item.get("query", "")))
    gold_fail = sum(1 for v in gold_cache.values() if v[1] is None)
    print(f"gold 可执行: {n - gold_fail}/{n}")

    ex_strict = pred_exec = missing = 0
    for i, sql in enumerate(preds):
        if not sql or not sql.strip():
            missing += 1
            continue
        db_path, gold_res = gold_cache[i]
        pred_res = run_sql(db_path, sql)
        if pred_res is None:
            continue
        pred_exec += 1
        if gold_res is None:
            continue
        if pred_res == gold_res:
            ex_strict += 1
    print(f"\nSQLCoder-7B-2 (local): EX_strict {ex_strict}/{n} = {ex_strict/n*100:.1f}%  执行成功 {pred_exec}/{n}  空预测 {missing}")


if __name__ == "__main__":
    main()
