"""CHESS E1 300 题 EX 评估 (双骨干)。

对齐: dev_300.json 行序 = 评估 idx; merged.json 的 qid = question_id 字段
输入: results/chess_bird300_<tag>/merged.json (qid -> "sql\t----- bird -----\tdb_id" 原样, 取 \t 前的 SQL)
用法: python eval_chess_e1.py <tag1> [tag2 ...]
"""
import json
import sys
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


def load_preds(tag, qid2idx):
    """merged.json (qid->pred) 映射回按 idx 排序的 300 长度列表。"""
    p = BASE / "results" / f"chess_bird300_{tag}" / "merged.json"
    merged = json.load(open(p, encoding="utf-8"))
    preds = [None] * 300
    for qid, v in merged.items():
        idx = qid2idx.get(int(qid))
        if idx is not None:
            preds[idx] = v.split("\t----- bird -----")[0].strip() if v else ""
    return preds, len(merged)


def main():
    tags = sys.argv[1:]
    assert tags, "usage: eval_chess_e1.py <tag1> [tag2 ...]"
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json", encoding="utf-8"))
    n = len(dev)
    assert n == 300
    qid2idx = {d["question_id"]: i for i, d in enumerate(dev)}

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
    for tag in tags:
        preds, got = load_preds(tag, qid2idx)
        ex_strict = pred_exec = 0
        missing = sum(1 for p in preds if p is None)
        for i, sql in enumerate(preds):
            db_path, gold_res = gold_cache[i]
            pred_res = run_sql(db_path, sql)
            if pred_res is None:
                continue
            pred_exec += 1
            if gold_res is None:
                continue
            if pred_res == gold_res:
                ex_strict += 1
        print(f"  {tag:<24} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>16} {pred_exec}/{n}")
        print(f"    (merged={got}, missing={missing})")


if __name__ == "__main__":
    main()
