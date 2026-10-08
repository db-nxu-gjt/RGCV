"""MAC-SQL E1 300 题 EX 评估 (双骨干)。

对齐: shard jsonl 的 idx = dev_300.json 行序 (run.py enumerate 序号)
输入: baselines/DAIL-SQL/dataset/bird/dev/dev_300.json
      results/macsql_bird300_<tag>/shard*.jsonl (合并后按 idx 排序)
用法: python eval_macsql_e1.py <tag1> [tag2 ...]
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


def load_preds(tag):
    """合并 shard jsonl, 返回按 idx 排序的 pred 列表 (缺失位为 None)。"""
    out_dir = BASE / "results" / f"macsql_bird300_{tag}"
    rows = {}
    for f in sorted(out_dir.glob("shard*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            rows[o["idx"]] = o.get("pred", "")
    return [rows.get(i) for i in range(300)], len(rows)


def main():
    tags = sys.argv[1:]
    assert tags, "usage: eval_macsql_e1.py <tag1> [tag2 ...]"
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json", encoding="utf-8"))
    n = len(dev)
    assert n == 300

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
    for tag in tags:
        preds, got = load_preds(tag)
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
        print(f"    (rows={got}, missing={missing}, error_pred={sum(1 for p in preds if p and p.startswith('error'))})")


if __name__ == "__main__":
    main()
