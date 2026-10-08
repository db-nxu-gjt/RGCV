"""SQLCoder-7B-2 (Ollama) BIRD EX 评估。

输入: SQLCODER_results.jsonl
输出: EX_strict / 执行率,与 DAIL-SQL / MAC-SQL / ReFoRCE 横向对比
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
    results = [json.loads(l) for l in open(RES / "SQLCODER_results.jsonl", encoding="utf-8") if l.strip()]
    # 按 question_id 排序与 dev.json 对齐
    results.sort(key=lambda x: x["question_id"])
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev.json"))
    dev_by_qid = {d["question_id"]: d for d in dev}

    n = len(results)
    ex_strict = 0
    pred_exec = 0
    api_err = 0
    sql_err = []

    for r in results:
        item = dev_by_qid[r["question_id"]]
        db_id = r["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_res = run_sql(db_path, item["SQL"])

        if r["error"]:  # API 调用失败
            api_err += 1
            continue
        pred_res = run_sql(db_path, r["pred"])
        if pred_res is not None:
            pred_exec += 1
        elif not r["pred"]:
            sql_err.append((r["instance_id"], "empty"))
        else:
            sql_err.append((r["instance_id"], r["pred"][:80]))

        if gold_res is not None and pred_res is not None and gold_res == pred_res:
            ex_strict += 1

    print(f"{'='*55}")
    print(f"  SQLCoder-7B-2 (Ollama Q4) EX — BIRD 61 题子集")
    print(f"{'='*55}")
    print(f"  完成题数:        {n}")
    print(f"  API 调用失败:    {api_err}")
    print(f"  Pred 执行成功:   {pred_exec}/{n} ({pred_exec/max(n,1)*100:.1f}%)")
    print(f"  EX_strict:       {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    if sql_err:
        print(f"  执行失败样例 (前 5):")
        for inst, info in sql_err[:5]:
            print(f"    {inst}: {info}")

    print(f"\n{'='*55}")
    print(f"  横向对比 (同 61 题子集)")
    print(f"{'='*55}")
    print(f"  {'基线':<22} {'EX_strict':>14} {'执行率':>10} {'骨干':<16}")
    print(f"  {'-'*66}")
    print(f"  {'DAIL-SQL (Pass2)':<22} {'29/61 = 47.5%':>14} {'90.2%':>10} {'deepseek-chat':<16}")
    print(f"  {'ReFoRCE':<22} {'32/61 = 52.5%':>14} {'100.0%':>10} {'deepseek-chat':<16}")
    print(f"  {'MAC-SQL':<22} {'35/61 = 57.4%':>14} {'100.0%':>10} {'deepseek-chat':<16}")
    print(f"  {'SQLCoder-7B-2':<22} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>14} {f'{pred_exec/max(n,1)*100:.1f}%':>10} {'本地 7B Q4':<16}")


if __name__ == "__main__":
    main()
