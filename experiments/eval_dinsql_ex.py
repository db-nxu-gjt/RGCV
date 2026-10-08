"""DIN-SQL BIRD EX 评估。

输入: baselines/DIN-SQL/predict_dev.json
      格式 {"<子集行号>": "<sql>\t----- bird -----\t<db_id>"}
      (键是 chess_bird61.json 的行索引,非 BIRD question_id)
输出: EX_strict / 执行率 + 五基线横向对比
"""
import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
PRED = BASE / "baselines" / "DIN-SQL" / "predict_dev.json"


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
    preds = json.load(open(PRED, encoding="utf-8"))
    dev = json.load(open(BASE / "results" / "chess_bird61.json", encoding="utf-8"))

    n = len(dev)
    ex_strict = 0
    pred_exec = 0
    missing = 0
    failed = []

    for i, item in enumerate(dev):
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_res = run_sql(db_path, item["SQL"])

        raw = preds.get(str(i))
        if not raw or not isinstance(raw, str) or raw.startswith("SELECT * FROM table"):
            missing += 1
            continue
        pred_sql = raw.split("\t----- bird -----")[0].strip().rstrip(";")
        pred_res = run_sql(db_path, pred_sql)

        if pred_res is not None:
            pred_exec += 1
        else:
            failed.append((i, pred_sql[:80]))
        if gold_res is not None and pred_res is not None and gold_res == pred_res:
            ex_strict += 1

    done = n - missing
    print(f"{'='*55}")
    print(f"  DIN-SQL (deepseek-chat) EX — BIRD 61 题子集")
    print(f"{'='*55}")
    print(f"  完成题数:        {done}/{n}" + (f"  (缺失 {missing})" if missing else ""))
    print(f"  Pred 执行成功:   {pred_exec}/{done} ({pred_exec/max(done,1)*100:.1f}%)")
    print(f"  EX_strict:       {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    if failed:
        print(f"  执行失败样例 (前 5):")
        for qid, info in failed[:5]:
            print(f"    idx{qid}: {info}")

    print(f"\n{'='*55}")
    print(f"  横向对比 (同 61 题子集)")
    print(f"{'='*55}")
    print(f"  {'基线':<22} {'EX_strict':>14} {'执行率':>10} {'骨干':<18}")
    print(f"  {'-'*70}")
    print(f"  {'DAIL-SQL (Pass2)':<22} {'29/61 = 47.5%':>14} {'90.2%':>10} {'deepseek-chat':<18}")
    print(f"  {'ReFoRCE':<22} {'32/61 = 52.5%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'CHESS':<22} {'32/61 = 52.5%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'MAC-SQL':<22} {'35/61 = 57.4%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'SQLCoder-7B-2':<22} {'11/61 = 18.0%':>14} {'55.7%':>10} {'本地 7B Q4':<18}")
    print(f"  {'DIN-SQL':<22} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>14} {f'{pred_exec/max(done,1)*100:.1f}%':>10} {'deepseek-chat':<18}")


if __name__ == "__main__":
    main()
