"""DAIL-SQL 执行准确率(EX)评估。

语法精确匹配 ≠ EX:很多 SQL 写法不同但结果相同。
本脚本把 pass1/pass2 的预测 SQL 在对应 SQLite 上执行,与 gold SQL 结果比较。
"""
import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"


def run_sql(db_path: str, sql: str, timeout: int = 10):
    """在 SQLite 上执行 SQL,返回结果集或 None(执行失败)。"""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted([tuple(row) for row in rows])
    except Exception:
        return None


def evaluate_pass(pass_name: str, results_name: str, dev: list):
    """评估单个 pass 的 EX。"""
    results_path = (
        DAIL / "dataset" / "process" / results_name / "RESULTS_MODEL-deepseek-chat.txt"
    )
    preds = [l.strip() for l in open(results_path, encoding="utf-8")]

    ex = 0            # 执行准确率(双方可执行)
    ex_strict = 0      # 严格执行准确率(预测执行成功的才算对)
    pred_exec_ok = 0  # 预测执行成功数
    gold_exec_ok = 0  # gold 执行成功数
    n = min(len(preds), len(dev))

    details = []  # 存每题结果用于调试

    for i in range(n):
        item = dev[i]
        db_id = item["db_id"]
        gold_sql = item["SQL"].strip()
        pred_sql = preds[i].strip()
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")

        gold_res = run_sql(db_path, gold_sql)
        pred_res = run_sql(db_path, pred_sql)

        gold_ok = gold_res is not None
        pred_ok = pred_res is not None
        if gold_ok:
            gold_exec_ok += 1
        if pred_ok:
            pred_exec_ok += 1

        match = gold_ok and pred_ok and gold_res == pred_res
        if match:
            ex += 1
            ex_strict += 1
        elif pred_ok:
            ex_strict += 0  # pred 执行了但结果不对

        details.append({
            "idx": i,
            "db_id": db_id,
            "gold_ok": gold_ok,
            "pred_ok": pred_ok,
            "match": match,
            "gold_sql": gold_sql[:120],
            "pred_sql": pred_sql[:120],
        })

    # 双方可执行子集 EX(更严格)
    both_ok = [d for d in details if d["gold_ok"] and d["pred_ok"]]
    both_match = sum(1 for d in both_ok if d["match"])

    print(f"\n{'='*60}")
    print(f"  {pass_name} EX 评估")
    print(f"{'='*60}")
    print(f"  总题数:         {n}")
    print(f"  Gold 执行成功:  {gold_exec_ok}/{n} ({gold_exec_ok/n*100:.1f}%)")
    print(f"  Pred 执行成功:  {pred_exec_ok}/{n} ({pred_exec_ok/n*100:.1f}%)")
    print(f"  EX_strict(所有题): {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    if len(both_ok) > 0:
        print(f"  EX_both(双方可执行): {both_match}/{len(both_ok)} = {both_match/len(both_ok)*100:.1f}%")

    # 展示不匹配样本(前 5 个)
    mismatches = [d for d in details if d["gold_ok"] and d["pred_ok"] and not d["match"]]
    errors = [d for d in details if not d["pred_ok"]]
    print(f"\n  预测执行错误:  {len(errors)} 题")
    print(f"  双方可执行但结果不同: {len(mismatches)} 题")
    if mismatches:
        print(f"\n  [前 5 个不匹配样本]")
        for d in mismatches[:5]:
            print(f"    #{d['idx']} [{d['db_id']}]")
            print(f"      GOLD: {d['gold_sql'][:100]}...")
            print(f"      PRED: {d['pred_sql'][:100]}...")

    return ex_strict, both_match, len(both_ok), n


def main():
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev.json"))

    passes = [
        ("Pass1 (EUCDISQUESTIONMASK)", "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096"),
        ("Pass2 (EUCDISMASKPRESKLSIMTHR)", "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096"),
    ]

    results = {}
    for name, rname in passes:
        results[name] = evaluate_pass(name, rname, dev)

    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    for name, (ex_strict, both_match, both_total, n) in results.items():
        print(f"  {name}:")
        print(f"    EX_strict = {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
        if both_total > 0:
            print(f"    EX_both   = {both_match}/{both_total} = {both_match/both_total*100:.1f}%")


if __name__ == "__main__":
    main()
