"""MAC-SQL BIRD EX 评估。

输入: MACSQL_results.jsonl (含 pred 字段) + BIRD dev.json
输出: EX_strict / EX_both / 执行率 + 与 DAIL-SQL 对比
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
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted([tuple(row) for row in rows])
    except Exception:
        return None


def main():
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev.json"))

    # MAC-SQL 结果 (按 idx 排序)
    mac_lines = []
    with open(RES / "MACSQL_results.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                mac_lines.append(json.loads(line))
    mac_lines.sort(key=lambda x: int(x["idx"]))
    print(f"MAC-SQL results: {len(mac_lines)}")

    # 构建 idx → dev item 映射 (MAC-SQL idx 是 enumerate 序号)
    ex_strict = 0
    gold_exec = 0
    pred_exec = 0
    both_ok = 0
    both_match = 0
    execution_errors = []

    for i, (item, mac) in enumerate(zip(dev, mac_lines)):
        db_id = item["db_id"]
        gold_sql = item["SQL"].strip().rstrip(";").strip()
        pred_sql = mac.get("pred", "").strip().rstrip(";").strip()
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")

        gold_res = run_sql(db_path, gold_sql)
        pred_res = run_sql(db_path, pred_sql)

        g_ok = gold_res is not None
        p_ok = pred_res is not None
        if g_ok:
            gold_exec += 1
        if p_ok:
            pred_exec += 1

        match = g_ok and p_ok and gold_res == pred_res
        if match:
            ex_strict += 1
        if g_ok and p_ok:
            both_ok += 1
            if match:
                both_match += 1
        elif p_ok is None and not match:
            execution_errors.append((i, db_id, pred_sql[:100]))

    n = min(len(mac_lines), len(dev))

    print(f"\n{'='*55}")
    print(f"  MAC-SQL (deepseek-chat) EX  —  BIRD 61 题子集")
    print(f"{'='*55}")
    print(f"  Gold 执行成功:  {gold_exec}/{n} ({gold_exec/n*100:.1f}%)")
    print(f"  Pred 执行成功:  {pred_exec}/{n} ({pred_exec/n*100:.1f}%)")
    print(f"  EX_strict:       {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    if both_ok > 0:
        print(f"  EX_both:         {both_match}/{both_ok} = {both_match/both_ok*100:.1f}%")
    print(f"  Pred 执行错误:   {len(execution_errors)} 题")
    print(f"  总 token 消耗:   {mac_lines[0].get('cur_total_prompt_tokens','?')} (per question log)")

    # ===== 横向对比 =====
    print(f"\n{'='*55}")
    print(f"  横向对比 (同 61 题子集, deepseek-chat)")
    print(f"{'='*55}")
    print(f"  {'基线':<15} {'EX_strict':>12} {'执行率':>12}")
    print(f"  {'-'*40}")
    print(f"  {'DAIL-SQL (Pass1)':<15} {'29/61 = 47.5%':>12} {'86.9%':>12}")
    print(f"  {'DAIL-SQL (Pass2)':<15} {'29/61 = 47.5%':>12} {'90.2%':>12}")
    print(f"  {'MAC-SQL':<15} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>12} {f'{pred_exec/n*100:.1f}%':>12}")
    print(f"\n  注:MAC-SQL 多智能体协作 vs DAIL-SQL 单智能体 EUCDIS pipeline")


if __name__ == "__main__":
    main()
