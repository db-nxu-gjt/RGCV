"""DAIL-SQL 完整结果处理 + Excel 导出。

合并两遍管线(EUCDISQUESTIONMASK → EUCDISMASKPRESKLSIMTHR)的 SQL 结果,
加上真实答案(Gold SQL),输出 results/DAIL-SQL.xlsx。
"""
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件
dev_json_path = DAIL / "dataset" / "bird" / "dev" / "dev.json"
pass1_questions_path = DAIL / "dataset" / "process" / "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096" / "questions.json"
pass1_sql_path = DAIL / "dataset" / "process" / "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096" / "RESULTS_MODEL-deepseek-chat.txt"
pass2_questions_path = DAIL / "dataset" / "process" / "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096" / "questions.json"
pass2_sql_path = DAIL / "dataset" / "process" / "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096" / "RESULTS_MODEL-deepseek-chat.txt"

print("=== 加载数据 ===")
dev = json.load(open(dev_json_path))
print(f"dev.json: {len(dev)} 题")

pass1_qs = json.load(open(pass1_questions_path))["questions"]
pass2_qs = json.load(open(pass2_questions_path))["questions"]
print(f"pass1 questions: {len(pass1_qs)}")
print(f"pass2 questions: {len(pass2_qs)}")

pass1_sql = [l.strip() for l in open(pass1_sql_path)]
pass2_sql = [l.strip() for l in open(pass2_sql_path)]
print(f"pass1 SQL: {len(pass1_sql)} 行")
print(f"pass2 SQL: {len(pass2_sql)} 行")

# === 合并 ===
rows = []
for i, dev_item in enumerate(dev):
    row = OrderedDict()
    row["question_id"] = dev_item.get("question_id", i)
    row["db_id"] = dev_item["db_id"]
    row["difficulty"] = dev_item.get("difficulty", "")
    row["question"] = dev_item["question"][:200]
    row["gold_sql"] = dev_item.get("SQL", "")
    row["evidence"] = dev_item.get("evidence", "")[:100] if dev_item.get("evidence") else ""

    # pass1 SQL 和元信息
    if i < len(pass1_sql):
        row["pass1_prompt_token"] = pass1_qs[i].get("prompt_tokens", "")
        row["pass1_n_examples"] = pass1_qs[i].get("n_examples", "")
        row["pass1_predicted_sql"] = pass1_sql[i]
    else:
        row["pass1_predicted_sql"] = ""

    # pass2 SQL 和元信息
    if i < len(pass2_sql):
        row["pass2_prompt_token"] = pass2_qs[i].get("prompt_tokens", "")
        row["pass2_n_examples"] = pass2_qs[i].get("n_examples", "")
        row["pass2_predicted_sql"] = pass2_sql[i]
    else:
        row["pass2_predicted_sql"] = ""

    row["pass1_match_gold"] = pass1_sql[i].strip().rstrip(";").strip() == dev_item.get("SQL", "").strip().rstrip(";").strip() if i < len(pass1_sql) else False
    row["pass2_match_gold"] = pass2_sql[i].strip().rstrip(";").strip() == dev_item.get("SQL", "").strip().rstrip(";").strip() if i < len(pass2_sql) else False

    rows.append(row)

print(f"\n=== 合并完成: {len(rows)} 行 ===")

# === 导出 Excel ===
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "DAIL-SQL Results"

    headers = list(rows[0].keys())
    ws.append(headers)

    header_font = Font(bold=True)
    for col_idx in range(1, len(headers) + 1):
        ws.cell(row=1, column=col_idx).font = header_font
        ws.cell(row=1, column=col_idx).fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")

    for row in rows:
        ws.append([row[h] for h in headers])

    # 自动列宽
    for col_idx, h in enumerate(headers, 1):
        max_len = max(
            [len(str(h))] + [len(str(r.get(h, ""))) for r in rows[:10]]
        )
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 60)

    out_path = RESULTS_DIR / "DAIL-SQL.xlsx"
    wb.save(out_path)
    print(f"\n✅ Excel 已保存: {out_path}")
except ImportError:
    # fallback: CSV
    out_path = RESULTS_DIR / "DAIL-SQL.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write("\t".join(str(v).replace("\n", " ").replace("\t", " ") for v in r.values()) + "\n")
    print(f"\n✅ CSV fallback 已保存: {out_path}")

# === 统计 ===
pass1_hits = sum(1 for r in rows if r["pass1_match_gold"])
pass2_hits = sum(1 for r in rows if r["pass2_match_gold"])
print(f"\n=== 通过率(语法匹配 Gold SQL) ===")
print(f"Pass1 (EUCDISQUESTIONMASK): {pass1_hits}/{len(rows)} = {pass1_hits/len(rows)*100:.1f}%")
print(f"Pass2 (EUCDISMASKPRESKLSIMTHR): {pass2_hits}/{len(rows)} = {pass2_hits/len(rows)*100:.1f}%")
print(f"注意:语法精确匹配率 < 执行准确率(EX),真实 EX 见 eval_dailsql_ex.py;")
print(f"      本结果由 DeepSeek-chat 真实 LLM 生成(2026-09-16,网络恢复后重跑)。")
