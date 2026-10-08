"""DIN-SQL placeholder 回收:从 logs.csv 重新提取 SQL,无需重跑 LLM。

根因: deepseek 不遵守 "SQL:" 前缀格式,改用 ```sql 围栏或裸 SQL 输出,
extract_sql_query 提取不到 → 36 题被记为 placeholder "SELECT * FROM table"。

提取优先级 (每行):
  1. self_correction 列的 Revised_SQL (原管线自纠错结果)
  2. sql_generation 列 "SQL:" 前缀 (原逻辑)
  3. ```sql 围栏内容 (取最后一个围栏)
  4. 裸 SELECT/WITH 块 (大小写敏感,截到最后一个分号)
行号映射: logs.csv 行号 ≠ dev 索引 (断点跳题),按 (question, db_id) 对齐。
产物: 覆盖 baselines/DIN-SQL/predict_dev.json 中 placeholder 行。
"""
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]   # repo root
DIN = BASE / "baselines" / "DIN-SQL"


def clean_sql(sql):
    if not sql:
        return None
    sql = str(sql)
    # 增强清洗: markdown 加粗符号 + 任意位置的代码围栏标记
    sql = sql.replace("**", "")
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE)
    return sql.strip() or None


def extract_revised(text):
    if pd.isna(text):
        return None
    m = re.search(r"Revised_SQL:\s*(.*?)$", str(text), re.DOTALL)
    return m.group(1).strip() if m else None


def extract_sql_prefix(text):
    if pd.isna(text):
        return None
    m = re.search(r"SQL:\s*(.*?)$", str(text), re.DOTALL)
    return m.group(1).strip() if m else None


def extract_fence(text):
    if pd.isna(text):
        return None
    blocks = re.findall(r"```(?:sql)?\s*(.*?)```", str(text), re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1]
    # 有尾围栏无首围栏: 截掉 ``` 之后内容再看
    t = re.sub(r"```[\s\S]*$", "", str(text)).strip()
    return t if re.search(r"\bSELECT\b", t, re.IGNORECASE) else None


def extract_bare(text):
    """裸 SQL: 移除围栏标记后取最后一个全大写 SELECT/WITH 起点截到最后分号。"""
    if pd.isna(text):
        return None
    t = str(text).replace("```", "")
    matches = list(re.finditer(r"\b(?:SELECT|WITH)\b", t))
    if not matches:
        return None
    seg = t[matches[-1].start():].strip()
    if ";" in seg:
        seg = seg[: seg.rindex(";") + 1]
    seg = seg.strip()
    # 剔除混入的尾注文本行 (含中文/问号/明显非 SQL 的行)
    lines = []
    for ln in seg.splitlines():
        if re.search(r"[\u4e00-\u9fff?（]|\*\*Final", ln):
            break
        lines.append(ln)
    return "\n".join(lines).strip() or None


def recover_one(row):
    # 收集所有候选;优先取以分号结尾的 (correction 响应可能被 max_tokens=2000
    # 截断 → Revised_SQL 不完整,此时应 fallback 到生成阶段的完整围栏)
    cands = []
    revised = clean_sql(extract_revised(row["self_correction"]))
    if revised:
        cands.append(revised)
    prefix = clean_sql(extract_sql_prefix(row["sql_generation"]))
    if prefix:
        cands.append(prefix)
    fence = clean_sql(extract_fence(row["sql_generation"]))
    if fence:
        cands.append(fence)
    bare = clean_sql(extract_bare(row["sql_generation"]))
    if bare:
        cands.append(bare)
    for c in cands:
        if c.rstrip().endswith(";"):
            return c, "complete"
    return (cands[0], "incomplete") if cands else (None, None)


def main():
    logs = pd.read_csv(DIN / "logs.csv")
    preds = json.load(open(DIN / "predict_dev.json", encoding="utf-8"))
    dev = json.load(open(BASE / "results" / "chess_bird61.json", encoding="utf-8"))

    # (question, db_id) → dev 索引
    qmap = {}
    for i, d in enumerate(dev):
        qmap[(d["question"], d["db_id"])] = i

    fixed, still = 0, []
    for _, row in logs.iterrows():
        key = (row["question"], row["db_id"])
        dev_idx = qmap.get(key)
        if dev_idx is None:
            still.append(f"未匹配 dev 索引: {str(row['question'])[:50]}")
            continue
        # 全量重提: 原管线 one_liner 用 replace('\n','') 导致关键字粘连
        # (如 "T1INNER JOIN"),必须从 logs.csv 原始多行文本重新提取
        sql, src = recover_one(row)
        if sql:
            # 规范单行化: \s+ → 单空格 (保留关键字间距)
            one_liner = re.sub(r"\s+", " ", sql).strip()
            preds[str(dev_idx)] = f"{one_liner}\t----- bird -----\t{row['db_id']}"
            fixed += 1
        else:
            still.append(f"q{dev_idx} ({row['db_id']}): 无可提取 SQL")

    json.dump(preds, open(DIN / "predict_dev.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    valid = sum(1 for v in preds.values() if isinstance(v, str) and not v.startswith("SELECT * FROM table"))
    print(f"\n回收 {fixed} 题; 仍有问题 {len(still)}: ")
    for s in still:
        print(" ", s)
    print(f"predict_dev.json 有效预测: {valid}/{len(preds)}")


if __name__ == "__main__":
    main()
