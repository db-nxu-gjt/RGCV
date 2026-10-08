"""BIRD dev.json → ReFoRCE omnisql 格式转换。

run.py 的 --omnisql_format_pth 接受一个 JSON 数组,每个元素需要:
  instance_id: str, 如 "local_BIRD_0000"
  question:    str
  db_id:       str
  SQL:         str (gold SQL)
  input_seq:   str (schema 描述,用于替代 get_table_info 的 prompts.txt)

ReFoRCE 把 BIRD 的 instance_id 格式化为 "local_BIRD_{question_id:04d}",
SQLite 路径则用 get_sqlite_path() 里的 "{db_id}/{db_id}.sqlite"。
"""
import json
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RES = BASE / "results"


def build_db_desc(db_id: str, tables_json: list) -> str:
    """从 tables.json 生成 ReFoRCE 期望的 schema 描述文本。

    格式模仿 ReFoRCE prompts.txt 的结构:
      # Table: <name>
      [(col, type_or_desc), ...]
      # Foreign keys
      ...
    """
    db = next((t for t in tables_json if t["db_id"] == db_id), None)
    if not db:
        return f"# Table: {db_id}\n  (schema not found in tables.json)\n"

    table_names = db["table_names_original"]
    column_names = db["column_names_original"]
    column_types = db["column_types"]
    primary_keys = db["primary_keys"]
    foreign_keys = db["foreign_keys"]

    # 按表分组列
    tables = {}
    for (tid, col_name), col_type in zip(column_names, column_types):
        if tid == -1:
            continue
        tname = table_names[tid]
        tables.setdefault(tname, []).append((col_name, col_type))

    lines = [f"The table structure information is described as follows:"]
    for tname, cols in tables.items():
        lines.append(f"# Table: {tname}")
        col_strs = [f"({c}, {t})" for c, t in cols]
        lines.append("[" + ", ".join(col_strs) + "]")

    if foreign_keys:
        lines.append("# Foreign keys")
        # BIRD fk 格式: [[col_idx_1, col_idx_2], ...]
        # 两个索引都指向 column_names_original 数组位置
        for fk in foreign_keys:
            c1_idx, c2_idx = fk[0], fk[1]
            (t1, cn1), (t2, cn2) = column_names[c1_idx], column_names[c2_idx]
            tname1 = table_names[t1] if t1 >= 0 else "?"
            tname2 = table_names[t2] if t2 >= 0 else "?"
            lines.append(f"{tname1}.`{cn1}` = {tname2}.`{cn2}`")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bird61", "bird300"], default="bird300",
                    help="bird61: DAIL dev/dev.json 全量61题原型集; bird300: results/sample_bird300.json E1子集")
    src = ap.parse_args().source

    tables = json.load(open(DAIL / "dataset" / "bird" / "tables.json"))
    if src == "bird300":
        sample = json.load(open(RES / "sample_bird300.json", encoding="utf-8"))
        # sample 里 question_id 是字符串, ReFoRCE run.py 的 f"{q_id:04d}" 需要 int
        items = [{"question_id": int(q["question_id"]), "db_id": q["db_id"],
                  "question": q["question"], "evidence": q.get("evidence", ""),
                  "SQL": q["SQL"]} for q in sample["questions"]]
        out_path = RES / "reforce_bird300_omnisql.json"
    else:
        items = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev.json"))
        out_path = RES / "reforce_bird61_omnisql.json"

    # ReFoRCE prompt.py 的 omni_sql_input_prompt_template:
    # task=="BIRD" 且 omnisql_format_pth 时,get_self_refine_prompt 直接返回
    # table_info(即 input_seq),因此 input_seq 必须是含问题和指令的完整 prompt
    OMNI_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""

    out = []
    for item in items:
        qid = item["question_id"]
        instance_id = f"local_BIRD_{qid:04d}"
        db_desc = build_db_desc(item["db_id"], tables)
        # evidence 追加到问题后(BIRD 特有知识提示)
        question = item["question"]
        if item.get("evidence"):
            question = f"{question}\n(Evidence: {item['evidence']})"
        out.append({
            "instance_id": instance_id,
            "question_id": qid,
            "question": question,
            "db_id": item["db_id"],
            "SQL": item["SQL"],
            "input_seq": OMNI_TEMPLATE.format(db_details=db_desc.rstrip("\n"), question=question),
        })

    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"转换完成:{len(out)} 题")
    print(f"输出:   {out_path}")
    print(f"样例 instance_id: {out[0]['instance_id']}")
    print(f"样例 db_desc 前 200 字: {out[0]['input_seq'][:200]}...")


if __name__ == "__main__":
    main()
