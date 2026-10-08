"""SQLCoder-7B-2 (Ollama) BIRD E1 300 题推理脚本。

与 run_sqlcoder.py (61 题版) 差异:
  - 数据源: dev_300.json (E1 分层子集, 行序 = 评估 idx)
  - 输出: results/SQLCODER_results_300.jsonl (与 61 题结果隔离)
  - 4 workers; 断点续跑 (question_id 去重); 记录 Ollama usage token
模型: sqlcoder-7b-2 为专用 SFT 模型, 骨干即模型本身, E1 无双骨干维度 (单行)
"""
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RES = BASE / "results"
# sqlcoder 系列均为补全模型(StarCoder/CodeLlama 基座), 必须用 /v1/completions
MODEL = os.environ.get("SQLCODER_MODEL", "sqlcoder-7b-2:q4_K_M")

PROMPT_TEMPLATE = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a SQLite database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQLite SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""


def get_ddl(db_path: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND type IN ('table','index') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0] + ";" for r in rows)


def extract_sql(text: str) -> str:
    text = text.strip()
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    text = text.strip().rstrip(";").strip()
    # PostgreSQL → SQLite 方言清洗 (sqlcoder-7b-2 在 PG 风格语料上训练)
    text = re.sub(r"\bNULLS\s+(FIRST|LAST)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"::[A-Za-z_]+", "", text)
    return text.strip().rstrip(";").strip()


def run_one(client: OpenAI, idx: int, item: dict, ddl_cache: dict) -> dict:
    qid = item["question_id"]
    db_id = item["db_id"]

    if db_id not in ddl_cache:
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        ddl_cache[db_id] = get_ddl(db_path)
    schema = ddl_cache[db_id]

    question = item["question"]
    if item.get("evidence"):
        question = f"{question} (Evidence: {item['evidence']})"

    prompt = PROMPT_TEMPLATE.format(question=question, schema=schema)

    t0 = time.time()
    try:
        resp = client.completions.create(
            model=MODEL,
            prompt=prompt,
            temperature=0.0,
            max_tokens=512,
            stop=["[/SQL]", "###", "\n\n\n"],
        )
        raw = resp.choices[0].text or ""
        pred_sql = extract_sql(raw)
        u = getattr(resp, "usage", None)
        ptok = u.prompt_tokens if u else None
        ctok = u.completion_tokens if u else None
        err = None
    except Exception as e:
        raw = ""
        pred_sql = ""
        ptok = ctok = None
        err = str(e)[:200]

    dt = time.time() - t0
    print(f"[{idx:03d}] q{qid} {db_id} {dt:.1f}s {'ERR: '+err if err else 'ok'}", flush=True)
    return {
        "idx": idx,
        "question_id": qid,
        "db_id": db_id,
        "question": item["question"],
        "evidence": item.get("evidence", ""),
        "SQL": item["SQL"],
        "raw": raw,
        "pred": pred_sql,
        "error": err,
        "time_s": round(dt, 1),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
    }


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json", encoding="utf-8"))
    assert len(dev) == 300

    out_path = RES / "SQLCODER_results_300.jsonl"
    done_ids = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
    todo = [(i, d) for i, d in enumerate(dev) if d["question_id"] not in done_ids]
    print(f"总 {len(dev)} 题, 已完成 {len(done_ids)}, 待跑 {len(todo)}")

    if not todo:
        print("全部完成")
        return

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    ddl_cache = {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(lambda it: run_one(client, it[0], it[1], ddl_cache), todo))

    with open(out_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = time.time() - t0
    ok = sum(1 for r in results if not r["error"])
    print(f"\n完成 {len(results)} 题 (成功 {ok}), 总耗时 {total/60:.1f} min")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
