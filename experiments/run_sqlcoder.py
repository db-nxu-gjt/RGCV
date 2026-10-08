"""SQLCoder-7B-2 (Ollama) BIRD 61 题推理脚本。

模型: ollama sqlcoder:7b-code-q4_K_M (defog sqlcoder-7b-2 Q4 量化)
端点: http://localhost:11434/v1 (Ollama OpenAI 兼容端点,无需代理,api_key 占位)
Prompt: defog sqlcoder-7b-2 官方模板
Schema: 从各 SQLite 的 sqlite_master 提取真实 DDL (CREATE TABLE 语句)
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
# sqlcoder 系列均为补全模型(StarCoder/CodeLlama 基座),必须用 /v1/completions
# 默认用导入的 sqlcoder-7b-2;可 env 切换回 Ollama 库初代 sqlcoder:7b-q4_K_M
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
    """从 sqlite_master 提取 CREATE TABLE / CREATE INDEX DDL。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND type IN ('table','index') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0] + ";" for r in rows)


def extract_sql(text: str) -> str:
    """从模型输出提取纯 SQL(兜底清洗)。"""
    text = text.strip()
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    else:
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    text = text.strip().rstrip(";").strip()
    # PostgreSQL → SQLite 方言清洗(sqlcoder-7b-2 在 Postgres 风格语料上训练)
    text = re.sub(r"\bNULLS\s+(FIRST|LAST)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"::[A-Za-z_]+", "", text)  # PG 类型转换
    return text.strip().rstrip(";").strip()


def run_one(client: OpenAI, item: dict, ddl_cache: dict) -> dict:
    qid = item["question_id"]
    inst = f"local_BIRD_{qid:04d}"
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
        # 补全端点(sqlcoder 是 completion 模型,chat 模板会破坏输出格式)
        resp = client.completions.create(
            model=MODEL,
            prompt=prompt,
            temperature=0.0,
            max_tokens=512,
            stop=["[/SQL]", "###", "\n\n\n"],
        )
        raw = resp.choices[0].text or ""
        pred_sql = extract_sql(raw)
        err = None
    except Exception as e:
        raw = ""
        pred_sql = ""
        err = str(e)[:200]

    dt = time.time() - t0
    print(f"[{inst}] {db_id} {dt:.1f}s {'ERR: '+err if err else 'ok'}", flush=True)
    return {
        "instance_id": inst,
        "question_id": qid,
        "db_id": db_id,
        "question": item["question"],
        "evidence": item.get("evidence", ""),
        "SQL": item["SQL"],
        "raw": raw,
        "pred": pred_sql,
        "error": err,
        "time_s": round(dt, 1),
    }


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev.json"))

    out_path = RES / "SQLCODER_results.jsonl"
    done_ids = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
    todo = [d for d in dev if d["question_id"] not in done_ids]
    print(f"总 {len(dev)} 题, 已完成 {len(done_ids)}, 待跑 {len(todo)}")

    if not todo:
        print("全部完成")
        return

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    ddl_cache = {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(lambda it: run_one(client, it, ddl_cache), todo))

    with open(out_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = time.time() - t0
    ok = sum(1 for r in results if not r["error"])
    print(f"\n完成 {len(results)} 题 (成功 {ok}), 总耗时 {total/60:.1f} min")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
