"""SafeQL E1 300 题双骨干驱动 (初始 SQL -> PG safeql() 修正执行)。

基于 run_safeql_bird61.py (deepseek-chat 单骨干) 扩展:
  - 数据源 dev_300.json (E1 分层子集)
  - 双骨干: v4pro = deepseek-v4-pro (t=0, max_tokens=4096)
            kimi  = kimi-k2.6 (t=1 平台强制, thinking 默认开启, max_tokens=16384)
  - 输出隔离: results/safeql_bird300_<tag>.json
  - 记录 LLM usage token; 断点续跑 (status=ok 跳过)
  - gold 转换器与 61 题一致 (GOLD_SPECIAL 按题扩展)
用法: python run_safeql_e1.py <v4pro|kimi> [--gold-only]
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg2
import requests

BASE = Path(__file__).resolve().parents[1]              # src/rgcv_repro/
REPO = BASE.parents[1]                                  # RGCV/ 仓库根
DAIL = REPO / "baselines" / "DAIL-SQL"
BIRD_DIR = DAIL / "dataset" / "bird" / "database"
RES = BASE / "results"
MAX_WORKERS = 3
TIMEOUT = 600

_p = os.environ.get("HTTPS_PROXY")           # 可选:经环境变量注入代理
PROXY = {"http": _p, "https": _p} if _p else None
PG = dict(host="localhost", port=5432, user="postgres", password="safeql", dbname="postgres")
_lock = threading.Lock()

BACKBONES = {
    "v4pro": dict(
        tag="v4pro", url="https://api.deepseek.com/v1/chat/completions",
        key=os.environ.get("DEEPSEEK_API_KEY", ""), model="deepseek-v4-pro",
        temperature=0, max_tokens=16384,
    ),
    "kimi": dict(
        tag="kimi", url="https://api.moonshot.cn/v1/chat/completions",
        key=os.environ.get("MOONSHOT_API_KEY", ""), model="kimi-k2.6",
        temperature=1, max_tokens=16384,
    ),
}


def _load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def gen_sql(bb: dict, question: str, evidence: str, db_id: str):
    """生成初始 PG SQL, 返回 (sql, usage_dict)。"""
    schema_file = BIRD_DIR / db_id / "schema.sql"
    if schema_file.exists():
        schema = schema_file.read_text(encoding="utf-8", errors="replace")
    else:
        import sqlite3
        con = sqlite3.connect(str(BIRD_DIR / db_id / f"{db_id}.sqlite"))
        schema = "\n\n".join(
            r[0] for r in con.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"))
        con.close()

    prompt = (
        "You are an expert in SQL. The target DBMS is PostgreSQL 17. "
        "Generate ONE PostgreSQL query to answer the question.\n\n"
        f"Database schema:\n{schema}\n\n"
        f"External knowledge: {evidence or '(none)'}\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- Output ONLY the SQL query, no explanation, no markdown.\n"
        "- Use PostgreSQL syntax (e.g. EXTRACT instead of STRFTIME).\n"
        "- Use the exact table and column names from the schema.\n"
    )
    t0 = time.time()
    usage = None
    for attempt in range(2):  # E1: 偶发 content 空 (reasoning 吃满 max_tokens), 重试一次
        r = requests.post(
            bb["url"], headers={"Authorization": f"Bearer {bb['key']}"},
            json={"model": bb["model"],
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": bb["temperature"], "max_tokens": bb["max_tokens"]},
            proxies=PROXY, timeout=300)
        r.raise_for_status()
        resp = r.json()
        sql = resp["choices"][0]["message"]["content"].strip()
        u = resp.get("usage") or {}
        usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                 "secs": round(time.time() - t0, 1)}
        if sql:
            break
    sql = re.sub(r"^```(sql)?\s*|\s*```$", "", sql, flags=re.M).strip()
    return sql, usage


def to_pg_ident(sql: str) -> str:
    """语法级方言适配: 反引号/方括号/双引号标识符 -> 双引号小写 (与迁移后小写 schema 一致)。"""
    sql = re.sub(r"`([^`]*)`", lambda m: '"' + m.group(1).lower().replace('"', '""') + '"', sql)
    sql = re.sub(r"\[([A-Za-z_][\w ]*)\]", lambda m: '"' + m.group(1).lower() + '"', sql)
    sql = re.sub(r'"([^"]*)"', lambda m: '"' + m.group(1).lower().replace('"', '""') + '"', sql)
    return sql


GOLD_SPECIAL = {
    # 61 题特例 (保留, 300 题超集兼容)
    1004: (("SELECT SUM(T1.wins),T2.forename, T2.surname FROM",
            "SELECT SUM(T1.wins),MIN(T2.forename), MIN(T2.surname) FROM"),
           ("ORDER BY T2.dob ASC", "ORDER BY MIN(T2.dob) ASC")),
    1026: (("SELECT teamDetails.team_long_name FROM",
            "SELECT MIN(teamDetails.team_long_name) FROM"),),
    1123: (("SELECT DISTINCT t1.player_name FROM", "SELECT t1.player_name FROM"),),
    1125: (("SELECT DISTINCT t1.player_name FROM", "SELECT t1.player_name FROM"),),
    1481: (("T2.Date BETWEEN 201301 AND 201312", "(T2.Date)::integer BETWEEN 201301 AND 201312"),),
}


def to_pg_gold(sql: str, qid: int = 0) -> str:
    """gold SQL 的 SQLite->PG 方言转换 (函数级 + 语法级), 使 gold 可在 PG 直接执行。"""
    for pair in GOLD_SPECIAL.get(qid, ()):
        sql = sql.replace(*pair)
    sql = re.sub(r"\bSTRFTIME\s*\(\s*'%Y'\s*,\s*([^)]+?)\)",
                 r"(EXTRACT(YEAR FROM (\1)::timestamp))::text", sql, flags=re.I)
    sql = re.sub(r"\bDATETIME\s*\(\s*\)\s*-\s*([A-Za-z_][\w.]*)",
                 r"(CURRENT_DATE - (\1)::date)", sql, flags=re.I)
    sql = re.sub(r"\bLIKE\b", "ILIKE", sql, flags=re.I)
    return to_pg_ident(sql)


def norm_cell(v) -> str:
    return "" if v is None else str(v)


def fetch_rows(cur, sql: str, timeout_s: int):
    cur.execute(f"SET statement_timeout TO {timeout_s * 1000}")
    cur.execute(sql)
    return sorted(tuple(norm_cell(v) for v in row) for row in cur.fetchall())


def run_one(bb: dict, out: Path, idx: int, item: dict) -> dict:
    qid = item["question_id"]
    db_id = item["db_id"]
    rec = {"idx": idx, "qid": qid, "db_id": db_id, "status": "error", "init_sql": None,
           "pg_sql": None, "pred_rows": None, "gold_rows": None, "gold_status": None,
           "usage": None, "secs": 0}
    t0 = time.time()
    try:
        init_sql, usage = gen_sql(bb, item["question"], item.get("evidence", ""), db_id)
        rec["init_sql"] = init_sql
        rec["usage"] = usage
        rec["pg_sql"] = to_pg_ident(init_sql)

        conn = psycopg2.connect(**PG)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("LOAD 'vectors'")
        cur.execute('SET search_path TO "$user", public, vectors, "%s"' % db_id)

        # pred: safeql() 修正执行
        try:
            rec["pred_rows"] = fetch_rows(
                cur, f"SELECT * FROM safeql($${rec['pg_sql']}$$) AS t(result TEXT)", 240)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "safeql_fail: " + str(e).splitlines()[0][:150]
        finally:
            conn.rollback()

        # gold: 直接执行 (不经 safeql)
        try:
            rec["gold_rows"] = fetch_rows(cur, to_pg_gold(item["SQL"], qid), 60)
            rec["gold_status"] = "ok"
        except Exception as e:
            rec["gold_status"] = "gold_fail: " + str(e).splitlines()[0][:150]
        finally:
            conn.rollback()

        cur.close()
        conn.close()
    except Exception as e:
        if rec["status"] == "error":
            rec["status"] = "error: " + str(e).splitlines()[0][:150]
    rec["secs"] = round(time.time() - t0)
    with _lock:
        data = _load(out, {})
        data[str(qid)] = rec
        json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{idx:03d}][{bb['tag']}] q{qid} ({db_id}) {rec['secs']}s {rec['status'][:60]}", flush=True)
    return rec


def repair_gold(out: Path, dev_by_qid: dict) -> None:
    """只重算 gold (不动 pred): gold 转换器修复后补齐。"""
    data = _load(out, {})
    for qid_s, rec in data.items():
        if rec.get("gold_status") == "ok":
            continue
        item = dev_by_qid[int(qid_s)]
        conn = psycopg2.connect(**PG)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("LOAD 'vectors'")
        cur.execute('SET search_path TO "$user", public, vectors, "%s"' % item["db_id"])
        try:
            rec["gold_rows"] = fetch_rows(cur, to_pg_gold(item["SQL"], int(qid_s)), 60)
            rec["gold_status"] = "ok"
        except Exception as e:
            rec["gold_status"] = "gold_fail: " + str(e).splitlines()[0][:150]
        finally:
            conn.rollback()
            cur.close()
            conn.close()
        print(f"gold repair q{qid_s}: {rec['gold_status'][:80]}", flush=True)
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    tag = sys.argv[1]
    assert tag in BACKBONES, f"usage: python run_safeql_e1.py <{'|'.join(BACKBONES)}> [--gold-only]"
    bb = BACKBONES[tag]
    out = RES / f"safeql_bird300_{bb['tag']}.json"
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json", encoding="utf-8"))
    assert len(dev) == 300
    dev_by_qid = {d["question_id"]: d for d in dev}

    if "--gold-only" in sys.argv:
        repair_gold(out, dev_by_qid)
        return

    done = {int(k) for k, v in _load(out, {}).items() if v.get("status") == "ok"}
    todo = [(i, d) for i, d in enumerate(dev) if d["question_id"] not in done]
    print(f"[{tag}] 已完成 {len(done)}; 待跑 {len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, bb, out, i, d) for i, d in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"驱动异常: {e}", flush=True)

    data = _load(out, {})
    ok = sum(1 for v in data.values() if v.get("status") == "ok")
    gold_ok = sum(1 for v in data.values() if v.get("gold_status") == "ok")
    ptok = sum((v.get("usage") or {}).get("prompt_tokens") or 0 for v in data.values())
    ctok = sum((v.get("usage") or {}).get("completion_tokens") or 0 for v in data.values())
    print(f"累计 pred 成功: {ok}/300, gold PG 可执行: {gold_ok}/300, "
          f"prompt_tok={ptok:,} comp_tok={ctok:,}", flush=True)


if __name__ == "__main__":
    main()
