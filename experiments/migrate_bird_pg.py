"""BIRD SQLite -> PostgreSQL 17 迁移 (SafeQL 基线)。

- 每库一个 PG schema (跨库表名隔离), 驱动脚本连接后 SET search_path 到对应 schema
- 表名/列名全部双引号包裹 (BIRD 有 "FRPM Count (Ages 5-17)" 等特殊列名)
- 类型映射: SQLite TEXT/INTEGER/REAL/BLOB -> PG TEXT/BIGINT/DOUBLE PRECISION/TEXT
- 不迁移主键/外键/索引 (SafeQL 修正搜索只依赖表结构 + 数据)
- COPY FROM STDIN CSV 批量导入, 幂等 (DROP SCHEMA CASCADE)
"""
import csv
import io
import sqlite3
import sys
import time
from pathlib import Path

import psycopg2

BASE = Path(__file__).resolve().parents[1]   # repo root
BIRD_DIR = BASE / "baselines" / "DAIL-SQL" / "dataset" / "bird" / "database"
DBS = [
    "california_schools", "card_games", "codebase_community",
    "debit_card_specializing", "european_football_2", "financial",
    "formula_1", "student_club", "superhero", "thrombosis_prediction",
    "toxicology",
]
PG = dict(host="localhost", port=5432, user="postgres", password="safeql", dbname="postgres")

TYPE_MAP = {
    "TEXT": "TEXT", "INTEGER": "BIGINT", "INT": "BIGINT", "REAL": "DOUBLE PRECISION",
    "FLOAT": "DOUBLE PRECISION", "DOUBLE": "DOUBLE PRECISION", "NUMERIC": "DOUBLE PRECISION",
    "BLOB": "TEXT", "DATETIME": "TEXT", "DATE": "TEXT", "BOOLEAN": "TEXT", "": "TEXT",
}


def qi(name: str) -> str:
    """PostgreSQL 未加引号标识符折叠小写; 统一小写 + 引号, 避免 LLM 生成的大小写噪音触发无谓修正。"""
    return '"' + name.lower().replace('"', '""') + '"'


def map_type(decl: str) -> str:
    d = (decl or "").strip().upper()
    for k, v in TYPE_MAP.items():
        if d.startswith(k):
            return v
    return "TEXT"


def infer_type(con, table: str, col: str, decl: str) -> str:
    """SQLite 弱类型: 声明 INTEGER 的列可能存 '182.88' 等浮点/字符串, 采样检测升级类型。"""
    t = map_type(decl)
    if t != "BIGINT":
        return t
    try:
        rows = con.execute(f"SELECT {qi(col)} FROM {qi(table)} LIMIT 5000").fetchall()
        for (v,) in rows:
            if v is None or isinstance(v, int):
                continue
            if isinstance(v, float):
                return "DOUBLE PRECISION"
            s = str(v).strip()
            if s == "":
                continue
            try:
                int(s)
                continue
            except ValueError:
                pass
            try:
                float(s)
                return "DOUBLE PRECISION"
            except ValueError:
                return "TEXT"
    except Exception:
        pass
    return t


def migrate_db(conn, db_id: str) -> None:
    t0 = time.time()
    con = sqlite3.connect(str(BIRD_DIR / db_id / f"{db_id}.sqlite"))
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    cur = conn.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {qi(db_id)} CASCADE")
    cur.execute(f"CREATE SCHEMA {qi(db_id)}")

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    total_rows = 0
    for t in tables:
        cols = con.execute(f"PRAGMA table_info({qi(t)})").fetchall()
        col_defs = ", ".join(f"{qi(c[1])} {infer_type(con, t, c[1], c[2])}" for c in cols)
        cur.execute(f"CREATE TABLE {qi(db_id)}.{qi(t)} ({col_defs})")

        collist = ", ".join(qi(c[1]) for c in cols)
        buf = io.StringIO()
        writer = csv.writer(buf)
        n = 0
        for row in con.execute(f"SELECT {collist} FROM {qi(t)}"):
            writer.writerow(["" if v is None else v for v in row])
            n += 1
            total_rows += 1
            if n % 100000 == 0:
                buf.seek(0)
                cur.copy_expert(
                    f"COPY {qi(db_id)}.{qi(t)} ({collist}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)
                buf.seek(0)
                buf.truncate(0)
        if buf.tell() > 0:
            buf.seek(0)
            cur.copy_expert(
                f"COPY {qi(db_id)}.{qi(t)} ({collist}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)
        print(f"  [{db_id}] {t}: {n} rows")

    conn.commit()
    con.close()
    print(f"[{db_id}] done: {len(tables)} tables, {total_rows} rows, {time.time()-t0:.1f}s")


def main():
    only = sys.argv[1:] or DBS
    conn = psycopg2.connect(**PG)
    conn.autocommit = False
    for db_id in only:
        migrate_db(conn, db_id)
    conn.close()
    print("ALL DONE")


if __name__ == "__main__":
    main()
