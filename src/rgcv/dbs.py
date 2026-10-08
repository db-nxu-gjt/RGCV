"""统一数据库连接器:SQLite / DuckDB 目录抽取与值签名采样。

y.docx 定义 2(模式异构图)需要表/列/外键/值签名(采样高频值与基数、空值率
统计)作为节点;本模块提供方言无关的目录抽取接口。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# duckdb 延迟导入:仅 DuckDB 路径需要;SQLite-only 环境(CHESS env)无 duckdb
try:
    import duckdb as _duckdb
    _DB_ERRORS = (sqlite3.Error, _duckdb.Error)
except ImportError:
    _duckdb = None
    _DB_ERRORS = (sqlite3.Error,)


@dataclass
class ColumnMeta:
    table: str
    name: str
    dtype: str
    notnull: bool = False
    is_pk: bool = False
    comment: str = ""
    # 值签名(y.docx 4.1.1):采样高频值 + 基数 + 空值率
    n_distinct: Optional[int] = None
    null_rate: Optional[float] = None
    top_values: List[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class FKMeta:
    src: Tuple[str, str]  # (table, column)
    dst: Tuple[str, str]


class Catalog:
    """数据库模式目录(离线构建,每库一次)。"""

    def __init__(self, dialect: str, tables: Dict[str, List[ColumnMeta]],
                 fks: List[FKMeta], db_path: str):
        self.dialect = dialect
        self.tables = tables
        self.fks = fks
        self.db_path = db_path
        self.n_columns = sum(len(cols) for cols in tables.values())

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    def columns_of(self, table: str) -> List[ColumnMeta]:
        return self.tables.get(table, [])

    def column(self, table: str, col: str) -> Optional[ColumnMeta]:
        for c in self.tables.get(table, []):
            if c.name == col:
                return c
        return None

    def fk_map(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        m: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for fk in self.fks:
            m.setdefault(fk.src, []).append(fk.dst)
            # 无向化:join 路径可双向使用
            m.setdefault(fk.dst, []).append(fk.src)
        return m


def load_sqlite_catalog(db_path: str, sample_rows: int = 200,
                         sample_values: int = 8) -> Catalog:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    tcols: Dict[str, List[ColumnMeta]] = {}
    fks: List[FKMeta] = []
    for t in tables:
        cols = cur.execute(f'PRAGMA table_info("{t}")').fetchall()
        n_rows = cur.execute(
            f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        for (_cid, cname, ctype, notnull, _dflt, pk) in cols:
            sig = _signature_sqlite(con, t, cname, n_rows, sample_rows,
                                    sample_values)
            tcols.setdefault(t, []).append(
                ColumnMeta(t, cname, ctype or "TEXT", bool(notnull), bool(pk),
                           n_distinct=sig[0], null_rate=sig[1],
                           top_values=sig[2]))
        for fk in cur.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
            # (id, seq, table, from, to, on_update, on_delete, match)
            fks.append(FKMeta((t, fk[3]), (fk[2], fk[4] or "?")))
    con.close()
    return Catalog("sqlite", tcols, fks, db_path)


def _signature_sqlite(con, table: str, col: str, n_rows: int,
                      sample_rows: int, sample_values: int):
    """采样值签名:基数 / 空值率 / 高频值。"""
    try:
        row = con.execute(
            f'SELECT COUNT(DISTINCT "{col}"), '
            f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) '
            f'FROM (SELECT "{col}" FROM "{table}" LIMIT {sample_rows})'
        ).fetchone()
        n_sample, n_null = row
        top = [str(v) for (v,) in con.execute(
            f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL '
            f'LIMIT {sample_values}')]
        return int(n_sample), (n_null / sample_rows if sample_rows else 0.0), top
    except sqlite3.Error:
        return None, None, []


def load_duckdb_catalog(db_path: str, sample_rows: int = 200,
                        sample_values: int = 8) -> Catalog:
    assert _duckdb is not None, "duckdb not installed"
    con = _duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_name").fetchall()]
    tcols: Dict[str, List[ColumnMeta]] = {}
    fks: List[FKMeta] = []
    for t in tables:
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? ORDER BY "
            "ordinal_position", [t]).fetchall()
        for (cname, ctype) in cols:
            sig = _signature_duckdb(con, t, cname, sample_rows, sample_values)
            tcols.setdefault(t, []).append(
                ColumnMeta(t, cname, ctype, False, False,
                           n_distinct=sig[0], null_rate=sig[1],
                           top_values=sig[2]))
        try:
            fks_db = con.execute(
                "SELECT source_table, source_column, target_table, "
                "target_column FROM duckdb_constraints() "
                "WHERE constraint_type='FOREIGN KEY'").fetchall()
            for (st, sc, tt, tc) in fks_db:
                fks.append(FKMeta((st, sc), (tt, tc)))
        except duckdb.Error:
            pass
    con.close()
    return Catalog("duckdb", tcols, fks, db_path)


def _signature_duckdb(con, table, col, sample_rows, sample_values):
    try:
        row = con.execute(
            f'SELECT COUNT(DISTINCT "{col}"), '
            f'AVG(CASE WHEN "{col}" IS NULL THEN 1.0 ELSE 0.0 END) '
            f'FROM (SELECT "{col}" FROM "{table}" LIMIT {sample_rows})'
        ).fetchone()
        top = [str(v) for (v,) in con.execute(
                f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL '
                f'LIMIT {sample_values}').fetchall()]
        return int(row[0]), float(row[1] or 0.0), top
    except _DB_ERRORS:
        return None, None, []


class DBExecutor:
    """执行 SQL 并返回结构化反馈(错误消息 / 结果 / 执行计划)。"""

    def __init__(self, dialect: str, db_path: str, timeout_s: float = 15.0):
        self.dialect = dialect
        self.db_path = db_path
        self.timeout_s = timeout_s

    def _connect(self):
        if self.dialect == "sqlite":
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            con.execute("PRAGMA query_only=ON")
            # 查询级超时:progress_handler 在 VM 步进中检查墙钟,
            # 超时抛 OperationalError("interrupted") — 防笛卡尔积失控 SQL
            # 阻塞修复搜索(实测 q1481 100% CPU 空转 38min)
            if self.timeout_s and self.timeout_s > 0:
                import time as _time
                deadline = _time.monotonic() + self.timeout_s
                con.set_progress_handler(
                    lambda: 1 if _time.monotonic() > deadline else 0, 100_000)
            return con
        assert _duckdb is not None, "duckdb not installed"
        return _duckdb.connect(self.db_path, read_only=True)

    def execute(self, sql: str):
        """返回 (columns, rows, rowcount) 或抛出 DBError。"""
        con = self._connect()
        try:
            if self.dialect == "sqlite":
                con.create_function("_timeout", 0, lambda: int(
                    __import__("time").time() > 0))  # 占位,兼容旧签名
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(501)
            cols = [d[0] for d in cur.description] if cur.description else []
            n = len(rows)
            return cols, rows, n
        except _DB_ERRORS as e:
            raise DBError(str(e)) from e
        finally:
            con.close()

    def count(self, sql: str) -> int:
        cols, rows, _ = self.execute(sql)
        return int(rows[0][0]) if rows else 0

    def explain(self, sql: str) -> str:
        """S2 执行计划信号。"""
        con = self._connect()
        try:
            plan = con.execute(f"EXPLAIN {sql}").fetchall()
            if self.dialect == "sqlite":
                return "\n".join(str(r[-1]) for r in plan)
            return "\n".join(" ".join(str(x) for x in r) for r in plan)
        except _DB_ERRORS as e:
            return f"EXPLAIN_ERROR: {e}"
        finally:
            con.close()


class DBError(Exception):
    pass


def attach_bird_descriptions(catalog: Catalog, desc_dir: str) -> int:
    """BIRD 官方 database_description CSV → 列注释(文献[3]:注释是
    企业场景第一杠杆;A1 消融'去注释'的开关)。"""
    import csv
    import os
    n = 0
    if not os.path.isdir(desc_dir):
        return 0
    for fn in os.listdir(desc_dir):
        if not fn.lower().endswith(".csv"):
            continue
        table = os.path.splitext(fn)[0]
        try:
            with open(os.path.join(desc_dir, fn), encoding="utf-8-sig",
                      errors="ignore", newline="") as f:
                for row in csv.DictReader(f):
                    col = (row.get("original_column_name") or "").strip()
                    desc = (row.get("column_description") or "").strip()
                    if not col or not desc:
                        continue
                    c = catalog.column(table, col)
                    if c is not None and not c.comment:
                        c.comment = desc
                        n += 1
        except (OSError, csv.Error):
            continue
    return n
