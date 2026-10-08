"""E-B(方案二):19 条 PG17 不可译 gold 的等价改写 + 双跑一致性校验。

背景:SafeQL 复现在 PG17 侧评分,19 条 gold(SQLite 方言)经 to_pg_gold 转换后
仍不可执行(原口径 gold_fail 计 0,L275 的 19/300)。本脚本对这 19 条做
人工辅助的 SQLite->PG17 等价改写,并校验:
  1) sqlglot(pg 方言)解析通过;
  2) 双跑一致:改写 SQL 在 PG17 的结果集 ≡ 原始 gold 在 SQLite 的结果集
     (多重集 + 数值 1e-6 相对容差;TEXT 亲和性列用 COLLATE "C" 复刻 SQLite
     BINARY 字节序);
  3) 通过后写 results/safeql_gold19_fixed.json(qid -> {pg_sql, gold_rows}),
     gold_rows 为 PG 侧 fetch_rows 同格式(排序归一化字符串),供对称计分
     eval_safeql_sym.py 直接替换。

语义复刻要点(与 SQLite 列亲和性一一对应):
  - schools.soc/doc、Laboratory.dna 为 SQLite TEXT 亲和性:soc='69'、
    doc='54'、dna COLLATE "C" < '8'(文本比较,非数值比较);
  - frpm."academic year" BETWEEN '2014' AND '2015':TEXT 亲和性下与
    整数字面量的比较被亲和为文本比较(BIRD gold 实际恒真);
  - district.a4 TEXT 亲和性:ORDER BY a4 COLLATE "C" DESC;
  - julianday/datetime('localtime') 以 epoch/EXTRACT 精确复刻(julianday
    午夜= X.5 的偏移已计入);年差窗口受运行日期影响,与 BIRD 官方语义一致。

用法:python translate_gold_19.py [--write]
  默认只校验;--write 在全部通过后写 safeql_gold19_fixed.json。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import psycopg2
import sqlglot

BASE = Path(__file__).resolve().parents[1]              # src/rgcv_repro/
DAIL = BASE / "baselines" / "DAIL-SQL"
BIRD_DIR = DAIL / "dataset" / "bird" / "database"
RES = BASE / "results"
PG = dict(host="localhost", port=5432, user="postgres", password="safeql",
          dbname="postgres")

# q19 清单(gold_fail 的 19 条,按 qid 升序)
Q19 = [57, 70, 72, 87, 115, 138, 388, 693, 880, 960, 1028, 1118, 1121,
       1170, 1234, 1277, 1367, 1404, 1501]

# 人工辅助等价改写(标识符一律小写;PG17 方言)
GOLD_19_PG = {
    # LIMIT 332,1 -> LIMIT 1 OFFSET 332(AvgScrWrite 为 INTEGER,数值排序;
    # SQLite DESC 将 NULL 排最后,PG 需 NULLS LAST);并列分按 SQLite 观测
    # 排序复刻(tie 组内 cds 升序)
    57: """SELECT t2.phone, t2.ext
FROM satscores AS t1 INNER JOIN schools AS t2 ON t1.cds = t2.cdscode
ORDER BY t1.avgscrwrite DESC NULLS LAST, t1.cds ASC
LIMIT 1 OFFSET 332""",
    # soc TEXT 亲和性:SQLite `SOC = 69` 亲和为文本比较 '69'
    70: """SELECT COUNT(school) FROM schools
WHERE (statustype = 'Closed' OR statustype = 'Active')
  AND soc = '69' AND county = 'Alpine'""",
    # "Academic Year" TEXT 亲和性:与 2014/2015 整数比较亲和为文本比较
    # (BIRD gold 语义:'2014-2015' 与两边界字符串比较均真)
    72: """SELECT t1."enrollment (ages 5-17)"
FROM frpm AS t1 INNER JOIN schools AS t2 ON t1.cdscode = t2.cdscode
WHERE t2.edopscode = 'SSS' AND t2.city = 'Fremont'
  AND t1."academic year" COLLATE "C" BETWEEN '2014' AND '2015'""",
    # doc/soc TEXT 亲和性 -> '54'/'62';strftime('%Y', OpenDate) -> EXTRACT
    87: """SELECT t2.admemail1, t2.admemail2
FROM frpm AS t1 INNER JOIN schools AS t2 ON t1.cdscode = t2.cdscode
WHERE t2.county = 'San Bernardino' AND t2.city = 'San Bernardino'
  AND t2.doc = '54'
  AND (EXTRACT(YEAR FROM (t2.opendate)::timestamp))::text BETWEEN '2009' AND '2010'
  AND t2.soc = '62'""",
    # SQLite SUM(boolean) -> SUM(CASE);CAST AS REAL -> double precision
    # (float4 会丢精度导致与 SQLite 结果串不等);a4 TEXT 亲和性排序
    115: """SELECT CAST(SUM(CASE WHEN t1.gender = 'M' THEN 1 ELSE 0 END)
             AS double precision) * 100 / COUNT(t1.client_id)
FROM client AS t1 INNER JOIN district AS t2 ON t1.district_id = t2.district_id
WHERE t2.a3 = 'south Bohemia'
GROUP BY t2.a4
ORDER BY t2.a4 COLLATE "C" DESC
LIMIT 1""",
    # LIMIT 1,1 -> LIMIT 1 OFFSET 1(a15 INTEGER/bigint,数值排序;
    # SQLite DESC 将 NULL 排最后,PG 需显式 NULLS LAST)
    138: """SELECT COUNT(t1.client_id)
FROM client AS t1 INNER JOIN district AS t2 ON t1.district_id = t2.district_id
WHERE t1.gender = 'M'
  AND t2.a15 = (SELECT t3.a15 FROM district AS t3
                ORDER BY t3.a15 DESC NULLS LAST LIMIT 1 OFFSET 1)""",
    # 标量子查询多行:SQLite 取扫描首行(cards.id 为 rowid 别名 -> 最小 id);
    # ORDER BY id LIMIT 1 确定性复刻
    388: """SELECT id, language FROM set_translations
WHERE id = (SELECT id FROM cards WHERE convertedmanacost = 5
            ORDER BY id LIMIT 1)
  AND setcode = '10E'""",
    # 聚合无 GROUP BY 时 SQLite 允许 ORDER BY 裸列(单行,LIMIT 1 无效) ->
    # PG 严格模式下去掉 ORDER BY,语义不变
    693: """SELECT COUNT(t2.id)
FROM users AS t1
INNER JOIN posts AS t2 ON t1.id = t2.owneruserid
INNER JOIN comments AS t3 ON t3.postid = t2.id""",
    # IIF -> CASE;fastestlapspeed TEXT -> double precision
    880: """SELECT (SUM(CASE WHEN t2.raceid = 853 THEN t2.fastestlapspeed::double precision ELSE 0 END)
       - SUM(CASE WHEN t2.raceid = 854 THEN t2.fastestlapspeed::double precision ELSE 0 END)) * 100
     / SUM(CASE WHEN t2.raceid = 853 THEN t2.fastestlapspeed::double precision ELSE 0 END)
FROM drivers AS t1 INNER JOIN results AS t2 ON t2.driverid = t1.driverid
WHERE t1.forename = 'Paul' AND t1.surname = 'di Resta'""",
    # AVG(text) -> AVG(::double precision);NULL 跳过语义一致
    960: """SELECT AVG(t1.fastestlapspeed::double precision)
FROM results AS t1 INNER JOIN races AS t2 ON t1.raceid = t2.raceid
WHERE t2.year = 2009 AND t2.name = 'Spanish Grand Prix'""",
    # 裸列与 GROUP BY 不一致:组键 away_team_api_id 函数决定 team_long_name,
    # MIN() 与 SQLite 任意行取值等价;COUNT(*) 并列时按 SQLite 参考输出
    # (Celtic, id 9925)复刻为 id DESC
    1028: """SELECT MIN(teaminfo.team_long_name)
FROM league AS leaguedata
INNER JOIN match AS matchdata ON leaguedata.id = matchdata.league_id
INNER JOIN team AS teaminfo ON matchdata.away_team_api_id = teaminfo.team_api_id
WHERE leaguedata.name = 'Scotland Premier League'
  AND matchdata.season = '2009/2010'
  AND matchdata.away_team_goal - matchdata.home_team_goal > 0
GROUP BY matchdata.away_team_api_id
ORDER BY COUNT(*) DESC, matchdata.away_team_api_id DESC
LIMIT 1""",
    # JULIANDAY('now')-JULIANDAY(birthday):julianday(now) 的 X.5 偏移与
    # julianday(午夜生日的 X.5)相减抵消,julianday 差 ≡ epoch/86400,
    # 直接按天数/365 与 SQLite 同口径;'now' 为 UTC
    1118: """SELECT player_name FROM player
WHERE EXTRACT(EPOCH FROM ((now() AT TIME ZONE 'UTC') - (birthday)::timestamp))
      / 86400.0 / 365.0 >= 35""",
    # datetime(CURRENT_TIMESTAMP,'localtime') - datetime(birthday):
    # 文本相减被数值截断为年差;'localtime' 按 SQLite 参考运行宿主时区
    # (Asia/Shanghai)复刻
    1121: """SELECT SUM(t2.home_team_goal)
FROM player AS t1 INNER JOIN match AS t2 ON t1.player_api_id = t2.away_player_1
WHERE EXTRACT(YEAR FROM now() AT TIME ZONE 'Asia/Shanghai')
    - EXTRACT(YEAR FROM (t1.birthday)::timestamp) < 31""",
    # STRFTIME('%Y',x) 相减被数值截断为年差(数值比较,非 to_pg_gold 的文本)
    1170: """SELECT COUNT(DISTINCT t1.id)
FROM patient AS t1 INNER JOIN examination AS t2 ON t1.id = t2.id
WHERE t1.admission = '+'
  AND EXTRACT(YEAR FROM (t2."examination date")::timestamp)
    - EXTRACT(YEAR FROM (t1."first date")::timestamp) >= 1""",
    # 裸列 birthday 由组键 t1.id(Patient 主键)函数决定 -> 并入 GROUP BY
    1234: """SELECT DISTINCT t1.id, t1.sex, t1.birthday
FROM patient AS t1 INNER JOIN laboratory AS t2 ON t1.id = t2.id
WHERE t2.wbc <= 3.5 OR t2.wbc >= 9.0
GROUP BY t1.sex, t1.id, t1.birthday
ORDER BY t1.birthday ASC""",
    # dna 声明为 TEXT(亲和性):SQLite `DNA < 8` 为文本比较 -> COLLATE "C" < '8'
    1277: """SELECT COUNT(DISTINCT t1.id)
FROM patient AS t1 INNER JOIN laboratory AS t2 ON t1.id = t2.id
WHERE t2.dna COLLATE "C" < '8' AND t1.description IS NULL""",
    # 裸列 college 由组键 major_id(major 主键)函数决定 -> 并入 GROUP BY
    1367: """SELECT t2.college
FROM member AS t1 INNER JOIN major AS t2 ON t1.link_to_major = t2.major_id
GROUP BY t2.major_id, t2.college
ORDER BY COUNT(t2.college) DESC
LIMIT 1""",
    # 聚合无 GROUP BY:单事件 type 唯一,GROUP BY t1.type 与 SQLite 单行等价
    1404: """SELECT t1.type, SUM(t3.cost)
FROM event AS t1
INNER JOIN budget AS t2 ON t1.event_id = t2.link_to_event
INNER JOIN expense AS t3 ON t2.budget_id = t3.link_to_budget
WHERE t1.event_name = 'October Meeting'
GROUP BY t1.type""",
    # 原 gold 无语法问题(原失败为 pred 失败后事务污染的连锁报错),直接重跑
    1501: """SELECT DISTINCT t2.country
FROM transactions_1k AS t1
INNER JOIN gasstations AS t2 ON t1.gasstationid = t2.gasstationid
INNER JOIN yearmonth AS t3 ON t1.customerid = t3.customerid
WHERE t3.date = '201306'""",
}


def norm_cell(v) -> str:
    """PG 侧存储口径(与 run_safeql_e1.fetch_rows 一致)。"""
    return "" if v is None else str(v)


def cell_key(v):
    """双跑比较口径:数值化(None/数值/数值串 -> float),否则字符串。"""
    if v is None:
        return ("s", "")
    if isinstance(v, (int, float)):
        return ("n", float(v))
    s = str(v).strip()
    try:
        return ("n", float(s))
    except ValueError:
        return ("s", s)


def rows_close(a, b, rtol=1e-6) -> bool:
    ka, kb = cell_key(a), cell_key(b)
    if ka[0] != kb[0]:
        return False
    if ka[0] == "n":
        return abs(ka[1] - kb[1]) <= rtol * max(1.0, abs(ka[1]))
    return ka[1] == kb[1]


def rows_multiset_eq(rows_a, rows_b, rtol=1e-6) -> bool:
    """多重集等价(容差):排序后双指针。"""
    if len(rows_a) != len(rows_b):
        return False
    sa = sorted(tuple(cell_key(c) for c in r) for r in rows_a)
    sb = sorted(tuple(cell_key(c) for c in r) for r in rows_b)
    i = j = 0
    while i < len(sa) and j < len(sb):
        ra, rb = sa[i], sb[j]
        if len(ra) == len(rb) and all(rows_close(x, y, rtol) for x, y in zip(ra, rb)):
            i += 1
            j += 1
        else:                       # 容差下排序次序可能局部错位 -> 线性探测
            if ra <= rb:
                i += 1
            else:
                j += 1
    return i == len(sa) and j == len(sb)


def pg_exec(cur, db_id: str, sql: str, timeout_s: int = 120):
    cur.execute(f"SET statement_timeout TO {timeout_s * 1000}")
    cur.execute(sql)
    return [tuple(norm_cell(v) for v in row) for row in cur.fetchall()]


def sqlite_exec(db_id: str, sql: str):
    con = sqlite3.connect(str(BIRD_DIR / db_id / f"{db_id}.sqlite"))
    rows = con.execute(sql).fetchall()
    con.close()
    return rows


def main():
    do_write = "--write" in sys.argv
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                         encoding="utf-8"))
    dev_by_qid = {d["question_id"]: d for d in dev}

    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()

    fixed, n_pass, n_fail = {}, 0, 0
    print(f"{'qid':>5} {'db':<26} {'nrows':>6} {'parse':>6} {'dualrun':>8}")
    for qid in Q19:
        item = dev_by_qid[qid]
        db_id = item["db_id"]
        pg_sql = GOLD_19_PG[qid]
        parse_ok = dual_ok = False
        try:
            sqlglot.parse_one(pg_sql, read="postgres")
            parse_ok = True
        except Exception as e:
            print(f"  sqlglot 解析失败 q{qid}: {str(e)[:120]}")
        try:
            cur.execute('SET search_path TO "$user", public, vectors, "%s"' % db_id)
            gold_rows = pg_exec(cur, db_id, pg_sql)
            rows_sqlite = sqlite_exec(db_id, item["SQL"])
            dual_ok = rows_multiset_eq(rows_sqlite, gold_rows)
            if not dual_ok:
                print(f"  [q{qid}] 双跑不一致: sqlite={len(rows_sqlite)}行 "
                      f"pg={len(gold_rows)}行")
                print(f"    sqlite 前3: {rows_sqlite[:3]}")
                print(f"    pg     前3: {gold_rows[:3]}")
        except Exception as e:
            print(f"  [q{qid}] 执行失败: {str(e)[:150]}")
        status = "PASS" if (parse_ok and dual_ok) else "FAIL"
        if status == "PASS":
            n_pass += 1
            fixed[str(qid)] = {"db_id": db_id, "pg_sql": pg_sql,
                               "gold_rows": [list(r) for r in gold_rows]}
        else:
            n_fail += 1
        print(f"{qid:>5} {db_id:<26} {len(gold_rows) if parse_ok and dual_ok else '-':>6} "
              f"{str(parse_ok):>6} {status:>8}")

    cur.close()
    conn.close()
    print(f"\n校验通过 {n_pass}/19,失败 {n_fail}")
    if n_fail == 0 and do_write:
        out = RES / "safeql_gold19_fixed.json"
        out.write_text(json.dumps(fixed, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"已写 {out}")
    elif do_write and n_fail:
        print("存在失败项,未写文件。")


if __name__ == "__main__":
    main()
