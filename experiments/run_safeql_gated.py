"""E-B 第二部分:SafeQL+门控消融(执行成功即跳过修正搜索)。

动机:RGCV 门控 = "初始 SQL 执行成功则不启动修复搜索"。SafeQL 复现
(safeql()) 默认对每条 SQL 都做搜索式修正。给 SafeQL 加同一门控开关,
可把 RGCV 相对 SafeQL 的净增量分解为 [门控选择策略] 与 [S2/S3/S4 信号]
两部分 —— 回应 R1-M1。

协议(配对消融,零额外生成成本):
  复用 run_safeql_e1.py 已生成的 init_sql(同题同候选),仅改变执行侧策略:
    gate=direct_exec   init_sql 在 PG17 直接执行成功 -> 直接采用,不调 safeql()
    gate=safeql_refine 直接执行失败 -> 走 safeql() 修正执行(与原口径一致)
  gold:281 条沿用原 PG gold;19 条用 translate_gold_19.py 的等价改写
  (双跑校验通过),使 300 条 gold 全部可执行(对称计分)。

用法:python run_safeql_gated.py <v4pro|kimi>
输出:results/safeql_bird300_<tag>_gated.json(同 run_safeql_e1 记录结构
     + gate 字段;pred_rows 口径与 base 一致)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg2

BASE = Path(__file__).resolve().parents[1]              # src/rgcv_repro/
RES = BASE / "results"
PG = dict(host="localhost", port=5432, user="postgres", password="safeql",
          dbname="postgres")
MAX_WORKERS = 3
_lock = threading.Lock()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_safeql_e1 import fetch_rows, to_pg_ident  # noqa: E402


def load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def run_one(tag: str, out: Path, qid: str, rec: dict, gold19: dict) -> dict:
    db_id = rec["db_id"]
    init_sql = rec.get("init_sql") or ""
    new = dict(rec)
    new["gate"] = "safeql_refine"
    t0 = time.time()
    conn = psycopg2.connect(**PG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("LOAD 'vectors'")
    cur.execute('SET search_path TO "$user", public, vectors, "%s"' % db_id)

    # gold:19 条用等价改写,其余沿用原记录
    if qid in gold19:
        new["gold_rows"] = [tuple(r) for r in gold19[qid]["gold_rows"]]
        new["gold_status"] = "ok"
    # else: 沿用 rec 的 gold_rows / gold_status

    # 门控:直接执行 -> 成功即用;失败才 safeql() 修正
    try:
        new["pred_rows"] = fetch_rows(cur, to_pg_ident(init_sql), 240)
        new["status"] = "ok"
        new["gate"] = "direct_exec"
    except Exception:
        conn.rollback()
        try:
            new["pred_rows"] = fetch_rows(
                cur, f"SELECT * FROM safeql($${to_pg_ident(init_sql)}$$) "
                     f"AS t(result TEXT)", 240)
            new["status"] = "ok"
            new["gate"] = "safeql_refine"
        except Exception as e:
            new["status"] = "safeql_fail: " + str(e).splitlines()[0][:150]
            new["pred_rows"] = None
        finally:
            conn.rollback()
    cur.close()
    conn.close()
    new["secs"] = round(time.time() - t0)
    with _lock:
        data = load(out, {})
        data[qid] = new
        json.dump(data, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"[{tag}] q{qid} ({db_id}) {new['secs']}s gate={new['gate']} "
          f"{new['status'][:60]}", flush=True)
    return new


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    assert tag in ("v4pro", "kimi"), "usage: python run_safeql_gated.py <v4pro|kimi>"
    src = RES / f"safeql_bird300_{tag}.json"
    out = RES / f"safeql_bird300_{tag}_gated.json"
    base = load(src, None)
    assert base, f"missing {src}"
    gold19 = load(RES / "safeql_gold19_fixed.json", {})
    done = {q for q, r in load(out, {}).items()
            if r.get("gate") in ("direct_exec", "safeql_refine")}
    todo = [(q, r) for q, r in base.items() if q not in done]
    print(f"[{tag}] 已完成 {len(done)}; 待跑 {len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_one, tag, out, q, r, gold19) for q, r in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"驱动异常 q: {e}", flush=True)

    data = load(out, {})
    n_direct = sum(1 for r in data.values() if r.get("gate") == "direct_exec")
    ok = sum(1 for r in data.values() if r.get("status") == "ok")
    gold_ok = sum(1 for r in data.values() if r.get("gold_status") == "ok")
    print(f"[{tag}] gated 完成: direct_exec={n_direct}, "
          f"safeql_refine={ok - n_direct}, pred_ok={ok}/300, "
          f"gold_ok={gold_ok}/300", flush=True)


if __name__ == "__main__":
    main()
