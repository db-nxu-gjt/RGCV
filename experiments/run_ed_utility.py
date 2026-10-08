"""E-D (T1-2a):告警子集受控再生成 — 下游效用与风险。

在 eA replay 的 V1+V2 告警题(ds 36 = 22 silent + 14 correct;kimi 43 =
28 + 15)上,每题 1 次 alarm-conditioned rewrite(输入 = 问题 + 全 schema
+ 冻结 SQL + 告警原因,T=0.6,seed=20000*qid),执行后测:
  silent 告警题 → rescued(EX 变 True,净收益);
  correct 告警题 → broken(EX 变 False 或执行失败,净损失);
  net ΔEX;V1+V2 重查告警是否消除。

输出 results/eD_alarm_regen.json;
中间产物 results/eD_alarm_regen_<slug>/q<qid>.json(断点续跑)。

运行:python run_ed_utility.py --engine deepseek-v4-pro [--smoke] [--retry-failed]
      python run_ed_utility.py --engine kimi-k2.6
      python run_ed_utility.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from common import RESULTS, bird_catalog, bird_executor  # noqa: E402

import common as _common  # noqa: E402

REPO = BASE.parents[1]
if not _common.BIRD_DB.exists():
    for _cand in (REPO / "data" / "bird" / "dev_20240627" / "dev_databases",
                  REPO / "baselines" / "DAIL-SQL" / "dataset" / "bird" / "dev"
                  / "dev_databases"):
        if _cand.exists():
            _common.BIRD_DB = _cand
            _common.BIRD_DIR = _cand.parent
            break
    assert _common.BIRD_DB.exists(), "BIRD dev_databases not found"

from exp_tier2_real import boot_ci  # noqa: E402
from rgcv.dbs import DBError  # noqa: E402
from rgcv.evalx import EXEvaluator  # noqa: E402
from rgcv.llm import LLMClient, extract_sqls  # noqa: E402
from rgcv.schema_graph import build_graph, full_schema_result  # noqa: E402
from rgcv.verifier import Verifier  # noqa: E402

EA_FILE = RESULTS / "eA_tier2_real_verification.json"
ENGINES = {
    "deepseek-v4-pro": ("deepseek-v4-pro",
                        "rgcv_bird300_deepseek-v4-pro_gated_fullschema"),
    "kimi-k2.6": ("kimi-k26", "rgcv_bird300_kimi-k26_gated_fullschema"),
}
TEMPERATURE = 0.6          # kimi 平台上限 0.6;两骨干统一
SEED_BASE = 20_000
MAX_TOKENS = 1024
WORKERS = 4

SYS_REPAIR = (
    "You are a SQL repair assistant for SQLite. A verification layer flagged "
    "the candidate SQL below. Given the database schema, the question, the "
    "candidate SQL and the alarm reasons, rewrite the SQL so that it "
    "preserves the question's intent while addressing the alarms. Only fix "
    "what the alarms point to; keep the rest of the query unchanged. Reply "
    "with the corrected SQL only, no explanations, no markdown fences.")

_schema_cache: dict = {}
_cache_lock = threading.Lock()


def schema_prompt(db_id: str) -> str:
    with _cache_lock:
        if db_id not in _schema_cache:
            graph = build_graph(bird_catalog(db_id), use_semantic=False)
            _schema_cache[db_id] = full_schema_result(graph, "").prompt
        return _schema_cache[db_id]


def load_worklist(tag: str) -> list:
    ea = json.loads(EA_FILE.read_text(encoding="utf-8"))
    tier2_dir = RESULTS / tag
    items = []
    for t in sorted(ea["traces"], key=lambda x: int(x["question_id"])):
        if t.get("tag") != tag or not t.get("V1+V2", {}).get("flagged"):
            continue
        if t.get("kind") not in ("silent", "correct"):
            continue
        qid = int(t["question_id"])
        rec = json.loads((tier2_dir / f"q_{qid}.json").read_text(
            encoding="utf-8"))
        if rec.get("error") or not rec.get("pred_sql"):
            continue
        items.append({
            "question_id": qid, "db_id": rec["db_id"], "kind": t["kind"],
            "question": rec["question"], "pred_sql": rec["pred_sql"],
            "gold_sql": rec["gold_sql"], "alarms": t.get("alarms", []),
        })
    return items


def run_one(llm_engine: str, item: dict) -> dict:
    qid = item["question_id"]
    cli = LLMClient(llm_engine)
    user = (f"{schema_prompt(item['db_id'])}\n\n"
            f"Question: {item['question']}\n\n"
            f"Candidate SQL:\n{item['pred_sql']}\n\n"
            "Alarm reasons:\n" +
            "\n".join(f"- {a}" for a in item["alarms"]))
    t0 = time.time()
    try:
        text, uin, uout = cli.chat("repair", SYS_REPAIR, user,
                                   max_tokens=MAX_TOKENS,
                                   temperature=TEMPERATURE,
                                   seed=SEED_BASE * qid)
    except Exception as e:
        return {**item, "status": "llm_error", "error": str(e)[:200]}
    sqls = extract_sqls(text, 1)
    new_sql = sqls[0] if sqls else ""
    if not new_sql:
        return {**item, "status": "llm_error", "error": "empty_sql"}
    rec = {**item, "status": "done", "rewrite_sql": new_sql,
           "secs": round(time.time() - t0, 1),
           "usage": {"calls": 1, "tokens_in": uin, "tokens_out": uout}}
    # 执行 + EX + V1+V2 重查
    ex = EXEvaluator(bird_executor(item["db_id"]))
    gold = ex.gold_result(item["gold_sql"])
    exec_ok, ex_ok = False, False
    if new_sql:
        try:
            bird_executor(item["db_id"]).execute(new_sql)
            exec_ok = True
            v = ex.ex(new_sql, item["gold_sql"])
            ex_ok = v is True
        except DBError:
            exec_ok = False
        except Exception:
            exec_ok = False
    rec["exec_ok"] = exec_ok
    rec["ex_correct"] = ex_ok
    rec["same_as_original"] = " ".join(new_sql.split()).lower() == \
        " ".join(item["pred_sql"].split()).lower()
    # V1+V2 重查(规则式,无 V3)
    still = None
    if exec_ok:
        cat = bird_catalog(item["db_id"])
        v = Verifier(bird_executor(item["db_id"]), cat,
                     layers=("v1", "v2"))
        rep = v.verify(item["question"], new_sql)
        still = bool(rep.alarmed_clauses)
        rec["alarms_after"] = rep.alarmed_clauses
    rec["v12_flagged_after"] = still
    return rec


def aggregate(engine: str, slug: str) -> dict:
    rdir = RESULTS / f"eD_alarm_regen_{slug}"
    recs = []
    for p in sorted(rdir.glob("q*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "done":
            recs.append(r)
    silent = [r for r in recs if r["kind"] == "silent"]
    correct = [r for r in recs if r["kind"] == "correct"]
    rescued = sum(1 for r in silent if r["ex_correct"])
    broken = sum(1 for r in correct if not r["exec_ok"] or not r["ex_correct"])
    still_s = sum(1 for r in silent if r.get("v12_flagged_after"))
    still_c = sum(1 for r in correct if r.get("v12_flagged_after"))
    noop = sum(1 for r in recs if r.get("same_as_original"))
    tok = sum(r["usage"]["tokens_in"] + r["usage"]["tokens_out"] for r in recs)
    return {
        "engine": engine,
        "n_done": len(recs),
        "silent_alarms": {"n": len(silent), "rescued": rescued,
                          "rescue_rate": round(rescued / len(silent), 4)
                          if silent else None,
                          "rescue_rate_ci95": boot_ci(
                              [r["ex_correct"] for r in silent])
                          if silent else [0, 0]},
        "correct_alarms": {"n": len(correct), "broken": broken,
                           "break_rate": round(broken / len(correct), 4)
                           if correct else None},
        "net_delta_ex": rescued - broken,
        "alarm_cleared_after": {
            "silent": still_s, "correct": still_c,
            "still_flagged_share": round((still_s + still_c) / len(recs), 3)
            if recs else None},
        "noop_rewrites": noop,
        "total_tokens": tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ids", nargs="*", type=int)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    engine, tag = ENGINES[a.engine]
    slug = "deepseek-v4-pro" if engine == "deepseek-v4-pro" else "kimi-k26"
    rdir = RESULTS / f"eD_alarm_regen_{slug}"
    (rdir / "responses").mkdir(parents=True, exist_ok=True)

    work = load_worklist(tag)
    if a.ids:
        work = [w for w in work if w["question_id"] in set(a.ids)]
    if a.smoke:
        work = [w for w in work if w["kind"] == "silent"][:2] + \
               [w for w in work if w["kind"] == "correct"][:1]

    if a.aggregate_only:
        print(json.dumps(aggregate(engine, slug), indent=1))
        return

    todo = []
    for w in work:
        p = rdir / f"q{w['question_id']}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") == "done" and not a.retry_failed:
                continue
        todo.append(w)
    print(f"[ed] worklist={len(work)} todo={len(todo)}", flush=True)

    lock = threading.Lock()
    done = [0]

    def job(w):
        rec = run_one(a.engine, w)
        p = rdir / f"q{w['question_id']}.json"
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        with lock:
            done[0] += 1
            print(f"[ed {engine}] {done[0]}/{len(todo)} "
                  f"q{w['question_id']} ({w['kind']}) "
                  f"exec_ok={rec.get('exec_ok')} ex={rec.get('ex_correct')}",
                  flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(job, w) for w in todo]
        for f in as_completed(futs):
            f.result()

    print(json.dumps(aggregate(engine, slug), indent=1))
    # 合并双骨干 aggregate
    out = RESULTS / "eD_alarm_regen.json"
    merged = {}
    for eng_key, (eng, _tag) in ENGINES.items():
        s = "deepseek-v4-pro" if eng == "deepseek-v4-pro" else "kimi-k26"
        rd = RESULTS / f"eD_alarm_regen_{s}"
        if rd.exists() and any(rd.glob("q*.json")):
            merged[eng] = aggregate(eng, s)
    out.write_text(json.dumps({
        "config": {"protocol": "alarm-conditioned rewrite, T=0.6, "
                               "seed=20000*qid, 1 call/alarm question",
                   "worklist": "eA replay V1+V2-flagged questions"},
        "summary": merged}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
