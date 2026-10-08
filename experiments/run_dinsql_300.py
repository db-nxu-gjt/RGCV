"""T1-6a: DIN-SQL BIRD-300 全量跑批 (统一骨干 deepseek-v4-pro) + EX 评分。

用法 (在 paper/github/experiments 下):
  python run_dinsql_300.py --smoke            # 3 题冒烟 (qid 36,37,10)
  python run_dinsql_300.py --ids 36,37        # 指定 question_id
  python run_dinsql_300.py                    # 全量 300 (断点续跑, workers=4)
  python run_dinsql_300.py --eval-only        # 仅对已有 merged.json 评分出报告

- 题集: baselines/DAIL-SQL/dataset/bird/dev/dev_300.json (行序 = 评估 idx)
- 管线: baselines/DIN-SQL/DIN-SQL_BIRD.py (T1-6a 改造: langchain 0.x → OpenAI
  1.x 薄适配; prompt 模板与四阶段管线不变; 骨干 deepseek-v4-pro, t=0,
  max_tokens=8192, thinking disabled —— 与 src/rgcv/llm.py 同骨干同解码)
- 中间产物: results/dinsql_bird300_v4pro/
    merged.json           qid -> "sql\t----- bird -----\tdb_id" (同 CHESS 格式)
    state.json            qid -> {status, secs, calls, tokens_in, tokens_out}
    trace/api_trace.jsonl 行级 usage (含失败重试行)
- 最终结果: results/dinsql_bird300_v4pro.json
    EX 口径与 analyze_t15_significance.run_sql 完全一致 (SQLite, sorted
    result-set 相等, 60s 软超时); 含逐题 0/1 向量、成本、模型与解码参数、
    旧 61 题参考预测交叉一致性、与 DAIL-SQL 44.7 / RGCV 54.7 的对比。
"""
import argparse
import json
import threading
import time
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]            # paper/github
REPO = BASE.parents[1]                                # RGCV root
DAIL_BIRD = REPO / "baselines" / "DAIL-SQL" / "dataset" / "bird"
RES = BASE / "results"
RUN_DIR = RES / "dinsql_bird300_v4pro"
OLD61 = REPO / "src" / "rgcv_repro" / "results" / "chess_bird61.json"
OLD61_PRED = REPO / "baselines" / "DIN-SQL" / "predict_dev.json"

from analyze_t15_significance import run_sql, GoldCache   # noqa: E402  同口径判定

_spec = importlib.util.spec_from_file_location(
    "dinsql_bird", REPO / "baselines" / "DIN-SQL" / "DIN-SQL_BIRD.py")
din = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(din)

_lock = threading.Lock()


def _load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


# ------------------------------------------------------------------ 跑批
def run_one(qid: int, item: dict, trace_path: Path) -> dict:
    t0 = time.time()

    def on_usage(phase, uin, uout, lat, ok, err):
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "qid": qid,
               "db_id": item["db_id"], "phase": phase, "tokens_in": uin,
               "tokens_out": uout, "latency_s": round(lat, 3), "ok": ok,
               "err": err}
        try:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    sql, qu, stages, status = "", {"calls": 0, "tokens_in": 0,
                                   "tokens_out": 0}, {}, "ok"
    try:
        sql, qu, stages = din.process_question(item, on_usage=on_usage)
    except Exception as e:
        status = f"error:{type(e).__name__}"[:60]
        print(f"q{qid} pipeline error: {e}", flush=True)
    # 各阶段原始响应落盘: 解析层离线重放用, 避免任何后续调整重烧 API
    if stages:
        rdir = RUN_DIR / "responses"
        rdir.mkdir(exist_ok=True)
        json.dump(stages, open(rdir / f"q{qid}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)

    pred = ""
    if isinstance(sql, str) and sql.strip() and \
            not sql.startswith("SELECT * FROM table"):
        pred = f"{sql}\t----- bird -----\t{item['db_id']}"
    rec = {"qid": qid, "db_id": item["db_id"], "status": status,
           "secs": round(time.time() - t0, 1), **qu}
    with _lock:
        if pred:
            merged = _load(RUN_DIR / "merged.json", {})
            merged[str(qid)] = pred
            json.dump(merged, open(RUN_DIR / "merged.json", "w",
                                   encoding="utf-8"), ensure_ascii=False, indent=1)
        state = _load(RUN_DIR / "state.json", {})
        state[str(qid)] = rec
        json.dump(state, open(RUN_DIR / "state.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"[{rec['status']}] q{qid} ({item['db_id']}) {rec['secs']}s "
          f"calls={qu['calls']} tok={qu['tokens_in']}+{qu['tokens_out']} "
          f"pred={'yes' if pred else 'NO'}", flush=True)
    return rec


def run_batch(dev, ids, workers):
    dev_by_qid = {d["question_id"]: d for d in dev}
    trace = RUN_DIR / "trace" / "api_trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    print(f"engine={din.ENGINE} workers={workers} todo={len(ids)}", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, q, dev_by_qid[q], trace): q for q in ids}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"q{futs[fut]} driver error: {e}", flush=True)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"batch done: ok {ok}/{len(ids)} elapsed {(time.time() - t0) / 60:.1f}min "
          f"total_usage={din.total_usage}", flush=True)


# ------------------------------------------------------------------ 评分
def _pred_sql_of(raw):
    if not raw or not isinstance(raw, str):
        return ""
    return raw.split("\t----- bird -----")[0].strip().rstrip(";")


def evaluate(dev):
    merged = _load(RUN_DIR / "merged.json", {})
    state = _load(RUN_DIR / "state.json", {})
    gcx = GoldCache()
    ex_vec = [0] * 300
    per_qid = {}
    exec_ok = missing = 0
    for i, item in enumerate(dev):
        qid = str(item["question_id"])
        raw = merged.get(qid, "")
        if not raw:
            missing += 1
            per_qid[item["question_id"]] = 0
            continue
        pred_sql = _pred_sql_of(raw)
        db_path = str(DAIL_BIRD / "database" / item["db_id"] /
                      f"{item['db_id']}.sqlite")
        gold = gcx.get(item["db_id"], item["SQL"])
        pred = run_sql(db_path, pred_sql) if pred_sql else None
        if pred is not None:
            exec_ok += 1
        v = 1 if (pred is not None and gold is not None and pred == gold) else 0
        ex_vec[i] = v
        per_qid[item["question_id"]] = v
    cost = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    for rec in state.values():
        for k in cost:
            cost[k] += int(rec.get(k, 0) or 0)
    return merged, ex_vec, per_qid, exec_ok, missing, cost


def consistency61(dev, per_qid):
    """旧 61 题预测 (deepseek-chat, chess_bird61 行序) vs 新 300 题重叠题 EX。"""
    old61 = _load(OLD61, [])
    old_preds = _load(OLD61_PRED, {})
    gcx = GoldCache()
    n = both_c = both_w = old_only = new_only = 0
    diffs = []
    for i, item in enumerate(old61):
        qid = item["question_id"]
        old_sql = _pred_sql_of(old_preds.get(str(i), ""))
        db_path = str(DAIL_BIRD / "database" / item["db_id"] /
                      f"{item['db_id']}.sqlite")
        gold = gcx.get(item["db_id"], item["SQL"])
        old_res = run_sql(db_path, old_sql) if old_sql else None
        old_ex = 1 if (old_res is not None and gold is not None
                       and old_res == gold) else 0
        new_ex = per_qid.get(qid, 0)
        n += 1
        both_c += old_ex and new_ex
        both_w += (not old_ex) and (not new_ex)
        old_only += old_ex and not new_ex
        new_only += (not old_ex) and new_ex
        if old_ex != new_ex:
            diffs.append({"qid": qid, "db_id": item["db_id"],
                          "old_ex": old_ex, "new_ex": new_ex})
    return {"n_overlap": n, "both_correct": both_c, "both_wrong": both_w,
            "old_only_correct": old_only, "new_only_correct": new_only,
            "diff_rate_pct": round(sum(1 for d in diffs) / max(n, 1) * 100, 1),
            "note": "old 61 题预测骨干为 deepseek-chat 且含列描述资源; "
                    "新 300 题骨干 deepseek-v4-pro (列描述缺失)",
            "diffs": diffs}


def finalize(dev):
    merged, ex_vec, per_qid, exec_ok, missing, cost = evaluate(dev)
    ex_n = sum(ex_vec)
    n_pred = sum(1 for v in merged.values() if v and str(v).strip())

    comp = {"dail_sql_v4pro_paper_pct": 44.7, "rgcv_v4pro_paper_pct": 54.7}
    t15 = _load(RES / "t15_significance.json", {})
    for t in t15.get("tests", []):
        if t.get("backbone") == "v4pro":
            if t.get("baseline") == "DAIL-SQL":
                comp["dail_sql_v4pro_replay_ex_n"] = t.get("baseline_ex")
            if t.get("baseline") == "SafeQL":
                comp["rgcv_v4pro_paper_ex_n"] = t.get("rgcv_ex")

    # phase 级成本分解 (从 trace 行级聚合, 仅成功调用)
    by_phase = {}
    for line in open(RUN_DIR / "trace" / "api_trace.jsonl",
                     encoding="utf-8").read().splitlines() if \
            (RUN_DIR / "trace" / "api_trace.jsonl").exists() else []:
        try:
            o = json.loads(line)
        except Exception:
            continue
        if not o.get("ok"):
            continue
        p = by_phase.setdefault(o["phase"], {"calls": 0, "tokens_in": 0,
                                             "tokens_out": 0})
        p["calls"] += 1
        p["tokens_in"] += o.get("tokens_in", 0)
        p["tokens_out"] += o.get("tokens_out", 0)

    cons = consistency61(dev, per_qid) if n_pred else {}

    out = {
        "task": "T1-6a DIN-SQL on BIRD-300 (unified backbone deepseek-v4-pro)",
        "model": {"engine": din.ENGINE, "api_base": din.API_BASE,
                  "temperature": din.TEMPERATURE, "max_tokens": din.MAX_TOKENS,
                  "thinking": "disabled",
                  "same_backbone_as": "src/rgcv/llm.py ENGINES[deepseek-v4-pro]"},
        "pipeline": "DIN-SQL (schema linking -> classification -> "
                    "easy/non-nested/nested generation -> self-correction); "
                    "prompt 模板未改动 (T1-6a)",
        "criteria": "EX_strict same as analyze_t15_significance.run_sql: "
                    "SQLite, sorted result-set equality, 60s soft timeout",
        "n_pred": n_pred,
        "missing_qids": sorted(set(str(d["question_id"]) for d in dev)
                               - set(k for k, v in merged.items()
                                     if v and str(v).strip()),
                               key=int),
        "ex_n": ex_n, "ex_pct": round(ex_n / 3.0, 2),
        "exec_rate_pct": round(exec_ok / max(n_pred, 1) * 100, 1),
        "per_question": ex_vec,
        "per_qid_ex": {str(k): v for k, v in sorted(per_qid.items())},
        "cost": {"total": cost,
                 "per_question_avg": {k: round(v / max(n_pred, 1), 1)
                                      for k, v in cost.items()},
                 "by_phase": by_phase},
        "consistency_vs_old61": cons,
        "comparison": comp,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    outp = RES / "dinsql_bird300_v4pro.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"\n==== DIN-SQL BIRD-300 ({din.ENGINE}) ====")
    print(f"  pred: {n_pred}/300  missing: {missing}")
    print(f"  EX_strict: {ex_n}/300 = {ex_n / 3.0:.2f}%   "
          f"(DAIL-SQL v4pro 44.7%, RGCV v4pro 54.7%)")
    print(f"  cost: {cost['calls']} calls, "
          f"{cost['tokens_in']} tok_in + {cost['tokens_out']} tok_out")
    if cons:
        print(f"  61 题交叉一致: diff {sum(1 for d in cons['diffs'])}/{cons['n_overlap']}"
              f" = {cons['diff_rate_pct']}%  "
              f"(old✓new✗ {cons['old_only_correct']}, old✗new✓ {cons['new_only_correct']})")
    print(f"saved -> {outp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated question_id list")
    ap.add_argument("--smoke", action="store_true", help="3 题冒烟 (36,37,10)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    dev = json.load(open(DAIL_BIRD / "dev" / "dev_300.json", encoding="utf-8"))
    assert len(dev) == 300
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        if args.smoke:
            ids = [36, 37, 10]
        elif args.ids:
            ids = [int(x) for x in args.ids.split(",") if x.strip()]
        else:
            merged = _load(RUN_DIR / "merged.json", {})
            done = {int(k) for k, v in merged.items() if v and str(v).strip()}
            ids = [d["question_id"] for d in dev if d["question_id"] not in done]
        run_batch(dev, ids, args.workers)

    finalize(dev)


if __name__ == "__main__":
    main()
