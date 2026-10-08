"""CHESS E1 300 题驱动 (双骨干): 每题独立子进程 + 线程池控并发, 规避 Pool 管道死锁。

用法: python run_chess_bird300.py <v4pro|kimi> [--ids qid1,qid2] [--workers N]
- 题集: baselines/DAIL-SQL/dataset/bird/dev/dev_300.json (行序=评估对齐 idx)
- 结果: results/chess_bird300_<tag>/merged.json  (qid -> "sql\t----- bird -----\tdb_id" 原样)
        results/chess_bird300_<tag>/state.json   (qid -> {status, secs, db_id})
        results/chess_bird300_<tag>/trace/q{qid}.jsonl (API token usage 行级 append)
- 断点: merged 已有 SQL 的题自动跳过; 失败题下次重跑
- 超时: 单题 3600s 硬超时
- 双骨干各起一个进程实例 (setting/结果目录/engine 完全隔离, 可并行)
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
CHESS = BASE / "baselines" / "CHESS" / "src"
RES = BASE / "results"
DAIL_BIRD = BASE / "baselines" / "DAIL-SQL" / "dataset" / "bird"
PY = os.environ.get("CHESS_PYTHON", sys.executable)
TIMEOUT = 3600

RUNS = {
    "v4pro": dict(
        setting="CHESS_IR_CG_UT_DS_V4PRO",
        api_base="https://api.deepseek.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        workers=3,
    ),
    "kimi": dict(
        setting="CHESS_IR_CG_UT_DS_KIMI",
        api_base="https://api.moonshot.cn/v1",
        api_key=os.environ.get("MOONSHOT_API_KEY", ""),
        workers=3,
    ),
}

_lock = threading.Lock()


def _load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def collect_result(qid: int, setting: str):
    """从 chess_q{qid} 最新含预测的 run 目录提取预测串(0 表示无 final SQL)。"""
    run_root = CHESS / "results" / "dev" / setting / f"chess_q{qid}"
    if not run_root.exists():
        return None
    for run_dir in sorted(run_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        pred_file = run_dir / "-predictions.json"
        if not pred_file.exists():
            continue
        preds = _load(pred_file, {})
        v = preds.get(str(qid))
        if isinstance(v, str):
            return v
    return None


def run_one(qid: int, item: dict, cfg: dict, tag: str) -> dict:
    out_dir = RES / f"chess_bird300_{tag}"
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    db_id = item["db_id"]
    tmp = tmp_dir / f"chess_q{qid}.json"
    json.dump([item], open(tmp, "w", encoding="utf-8"), ensure_ascii=False)

    env = {
        **os.environ,               # 代理经 HTTP(S)_PROXY 环境变量透传
        "DB_ROOT_PATH": str(DAIL_BIRD),
        "OPENAI_API_BASE": cfg["api_base"],
        "OPENAI_API_KEY": cfg["api_key"],
        "CHESS_TRACE_DIR": str(trace_dir),
        "CHESS_TRACE_QID": str(qid),
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        PY, "main.py",
        "--data_mode", "dev",
        "--data_path", str(tmp),
        "--config", rf"..\run\configs\{cfg['setting']}.yaml",
        "--num_workers", "1",
        "--log_level", "warning",
    ]
    t0 = time.time()
    status = "error"
    try:
        r = subprocess.run(
            cmd, cwd=str(CHESS), env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT,
        )
        status = "ok" if r.returncode == 0 else f"rc{r.returncode}"
        if r.returncode != 0:
            (out_dir / f"stderr_q{qid}.log").write_text(
                (r.stderr or "")[-8000:], encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        status = "timeout"
    pred = collect_result(qid, cfg["setting"])
    if pred:
        status = "ok"
    rec = {"qid": qid, "db_id": db_id, "status": status, "secs": round(time.time() - t0)}
    with _lock:
        merged_path = out_dir / "merged.json"
        merged = _load(merged_path, {})
        if pred:
            merged[str(qid)] = pred
        json.dump(merged, open(merged_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        state_path = out_dir / "state.json"
        state = _load(state_path, {})
        state[str(qid)] = rec
        json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{rec['status']}] q{qid} ({db_id}) {rec['secs']}s pred={'yes' if pred else 'NO'}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=list(RUNS))
    ap.add_argument("--ids", default="", help="comma-separated question_id list (smoke mode)")
    ap.add_argument("--workers", type=int, default=0, help="override concurrency")
    args = ap.parse_args()

    cfg = RUNS[args.tag]
    out_dir = RES / f"chess_bird300_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or cfg["workers"]

    dev = json.load(open(DAIL_BIRD / "dev" / "dev_300.json", encoding="utf-8"))
    assert len(dev) == 300
    dev_by_qid = {d["question_id"]: d for d in dev}

    if args.ids:
        todo = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        merged = _load(out_dir / "merged.json", {})
        done = {int(k) for k, v in merged.items() if v and str(v).strip()}
        todo = [d["question_id"] for d in dev if d["question_id"] not in done]
    print(f"[{args.tag}] setting={cfg['setting']} workers={workers} todo={len(todo)}", flush=True)

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, qid, dev_by_qid[qid], cfg, args.tag): qid for qid in todo}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"q{futs[fut]} driver error: {e}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    merged = _load(out_dir / "merged.json", {})
    wall = time.time() - t_start
    print(f"\n[{args.tag}] done: ok {ok}/{len(todo)} elapsed {wall/3600:.2f}h total_pred {len(merged)}/300", flush=True)
    for r in sorted(results, key=lambda x: x["qid"]):
        if r["status"] != "ok":
            print(f"  FAIL: q{r['qid']} ({r['db_id']}) {r['status']}", flush=True)


if __name__ == "__main__":
    main()
