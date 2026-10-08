"""CHESS BIRD EX 评估。

输入: src/results/dev/<setting>/<dataset>/<run_time>/-predictions.json
      格式 {"<question_id>": "<sql>\t----- bird -----\t<db_id>"}
输出: EX_strict / 执行率 + 四基线横向对比
"""
import json
import sqlite3
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
CHESS_RES = BASE / "baselines" / "CHESS" / "src" / "results" / "dev"


def run_sql(db_path: str, sql: str, timeout: int = 10):
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted([tuple(r) for r in rows])
    except Exception:
        return None


def find_latest_run(dataset_name: str):
    """在 CHESS results 下找指定 dataset 的最新一次运行。"""
    cands = []
    for setting_dir in CHESS_RES.iterdir():
        ds_dir = setting_dir / dataset_name
        if not ds_dir.exists():
            continue
        for run_dir in ds_dir.iterdir():
            pred = run_dir / "-predictions.json"
            if pred.exists():
                cands.append(pred)
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "chess_bird61"
    # 优先用逐题驱动合并结果 (run_chess_bird61.py 产物)
    merged_file = BASE / "results" / "chess_bird61_merged.json"
    if merged_file.exists():
        preds = json.load(open(merged_file, encoding="utf-8"))
        print(f"评估文件: {merged_file} ({len(preds)} 题)\n")
    else:
        pred_file = find_latest_run(dataset_name)
        if not pred_file:
            print(f"未找到 {dataset_name} 的 predictions.json")
            return
        print(f"评估文件: {pred_file}\n")
        preds = json.load(open(pred_file, encoding="utf-8"))
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev.json"))
    dev_by_qid = {d["question_id"]: d for d in dev}

    n = len(dev)
    ex_strict = 0
    pred_exec = 0
    missing = 0
    failed = []

    for item in dev:
        qid = str(item["question_id"])
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database" / db_id / f"{db_id}.sqlite")
        gold_res = run_sql(db_path, item["SQL"])

        raw = preds.get(qid)
        if not raw or not isinstance(raw, str):
            missing += 1
            continue
        pred_sql = raw.split("\t----- bird -----")[0].strip().rstrip(";")
        pred_res = run_sql(db_path, pred_sql)

        if pred_res is not None:
            pred_exec += 1
        else:
            failed.append((qid, pred_sql[:80]))
        if gold_res is not None and pred_res is not None and gold_res == pred_res:
            ex_strict += 1

    done = n - missing
    print(f"{'='*55}")
    print(f"  CHESS (deepseek-chat) EX — BIRD 61 题子集")
    print(f"{'='*55}")
    print(f"  完成题数:        {done}/{n}" + (f"  (缺失 {missing})" if missing else ""))
    print(f"  Pred 执行成功:   {pred_exec}/{done} ({pred_exec/max(done,1)*100:.1f}%)")
    print(f"  EX_strict:       {ex_strict}/{n} = {ex_strict/n*100:.1f}%")
    if failed:
        print(f"  执行失败样例 (前 3):")
        for qid, info in failed[:3]:
            print(f"    q{qid}: {info}")

    print(f"\n{'='*55}")
    print(f"  横向对比 (同 61 题子集)")
    print(f"{'='*55}")
    print(f"  {'基线':<22} {'EX_strict':>14} {'执行率':>10} {'骨干':<18}")
    print(f"  {'-'*70}")
    print(f"  {'DAIL-SQL (Pass2)':<22} {'29/61 = 47.5%':>14} {'90.2%':>10} {'deepseek-chat':<18}")
    print(f"  {'ReFoRCE':<22} {'32/61 = 52.5%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'MAC-SQL':<22} {'35/61 = 57.4%':>14} {'100.0%':>10} {'deepseek-chat':<18}")
    print(f"  {'SQLCoder-7B-2':<22} {'11/61 = 18.0%':>14} {'55.7%':>10} {'本地 7B Q4':<18}")
    print(f"  {'CHESS':<22} {f'{ex_strict}/{n} = {ex_strict/n*100:.1f}%':>14} {f'{pred_exec/max(done,1)*100:.1f}%':>10} {'deepseek-chat':<18}")


if __name__ == "__main__":
    main()
