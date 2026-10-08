"""CHESS 补跑:找出主运行中崩溃/未产出 final SQL 的题目,逐题补跑。

用法: python rerun_chess_missing.py <main_run_dir_name>
在主 run 目录旁生成 rerun_missing 目录;每题一个 data json 顺序跑
(evaluate.py 已有 comparison_matrix 空值防御,补跑不会重复崩溃)。
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
CHESS = BASE / "baselines" / "CHESS" / "src"
RES = BASE / "results"
PY = os.environ.get("CHESS_PYTHON", sys.executable)


def main():
    main_run = sys.argv[1] if len(sys.argv) > 1 else None
    # 找主 run 目录(最新)
    if main_run:
        run_dir = CHESS / "results" / "dev" / "CHESS_IR_CG_UT_DS" / "chess_bird61" / main_run
    else:
        run_dir = max(
            (CHESS / "results" / "dev" / "CHESS_IR_CG_UT_DS" / "chess_bird61").iterdir(),
            key=lambda p: p.stat().st_mtime,
        )
    preds = json.load(open(run_dir / "-predictions.json", encoding="utf-8"))
    missing = sorted(int(k) for k, v in preds.items() if not isinstance(v, str))
    print(f"主 run: {run_dir.name}")
    print(f"缺失题: {missing} (共 {len(missing)})")
    if not missing:
        print("无缺失,退出")
        return

    dev = json.load(open(RES / "chess_bird61.json", encoding="utf-8"))
    dev_by_qid = {d["question_id"]: d for d in dev}

    for qid in missing:
        data = [dev_by_qid[qid]]
        tmp = RES / f"chess_fill_q{qid}.json"
        json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        cmd = [
            PY, "main.py",
            "--data_mode", "dev",
            "--data_path", str(tmp),
            "--config", r"..\run\configs\CHESS_IR_CG_UT_DS.yaml",
            "--num_workers", "1",
            "--log_level", "warning",
        ]
        print(f"\n=== 补跑 q{qid} ({dev_by_qid[qid]['db_id']}) ===", flush=True)
        r = subprocess.run(cmd, cwd=str(CHESS), capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = (r.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        print(tail[0], flush=True)
        if r.returncode != 0:
            print(f"  退出码 {r.returncode}, stderr 尾部: {(r.stderr or '')[-300:]}", flush=True)

    print("\n补跑完成。各补跑结果位于 src/results/dev/CHESS_IR_CG_UT_DS/chess_fill_q*/")


if __name__ == "__main__":
    main()
