"""E-E 评测:DAIL-SQL v4-pro 全量 1534 EX + 300 子集 vs 全量 + 排名一致性。

输入:
  process/..._FULL/SHARD{0..3}/RESULTS_MODEL-deepseek-v4-pro.txt(按序拼接=1534 行)
  dataset/bird/dev.json(全量预处理版,行序=questions.json 行序)
  results/rgcv_bird300_*/ 不用;300 旧跑结果在 process/...(无 _FULL)/RESULTS_MODEL-deepseek-v4-pro.txt
输出:
  results/eE_full_dev.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
DAIL = BASE.parents[1] / "baselines" / "DAIL-SQL"
sys.path.insert(0, str(BASE / "experiments"))

from eval_rgcv_e1 import run_sql  # noqa: E402

P1FULL = DAIL / "dataset" / "process" / \
    "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096_FULL"
P1OLD = DAIL / "dataset" / "process" / \
    "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096"
NQ = 1534


def merge_shards() -> list[str]:
    lines: list[str] = []
    for s in range(4):
        f = P1FULL / f"SHARD{s}" / "RESULTS_MODEL-deepseek-v4-pro.txt"
        ls = [l.rstrip("\n") for l in open(f, encoding="utf-8") if l.strip()]
        print(f"SHARD{s}: {len(ls)} 行")
        lines.extend(ls)
    return lines


def eval_predictions(preds: list[str], dev: list[dict], dbroot) -> list[int]:
    gold_cache: dict[int, object] = {}
    ex = []
    for i, item in enumerate(dev):
        db = dbroot / item["db_id"] / f"{item['db_id']}.sqlite"
        if i not in gold_cache:
            gold_cache[i] = run_sql(str(db), item["SQL"])
        g = gold_cache[i]
        p = run_sql(str(db), preds[i]) if i < len(preds) else None
        ex.append(int(p is not None and g is not None and p == g))
        if (i + 1) % 200 == 0:
            print(f"  eval {i + 1}/{len(dev)}: 累计 EX={sum(ex)}/{i + 1}")
    return ex


def boot_ci(pairs, n=2000, seed=0):
    """pairs: [(ex_full_i, ex_300_i)] 同题配对 → |full-300| 的 bootstrap CI"""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        f = sum(a for a, _ in s) / len(s)
        t = sum(b for _, b in s) / len(s)
        diffs.append(f - t)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev.json",
                         encoding="utf-8"))
    assert len(dev) == NQ, f"dev.json {len(dev)} != {NQ}"
    dev300 = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                            encoding="utf-8"))
    qid300 = {str(q["question_id"]) for q in dev300}
    dbroot = DAIL / "dataset" / "bird" / "database"

    preds = merge_shards()
    print(f"合并分片: {len(preds)} 行 (期望 {NQ})")
    degen = sum(1 for p in preds if p.strip().upper() in ("SELECT", ""))
    print(f"退化行(SELECT/空): {degen}")
    assert len(preds) == NQ, "行数不符,分片未完成"

    print("评测全量 1534 ...")
    ex_full = eval_predictions(preds, dev, dbroot)
    n = len(ex_full)
    print(f"EX_full = {sum(ex_full)}/{n} = {sum(ex_full)/n*100:.2f}%")

    # 300 子集(全量跑内) vs 旧 300 跑
    idx300 = [i for i, d in enumerate(dev)
              if str(d["question_id"]) in qid300]
    ex300_from_full = [ex_full[i] for i in idx300]
    old_lines = [l.strip() for l in
                 open(P1OLD / "RESULTS_MODEL-deepseek-v4-pro.txt",
                      encoding="utf-8") if l.strip()]
    print(f"旧 300 跑结果: {len(old_lines)} 行")
    # 旧跑顺序 = dev_300.json 顺序
    old_dev = dev300
    ex300_old = []
    for i, item in enumerate(old_dev):
        db = dbroot / item["db_id"] / f"{item['db_id']}.sqlite"
        g = run_sql(str(db), item["SQL"])
        p = run_sql(str(db), old_lines[i]) if i < len(old_lines) else None
        ex300_old.append(int(p is not None and g is not None and p == g))

    m_full = sum(ex300_from_full) / len(ex300_from_full) * 100
    m_old = sum(ex300_old) / len(ex300_old) * 100
    pairs = list(zip(ex300_from_full, ex300_old))
    lo, hi = boot_ci(pairs)
    print(f"EX_300(全量跑内) = {m_full:.2f}%  EX_300(旧跑) = {m_old:.2f}%  "
          f"Δ={m_full - m_old:+.2f}pp CI95=[{lo * 100:+.2f}, {hi * 100:+.2f}]")
    print(f"外推偏差: EX_300 - EX_full = {m_full - sum(ex_full)/n*100:+.2f}pp")

    # 难度分层(全量)
    by_diff = {}
    for i, d in enumerate(dev):
        by_diff.setdefault(d.get("difficulty", "?"), []).append(ex_full[i])
    diff_stat = {k: f"{sum(v)}/{len(v)}={sum(v)/len(v)*100:.1f}%"
                 for k, v in sorted(by_diff.items())}
    print(f"难度分层: {diff_stat}")

    out = RES / "eE_full_dev.json"
    out.write_text(json.dumps({
        "system": "DAIL-SQL (deepseek-v4-pro, pass1 EUCDISQUESTIONMASK, T=0)",
        "n_full": n,
        "ex_full": sum(ex_full),
        "pct_full": round(sum(ex_full) / n * 100, 2),
        "by_difficulty": {k: {"n": len(v), "correct": sum(v),
                              "pct": round(sum(v) / len(v) * 100, 2)}
                          for k, v in by_diff.items()},
        "subset_300": {
            "n": len(idx300),
            "ex_from_full_run": sum(ex300_from_full),
            "pct_from_full_run": round(m_full, 2),
            "ex_old_300_run": sum(ex300_old),
            "pct_old_300_run": round(m_old, 2),
            "delta_pp": round(m_full - m_old, 2),
            "boot_ci95_delta_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        },
        "extrapolation_gap_pp": round(m_full - sum(ex_full) / n * 100, 2),
        "degenerate_lines": degen,
        "ex_full_per_question": ex_full,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {out}")


if __name__ == "__main__":
    main()
