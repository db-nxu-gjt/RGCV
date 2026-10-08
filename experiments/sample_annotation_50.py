"""E-A 配套:50 例人工标注抽样(25 拦截 + 25 误报)与 Cohen's κ。

兑现 R4-M2"50 例人工校准"承诺:自动判定 vs 人工判定的一致率。

模式 1(默认,generate):从 results/eA_tier2_real_verification.json 抽样
  - stratum intercepted  = silent 且 V1+V2+V3 flagged(拦截成功)
  - stratum false_positive = correct 且 flagged(误报)
  各抽 25(seed=0;不足则全取)。输出盲评 CSV:
  results/eA_annotation_50.csv(不含 stratum/验证器结论 → 盲评),
  映射与协议存 results/eA_annotation_50_manifest.json。

标注协议(写入 manifest):
  1. 任务:判断 pred_sql 在语义上是否正确回答 question(gold_sql 仅为
     参考;执行语义判定,非语法比对);
  2. annotator_is_wrong:1 = 预测答案错误(行/列/值/行集任一不对),
     0 = 正确;不确定时可在本地 SQLite 执行两查询比对后判定;
  3. annotator_error_type(自由文本):wrong-filter / wrong-aggregation /
     wrong-join / missing-rows / value-shift / other;
  4. 标注者不可见验证器任何输出(盲评);争议题双人仲裁。

模式 2(--score <filled.csv>):读取已填 CSV,按 manifest 分层计算
  一致率与 Cohen's κ(自动判定为参考标准:silent→wrong=1,correct→0)。

运行:python experiments/sample_annotation_50.py
      python experiments/sample_annotation_50.py --score results/eA_annotation_50_filled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402

E_A_FILE = RESULTS / "eA_tier2_real_verification.json"
CSV_FILE = RESULTS / "eA_annotation_50.csv"
MANIFEST_FILE = RESULTS / "eA_annotation_50_manifest.json"
N_PER_STRATUM = 25

PROTOCOL = [
    "Judge pred_sql semantics against question; gold_sql is reference only.",
    "annotator_is_wrong: 1=wrong answer (rows/cols/values/row-set), 0=correct;"
    " execute both queries on the local SQLite DB when unsure.",
    "annotator_error_type (free text): wrong-filter / wrong-aggregation / "
    "wrong-join / missing-rows / value-shift / other.",
    "Annotator is blind to verifier output; disputes resolved by second "
    "annotator.",
]


def load_pred(tag: str, qid: int) -> dict:
    p = RESULTS / tag / f"q_{qid}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def generate():
    data = json.loads(E_A_FILE.read_text(encoding="utf-8"))
    traces = [t for t in data["traces"] if not t.get("skipped")]
    strata = {"intercepted": [], "false_positive": []}
    for t in traces:
        flagged = t["V1+V2+V3"]["flagged"]
        if t["kind"] == "silent" and flagged:
            strata["intercepted"].append(t)
        elif t["kind"] == "correct" and flagged:
            strata["false_positive"].append(t)

    rng = random.Random(0)
    sample = []
    for name, pool in strata.items():
        rng.shuffle(pool)
        # 跨配置同题去重:traces 按传入顺序排列,主配置在前优先保留
        seen_qids = set()
        taken = []
        for t in pool:
            if t["question_id"] in seen_qids:
                continue
            seen_qids.add(t["question_id"])
            taken.append(t)
        taken = taken[:N_PER_STRATUM]
        print(f"[ann] {name}: pool={len(pool)} unique={len(seen_qids)} "
              f"sampled={len(taken)}")
        for t in taken:
            rec = load_pred(t["tag"], t["question_id"])
            sample.append({
                "question_id": t["question_id"], "db_id": t["db_id"],
                "tag": t["tag"], "_stratum": name,
                "question": rec["question"],
                "pred_sql": rec["pred_sql"], "gold_sql": rec["gold_sql"],
            })

    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "db_id", "tag", "question", "pred_sql",
                    "gold_sql", "annotator_is_wrong", "annotator_error_type",
                    "notes"])
        for s in sample:
            w.writerow([s["question_id"], s["db_id"], s["tag"], s["question"],
                        s["pred_sql"], s["gold_sql"], "", "", ""])

    manifest = {
        "config": {
            "source": str(E_A_FILE.name),
            "seed": 0,
            "n_per_stratum": N_PER_STRATUM,
            "pool_sizes": {k: len(v) for k, v in strata.items()},
            "auto_label_ref": "silent→wrong=1, correct→wrong=0",
        },
        "protocol": PROTOCOL,
        "stratum_map": {str(s["question_id"]): s["_stratum"]
                        for s in sample},
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False,
                                        indent=1), encoding="utf-8")
    print(f"[saved] {CSV_FILE.name} (n={len(sample)}, blind: 无分层列)")
    print(f"[saved] {MANIFEST_FILE.name}")


def score(filled: str):
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    smap = manifest["stratum_map"]
    rows = list(csv.DictReader(open(filled, encoding="utf-8-sig")))
    out = {}
    for stratum in ("intercepted", "false_positive"):
        pairs = []          # (auto, human): wrong=1/0
        for r in rows:
            qid = str(r["question_id"])
            if smap.get(qid) != stratum:
                continue
            v = (r.get("annotator_is_wrong") or "").strip()
            if v not in ("0", "1"):
                continue
            auto = 1 if stratum == "intercepted" else 0
            pairs.append((auto, int(v)))
        n = len(pairs)
        if not n:
            continue
        agree = sum(1 for a, h in pairs if a == h) / n
        # Cohen's κ(二分类)
        n11 = sum(1 for a, h in pairs if a == 1 and h == 1)
        n10 = sum(1 for a, h in pairs if a == 1 and h == 0)
        n01 = sum(1 for a, h in pairs if a == 0 and h == 1)
        n00 = sum(1 for a, h in pairs if a == 0 and h == 0)
        po = agree
        pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / n ** 2
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        out[stratum] = {"n": n, "agreement": round(agree, 4),
                        "kappa": round(kappa, 4),
                        "confusion": {"1-1": n11, "1-0": n10,
                                      "0-1": n01, "0-0": n00}}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (RESULTS / "eA_annotation_50_kappa.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] eA_annotation_50_kappa.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=None,
                    help="path to filled annotation csv → compute κ")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    else:
        generate()
