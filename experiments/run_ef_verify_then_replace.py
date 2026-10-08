"""E-F:门控 vs verify-then-replace 选择策略对照(离线重建,零生成成本)。

语义(P1-2 步骤 1):修复搜索照常进行(复用 ungated 轨迹),但仅当新候选
通过 V1–V3 验证(verdict=="pass")才替换原查询;否则保留生成器候选起点。
数据源:ungated run 的 q_*.json——candidates[0]=起点 orig,final_sql=修复
后 repaired,verification=修复后 SQL 的 V1–V3 报告。关键闭环:ungated 中
diff-repair 仅在 verdict∈{alarm,reject} 时触发,而 vtR 在该情形直接保留
orig,故 verdict=="pass" 时 final_sql 必为 diff-repair 前的 repaired——
无需重放修复即可精确重建 vtR。

指标:
  - 逐题 EX:EX_orig / EX_ungated / EX_vtR / EX_gated(gated run 对照)
  - 47/9 型翻转归因:broken(orig对→ungated错)/rescued(orig错→ungated对),
    及其 verdict 分布(vtR 能拦回多少 broken / 丢多少 rescued)
  - McNemar + paired bootstrap(vtR vs gated / vtR vs ungated)

用法:python run_ef_verify_then_replace.py <engine_tag> [...] [--fullschema]
  engine_tag ∈ {deepseek-v4-pro, kimi-k2.6}
  --fullschema 时 ungated/gated tag 加 _fullschema(full 注入复测,步骤 5)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]              # src/rgcv_repro/
RES = BASE / "results"
DAIL = BASE.parents[1] / "baselines" / "DAIL-SQL"
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from eval_rgcv_e1 import run_sql  # noqa: E402
from rgcv.evalx import mcnemar_test, paired_bootstrap  # noqa: E402


def load_run(tag: str, qid2idx: dict) -> dict:
    """tag → {qid: {"orig","final","verdict","error"}}(仅 dev300 内)。"""
    outdir = RES / f"rgcv_bird300_{tag}"
    recs = {}
    for p in outdir.glob("q_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        qid = str(d.get("question_id"))
        if qid not in qid2idx:
            continue
        err = bool(d.get("error"))
        cands = d.get("candidates") or []
        orig = cands[0] if cands else ""
        v = d.get("verification") or {}
        recs[qid] = {
            "orig": "" if err else orig,
            "final": "" if err else (d.get("pred_sql") or ""),
            "verdict": None if err else v.get("verdict"),
            "error": err,
        }
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="+",
                    choices=["deepseek-v4-pro", "kimi-k2.6"])
    ap.add_argument("--fullschema", action="store_true",
                    help="full 注入复测(步骤 5):tag 加 _fullschema")
    a = ap.parse_args()

    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                         encoding="utf-8"))
    qid2idx = {str(d["question_id"]): i for i, d in enumerate(dev)}

    # gold 缓存(300 题)
    gold_cache = {}
    for i, item in enumerate(dev):
        db_id = item["db_id"]
        db_path = str(DAIL / "dataset" / "bird" / "database"
                      / db_id / f"{db_id}.sqlite")
        gold_cache[i] = (db_path,
                         run_sql(db_path, item.get("SQL")
                                 or item.get("query", "")))
    gold_fail = sum(1 for v in gold_cache.values() if v[1] is None)
    print(f"gold 可执行: {300 - gold_fail}/300")

    suffix = "_fullschema" if a.fullschema else ""
    summary = {
        "note": "E-F verify-then-replace vs gating (offline rebuild)",
        "injection": "full" if a.fullschema else "compressed",
        "engines": {},
    }
    for engine in a.engines:
        base = engine.replace(".", "")
        ung = load_run(base + suffix, qid2idx)
        gat = load_run(base + "_gated" + suffix, qid2idx)   # make_tag: gated 在 ablation 前
        print(f"\n[{engine}] ungated={len(ung)} gated={len(gat)}")

        a_orig, a_ung, a_vtr, a_gat = [], [], [], []
        rows = []
        for i, item in enumerate(dev):
            qid = str(item["question_id"])
            db_path, gold_res = gold_cache[i]
            u = ung.get(qid, {"orig": "", "final": "", "verdict": None,
                              "error": True})
            g = gat.get(qid)
            # vtR:仅当修复候选通过 V1–V3(pass)才替换,否则保留起点
            vtr = u["final"] if u["verdict"] == "pass" else u["orig"]
            exo = int(run_sql(db_path, u["orig"]) == gold_res
                      if gold_res is not None else False)
            exu = int(run_sql(db_path, u["final"]) == gold_res
                      if gold_res is not None else False)
            ext = int(run_sql(db_path, vtr) == gold_res
                      if gold_res is not None else False)
            exg = 0
            if g:
                exg = int(run_sql(db_path, g["final"]) == gold_res
                          if gold_res is not None else False)
            a_orig.append(bool(exo))
            a_ung.append(bool(exu))
            a_vtr.append(bool(ext))
            a_gat.append(bool(exg))
            rows.append({"qid": item["question_id"], "db_id": item["db_id"],
                         "difficulty": item.get("difficulty", ""),
                         "verdict": u["verdict"], "ex_orig": exo,
                         "ex_ungated": exu, "ex_vtr": ext, "ex_gated": exg})

        n = len(a_orig)
        # 翻转归因(47/9 型):相对生成器起点 orig
        broken = [(r["verdict"]) for r in rows
                  if r["ex_orig"] and not r["ex_ungated"]]
        rescued = [(r["verdict"]) for r in rows
                   if not r["ex_orig"] and r["ex_ungated"]]
        vt_broken = sum(1 for r in rows
                        if r["ex_orig"] and not r["ex_vtr"])
        vt_rescued = sum(1 for r in rows
                         if not r["ex_orig"] and r["ex_vtr"])
        # broken/rescued 在 vtR 下的走向
        b_verdicts = {}
        for v in broken:
            b_verdicts[v] = b_verdicts.get(v, 0) + 1
        r_verdicts = {}
        for v in rescued:
            r_verdicts[v] = r_verdicts.get(v, 0) + 1

        def mcn(a1, a2):
            return mcnemar_test(a1, a2)

        print(f"  EX_orig={sum(a_orig)}/{n}={sum(a_orig)/n*100:.1f}%  "
              f"EX_ungated={sum(a_ung)}/{n}={sum(a_ung)/n*100:.1f}%  "
              f"EX_vtR={sum(a_vtr)}/{n}={sum(a_vtr)/n*100:.1f}%  "
              f"EX_gated={sum(a_gat)}/{n}={sum(a_gat)/n*100:.1f}%")
        print(f"  ungated 翻转: broken={len(broken)} rescued={len(rescued)} "
              f"net={len(rescued) - len(broken)}")
        print(f"    broken verdict 分布: {b_verdicts}")
        print(f"    rescued verdict 分布: {r_verdicts}")
        print(f"  vtR 翻转: broken={vt_broken} rescued={vt_rescued} "
              f"net={vt_rescued - vt_broken}")
        bs_tg = paired_bootstrap(a_vtr, a_gat)   # vtR - gated
        bs_tu = paired_bootstrap(a_vtr, a_ung)
        print(f"  vtR-gated delta="
              f"{(sum(a_vtr)-sum(a_gat))/n*100:+.1f}pp CI95={bs_tg['ci95']} "
              f"McNemar p={mcn(a_vtr, a_gat).get('p_value', '?')}")
        print(f"  vtR-ungated delta="
              f"{(sum(a_vtr)-sum(a_ung))/n*100:+.1f}pp CI95={bs_tu['ci95']} "
              f"McNemar p={mcn(a_vtr, a_ung).get('p_value', '?')}")

        summary["engines"][engine] = {
            "ungated_tag": f"rgcv_bird300_{base}{suffix}",
            "gated_tag": f"rgcv_bird300_{base}_gated{suffix}",
            "n": n,
            "ex_orig": sum(a_orig), "ex_ungated": sum(a_ung),
            "ex_vtr": sum(a_vtr), "ex_gated": sum(a_gat),
            "pct": {k: round(v / n * 100, 2) for k, v in
                    [("orig", sum(a_orig)), ("ungated", sum(a_ung)),
                     ("vtr", sum(a_vtr)), ("gated", sum(a_gat))]},
            "ungated_broken": len(broken), "ungated_rescued": len(rescued),
            "broken_verdict_dist": b_verdicts,
            "rescued_verdict_dist": r_verdicts,
            "vtr_broken": vt_broken, "vtr_rescued": vt_rescued,
            "mcnemar_vtr_vs_gated": mcn(a_vtr, a_gat),
            "mcnemar_vtr_vs_ungated": mcn(a_vtr, a_ung),
            "bootstrap_vtr_minus_gated": bs_tg,
            "bootstrap_vtr_minus_ungated": bs_tu,
            "per_question": rows,
        }

    name = ("eF_verify_then_replace_full.json" if a.fullschema
            else "eF_verify_then_replace.json")
    out = RES / name
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
