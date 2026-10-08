"""E-D-a:改写稳健性分析(逐题配对 EX 差值分布)。

口径:原跑与改写跑都用同一新算的 SQLite 侧 EX(eval_rgcv_e1.run_sql 严格
结果集匹配,同一 gold),题级配对,只统计改写集 150(auto_pass)题。
输出(每骨干):
  - EX_orig_subset / EX_para / 差值(pp) + paired bootstrap 95% CI
  - 翻转分解:regress(原对->改写错,一致于记忆化敏感)vs gain(原错->改写对)
  - McNemar(orig vs para)
  - 按难度分解
写入:results/eD_paraphrase_summary.json

用法:python analyze_paraphrase.py <engine_tag> [<engine_tag> ...]
  engine_tag ∈ {deepseek-v4-pro, kimi-k2.6}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]              # src/rgcv_repro/
RES = BASE / "results"
DAIL = BASE / "baselines" / "DAIL-SQL"
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from eval_rgcv_e1 import load_preds, run_sql  # noqa: E402
from rgcv.evalx import mcnemar_test, paired_bootstrap  # noqa: E402


def main():
    engines = sys.argv[1:]
    assert engines, "usage: python analyze_paraphrase.py <deepseek-v4-pro|kimi-k2.6> ..."
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                         encoding="utf-8"))
    qid2idx = {d["question_id"]: i for i, d in enumerate(dev)}
    manifest = json.load(open(RES / "paraphrase_150.json", encoding="utf-8"))
    para_qids = [q for q, m in manifest.items()
                 if m.get("status") == "auto_pass"]
    para_idx = sorted(qid2idx[int(q)] for q in para_qids)
    print(f"改写集 auto_pass: {len(para_idx)} 题")

    # gold 缓存(150 题,每库执行一次)
    gold_cache = {}
    for i in para_idx:
        item = dev[i]
        db_path = str(DAIL / "dataset" / "bird" / "database"
                      / item["db_id"] / f"{item['db_id']}.sqlite")
        gold_cache[i] = (db_path, run_sql(db_path,
                                          item.get("SQL") or item.get("query", "")))
    gold_fail = sum(1 for v in gold_cache.values() if v[1] is None)
    print(f"gold 可执行: {len(gold_cache) - gold_fail}/{len(gold_cache)}")

    summary = {"note": "E-D-a paraphrase robustness; paired per-question EX "
                       "(same fresh SQLite-side criterion on both runs)",
               "n_paraphrase": len(para_idx),
               "engines": {}}
    for engine in engines:
        tag_engine = engine.replace(".", "")            # make_tag 同规则
        orig_tag = f"{tag_engine}_gated_fullschema"      # load_preds 内部补前缀
        para_tag = f"{orig_tag}_para"
        preds_o, _ = load_preds(orig_tag, qid2idx)
        preds_p, _ = load_preds(para_tag, qid2idx)
        a = []                       # orig EX per paraphrase-question
        b = []                       # para EX
        rows = []
        for i in para_idx:
            db_path, gold_res = gold_cache[i]
            ro = run_sql(db_path, preds_o[i])
            rp = run_sql(db_path, preds_p[i])
            xo = int(ro is not None and gold_res is not None and ro == gold_res)
            xp = int(rp is not None and gold_res is not None and rp == gold_res)
            a.append(bool(xo))
            b.append(bool(xp))
            rows.append({"qid": dev[i]["question_id"],
                         "db_id": dev[i]["db_id"],
                         "difficulty": dev[i].get("difficulty", ""),
                         "ex_orig": xo, "ex_para": xp})
        n = len(a)
        ex_o, ex_p = sum(a), sum(b)
        regress = sum(1 for x, y in zip(a, b) if x and not y)
        gain = sum(1 for x, y in zip(a, b) if y and not x)
        mc = mcnemar_test(a, b)
        bs = paired_bootstrap(b, a)   # para - orig
        by_diff = {}
        for r in rows:
            d = by_diff.setdefault(r["difficulty"],
                                   {"n": 0, "ex_orig": 0, "ex_para": 0,
                                    "regress": 0, "gain": 0})
            d["n"] += 1
            d["ex_orig"] += r["ex_orig"]
            d["ex_para"] += r["ex_para"]
            d["regress"] += int(r["ex_orig"] and not r["ex_para"])
            d["gain"] += int(r["ex_para"] and not r["ex_orig"])
        print(f"\n[{engine}]")
        print(f"  EX_orig(150)={ex_o}/{n}={ex_o/n*100:.1f}%  "
              f"EX_para={ex_p}/{n}={ex_p/n*100:.1f}%  "
              f"delta={(ex_p-ex_o)/n*100:+.1f}pp  CI95={bs['ci95']}")
        print(f"  regress(原对->改写错)={regress}  gain(原错->改写对)={gain}  "
              f"McNemar p={mc.get('p_value', mc.get('p'))}")
        for d, v in sorted(by_diff.items()):
            print(f"    {d:<12} n={v['n']:>3} orig={v['ex_orig']:>3} "
                  f"para={v['ex_para']:>3} regress={v['regress']} "
                  f"gain={v['gain']}")
        summary["engines"][engine] = {
            "orig_tag": f"rgcv_bird300_{orig_tag}",
            "para_tag": f"rgcv_bird300_{para_tag}", "n": n,
            "ex_orig": ex_o, "ex_para": ex_p,
            "ex_orig_pct": round(ex_o / n * 100, 2),
            "ex_para_pct": round(ex_p / n * 100, 2),
            "delta_pp": round((ex_p - ex_o) / n * 100, 2),
            "bootstrap_para_minus_orig": bs,
            "regress": regress, "gain": gain, "mcnemar": mc,
            "by_difficulty": by_diff, "per_question": rows,
        }

    out = RES / "eD_paraphrase_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
