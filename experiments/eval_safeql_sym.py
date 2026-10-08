"""E-B:SafeQL 对称计分评估(EX_sym / 剔除 19 条敏感性 / McNemar vs RGCV)。

输入:
  results/safeql_bird300_<tag>.json   SafeQL 复现结果(base 或 _gated)
  results/safeql_gold19_fixed.json    19 条 gold 等价改写(translate_gold_19.py)
  results/eA_tier2_real_verification.json
      traces 中 kind ∈ {correct, silent} 重建 RGCV 逐题 EX
      (correct=1, silent=0, 缺失/执行失败=0)

口径:
  EX_sym(300)  = pred 在 PG 执行成功且 pred_rows == gold_rows
                 (19 条的 gold_rows 用改写后 PG gold;其余 281 条用原 PG gold)
  EX_drop19(281) = 剔除 19 条后的敏感性口径
  McNemar:RGCV(SQLite 侧逐题 EX)vs SafeQL_sym(PG 侧逐题 EX),配对题级;
  paired bootstrap 10^4 次 95% CI。

用法:python eval_safeql_sym.py <tag> [tag2 ...]
  例:eval_safeql_sym.py v4pro kimi v4pro_gated kimi_gated
输出:results/eB_symmetric_summary.json
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

from rgcv.evalx import mcnemar_test, paired_bootstrap  # noqa: E402

RGCV_MAIN = {
    "v4pro": "rgcv_bird300_deepseek-v4-pro_gated_fullschema",
    "kimi": "rgcv_bird300_kimi-k26_gated_fullschema",
}
# SafeQL 复现 tag -> 对应 RGCV 骨干 key(用于 McNemar 配对)
TAG2BB = {"v4pro": "v4pro", "kimi": "kimi",
          "v4pro_gated": "v4pro", "kimi_gated": "kimi"}


def rgcv_ex_map() -> dict:
    """从 eA traces 重建两个主配置的逐题 EX:{bb_key: {qid: bool}}。"""
    ea = json.load(open(RES / "eA_tier2_real_verification.json", encoding="utf-8"))
    out = {}
    for bb, tag in RGCV_MAIN.items():
        m = {}
        for tr in ea["traces"]:
            if tr["tag"] != tag:
                continue
            m[str(tr["question_id"])] = 1 if tr["kind"] == "correct" else 0
        out[bb] = m
    return out


def main():
    tags = sys.argv[1:]
    assert tags, "usage: python eval_safeql_sym.py <v4pro|kimi|v4pro_gated|kimi_gated ...>"
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                         encoding="utf-8"))
    qids = [str(d["question_id"]) for d in dev]          # dev 行序
    q19 = set(json.load(open(RES / "safeql_gold19_fixed.json",
                             encoding="utf-8")).keys())
    rgcv = rgcv_ex_map()
    for bb, m in rgcv.items():
        print(f"RGCV[{bb}] EX 重建: {sum(m.values())}/{len(m)} traces "
              f"-> {sum(m.values())}/300")

    summary = {"note": "E-B symmetric scoring; SafeQL EX on PG17 with "
                       "19 rewritten golds (translate_gold_19.py, dual-run "
                       "verified); RGCV EX from eA per-question replay",
               "rgcv_replay_ex": {}, "tags": {}}
    for bb, m in rgcv.items():
        c19 = sum(1 for q in q19 if m.get(q))
        ex300 = sum(m.values())
        summary["rgcv_replay_ex"][bb] = {
            "tag": RGCV_MAIN[bb], "ex300": ex300, "correct_in_19": c19,
            "ex281": ex300 - c19,
            "ex281_pct": round((ex300 - c19) / (300 - len(q19)) * 100, 2),
        }
    for tag in tags:
        data = json.load(open(RES / f"safeql_bird300_{tag}.json",
                              encoding="utf-8"))
        bb = TAG2BB[tag]
        ex_sym = ex_drop = pred_ok = 0
        per_qid = {}
        for qid in qids:
            rec = data.get(qid, {})
            ok = (rec.get("status") == "ok" and rec.get("pred_rows") is not None
                  and rec.get("gold_rows") is not None
                  and rec["pred_rows"] == rec["gold_rows"])
            per_qid[qid] = int(ok)
            if rec.get("status") == "ok" and rec.get("pred_rows") is not None:
                pred_ok += 1
            if ok:
                ex_sym += 1
            if ok and qid not in q19:
                ex_drop += 1
        n300, n281 = 300, 300 - len(q19)
        rgcv_map = rgcv[bb]
        # McNemar / bootstrap(题级配对,按 dev 行序对齐)
        a = [bool(rgcv_map.get(q, 0)) for q in qids]          # RGCV
        b = [bool(per_qid.get(q, 0)) for q in qids]           # SafeQL
        keep = [i for i, q in enumerate(qids) if q not in q19]
        a281 = [a[i] for i in keep]
        b281 = [b[i] for i in keep]
        mc300 = mcnemar_test(a, b)
        mc281 = mcnemar_test(a281, b281)
        bs300 = paired_bootstrap(a, b)
        bs281 = paired_bootstrap(a281, b281)
        print(f"\n[{tag}] EX_sym={ex_sym}/{n300}={ex_sym/n300*100:.1f}%  "
              f"EX_drop19={ex_drop}/{n281}={ex_drop/n281*100:.1f}%  "
              f"pred_ok={pred_ok}")
        print(f"  RGCV[{bb}]-SafeQL: gap300="
              f"{(sum(a)-sum(b))/3:+.1f}pp  gap281="
              f"{(sum(a281)-sum(b281))*100/n281:+.1f}pp")
        print(f"  McNemar300 b/c={mc300.get('b')}/{mc300.get('c')} "
              f"p={mc300.get('p_value', mc300.get('p'))}  "
              f"bootstrap CI300={bs300['ci95']}")
        print(f"  McNemar281 b/c={mc281.get('b')}/{mc281.get('c')} "
              f"p={mc281.get('p_value', mc281.get('p'))}  "
              f"bootstrap CI281={bs281['ci95']}")
        summary["tags"][tag] = {
            "backbone": bb, "ex_sym": ex_sym, "n": n300,
            "ex_sym_pct": round(ex_sym / n300 * 100, 2),
            "ex_drop19": ex_drop, "n_drop19": n281,
            "ex_drop19_pct": round(ex_drop / n281 * 100, 2),
            "pred_exec_ok": pred_ok,
            "gap_vs_rgcv_300pp": round((sum(a) - sum(b)) / 3, 2),
            "gap_vs_rgcv_281pp": round((sum(a281) - sum(b281)) * 100 / n281, 2),
            "mcnemar_300": mc300, "mcnemar_281": mc281,
            "bootstrap_300": bs300, "bootstrap_281": bs281,
            "per_qid_ex": per_qid,
        }

    out = RES / "eB_symmetric_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
