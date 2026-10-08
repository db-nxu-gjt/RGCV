"""E-H:强度-收益曲线与 p* 实测(Prop.4)。

三个(或四个)生成器强度点 × {ungated, gated} × BIRD-300:
  ruleproxy          RuleGenerator(确定性模板,弱) — run_eh_ruleproxy.py 产物
  deepseek-v4-flash  中强度 — run_rgcv_e1.py --engine deepseek-v4-flash{,_gated}
  deepseek-v4-pro    旗舰 — 已有 ungated/gated compressed runs
  kimi-k2.6          第二旗舰 — 已有

逐题估计 Prop.4 参数(以候选起点 orig 为条件):
  e   = Pr[orig 不可执行]
  p   = Pr[orig EX=1 | orig 可执行]
  α   = Pr[ungated final EX=0 | orig 可执行且 EX=1]  (correct→wrong)
  β   = Pr[ungated final EX=1 | orig 可执行且 EX=0]  (wrong→correct)
  η   = Pr[ungated final EX=1 | orig 不可执行]        (restore)
  p*  = (β + η·e/(1−e)) / (α + β)
  Δ_u = EX(ungated) − EX(orig)   (实测,与 e·η−(1−e)[pα−(1−p)β] 对照)
  Δ_g = EX(gated) − EX(ungated)  (门控收益,曲线 y 轴)

输出:results/eH_strength_curve.json
用法:python run_eh_strength_curve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"
DAIL = BASE.parents[1] / "baselines" / "DAIL-SQL"
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from eval_rgcv_e1 import run_sql  # noqa: E402
from rgcv.evalx import mcnemar_test  # noqa: E402

# (显示名, ungated tag, gated tag)
POINTS = [
    ("ruleproxy", "ruleproxy", "ruleproxy_gated"),
    ("deepseek-v4-flash", "deepseek-v4-flash", "deepseek-v4-flash_gated"),
    ("deepseek-v4-pro", "deepseek-v4-pro", "deepseek-v4-pro_gated"),
    ("kimi-k2.6", "kimi-k26", "kimi-k26_gated"),
]


def load_run(tag: str, qid2idx: dict) -> dict:
    outdir = RES / f"rgcv_bird300_{tag}"
    recs = {}
    for p in outdir.glob("q_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        qid = str(d.get("question_id"))
        if qid not in qid2idx:
            continue
        err = bool(d.get("error"))
        cands = d.get("candidates") or []
        recs[qid] = {
            "orig": "" if err else (cands[0] if cands else ""),
            "final": "" if err else (d.get("pred_sql") or ""),
            "ex_orig_saved": d.get("ex_orig"),
            "ex_final_saved": d.get("ex_final"),
            "verdict": None if err else (d.get("verification") or {})
            .get("verdict"),
        }
    return recs


def main():
    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                         encoding="utf-8"))
    qid2idx = {str(d["question_id"]): i for i, d in enumerate(dev)}
    DBROOT = DAIL / "dataset" / "bird" / "database"

    gold_cache = {}
    for i, item in enumerate(dev):
        db_id = item["db_id"]
        db_path = str(DBROOT / db_id / f"{db_id}.sqlite")
        gold_cache[i] = run_sql(db_path, item.get("SQL")
                                or item.get("query", ""))

    summary = {"note": "E-H strength-gain curve + p* measurement (Prop.4)",
               "points": {}}
    for name, tag_u, tag_g in POINTS:
        ung = load_run(tag_u, qid2idx)
        gat = load_run(tag_g, qid2idx)
        if not ung or not gat:
            print(f"[{name}] 缺数据 ung={len(ung)} gat={len(gat)} — 跳过")
            continue
        a_o, a_u, a_g = [], [], []
        n_exec = n_exec_corr = 0
        n_alpha = n_alpha_hit = 0          # 可执行且 orig 对 → final 错
        n_beta = n_beta_hit = 0            # 可执行且 orig 错 → final 对
        n_eta = n_eta_hit = 0              # 不可执行 → final 对
        for i, item in enumerate(dev):
            qid = str(item["question_id"])
            db_path = str(DBROOT / item["db_id"]
                          / f"{item['db_id']}.sqlite")
            gold_res = gold_cache[i]
            u = ung.get(qid, {"orig": "", "final": ""})
            g = gat.get(qid, {"final": ""})
            exo_saved = u.get("ex_orig_saved")
            if exo_saved is not None and u["orig"]:
                exo = int(exo_saved)
            else:
                exo = int(run_sql(db_path, u["orig"]) == gold_res
                          if gold_res is not None else False)
            exf_saved = u.get("ex_final_saved")
            if exf_saved is not None and u["final"]:
                exu = int(exf_saved)
            else:
                exu = int(run_sql(db_path, u["final"]) == gold_res
                          if gold_res is not None else False)
            exg = int(run_sql(db_path, g["final"]) == gold_res
                      if gold_res is not None else False)
            a_o.append(bool(exo))
            a_u.append(bool(exu))
            a_g.append(bool(exg))

            orig_exec = bool(u["orig"]) and \
                run_sql(db_path, u["orig"]) is not None
            if orig_exec:
                n_exec += 1
                if exo:
                    n_exec_corr += 1
                    n_alpha += 1
                    n_alpha_hit += int(not exu)
                else:
                    n_beta += 1
                    n_beta_hit += int(exu)
            else:
                n_eta += 1
                n_eta_hit += int(exu)

        n = len(a_o)
        e_hat = n_eta / n if n else 0.0
        p_hat = (n_exec_corr / n_exec) if n_exec else 0.0
        alpha = (n_alpha_hit / n_alpha) if n_alpha else 0.0
        beta = (n_beta_hit / n_beta) if n_beta else 0.0
        eta = (n_eta_hit / n_eta) if n_eta else 0.0
        p_star = ((beta + eta * e_hat / max(1 - e_hat, 1e-9))
                  / (alpha + beta)) if (alpha + beta) > 0 else None
        # Δ_u 预测(式 8)
        du_pred = (e_hat * eta
                   - (1 - e_hat) * (p_hat * alpha - (1 - p_hat) * beta))
        mc_gu = mcnemar_test(a_g, a_u)
        mc_uo = mcnemar_test(a_u, a_o)
        rec = {
            "ungated_tag": f"rgcv_bird300_{tag_u}",
            "gated_tag": f"rgcv_bird300_{tag_g}", "n": n,
            "ex_orig": sum(a_o), "ex_ungated": sum(a_u),
            "ex_gated": sum(a_g),
            "pct_orig": round(sum(a_o) / n * 100, 2),
            "pct_ungated": round(sum(a_u) / n * 100, 2),
            "pct_gated": round(sum(a_g) / n * 100, 2),
            "delta_u_pp": round((sum(a_u) - sum(a_o)) / n * 100, 2),
            "delta_g_pp": round((sum(a_g) - sum(a_u)) / n * 100, 2),
            "params": {
                "e": round(e_hat, 4), "p": round(p_hat, 4),
                "alpha": round(alpha, 4), "beta": round(beta, 4),
                "eta": round(eta, 4),
                "n_exec": n_exec, "n_exec_correct": n_exec_corr,
                "n_alpha": n_alpha, "n_beta": n_beta, "n_eta": n_eta,
            },
            "p_star": round(p_star, 4) if p_star is not None else None,
            "delta_u_predicted": round(du_pred, 4),
            "gate_beyond_p_star": (p_hat > p_star) if p_star is not None
            else None,
            "mcnemar_gated_vs_ungated": mc_gu,
            "mcnemar_ungated_vs_orig": mc_uo,
        }
        summary["points"][name] = rec
        print(f"\n[{name}] n={n}")
        print(f"  EX orig={rec['pct_orig']} ungated={rec['pct_ungated']} "
              f"gated={rec['pct_gated']}")
        print(f"  Δ_u={rec['delta_u_pp']}pp (预测 {du_pred:+.3f})  "
              f"Δ_g(门控收益)={rec['delta_g_pp']}pp  "
              f"McNemar(g,u) p={mc_gu.get('p_value')}")
        print(f"  e={e_hat:.3f} p={p_hat:.3f} α={alpha:.3f} β={beta:.3f} "
              f"η={eta:.3f}  →  p*={p_star if p_star is None else round(p_star, 3)}  "
              f"p>p*? {rec['gate_beyond_p_star']}")

    out = RES / "eH_strength_curve.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
