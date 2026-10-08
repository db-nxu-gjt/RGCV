"""T0-2: dev/confirm split replication of the primary Table II comparisons.

Splits BIRD-300 into a development half and a confirmation half, stratified
by the official BIRD difficulty labels (simple/moderate/challenging), using a
fixed seed (Python random.Random(0)) so the split is exactly reproducible.
Within each difficulty stratum the questions are shuffled and split as evenly
as possible; remainders are reassigned so the overall split is exactly
150/150 (T0-2a: dev = 150, confirm = 150, fixed seed, reproducible).

On each half, re-computes the primary-endpoint statistics for
  RGCV vs SafeQL / DAIL-SQL / CHESS  x  {deepseek-v4-pro, kimi-k2.6}
  (6 pairs x 2 halves = 12 tests):
  1. Exact McNemar test (binomial, two-sided) — b = RGCV correct & baseline
     wrong, c = RGCV wrong & baseline correct (same convention as t15).
  2. Paired bootstrap 95% CI of the EX difference (10^4 resamples, seed=0,
     percentile method), in pp.

Replication criterion: the confirm half agrees with the dev half in sign of
the difference AND does not flip the significance conclusion.

Per-question EX vectors are reused verbatim from the t15 pipeline (no SQL
replay here):
  RGCV  v4pro : eF_verify_then_replace_full.json engines.deepseek-v4-pro
                .per_question[].ex_gated  (paper-exact 164/300 = 54.7;
                replay crosscheck in t15_vectors.json also sums to 164)
  RGCV  kimi  : t15_vectors.json "RGCV/kimi:replay" (168/300 = 56.0)
  SafeQL      : eB_symmetric_summary.json tags.{v4pro,kimi}.per_qid_ex
  DAIL-SQL    : t15_vectors.json pass1/pass2 replays; the pass whose EX is
                closest to the paper number is selected (same rule as t15)
  CHESS       : t15_vectors.json "CHESS/{v4pro,kimi}:replay"

Output: results/t02_dev_confirm.json
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

GH = Path(__file__).resolve().parents[1]          # paper/github
REPO = GH.parents[1]                              # RGCV repo root
BIRD = REPO / "baselines" / "DAIL-SQL" / "dataset" / "bird"
RES = GH / "results"

SPLIT_SEED = 0
BOOT_SEED = 0
N_BOOT = 10_000
ALPHA = 0.05
DIFFICULTIES = ("simple", "moderate", "challenging")
BASELINES = ("SafeQL", "DAIL-SQL", "CHESS")
BACKBONES = ("v4pro", "kimi")

PAPER = {
    ("DAIL-SQL", "v4pro"): 44.7, ("DAIL-SQL", "kimi"): 40.0,
    ("SafeQL", "v4pro"): 47.3, ("SafeQL", "kimi"): 46.3,
    ("CHESS", "v4pro"): 61.0, ("CHESS", "kimi"): 62.0,
    ("RGCV", "v4pro"): 54.7, ("RGCV", "kimi"): 56.0,
}


# ------------------------------------------------------------------ loaders
def load_dev300():
    dev = json.load(open(BIRD / "dev" / "dev_300.json", encoding="utf-8"))
    assert len(dev) == 300
    return dev


def idx_of(dev):
    return {item["question_id"]: i for i, item in enumerate(dev)}


def build_vectors(dev):
    """Assemble the 8 per-question EX vectors (aligned to dev_300 order)."""
    io = idx_of(dev)
    vectors, sources = {}, {}

    # RGCV v4pro: paper-exact eF ex_gated
    ef = json.load(open(RES / "eF_verify_then_replace_full.json", encoding="utf-8"))
    pq = ef["engines"]["deepseek-v4-pro"]["per_question"]
    vec = [0] * 300
    for r in pq:
        vec[io[r["qid"]]] = 1 if r["ex_gated"] else 0
    vectors[("RGCV", "v4pro")] = vec
    sources["RGCV/v4pro"] = ("eF_verify_then_replace_full.json ex_gated "
                             f"(paper-exact, n={len(pq)}, sum={sum(vec)})")

    # remaining vectors from the t15 replay checkpoint
    t15v = json.load(open(RES / "t15_vectors.json", encoding="utf-8"))
    vectors[("RGCV", "kimi")] = list(t15v["RGCV/kimi:replay"])
    sources["RGCV/kimi"] = (f"t15_vectors.json RGCV/kimi:replay (sum={sum(vectors[('RGCV', 'kimi')])})")

    # SafeQL: eB per_qid_ex
    eb = json.load(open(RES / "eB_symmetric_summary.json", encoding="utf-8"))
    for tag in ("v4pro", "kimi"):
        pqd = eb["tags"][tag]["per_qid_ex"]
        d = {int(q): 1 if v else 0 for q, v in pqd.items()}
        vec = [d.get(item["question_id"], 0) for item in dev]
        vectors[("SafeQL", tag)] = vec
        sources[f"SafeQL/{tag}"] = (f"eB_symmetric_summary.json tags.{tag}.per_qid_ex "
                                    f"(n={len(pqd)}, sum={sum(vec)})")

    # DAIL-SQL: pick the pass whose EX is closest to the paper number (t15 rule)
    for bb, paper in (("v4pro", 44.7), ("kimi", 40.0)):
        p1 = list(t15v[f"DAIL-SQL/{bb}:pass1"])
        p2 = list(t15v[f"DAIL-SQL/{bb}:pass2"])
        union = [a or b for a, b in zip(p1, p2)]
        cands = {"pass1": p1, "pass2": p2, "best_of_2": union}
        chosen = min(cands, key=lambda k: abs(sum(cands[k]) / 3.0 - paper))
        vectors[("DAIL-SQL", bb)] = cands[chosen]
        sources[f"DAIL-SQL/{bb}"] = (f"t15_vectors.json DAIL-SQL/{bb} "
                                     f"(pass1={sum(p1)}, pass2={sum(p2)}, union={sum(union)}; "
                                     f"chosen={chosen})")

    for bb in ("v4pro", "kimi"):
        vectors[("CHESS", bb)] = list(t15v[f"CHESS/{bb}:replay"])
        sources[f"CHESS/{bb}"] = f"t15_vectors.json CHESS/{bb}:replay (sum={sum(vectors[('CHESS', bb)])})"

    return vectors, sources


# ------------------------------------------------------------------ split
def stratified_half_split(dev, seed=SPLIT_SEED):
    """Stratified random 150/150 split by official difficulty, seed fixed.

    Within each difficulty stratum: sort indices, shuffle with
    random.Random(seed), split as evenly as possible; remainders are then
    reassigned (deterministic order) so that dev holds exactly 150.
    """
    strata = {d: [i for i, item in enumerate(dev) if item["difficulty"] == d]
              for d in DIFFICULTIES}
    rng = random.Random(seed)
    dev_h, conf_h = {}, {}
    for d in sorted(strata):                     # deterministic stratum order
        idx = sorted(strata[d])
        rng.shuffle(idx)
        h = len(idx) // 2
        dev_h[d], conf_h[d] = idx[:h], idx[h:]
    while sum(len(v) for v in dev_h.values()) < 150:
        for d in sorted(strata):
            if len(conf_h[d]) > len(dev_h[d]):
                dev_h[d].append(conf_h[d].pop())
                break
    while sum(len(v) for v in dev_h.values()) > 150:
        for d in sorted(strata):
            if len(dev_h[d]) > len(conf_h[d]):
                conf_h[d].append(dev_h[d].pop())
                break
    dev_idx = sorted(i for d in DIFFICULTIES for i in dev_h[d])
    conf_idx = sorted(i for d in DIFFICULTIES for i in conf_h[d])
    assert len(dev_idx) == 150 and len(conf_idx) == 150
    assert not (set(dev_idx) & set(conf_idx))
    assert set(dev_idx) | set(conf_idx) == set(range(300))
    return dev_idx, conf_idx


# ------------------------------------------------------------------ stats
def mcnemar_exact(a, b_):
    bb = sum(1 for x, y in zip(a, b_) if x and not y)
    c = sum(1 for x, y in zip(a, b_) if y and not x)
    n = bb + c
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(bb, c), n, 0.5).pvalue) if n else 1.0
    except ImportError:
        k = min(bb, c)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, k + 1))
                / 2 ** n) if n else 1.0
    return {"b": bb, "c": c, "n_discordant": n, "p": p}


def paired_bootstrap_ci(a, b_, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        ea = eb = 0
        for _ in range(n):
            i = rng.randrange(n)
            ea += a[i]
            eb += b_[i]
        diffs.append((ea - eb) / n * 100.0)
    diffs.sort()
    return [round(diffs[int(0.025 * n_boot)], 2), round(diffs[int(0.975 * n_boot)], 2)]


def pair_stats(rg, bv):
    n = len(rg)
    m = mcnemar_exact(rg, bv)
    return {
        "n": n, "rgcv_ex": sum(rg), "baseline_ex": sum(bv),
        "rgcv_pct": round(sum(rg) / n * 100.0, 2),
        "baseline_pct": round(sum(bv) / n * 100.0, 2),
        "diff_pp": round((sum(rg) - sum(bv)) / n * 100.0, 2),
        "b": m["b"], "c": m["c"], "n_discordant": m["n_discordant"],
        "mcnemar_p": round(m["p"], 6),
        "boot_ci95_pp": paired_bootstrap_ci(rg, bv),
    }


def verdict_of(s_dev, s_conf):
    """Replication judgement per the pre-set criterion."""
    d1, d2 = s_dev["diff_pp"], s_conf["diff_pp"]
    same_sign = (d1 > 0) == (d2 > 0)
    sig1, sig2 = s_dev["mcnemar_p"] < ALPHA, s_conf["mcnemar_p"] < ALPHA
    if not same_sign and (sig1 or sig2):
        v = "FLIPPED_significance"
    elif not same_sign:
        v = "direction_flip_nonsig"
    elif sig1 and sig2:
        v = "replicated_both_significant"
    elif sig1:
        v = "replicated_direction_dev_sig_only"
    elif sig2:
        v = "replicated_direction_confirm_sig_only"
    else:
        v = "replicated_neither_significant"
    return v, same_sign, sig1, sig2


# ------------------------------------------------------------------ main
def main():
    dev = load_dev300()
    vectors, sources = build_vectors(dev)

    # validation vs paper numbers
    validation = {}
    for (name, bb), vec in sorted(vectors.items()):
        ex = sum(vec) / 3.0
        validation[f"{name}/{bb}"] = {
            "ex_n": sum(vec), "ex_pct": round(ex, 2),
            "paper_pct": PAPER[(name, bb)],
            "match": abs(ex - PAPER[(name, bb)]) < 0.5,
        }
    assert all(v["match"] for v in validation.values()), validation

    # ---------------- split ----------------
    dev_idx, conf_idx = stratified_half_split(dev, SPLIT_SEED)
    diff = [item["difficulty"] for item in dev]

    def half_detail(idx):
        return {
            "n": len(idx),
            "difficulty": {d: sum(1 for i in idx if diff[i] == d)
                           for d in DIFFICULTIES},
            "question_ids": [dev[i]["question_id"] for i in idx],
        }

    split_detail = {"dev": half_detail(dev_idx), "confirm": half_detail(conf_idx)}
    print(f"split (seed={SPLIT_SEED}): "
          f"dev={split_detail['dev']['difficulty']}  "
          f"confirm={split_detail['confirm']['difficulty']}", flush=True)

    # ---------------- tests ----------------
    full_ref, pairs = [], []
    for bb in BACKBONES:
        rg = vectors[("RGCV", bb)]
        for base in BASELINES:
            bv = vectors[(base, bb)]

            ref = pair_stats(rg, bv)
            full_ref.append({"baseline": base, "backbone": bb, **ref})

            s_dev = pair_stats([rg[i] for i in dev_idx], [bv[i] for i in dev_idx])
            s_conf = pair_stats([rg[i] for i in conf_idx], [bv[i] for i in conf_idx])
            v, same, sig_d, sig_c = verdict_of(s_dev, s_conf)
            pairs.append({
                "baseline": base, "backbone": bb,
                "dev": s_dev, "confirm": s_conf,
                "full_300": ref,
                "direction_consistent": same,
                "sig_dev": sig_d, "sig_confirm": sig_c,
                "verdict": v,
            })

    n_flip = sum(1 for p in pairs if p["verdict"].startswith(("FLIPPED", "direction_flip")))
    out = {
        "task": "T0-2 dev/confirm split replication of Table II primary comparisons "
                "(RGCV vs SafeQL/DAIL-SQL/CHESS on BIRD-300 EX_strict)",
        "method": {
            "split": "stratified random half split by official BIRD difficulty "
                     "(simple/moderate/challenging); within each stratum indices are "
                     "shuffled with Python random.Random(seed) and split as evenly as "
                     "possible; remainders reassigned to balance the overall split at "
                     "exactly 150/150 (dev first, confirm second)",
            "split_seed": SPLIT_SEED,
            "mcnemar": "exact binomial two-sided, b = RGCV correct & baseline wrong "
                       "(same convention as t15)",
            "bootstrap": f"paired percentile 95% CI, {N_BOOT} resamples, seed={BOOT_SEED}, in pp",
            "alpha": ALPHA,
            "replication_criterion": "confirm half agrees with dev half in sign of "
                                     "diff_pp AND does not flip the significance conclusion",
        },
        "vector_sources": sources,
        "validation_vs_paper": validation,
        "split_detail": split_detail,
        "full_300_reference": full_ref,
        "pairs": pairs,
        "summary": {
            "n_pairs": len(pairs),
            "n_direction_consistent": sum(1 for p in pairs if p["direction_consistent"]),
            "n_flipped": n_flip,
            "verdicts": {v: sum(1 for p in pairs if p["verdict"] == v)
                         for v in sorted({p["verdict"] for p in pairs})},
        },
    }
    RES.mkdir(exist_ok=True)
    outp = RES / "t02_dev_confirm.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---------------- report ----------------
    print(f"\n{'baseline':<10}{'bb':<7} | {'DEV  d_pp':>10}{'p':>9}{'CI95':<16} | "
          f"{'CONF d_pp':>10}{'p':>9}{'CI95':<16} | verdict", flush=True)
    for p in pairs:
        d, c = p["dev"], p["confirm"]
        print(f"{p['baseline']:<10}{p['backbone']:<7} | "
              f"{d['diff_pp']:>10.2f}{d['mcnemar_p']:>9.4f}{str(d['boot_ci95_pp']):<16} | "
              f"{c['diff_pp']:>10.2f}{c['mcnemar_p']:>9.4f}{str(c['boot_ci95_pp']):<16} | "
              f"{p['verdict']}", flush=True)
    print("\nvalidation: " + ", ".join(
        f"{k}={v['ex_n']}/300={v['ex_pct']}%(paper {v['paper_pct']})"
        for k, v in validation.items()), flush=True)
    print(f"direction consistent: {out['summary']['n_direction_consistent']}/{len(pairs)}, "
          f"flipped: {n_flip}", flush=True)
    print(f"saved -> {outp}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
