"""T15: Table II (BIRD-300) RGCV vs baselines — paired significance tests.

Per backbone (deepseek-v4-pro / kimi-k2.6), for each baseline with per-question
data, computes:
  1. Exact McNemar test (binomial, two-sided) — b = RGCV correct & baseline
     wrong, c = RGCV wrong & baseline correct (same convention as rgcv.evalx).
  2. Paired bootstrap 95% CI of EX difference (10^4 resamples, seed=0,
     percentile method) — same as rgcv.evalx.paired_bootstrap, in pp.
  3. Benjamini-Hochberg FDR over the whole family (all RGCV-vs-baseline tests
     on both backbones). Primary endpoint (pre-registered): RGCV vs SafeQL;
     it is included in the family AND reported separately.

Per-question EX vectors (300-length, aligned to dev_300.json order, 0/1,
missing/failed prediction = 0):
  RGCV  v4pro : eF_verify_then_replace_full.json  engines.deepseek-v4-pro
                .per_question[].ex_gated  (paper-exact 164/300 = 54.7)
  RGCV  kimi  : offline replay of results/rgcv_bird300_kimi-k26_gated_fullschema
                /merged.json with eval_rgcv_e1.run_sql (60s timeout)
  SafeQL      : eB_symmetric_summary.json tags.{v4pro,kimi}.per_qid_ex
                (PG17 symmetric scoring, paper-exact 47.33 / 46.33)
  DAIL-SQL    : offline replay of E5 RESULTS_MODEL-*.txt (pass1/pass2; the
                pass matching the paper number is selected per backbone)
  ReFoRCE     : offline replay of reforce_bird300_{v4pro,kimi}_results.json
                (keys "local_BIRD_%04d" -> question_id)
  MAC-SQL     : v4pro  replay of src/rgcv_repro/results/MACSQL_results.jsonl
                kimi   replay of E5 macsql_bird300_kimi_predictions.json
  CHESS       : offline replay of chess_bird300_{v4pro,kimi}_merged.json
  SQLCoder    : offline replay of SQLCODER_results_300.jsonl (v4pro only;
                kimi column is "---" in Table II)

Replay criteria are byte-identical to eval_rgcv_e1.run_sql (SQLite, sorted
result-set equality, 60s soft timeout); pred sets are replayed in parallel
worker processes for wall-time only — per-question verdicts are unaffected.

Output: results/t15_significance.json  (+ resumable results/t15_vectors.json)
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path

GH = Path(__file__).resolve().parents[1]          # paper/github
REPO = GH.parents[1]                              # RGCV repo root
BIRD = REPO / "baselines" / "DAIL-SQL" / "dataset" / "bird"
E5 = REPO / "preResearch" / "doc" / "exp_results" / "E5_baselines"
SRC_RES = REPO / "src" / "rgcv_repro" / "results"
RES = GH / "results"

Q_TIMEOUT = 60
N_BOOT = 10_000
PRIMARY = ("SafeQL", "v4pro")
VECTORS_FILE = RES / "t15_vectors.json"

PAPER = {
    ("DAIL-SQL", "v4pro"): 44.7, ("DAIL-SQL", "kimi"): 40.0,
    ("SafeQL", "v4pro"): 47.3, ("SafeQL", "kimi"): 46.3,
    ("ReFoRCE", "v4pro"): 58.3, ("ReFoRCE", "kimi"): 59.3,
    ("MAC-SQL", "v4pro"): 58.0, ("MAC-SQL", "kimi"): 57.7,
    ("CHESS", "v4pro"): 61.0, ("CHESS", "kimi"): 62.0,
    ("SQLCoder-7B-2", "v4pro"): 28.0,
    ("RGCV", "v4pro"): 54.7, ("RGCV", "kimi"): 56.0,
}


# ------------------------------------------------------------------ run_sql
def _exec(db_path, sql):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return sorted(tuple(r) for r in rows)
    except Exception:
        return None


def run_sql(db_path, sql):
    """eval_rgcv_e1.py criteria: 60s timeout, sorted-tuple result set.

    Verdict semantics are identical to eval_rgcv_e1 (thread + result(60s)):
    a query not finished within 60s -> None. The only change is that the
    abandoned thread is a daemon (join instead of shutdown-wait), so a timed
    -out query does not stall the pipeline for its full (possibly hours-long)
    execution; its verdict was already None at the 60s mark either way.
    """
    if not sql or not sql.strip():
        return None
    out = {}
    t = threading.Thread(target=lambda: out.__setitem__("r", _exec(db_path, sql)),
                         daemon=True)
    t.start()
    t.join(Q_TIMEOUT)
    if t.is_alive():          # timed out: verdict None (same as paper eval)
        return None
    return out.get("r")       # finished (may still be None on exec error)


class GoldCache:
    """Lazy per-(db_id, gold_sql) cache; identical verdicts to a flat cache."""

    def __init__(self):
        self.by_db = {}
        self.slow = []

    def get(self, db_id, gold_sql):
        d = self.by_db.setdefault(db_id, {})
        if gold_sql not in d:
            db_path = str(BIRD / "database" / db_id / f"{db_id}.sqlite")
            t0 = time.time()
            d[gold_sql] = run_sql(db_path, gold_sql)
            dt = time.time() - t0
            if dt > 5:
                self.slow.append((db_id, dt, "GOLD"))
        return d[gold_sql]


# ------------------------------------------------------------------ replay
def replay_task(name, jobs, dev):
    """jobs: list of (idx, db_id, gold_sql, pred_sql). Returns (name, ex_vec)."""
    gcx = GoldCache()
    ex = [0] * 300
    t0 = time.time()
    for k, (i, db_id, gold_sql, pred_sql) in enumerate(jobs):
        tq = time.time()
        gold_res = gcx.get(db_id, gold_sql)
        pred_res = run_sql(str(BIRD / "database" / db_id / f"{db_id}.sqlite"),
                           pred_sql) if pred_sql else None
        dt = time.time() - tq
        if dt > 5:
            gcx.slow.append((db_id, round(dt, 1), f"q{dev[i]['question_id']}"))
        if pred_res is not None and gold_res is not None and pred_res == gold_res:
            ex[i] = 1
        if (k + 1) % 50 == 0:
            print(f"  [{name}] {k + 1}/300  elapsed={time.time() - t0:.0f}s",
                  flush=True)
    slow = sorted(gcx.slow, key=lambda x: -x[1])[:8]
    print(f"[END] {name}: ex={sum(ex)}/300  elapsed={time.time() - t0:.0f}s  "
          f"slow(>5s)={slow}", flush=True)
    return name, ex


# ------------------------------------------------------------------ loaders
def load_dev300():
    dev = json.load(open(BIRD / "dev" / "dev_300.json", encoding="utf-8"))
    assert len(dev) == 300
    return dev


def idx_of(dev):
    return {item["question_id"]: i for i, item in enumerate(dev)}


def qid_vec(pairs):
    d = {}
    for q, v in pairs:
        d[int(q)] = 1 if v else 0
    return d


# each loader returns (name -> dict of per-question sql by dev_300 idx)
def preds_rgcv_kimi(dev):
    merged = json.load(open(RES / "rgcv_bird300_kimi-k26_gated_fullschema" / "merged.json",
                            encoding="utf-8"))
    return {"RGCV/kimi:replay": {i: merged.get(str(item["question_id"]), "")
                                 for i, item in enumerate(dev)}}


def preds_rgcv_v4pro_crosscheck(dev):
    merged = json.load(open(RES / "rgcv_bird300_deepseek-v4-pro_gated_fullschema" / "merged.json",
                            encoding="utf-8"))
    return {"RGCV/v4pro:replay": {i: merged.get(str(item["question_id"]), "")
                                  for i, item in enumerate(dev)}}


def preds_dail(dev):
    files = {
        "DAIL-SQL/v4pro:pass1": E5 / "DAIL_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_RESULTS_MODEL-deepseek-v4-pro.txt",
        "DAIL-SQL/v4pro:pass2": E5 / "DAIL_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_RESULTS_MODEL-deepseek-v4-pro.txt",
        "DAIL-SQL/kimi:pass1": E5 / "DAIL_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_RESULTS_MODEL-kimi-k2.6.txt",
        "DAIL-SQL/kimi:pass2": E5 / "DAIL_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_KIMI_RESULTS_MODEL-kimi-k2.6.txt",
    }
    out = {}
    for name, f in files.items():
        lines = [l.strip() for l in open(f, encoding="utf-8").read().splitlines()]
        assert len(lines) == 300, f"{f.name}: {len(lines)} lines"
        out[name] = dict(enumerate(lines))
    return out


def preds_reforce(dev):
    out = {}
    for bb, f in (("v4pro", E5 / "reforce_bird300_v4pro_results.json"),
                  ("kimi", E5 / "reforce_bird300_kimi_results.json")):
        d = json.load(open(f, encoding="utf-8"))
        assert len(d) == 300, f"{f.name}: {len(d)}"
        qmap = {}
        for k, sql in d.items():
            qmap[int(re.match(r"local_BIRD_(\d+)", k).group(1))] = sql
        assert set(qmap) == {item["question_id"] for item in dev}, f.name
        out[f"ReFoRCE/{bb}:replay"] = {i: qmap[item["question_id"]]
                                       for i, item in enumerate(dev)}
    return out


def preds_macsql(dev):
    """Shard jsonl (idx = dev_300 order, pred field) — same loader as
    eval_macsql_e1.py; dirs live under src/rgcv_repro/results."""
    out = {}
    for bb, tag in (("v4pro", "macsql_bird300_deepseek-v4-pro"),
                    ("kimi", "macsql_bird300_kimi-k2.6")):
        rows = {}
        for f in sorted((SRC_RES / tag).glob("shard*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    o = json.loads(line)
                    rows[o["idx"]] = o.get("pred", "")
        assert len(rows) == 300, f"{tag}: {len(rows)}"
        out[f"MAC-SQL/{bb}:replay"] = {i: rows.get(i, "") for i in range(300)}
    return out


def preds_chess(dev):
    out = {}
    for bb, f in (("v4pro", E5 / "chess_bird300_v4pro_merged.json"),
                  ("kimi", E5 / "chess_bird300_kimi_merged.json")):
        d = json.load(open(f, encoding="utf-8"))
        preds = {}
        for i, item in enumerate(dev):
            raw = d.get(str(item["question_id"]))
            preds[i] = (raw.split("\t----- bird -----")[0].strip().rstrip(";")
                        if isinstance(raw, str) and raw else "")
        out[f"CHESS/{bb}:replay"] = preds
    return out


def preds_sqlcoder(dev):
    recs = [json.loads(l) for l in open(E5 / "SQLCODER_results_300.jsonl", encoding="utf-8")]
    assert len(recs) == 300
    io = idx_of(dev)
    return {"SQLCoder-7B-2/v4pro:replay": {io[r["question_id"]]: r["pred"] for r in recs}}


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


def paired_bootstrap_ci(a, b_, n_boot=N_BOOT, seed=0):
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


def bh_fdr(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        q[i] = val
        prev = val
    return [round(v, 6) for v in q]


# ------------------------------------------------------------------ main
def build_vectors(dev, workers):
    """Replay all pred sets in parallel; checkpoint to VECTORS_FILE."""
    if VECTORS_FILE.exists():
        done = json.load(open(VECTORS_FILE, encoding="utf-8"))
        print(f"resuming: {len(done)} vectors already computed", flush=True)
    else:
        done = {}

    sets = {}
    sets.update(preds_dail(dev))
    sets.update(preds_reforce(dev))
    sets.update(preds_macsql(dev))
    sets.update(preds_chess(dev))
    sets.update(preds_sqlcoder(dev))
    sets.update(preds_rgcv_kimi(dev))
    sets.update(preds_rgcv_v4pro_crosscheck(dev))

    todo = {k: v for k, v in sets.items() if k not in done}
    print(f"replay sets: {len(sets)} total, {len(todo)} to run, "
          f"workers={workers}", flush=True)

    tasks = []
    for name, pred_map in todo.items():
        jobs = [(i, item["db_id"], item.get("SQL") or item.get("query", ""),
                 pred_map.get(i, "")) for i, item in enumerate(dev)]
        tasks.append((name, jobs, dev))

    if todo:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(replay_task, n, j, d): n for n, j, d in tasks}
            for fut in cf.as_completed(futures):
                name, ex = fut.result()
                done[name] = ex
                json.dump(done, open(VECTORS_FILE, "w", encoding="utf-8"))
                print(f"checkpoint: {name} -> {VECTORS_FILE.name}", flush=True)
    return done


def main():
    workers = int(os.environ.get("T15_WORKERS", "6"))
    dev = load_dev300()
    t_all = time.time()

    replayed = build_vectors(dev, workers)

    # ---------------- assemble per-method vectors ----------------
    vectors, sources = {}, {}
    pick = {}

    # RGCV v4pro: paper-exact eF ex_gated
    ef = json.load(open(RES / "eF_verify_then_replace_full.json", encoding="utf-8"))
    pq = ef["engines"]["deepseek-v4-pro"]["per_question"]
    vec = [0] * 300
    for r in pq:
        vec[idx_of(dev)[r["qid"]]] = 1 if r["ex_gated"] else 0
    vectors[("RGCV", "v4pro")] = vec
    sources["RGCV/v4pro"] = f"eF_verify_then_replace_full.json ex_gated (n={len(pq)}, sum={sum(vec)})"
    sources["RGCV/v4pro_replay_crosscheck"] = f"replay merged.json -> {sum(replayed['RGCV/v4pro:replay'])}/300"

    vec = replayed["RGCV/kimi:replay"]
    vectors[("RGCV", "kimi")] = vec
    sources["RGCV/kimi"] = "replay of rgcv_bird300_kimi-k26_gated_fullschema/merged.json"

    # SafeQL: eB per_qid_ex
    eb = json.load(open(RES / "eB_symmetric_summary.json", encoding="utf-8"))
    for tag in ("v4pro", "kimi"):
        pq = eb["tags"][tag]["per_qid_ex"]
        d = qid_vec(pq.items())
        vec = [d.get(item["question_id"], 0) for item in dev]
        vectors[("SafeQL", tag)] = vec
        sources[f"SafeQL/{tag}"] = f"eB_symmetric_summary.json tags.{tag}.per_qid_ex (n={len(pq)}, sum={sum(vec)})"

    # DAIL-SQL: pick pass matching paper
    for bb, paper in (("v4pro", 44.7), ("kimi", 40.0)):
        p1 = replayed[f"DAIL-SQL/{bb}:pass1"]
        p2 = replayed[f"DAIL-SQL/{bb}:pass2"]
        union = [a or b for a, b in zip(p1, p2)]
        cands = {"pass1": p1, "pass2": p2, "best_of_2": union}
        pick[bb] = {k: sum(v) for k, v in cands.items()}
        chosen = min(cands, key=lambda k: abs(sum(cands[k]) / 3.0 - paper))
        vectors[("DAIL-SQL", bb)] = cands[chosen]
        sources[f"DAIL-SQL/{bb}"] = f"replay E5 DAIL {chosen} (pass1={sum(p1)}, pass2={sum(p2)}, union={sum(union)})"

    for base, prefix in (("ReFoRCE", "ReFoRCE"), ("MAC-SQL", "MAC-SQL"),
                         ("CHESS", "CHESS")):
        for bb in ("v4pro", "kimi"):
            vec = replayed[f"{prefix}/{bb}:replay"]
            vectors[(base, bb)] = vec
            sources[f"{base}/{bb}"] = f"offline replay (see t15_vectors.json key {prefix}/{bb}:replay)"
    vec = replayed["SQLCoder-7B-2/v4pro:replay"]
    vectors[("SQLCoder-7B-2", "v4pro")] = vec
    sources["SQLCoder-7B-2/v4pro"] = "offline replay of SQLCODER_results_300.jsonl"

    # ---------------- validation vs paper ----------------
    validation = {}
    for (name, bb), vec in sorted(vectors.items()):
        ex = sum(vec) / 3.0
        validation[f"{name}/{bb}"] = {
            "ex_n": sum(vec), "ex_pct": round(ex, 2),
            "paper_pct": PAPER.get((name, bb)),
            "match": abs(ex - PAPER[(name, bb)]) < 0.5 if (name, bb) in PAPER else None,
        }

    # ---------------- tests ----------------
    tests = []
    for bb in ("v4pro", "kimi"):
        rg = vectors[("RGCV", bb)]
        for base in ("DAIL-SQL", "SafeQL", "ReFoRCE", "MAC-SQL", "CHESS",
                     "SQLCoder-7B-2"):
            if (base, bb) not in vectors:
                continue
            bv = vectors[(base, bb)]
            m = mcnemar_exact(rg, bv)
            ci = paired_bootstrap_ci(rg, bv)
            tests.append({
                "baseline": base, "backbone": bb,
                "rgcv_ex": sum(rg), "baseline_ex": sum(bv),
                "diff_pp": round((sum(rg) - sum(bv)) / 3.0, 2),
                "b": m["b"], "c": m["c"], "n_discordant": m["n_discordant"],
                "mcnemar_p": round(m["p"], 6),
                "mcnemar_p_exact": m["p"],
                "boot_ci95_pp": ci,
                "primary": (base, bb) == PRIMARY,
            })
    for t, q in zip(tests, bh_fdr([t["mcnemar_p"] for t in tests])):
        t["bh_q"] = q

    out = {
        "task": "T15 Table II paired significance (RGCV vs baselines, BIRD-300 EX_strict)",
        "criteria": "McNemar exact binomial (b = RGCV correct & baseline wrong); "
                    f"paired bootstrap {N_BOOT} resamples seed=0 percentile CI in pp; "
                    "BH-FDR over the full family of RGCV-vs-baseline tests on both backbones; "
                    "primary endpoint (pre-registered): RGCV vs SafeQL (v4pro) — included "
                    "in the family and flagged separately",
        "sources": sources,
        "dail_pass_selection": pick,
        "validation_vs_paper": validation,
        "tests": tests,
    }
    RES.mkdir(exist_ok=True)
    outp = RES / "t15_significance.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\n{'baseline':<16}{'bb':<7}{'b':>4}{'c':>4}{'p':>10}  "
          f"{'CI95 pp':<18}{'q':>8}", flush=True)
    for t in tests:
        star = " *" if t["primary"] else ""
        print(f"{t['baseline']:<16}{t['backbone']:<7}{t['b']:>4}{t['c']:>4}"
              f"{t['mcnemar_p']:>10.4f}  {str(t['boot_ci95_pp']):<18}"
              f"{t['bh_q']:>8.4f}{star}", flush=True)
    print("\nvalidation: " + ", ".join(
        f"{k}={v['ex_n']}/300={v['ex_pct']}%(paper {v['paper_pct']})"
        for k, v in validation.items()), flush=True)
    print(f"total elapsed: {time.time() - t_all:.0f}s\nsaved -> {outp}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
