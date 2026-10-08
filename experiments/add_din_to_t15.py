"""T1-6a 后续：把 DIN-SQL/v4pro 检验并入 t15 显著性族并重算 BH（m=12）。"""
import json

P = r"paper/github/results/t15_significance.json"
t = json.load(open(P, encoding="utf-8"))

# 幂等：先移除已存在的 DIN 行
t["tests"] = [x for x in t["tests"] if x.get("baseline") != "DIN-SQL"]
t["tests"].append({
    "baseline": "DIN-SQL", "backbone": "v4pro",
    "rgcv_ex": 164, "baseline_ex": 171, "diff_pp": -2.33,
    "b": 20, "c": 27, "n_discordant": 47,
    "mcnemar_p": 0.3817, "boot_ci95_pp": None, "primary": False,
    "source": "dinsql_bird300_v4pro.json per_qid_ex aligned to dev_300.json order (T1-6a revision run)",
    "note": "Kimi side not run; bootstrap CI omitted (n.s., not quoted in paper)",
})

tests = sorted(t["tests"], key=lambda x: x["mcnemar_p"])
m = len(tests)
prev = 1.0
qs = [0.0] * m
for i in range(m - 1, -1, -1):
    q = tests[i]["mcnemar_p"] * m / (i + 1)
    prev = min(prev, q)
    qs[i] = prev
for test, q in zip(tests, qs):
    test["bh_q_12family"] = round(q, 4)

t["criteria"] = ("McNemar exact binomial (b = RGCV correct & baseline wrong); "
                 "paired bootstrap 10000 resamples seed=0 percentile CI in pp; "
                 "BH-FDR over the full family of RGCV-vs-baseline tests on both backbones "
                 "(recomputed at m=12 after the T1-6a DIN-SQL addition; original m=11 q values kept in bh_q); "
                 "primary endpoint (pre-registered): RGCV vs SafeQL (v4pro) — included in the family and flagged separately")

json.dump(t, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for test in sorted(t["tests"], key=lambda x: x["mcnemar_p"]):
    print("%-12s %-5s p=%-9s q11=%-8s q12=%s" % (
        test["baseline"], test["backbone"], test["mcnemar_p"],
        test.get("bh_q"), test["bh_q_12family"]))
