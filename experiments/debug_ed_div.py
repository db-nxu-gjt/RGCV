"""调试 T1-2c divergence=0:抽一道 SC 告警题手动算 hash。"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from common import RESULTS, bird_executor  # noqa: E402
from rgcv.verifier import _result_hash  # noqa: E402

ec = json.loads((RESULTS / "ec_verification_baselines.json").read_text(encoding="utf-8"))
# 找一道 sc_alarm 且 clusters>=2 的 ds 题
qid = None
for t in ec["traces"]:
    if t["engine"] == "deepseek-v4-pro" and t["sc_alarm"] and t["sc_clusters"] >= 2:
        qid = int(t["question_id"])
        break
print("qid:", qid)
r = json.loads((RESULTS / "ec_baselines_deepseek-v4-pro/responses" / f"q{qid}.json")
               .read_text(encoding="utf-8"))
pred = r["pred_sql"]
print("pred_sql:", pred[:100])
print("sampled_sqls:")
for i, s in enumerate(r["sampled_sqls"]):
    print(f"  [{i}]", (s or "<EMPTY>")[:100])
ex = bird_executor(r["db_id"])
sqls = [pred] + r["sampled_sqls"]
for i, s in enumerate(sqls):
    try:
        cols, rows, n = ex.execute(s)
        print(f"  [{i}] hash={_result_hash(cols, rows)} rows={n}")
    except Exception as e:
        print(f"  [{i}] EXEC_ERR: {type(e).__name__}: {str(e)[:120]}")
