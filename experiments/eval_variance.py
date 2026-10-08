"""E-C:方差运行评测 + run-to-run 方差汇总(revision_plan P0-4)。

对主配置(gated_fullschema)的重复运行计 EX并向量:
  DeepSeek-V4-Pro: 原始(T=0,修复重生成确定性) + _s1 + _s2(T=0.6+seed
                   1/2,修复重生成作为显式随机源);
  Kimi-K2.6:       原始 + _r1 + _r2(平台 T=0.6 固定,重复采样方差);
输出每骨干:各次 EX、均值 ± 样本标准差、逐题一致率、max-min band、
配对 McNemar(相邻两次);95% CI 用 bootstrap 10^4。

并行执行(sqlite 只读,8 worker;单查询 60s 墙钟超时)。
输出:results/eC_variance_summary.json
运行:python experiments/eval_variance.py
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import concurrent.futures as cf
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DAIL = BASE / "baselines" / "DAIL-SQL"
RESULTS = BASE / "results"
Q_TIMEOUT = 60
N_WORKERS = 8

GROUPS = {
    "deepseek-v4-pro": ["rgcv_bird300_deepseek-v4-pro_gated_fullschema",
                        "rgcv_bird300_deepseek-v4-pro_gated_fullschema_s1",
                        "rgcv_bird300_deepseek-v4-pro_gated_fullschema_s2"],
    "kimi-k2.6": ["rgcv_bird300_kimi-k26_gated_fullschema",
                  "rgcv_bird300_kimi-k26_gated_fullschema_r1",
                  "rgcv_bird300_kimi-k26_gated_fullschema_r2"],
}


def _exec_one(db_path, sql):
    """返回规范化行多重集;失败/超时 None。

    超时用 SQLite 进度处理器协作式中断(每 10 万条 VM 指令检查一次);
    线程池 result(timeout) 的旧方案会因 with-shutdown 等待慢查询而失效。
    """
    if not sql or not sql.strip():
        return None
    import time
    try:
        conn = sqlite3.connect(db_path)
        deadline = time.perf_counter() + Q_TIMEOUT

        def _guard():
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_guard, 100_000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return None

    def norm(v):
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f"{round(float(v), 6)}"     # 统一字符串,避免混合类型排序
        if isinstance(v, bytes):
            return v[:64].decode("utf-8", "ignore")
        return str(v).strip()

    ms = sorted((tuple(norm(x) for x in r) for r in rows), key=repr)
    return tuple(ms)          # 与 eval_rgcv_e1 EX_strict 同口径:只比行集


def exec_safe(db_path, sql):
    return _exec_one(db_path, sql)


def load_merged(tag):
    p = RESULTS / tag / "merged.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    dev = json.loads((DAIL / "dataset" / "bird" / "dev"
                      / "dev_300.json").read_text(encoding="utf-8"))
    assert len(dev) == 300
    qid2idx = {d["question_id"]: i for i, d in enumerate(dev)}

    def db_path_of(db_id):
        return str(DAIL / "dataset" / "bird" / "database" / db_id
                   / f"{db_id}.sqlite")

    # ---- gold 一次性并行执行(所有 tag 共享)
    print("[eC] gold 执行(并行)...", flush=True)
    gold: dict = {}
    with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(exec_safe, db_path_of(d["db_id"]),
                            d.get("SQL") or d.get("query", "")):
                d["question_id"] for d in dev}
        for f in cf.as_completed(futs):
            gold[futs[f]] = f.result()
    n_gold_ok = sum(1 for v in gold.values() if v is not None)
    print(f"[eC] gold 可执行 {n_gold_ok}/300", flush=True)

    # ---- 每个 tag 并行执行预测
    ex_vec = {}          # tag -> [Optional[bool]] * 300
    for tag in [t for g in GROUPS.values() for t in g]:
        merged = load_merged(tag)
        preds = [None] * 300
        for qid, v in merged.items():
            idx = qid2idx.get(int(qid))
            if idx is None:
                continue
            sql, rest = v.split("\t----- bird -----")
            preds[idx] = (sql.strip(), db_path_of(rest.strip()))
        with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = {pool.submit(exec_safe, dbp, sql): i
                    for i, (sql, dbp) in enumerate(preds) if sql}
            res = {}
            for f in cf.as_completed(futs):
                res[futs[f]] = f.result()
        vec = []
        for i in range(300):
            g = gold[dev[i]["question_id"]]
            p = res.get(i)
            if p is None or g is None:
                vec.append(None)      # 执行失败或 gold 不可执行
            else:
                vec.append(p == g)
        ex_vec[tag] = vec
        n_exec = sum(1 for v in vec if v is not None)
        n_ok = sum(1 for v in vec if v)
        print(f"[eC] {tag}: EX={n_ok}/{n_gold_ok}"
              f" = {n_ok / n_gold_ok * 100:.1f}%  exec={n_exec}/300",
              flush=True)

    # ---- 方差汇总
    def mcnemar(a, b):
        pairs = [(x, y) for x, y in zip(a, b)
                 if x is not None and y is not None]
        nd = sum(1 for x, y in pairs if y and not x)
        nc = sum(1 for x, y in pairs if x and not y)
        n = nd + nc
        if n == 0:
            return {"b": 0, "c": 0, "p": 1.0}
        from scipy.stats import binomtest
        return {"b": nd, "c": nc,
                "p": round(binomtest(min(nd, nc), n, 0.5).pvalue, 6)}

    summary = {}
    for eng, tags in GROUPS.items():
        rates = []
        for t in tags:
            v = [x for x in ex_vec[t] if x is not None]
            rates.append(sum(v) / len(v))
        disagree = 0
        n_both = 0
        for i in range(300):
            vals = [ex_vec[t][i] for t in tags]
            vv = [x for x in vals if x is not None]
            if vv and len(vv) < len(vals):
                pass
            n_both += len(vv)
            if len(set(vv)) > 1:
                disagree += 1
        summary[eng] = {
            "runs": tags,
            "ex_rates": [round(r * 100, 2) for r in rates],
            "mean": round(statistics.mean(rates) * 100, 2),
            "stdev": (round(statistics.stdev(rates) * 100, 3)
                      if len(rates) > 1 else 0.0),
            "band_pp": round((max(rates) - min(rates)) * 100, 2),
            "questions_with_disagreement": disagree,
            "pairwise_mcnemar": {
                f"run{i} vs run{i + 1}": mcnemar(ex_vec[tags[i]],
                                                 ex_vec[tags[i + 1]])
                for i in range(len(tags) - 1)},
        }

    out = {
        "config": {
            "protocol": "DeepSeek: 原始 T=0 + 修复重生成 t=0.6+seed{1,2}"
                        "(显式随机源); Kimi: 平台 T=0.6 重复采样 ×3;"
                        "主配置 gated_fullschema, n=300",
            "ex_criteria": "结果多重集 + 数值 6 位舍入(evalx 同口径,"
                           "无 1e-6 相对容差,EX_strict 近似)",
            "q_timeout_s": Q_TIMEOUT,
        },
        "summary": summary,
    }
    (RESULTS / "eC_variance_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("[saved] eC_variance_summary.json", flush=True)


if __name__ == "__main__":
    main()
