"""E-D-a:BIRD 污染改写稳健性测试 —— 分层抽 150 题做语义保持改写。

协议(revision_plan P0-5 / E-D-a):
  - 从 dev_300 按难度分层抽样 150 题(最大余数法,约 simple 90 / moderate 46
    / challenging 14,固定种子 0 可复现);
  - 每题 1 个语义保持改写:同义替换/语序调整/句式重组,仅改措辞,不改
    任何实体、数值、时间窗、比较方向与超/序数词(gold 与答案不变);
    用 DeepSeek-V4-Pro(t=0,确定性)辅助改写;
  - 自动守卫:改写前后 \\d+ 数字 token 多重集必须一致,否则标记
    auto_reject(不进入重跑集,留待人工处理);
  - 人工审核列:eD_paraphrase_150.csv(equivalent y/n + 备注),审核结果
    可在重跑前修订 manifest 的 status。
  - 口径说明:计划原文允许"数值时间窗平移",但平移会改变 gold 答案,与
    "EX 对原 gold"的差值分布协议冲突;本实现限定为纯文本改写(标准污染
    探针),如需平移需同步改 gold,另行扩展。

用法:python make_paraphrase_150.py [--smoke N]
输出:
  results/paraphrase_150.json        {qid: {question_para, evidence_para,
                                      status, difficulty, db_id, ...}}
  results/eD_paraphrase_150.csv      人工审核表
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]              # src/rgcv_repro/
RES = BASE / "results"
DAIL = BASE / "baselines" / "DAIL-SQL"
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from rgcv.llm import LLMClient  # noqa: E402

SEED = 0
N_TARGET = 150

SYSTEM = ("You are a meticulous paraphraser used for benchmark "
          "contamination probing. You rewrite a question WITHOUT changing "
          "its meaning in any way that could alter the correct answer.")

USER_TMPL = """Rewrite the question below in different words. Keep the SAME database question, the SAME answer, and the SAME level of difficulty.

Hard rules:
- Preserve ALL facts exactly: entity names, numbers, years, dates, units, currencies, comparison directions (>=, <=), superlatives (highest/lowest/second), ordinals (333rd), counting targets.
- Change only wording: synonyms, sentence structure, word order, voice.
- Do NOT answer the question. Do NOT add or remove constraints. Do NOT simplify away any condition.
- Keep it a natural, fluent database question.

Question: {question}
{evidence_block}
Output format (exactly two lines, no extra text):
QUESTION: <rewritten question>
EVIDENCE: <rewritten evidence, or NONE if no evidence>
"""


def digits_of(s: str) -> Counter:
    return Counter(re.findall(r"\d+", s or ""))


def parse_llm(text: str):
    q = e = None
    for line in text.splitlines():
        ls = line.strip()
        if ls.upper().startswith("QUESTION:"):
            q = ls[len("QUESTION:"):].strip()
        elif ls.upper().startswith("EVIDENCE:"):
            e = ls[len("EVIDENCE:"):].strip()
    return q, e


def stratified_sample(dev, n_target, seed):
    rng = random.Random(seed)
    by_diff = {}
    for d in dev:
        by_diff.setdefault(d.get("difficulty", "simple"), []).append(d)
    # 最大余数法分配名额
    n_total = len(dev)
    quotas = {k: n_target * len(v) // n_total for k, v in by_diff.items()}
    rem = n_target - sum(quotas.values())
    fracs = sorted(by_diff, key=lambda k: -(n_target * len(by_diff[k]) / n_total
                                            % 1))
    for k in fracs[:rem]:
        quotas[k] += 1
    picked = []
    for k, v in sorted(by_diff.items()):
        idx = list(range(len(v)))
        rng.shuffle(idx)
        picked.extend(v[i] for i in idx[:quotas[k]])
    return picked, quotas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    dev = json.load(open(DAIL / "dataset" / "bird" / "dev" / "dev_300.json",
                         encoding="utf-8"))
    picked, quotas = stratified_sample(dev, N_TARGET, SEED)
    print(f"分层配额: {quotas} -> {len(picked)} 题")
    if a.smoke:
        picked = picked[:a.smoke]
        print(f"[smoke] 前 {len(picked)} 题")

    out_json = RES / "paraphrase_150.json"
    manifest = json.load(open(out_json, encoding="utf-8")) if out_json.exists() else {}

    client = LLMClient("deepseek-v4-pro",
                       trace_path=str(RES / "api_trace_paraphrase.jsonl"))
    n_pass = n_reject = 0
    t0 = time.perf_counter()
    for i, item in enumerate(picked):
        qid = str(item["question_id"])
        if qid in manifest and manifest[qid].get("status") in ("auto_pass",):
            n_pass += 1
            continue
        question = item["question"]
        evidence = item.get("evidence", "") or ""
        ev_block = f"External knowledge: {evidence}\n" if evidence else ""
        text, _, _ = client.chat(
            "paraphrase", SYSTEM,
            USER_TMPL.format(question=question, evidence_block=ev_block),
            max_tokens=1024, temperature=0)
        q_para, e_para = parse_llm(text)
        status, reason = "auto_reject", ""
        if not q_para:
            reason = "llm_empty"
        else:
            e_para = "" if (not e_para or e_para.upper() == "NONE") else e_para
            if digits_of(question) != digits_of(q_para):
                reason = "digit_mismatch_question"
            elif evidence and digits_of(evidence) != digits_of(e_para):
                reason = "digit_mismatch_evidence"
            elif len(q_para) < 0.4 * len(question):
                reason = "too_short_suspicious"
            else:
                status, reason = "auto_pass", "digit_guard_ok"
        manifest[qid] = {
            "question_id": item["question_id"], "db_id": item["db_id"],
            "difficulty": item.get("difficulty", ""),
            "question_orig": question, "evidence_orig": evidence,
            "question_para": q_para or "", "evidence_para": e_para or "",
            "status": status, "reason": reason,
        }
        if status == "auto_pass":
            n_pass += 1
        else:
            n_reject += 1
        if (i + 1) % 10 == 0 or i + 1 == len(picked):
            json.dump(manifest, open(out_json, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"[{i+1}/{len(picked)}] pass={n_pass} reject={n_reject} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        time.sleep(0.2)

    json.dump(manifest, open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 人工审核 CSV(仅本次抽样集合)
    csv_path = RES / "eD_paraphrase_150.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "db_id", "difficulty", "status", "reason",
                    "question_orig", "question_para",
                    "evidence_orig", "evidence_para",
                    "equivalent(y/n)", "reviewer_note"])
        for item in picked:
            qid = str(item["question_id"])
            m = manifest.get(qid, {})
            w.writerow([qid, m.get("db_id", ""), m.get("difficulty", ""),
                        m.get("status", ""), m.get("reason", ""),
                        m.get("question_orig", ""), m.get("question_para", ""),
                        m.get("evidence_orig", ""), m.get("evidence_para", ""),
                        "", ""])
    print(f"manifest: {out_json}\n审核表: {csv_path}\n"
          f"auto_pass={n_pass} auto_reject={n_reject}")


if __name__ == "__main__":
    main()
