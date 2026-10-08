"""E-E: DAIL-SQL 全量 BIRD dev (1534) 运行准备与分片。

--stage prep:
  1. 备份当前 dev/dev.json(300 预处理版) → dev_300_preprocessed.json
  2. 读源 data/bird/dev_20240627/dev.json 全量,应用与 dailsql_manual_preprocess
     完全相同的预处理(evidence 并入 question / question_toks / query=SQL) → dev/dev.json
  3. sanity: 题数 / tables / database sqlite 覆盖检查
--stage shard:
  把 _FULL process 目录的 questions.json 复制进 4 个 SHARD 目录;
  ask_llm 以 --start_index/--end_index 取范围,各分片独立写 RESULTS 文件,
  完成后按 SHARD0..3 顺序拼接。
"""
import argparse
import json
import math
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
BIRD = DAIL / "dataset" / "bird"
DEV_DIR = BIRD / "dev"
SRC_FULL = BASE / "data" / "bird" / "dev_20240627" / "dev.json"
P1DIR = DAIL / "dataset" / "process" / \
    "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096_FULL"
N_SHARD = 4


def preprocess(items):
    out = []
    for item in items:
        # evidence 并入 question(同 dailsql_manual_preprocess / DAIL-SQL with_evidence)
        if item.get("evidence") and len(item["evidence"]) > 0:
            item["question"] = (item["question"] + " " + item["evidence"]).strip()
        tokens = []
        for token in item["question"].split(" "):
            if len(token) == 0:
                continue
            if token[-1] in ['?', '.', ':', ';', ','] and len(token) > 1:
                tokens.extend([token[:-1], token[-1:]])
            else:
                tokens.append(token)
        item["question_toks"] = tokens
        item["query"] = item["SQL"]
        out.append(item)
    return out


def stage_prep():
    cur = DEV_DIR / "dev.json"
    bak = DEV_DIR / "dev_300_preprocessed.json"
    if cur.exists() and not bak.exists():
        shutil.copy2(cur, bak)
        n_bak = len(json.load(open(bak)))
        print(f"备份 300 预处理版 dev.json → {bak.name} ({n_bak} 题)")

    full = json.load(open(SRC_FULL))
    print(f"源全量 dev.json: {len(full)} 题")
    proc = preprocess(full)
    # DAIL-SQL BirdDataset 读 dataset/bird/dev.json(顶层);dev/dev.json 同步写保持一致
    for target in (BIRD / "dev.json", DEV_DIR / "dev.json"):
        with open(target, "w") as f:
            json.dump(proc, f, indent=2, ensure_ascii=False)
        print(f"写入预处理全量 → {target}")

    tables = json.load(open(BIRD / "tables.json"))
    db_ids = {q["db_id"] for q in proc}
    # 兼容两种布局: database/<db_id>.sqlite(扁平) 或 database/<db_id>/<db_id>.sqlite(嵌套)
    def has_db(db):
        return ((BIRD / "database" / f"{db}.sqlite").exists()
                or (BIRD / "database" / db / f"{db}.sqlite").exists())
    missing = {db for db in db_ids if not has_db(db)}
    n_flat = len(list((BIRD / "database").glob("*.sqlite")))
    print(f"tables.json: {len(tables)} 张表 | 涉及 db_id: {len(db_ids)} "
          f"(扁平 sqlite {n_flat} 个)")
    assert not missing, f"缺数据库: {missing}"
    print("prep OK")


def stage_shard():
    qs = json.load(open(P1DIR / "questions.json"))
    n = len(qs["questions"])
    print(f"_FULL questions.json: {n} 题")
    per = math.ceil(n / N_SHARD)
    for s in range(N_SHARD):
        sd = P1DIR / f"SHARD{s}"
        sd.mkdir(exist_ok=True)
        shutil.copy2(P1DIR / "questions.json", sd / "questions.json")
        lo, hi = s * per, min((s + 1) * per, n)
        print(f"SHARD{s}: [{lo}, {hi}) 共 {hi - lo} 题")
    print("shard OK")


def stage_link():
    """用 CoreNLP 重建全量 schema linking(enc/test_schema-linking.jsonl)。
    需 CoreNLP server 已在 localhost:9000 运行。"""
    import os
    import socket
    import sys
    import time

    for _ in range(90):
        try:
            socket.create_connection(("localhost", 9000), timeout=2).close()
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError("CoreNLP server :9000 未就绪")

    sys.path.insert(0, str(DAIL))
    os.chdir(DAIL)
    from data_preprocess import schema_linking_producer

    schema_linking_producer(test="dev.json", train="train.json",
                            table="tables.json", db="database",
                            dataset_dir="./dataset/bird",
                            compute_cv_link=False)
    enc = BIRD / "enc" / "test_schema-linking.jsonl"
    n = sum(1 for line in open(enc) if line.strip())
    print(f"test_schema-linking.jsonl: {n} 行")
    assert n == 1534, f"linking 行数 {n} != 1534"
    print("link OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "shard", "link"], required=True)
    args = ap.parse_args()
    {"prep": stage_prep, "shard": stage_shard, "link": stage_link}[args.stage]()
