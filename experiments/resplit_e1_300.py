"""E1: DAIL-SQL BIRD 300 题重切分 — test=300 全测, train=dev 剩余 1234 (few-shot 检索池, 无泄漏)。

区别于 61 题方案 (dailsql_resplit_traintest.py 240/60): E1 协议要求 300 题全部作为测试集。
few-shot 检索池改用 BIRD dev 全量未抽中的 1234 题 (与 test 无重叠, 无信息泄漏)。
"""
import json, shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
BIRD = DAIL / "dataset" / "bird"
DEV_DIR = BIRD / "dev"
TRAIN_DIR = BIRD / "train"

subset = json.load(open(BASE / "results" / "sample_bird300.json"))
test_qids = {q["question_id"] for q in subset["questions"]}
dev_all = json.load(open(BASE / "data" / "bird" / "dev_20240627" / "dev.json", encoding="utf-8"))
train_data = [q for q in dev_all if q["question_id"] not in test_qids]
test_data = [q for q in dev_all if q["question_id"] in test_qids]
print(f"train={len(train_data)}, test={len(test_data)}")


def preprocess(items):
    new_data = []
    for item in items:
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
        new_data.append(item)
    return new_data


train_p = preprocess(train_data)
test_p = preprocess(test_data)

for path, data in [(TRAIN_DIR / "train.json", train_p), (DEV_DIR / "dev.json", test_p),
                   (BIRD / "train.json", train_p), (BIRD / "dev.json", test_p)]:
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# tables.json (11 库全量, train/test 都用)
all_tables = json.load(open(DEV_DIR / "dev_tables.json", encoding="utf-8"))
json.dump(all_tables, open(BIRD / "tables.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
json.dump(all_tables, open(TRAIN_DIR / "train_tables.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

enc_dir = BIRD / "enc"
if enc_dir.exists():
    shutil.rmtree(enc_dir)
    print("已清理旧 enc/")

print(f"完成: train.json={len(train_p)}, dev.json={len(test_p)}, tables={len(all_tables)}")
print("下一步: schema_linking_producer → generate_question ×2 → ask_llm ×2")
