"""重新准备 DAIL-SQL BIRD 数据:300 题切分成 train/test(240/60)。

之前手动 preprocess 把 300 题全放 dev 里,train 是空占位,导致 EUCDISQUESTIONMASK
做 k_shot example selection 时 train_embeddings 为空 → NaN。

本次:
  1. 从 sample_bird300.json 读 300 题
  2. 按 80/20 切分 → 240 train + 60 test
  3. 分别预处理(加 question_toks + query + evidence)
  4. 写入 train/train.json 和 dev/dev.json(DAIL-SQL bird_pre_process 格式)
  5. 重新生成 tables.json
  6. 重新跑 schema_linking_producer(因为 train 现在不再是空的)
"""
import json, os, shutil, random
from pathlib import Path

random.seed(42)
BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
BIRD = DAIL / "dataset" / "bird"
DEV_DIR = BIRD / "dev"
TRAIN_DIR = BIRD / "train"

SUBSET = json.load(open(BASE / "results" / "sample_bird300.json"))
full_data = [q for q in json.load(open(DEV_DIR / "dev_300.json"))]  # 300 题原始 BIRD 格式
print(f"=== 1. 原始数据 {len(full_data)} 题 ===")

# 按 db_id 分组后分层切分,保证 train/test 都有各数据库
by_db = {}
for q in full_data:
    by_db.setdefault(q["db_id"], []).append(q)

train_data, test_data = [], []
for db_id, items in by_db.items():
    random.shuffle(items)
    split = max(1, int(len(items) * 0.8))
    train_data.extend(items[:split])
    test_data.extend(items[split:])

print(f"分层切分:train={len(train_data)},test={len(test_data)}")
assert len(train_data) + len(test_data) == len(full_data)

# === 2. JSON 预处理 ===
def preprocess(items):
    new_data = []
    for item in items:
        if item.get("evidence") and len(item["evidence"]) > 0:
            item["question"] = (item["question"] + " " + item["evidence"]).strip()
        question = item["question"]
        tokens = []
        for token in question.split(" "):
            if len(token) == 0: continue
            if token[-1] in ['?', '.', ':', ';', ','] and len(token) > 1:
                tokens.extend([token[:-1], token[-1:]])
            else:
                tokens.append(token)
        item["question_toks"] = tokens
        item["query"] = item["SQL"]
        new_data.append(item)
    return new_data

train_data_p = preprocess(train_data)
test_data_p = preprocess(test_data)

with open(TRAIN_DIR / "train.json", "w") as f:
    json.dump(train_data_p, f, indent=2, ensure_ascii=False)
with open(DEV_DIR / "dev.json", "w") as f:
    json.dump(test_data_p, f, indent=2, ensure_ascii=False)

# 根目录也各放一份(schema_linking_producer 读根目录)
with open(BIRD / "train.json", "w") as f:
    json.dump(train_data_p, f, indent=2, ensure_ascii=False)
with open(BIRD / "dev.json", "w") as f:
    json.dump(test_data_p, f, indent=2, ensure_ascii=False)

print(f"\n=== 2. 预处理完成 ===")
print(f"  train.json: {len(train_data_p)} 题")
print(f"  dev.json:   {len(test_data_p)} 题")

# === 3. tables.json ===
# 用所有被引用的 db_id 对应的 tables 子集
all_db_ids = {q["db_id"] for q in full_data}
with open(DEV_DIR / "dev_tables.json") as f:
    all_tables = json.load(f)
# BIRD dev_tables.json 是 list 格式,每个元素有 db_id
filtered_tables = [t for t in all_tables if t.get("db_id") in all_db_ids]
with open(BIRD / "tables.json", "w") as f:
    json.dump(filtered_tables, f, indent=2, ensure_ascii=False)
print(f"\n=== 3. tables.json: {len(filtered_tables)} 库 ===")

# 也写到 train/train_tables.json 占位
with open(TRAIN_DIR / "train_tables.json", "w") as f:
    json.dump(filtered_tables, f, indent=2, ensure_ascii=False)

# === 4. 清理旧 schema linking 产物 ===
enc_dir = BIRD / "enc"
if enc_dir.exists():
    shutil.rmtree(enc_dir)
    print(f"\n=== 4. 清理旧 enc/ ===")
else:
    print(f"\n=== 4. enc/ 不存在,跳过 ===")

print("\n=== 5. 下一步 ===")
print("重新跑 schema_linking_producer → generate_question.py → ask_llm.py")
