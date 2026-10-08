"""手动完成 DAIL-SQL bird_pre_process（绕过 train 数据硬依赖）,然后跑完整管线。

DAIL-SQL bird_pre_process 硬编码读 train/train.json,但我们只有 BIRD dev 数据。
本脚本手动完成等价工作:
  1. 从 dev/dev.json 生成 question_toks/query 字段 → 输出 dev.json(已在 dev 目录)
  2. 拷贝 dev_tables.json 生成 tables.json(因为没有 train tables,直接用 dev 的)
  3. 创建 database 目录,拷贝 dev_databases 下的 sqlite 到扁平化 database/
  4. 然后跑 schema_linking_producer
"""
import json
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
BIRD = DAIL / "dataset" / "bird"
DEV_DIR = BIRD / "dev"

# === Step 1: 预处理 dev.json(加 question_toks + query) ===
print("=== Step 1: JSON preprocess ===")
with open(DEV_DIR / "dev.json") as f:
    dev_data = json.load(f)
print(f"dev.json: {len(dev_data)} 题")

new_data = []
for item in dev_data:
    # 加 evidence 到 question(DAIL-SQL with_evidence=True 逻辑)
    if item.get("evidence") and len(item["evidence"]) > 0:
        item["question"] = (item["question"] + " " + item["evidence"]).strip()

    # tokenization(同 bird_pre_process)
    question = item["question"]
    tokens = []
    for token in question.split(" "):
        if len(token) == 0:
            continue
        if token[-1] in ['?', '.', ':', ';', ','] and len(token) > 1:
            tokens.extend([token[:-1], token[-1:]])
        else:
            tokens.append(token)
    item["question_toks"] = tokens
    item["query"] = item["SQL"]
    new_data.append(item)

with open(DEV_DIR / "dev.json", "w") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)
print(f"预处理完成:{len(new_data)} 题 → {DEV_DIR / 'dev.json'}")

# 生成空 train.json 和空占位 train 数据(schema_linking_producer 会读它们)
TRAIN_DIR = BIRD / "train"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
with open(TRAIN_DIR / "train.json", "w") as f:
    json.dump([], f)
print("创建空 train.json")

# === Step 2: 合并 tables.json ===
print("\n=== Step 2: tables.json ===")
with open(DEV_DIR / "dev_tables.json") as f:
    tables = json.load(f)
with open(BIRD / "tables.json", "w") as f:
    json.dump(tables, f, indent=2, ensure_ascii=False)
print(f"tables.json: {len(tables)} 张表 → {BIRD / 'tables.json'}")

# === Step 3: 拷贝 dev.sql ===
if (DEV_DIR / "dev.sql").exists():
    shutil.copy2(DEV_DIR / "dev.sql", BIRD / "dev.sql")
    print("拷贝 dev.sql")

# === Step 4: 创建 database 目录 ===
print("\n=== Step 4: database 目录 ===")
new_db_path = BIRD / "database"
if not new_db_path.exists():
    new_db_path.mkdir(parents=True)
    # 拷贝 dev_databases/*/xxx.sqlite 扁平化
    dev_dbs = DEV_DIR / "dev_databases"
    count = 0
    for db_dir in dev_dbs.iterdir():
        if db_dir.is_dir():
            for sqlite_file in db_dir.glob("*.sqlite"):
                dest = new_db_path / sqlite_file.name
                shutil.copy2(sqlite_file, dest)
                count += 1
    print(f"扁平化拷贝 {count} 个 sqlite → {new_db_path}")
else:
    print(f"database 已存在")

print("\n=== 预处理完成 ===")
print("现在启动 CoreNLP 并跑 schema_linking_producer")
print("然后: data_preprocess.py → generate_question.py → ask_llm.py")
