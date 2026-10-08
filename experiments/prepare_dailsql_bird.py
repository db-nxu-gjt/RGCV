"""DAIL-SQL 运行准备:BIRD 子集过滤 + 目录搭建 + API 适配。

步骤:
  1. 创建 DAIL-SQL/dataset/bird/dev/ 目录,拷贝 dev.databases + dev_tables
  2. 从 BIRD dev.json 按 sample_bird300.json 的 question_id 过滤生成 dev_300.json
  3. 创建空的 train 占位目录(DAIL-SQL bird_pre_process 需要但不影响 test-only 运行)
  4. 备份 + 修补 llm/chatgpt.py:加 openai.api_base,模型名改 deepseek-chat
"""
import json
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
BIRD_SRC = BASE / "data" / "bird" / "dev_20240627"
SUBSET = json.load(open(BASE / "results" / "sample_bird300.json"))
TARGET_DIR = DAIL / "dataset" / "bird"

print("=== 1. 目录结构 ===")
TARGET_DEV = TARGET_DIR / "dev"
TARGET_TRAIN = TARGET_DIR / "train"
TARGET_DEV.mkdir(parents=True, exist_ok=True)
TARGET_TRAIN.mkdir(parents=True, exist_ok=True)

# 拷贝 dev_databases(如果目标不存在)
db_src = BIRD_SRC / "dev_databases"
db_dst = TARGET_DEV / "dev_databases"
if not db_dst.exists():
    print(f"拷贝 dev_databases → {db_dst}")
    shutil.copytree(db_src, db_dst, dirs_exist_ok=True)
else:
    print(f"dev_databases 已存在 ({db_dst})")

# 拷贝 dev_tables.json
tables_src = BIRD_SRC / "dev_tables.json"
tables_dst = TARGET_DEV / "dev_tables.json"
if not tables_dst.exists():
    shutil.copy2(tables_src, tables_dst)
    print(f"拷贝 dev_tables.json")
else:
    print(f"dev_tables.json 已存在")

# 创建空 train 占位(train_databases 目录)
train_db_dst = TARGET_TRAIN / "train_databases"
train_db_dst.mkdir(parents=True, exist_ok=True)
print(f"创建 train 占位: {train_db_dst}")

# 拷贝 dev_tables.json 到 train(DAIL-SQL bird_pre_process 会从 train 也读 tables)
train_tables_dst = TARGET_TRAIN / "train_tables.json"
if not train_tables_dst.exists():
    shutil.copy2(tables_src, train_tables_dst)

print("\n=== 2. 过滤 dev.json → dev_300.json ===")
dev_all = json.load(open(BIRD_SRC / "dev.json"))
print(f"BIRD dev 全量: {len(dev_all)} 题")

subset_qids = {q["question_id"] for q in SUBSET["questions"]}
print(f"sample_bird300 question_id: {len(subset_qids)} 个")

dev_300 = [q for q in dev_all if q.get("question_id") in subset_qids]
print(f"过滤后: {len(dev_300)} 题")
assert len(dev_300) == 300, f"期望 300 题,实际 {len(dev_300)}"

dev_300_path = TARGET_DEV / "dev_300.json"
with open(dev_300_path, "w") as f:
    json.dump(dev_300, f, indent=2, ensure_ascii=False)
print(f"写入: {dev_300_path}")

# 同时用过滤后数据覆盖 dev.json(DAIL-SQL 硬编码读 dev/dev.json)
dev_json = TARGET_DEV / "dev.json"
shutil.copy2(dev_300_path, dev_json)
print(f"同时覆盖 dev.json → dev_300.json(DAIL-SQL 硬编码读 dev.json)")

# dev.sql 也拷贝(空文件, bird_pre_process 需要)
dev_sql_src = BIRD_SRC / "dev.sql"
if dev_sql_src.exists():
    shutil.copy2(dev_sql_src, TARGET_DEV / "dev.sql")

print("\n=== 3. 最终目录结构 ===")
for p in sorted(TARGET_DEV.rglob("*")):
    rel = p.relative_to(TARGET_DIR)
    marker = " DIR" if p.is_dir() else f" ({p.stat().st_size/1024:.0f}KB)"
    print(f"  {rel}{marker}")
print(f"  ... + {len(list(TARGET_DEV.rglob('*.sqlite')))} .sqlite 数据库")
