# Data setup

This repository does **not** redistribute the datasets or model weights.
Create the following layout before running the experiments:

```
data/
├── bird/
│   └── dev_20240627/                 # BIRD dev split
│       ├── dev.json                  # questions + gold SQL
│       └── dev_databases/
│           ├── <db_id>/<db_id>.sqlite
│           └── <db_id>/database_description/   # column descriptions
├── bge-m3/                           # BAAI/bge-m3 (embedding)
└── bge-reranker-v2-m3/               # BAAI/bge-reranker-v2-m3
```

## 1. BIRD (required for all experiments)

Download the **dev** split (`dev_20240627`) from the BIRD benchmark website
(https://bird-bench.github.io) and unpack it so that `data/bird/dev_20240627/`
matches the layout above. The paper's BIRD-300 stratified subset manifest is
`results/sample_bird300.json` (`sample_bird300.py` regenerates it from
`dev.json`).

## 2. Embedding / reranker models (required for retrieval)

```bash
huggingface-cli download BAAI/bge-m3      --local-dir data/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir data/bge-reranker-v2-m3
```

`experiments/common.py` points the embedding module at these two directories.

## 3. Third-party baselines (optional, only to re-run baseline rows)

Clone each baseline into `baselines/` at the repo root and follow its own
setup instructions; the drivers in `experiments/` reference these paths:

```
baselines/
├── CHESS/       # https://github.com/InsightLab/CHESS
├── DAIL-SQL/    # https://github.com/beachwang/DAIL-SQL (incl. dataset/bird)
├── DIN-SQL/     # https://github.com/njustkmg/DIN-SQL
├── MAC-SQL/     # https://github.com/wbbeyourself/MAC-SQL
├── ReFoRCE/     # per paper release
└── SafeQL/      # per paper release
```

Helper downloaders: `experiments/download_corenlp.py` (DAIL-SQL dependency),
`download_glove.py` (DAIL-SQL vector cache), `download_sqlcoder_gguf.py`
(local SQLCoder weights), `install_nltk_data.py`.

## 4. PG17 (optional, SafeQL substrate only)

`run_safeql_*.py` expect a local PostgreSQL 17 instance per
`experiments/migrate_bird_pg.py` (host/port/user/password are set at the top
of each driver) with the BIRD databases migrated via that script.
