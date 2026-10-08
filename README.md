# RGCV: Cost-Aware Retrieval, Strength-Gated Correction, and One-Way Verification for Enterprise Text-to-SQL

Reference implementation and experiment artifacts for the paper
*"RGCV: ..."* (TKDE submission). RGCV is a cost-aware
retrieval–generation–correction–verification loop for enterprise Text-to-SQL:

- **Schema-graph retrieval** (SchemaGraph-RAG) that pays exactly where schemas
  are large (thousand-column regime) and injects full schemas while they fit
  the prompt budget;
- a **priority-scheduled, strength-gated DBMS-collaborative corrector** —
  active repair collaboration for weak generators, execution-failure fallback
  plus verification reporting for strong ones;
- a **one-way three-layer verifier** (V1 result fingerprint → V2 differential
  execution → V3 LLM semantic check) for silent errors.

All Tier-2 experiments run on the BIRD-300 stratified subset with two unified
Chinese backbones (DeepSeek-V4-Pro, Kimi-K2.6) under measured API usage.

## Repository layout

```
github/
├── src/rgcv/            Core library (retrieval, generator, corrector,
│                        verifier, budget, LLM client, evaluation)
├── experiments/         Experiment drivers + analysis + plotting
├── results/             Frozen summary JSONs for every paper experiment
│                        + per-question logs of the two main configurations
├── data/                NOT included — download instructions (data/README.md)
└── figures/             Output directory created by experiments/plot_all.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ recommended (developed on 3.11).

### API keys

```bash
export DEEPSEEK_API_KEY=sk-...     # DeepSeek platform (deepseek-v4-pro / -v4-flash)
export MOONSHOT_API_KEY=sk-...     # Moonshot platform (kimi-k2.6)
```

Keys are read exclusively from the environment; the code contains no
credentials. The Tier-1 prototype experiments (E1–E4) use a rule-based proxy
for all LLM sites and run offline with **no** API key.

### Data (see `data/README.md` for details)

```
data/
├── bird/dev_20240627/          BIRD dev split (download from the BIRD site)
│   ├── dev.json
│   └── dev_databases/          SQLite databases + database_description/
├── bge-m3/                     BAAI/bge-m3 embedding model
└── bge-reranker-v2-m3/         BAAI/bge-reranker-v2-m3
```

`baselines/` (repo root, git-ignored) holds third-party baseline source trees
(CHESS, DAIL-SQL, DIN-SQL, MAC-SQL, ReFoRCE, SafeQL) if you want to re-run the
baseline rows; the frozen baseline results referenced by the paper are in
`results/`.

## Reproducing the experiments

Run everything from the repo root; each driver writes JSON summaries into
`results/`.

### Tier-1 prototype studies (offline, rule-proxy LLM sites)

| Exp | Command | Output |
|-----|---------|--------|
| E1 retrieval | `python experiments/exp_retrieval.py` | `e1_retrieval.json` |
| E2 correction | `python experiments/exp_correction.py` | `e2_correction_bird.json` |
| E3 verification | `python experiments/exp_verification.py` | `e3_verification_bird.json` |
| E4 pipeline | `python experiments/exp_pipeline.py --n 60` | `e4_pipeline_bird.json` |

Per-experiment analysis: `python experiments/analyze_e1.py` (likewise
`analyze_e2/3/4.py`), cost model: `estimate_e1_cost.py`.

### Tier-2 end-to-end (real LLMs, BIRD-300)

Main configuration (per-question logs in `results/rgcv_bird300_*_gated_fullschema/`):

```bash
python experiments/run_rgcv_e1.py --engine deepseek-v4-pro --segment 0 --n_segments 5 --ablation fullschema --gated
python experiments/run_rgcv_e1.py --engine kimi-k2.6      --segment 0 --n_segments 5 --ablation fullschema --gated
```

Ablations: `--ablation nc4 | norepair | noverify` (optionally combined with
`--gated`). The 300-question subset manifest is `results/sample_bird300.json`.

| Exp | Command | Output |
|-----|---------|--------|
| E-A real silent-error replay | `python experiments/exp_tier2_real.py` (`--v3_llm` for cross-backbone V3) | `eA_tier2_real_verification*.json` |
| E-B symmetric-protocol comparison | `python experiments/eval_safeql_sym.py` | `eB_symmetric_summary.json` |
| E-C variance / reruns | `python experiments/eval_variance.py` | `eC_variance_summary.json` |
| E-D paraphrase robustness | `python experiments/make_paraphrase_150.py && python experiments/run_rgcv_paraphrase.py && python experiments/analyze_paraphrase.py` | `eD_paraphrase_summary.json` |
| E-E full-dev calibration | `python experiments/run_ee_full_prepare.py && python experiments/run_ee_eval.py` | `eE_full_dev.json` |
| E-F verify-then-replace control | `python experiments/run_ef_verify_then_replace.py <engine>` | `eF_verify_then_replace*.json` |
| E-G edge-type ablation | `python experiments/run_eg_edge_ablation.py` | `eG_edge_ablation.json` |
| E-H strength–benefit curve | `python experiments/run_eh_strength_curve.py`, `run_eh_ruleproxy.py` | `eH_strength_curve.json` |

### Baselines

`experiments/` contains the drivers used for each baseline row
(`run_chess_bird300.py`, `run_safeql_e1.py`, `run_sqlcoder*.py`,
`prep_reforce_bird.py`, `prepare_dailsql_bird.py`, `eval_*_{ex,e1}.py`, …).
They assume the matching third-party source tree under `baselines/` and are
provided to document exactly how the published numbers were produced.

### Figures

```bash
python experiments/plot_all.py     # writes figures/ (PNG+PDF+SVG, 300 dpi)
```

## Results provenance

- `results/*.json` are the frozen artifacts behind every number in the paper;
  file names match the experiment IDs above.
- `results/rgcv_bird300_{deepseek-v4-pro,kimi-k26}_gated_fullschema/` contain
  the complete per-question records (`q_*.json`: question, candidates, final
  SQL, verification report, usage) plus `merged.json` and the line-level
  `api_trace.jsonl` usage ledger of the two main configurations.
- Tier-1 (E1–E4) per-question traces of all 1,425 prototype runs are available
  from the authors on request.

## License

MIT — see [LICENSE](LICENSE). BIRD and the third-party baseline codebases are
governed by their own licenses and are **not** redistributed here.
