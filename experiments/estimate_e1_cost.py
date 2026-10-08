"""E1 成本估算: DAIL-SQL (4 run) + ReFoRCE (2 run) 的 prompt/completion token 聚合。

口径:
  DAIL prompt_tokens = questions.json 每题 prompt_tokens 求和 (tiktoken 精确)
  ReFoRCE token      = 主 run 日志 chat_session len 的 prompt_len/response_len
                       字符数 / 4 估算 (log 只记字符, 无 API usage)
  completion 均为字符/4 估算下界 (不含 reasoning token)
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # repo root
DAIL = BASE / "baselines" / "DAIL-SQL"
RES = BASE / "results"
EXP = BASE / "experiments"
PROC = DAIL / "dataset" / "process"
P1 = PROC / "BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096"
P2 = PROC / "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096"
P2K = PROC / "BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096_KIMI"
CHARS_PER_TOKEN = 4

DAIL_RUNS = [
    ("DAIL pass1 v4-pro", P1, "deepseek-v4-pro", "2:08:59"),
    ("DAIL pass2 v4-pro", P2, "deepseek-v4-pro", "1:37:25"),
    ("DAIL pass1 kimi", P1, "kimi-k2.6", "3:24:24"),
    ("DAIL pass2 kimi", P2K, "kimi-k2.6", "3:19:09"),
]
REFORCE_RUNS = [
    ("ReFoRCE v4-pro", EXP / "run_reforce_e1_deepseek-v4-pro.log", "0:30:15"),
    ("ReFoRCE kimi", EXP / "run_reforce_e1_kimi-k2.6.log", "1:28:19"),
]

# MAC-SQL: 4 shard 混写同一 api_trace.json (行级 append, 损坏行跳过);
# 墙钟取 stdout_shard0.log 创建→最后写入
MACSQL_TAGS = [
    ("MAC-SQL v4-pro", "deepseek-v4-pro", None),
    ("MAC-SQL kimi", "kimi-k2.6", None),
]

# CHESS: trace/q{qid}.jsonl 行级 append (API 原生 token usage, 非估算);
# 墙钟取 out_dir 创建→merged.json 最后写入
CHESS_TAGS = [
    ("CHESS v4-pro", "v4pro"),
    ("CHESS kimi", "kimi"),
]

# ReFoRCE 墙钟用文件创建→最后写入时间核对
for _, log, _ in REFORCE_RUNS:
    st = log.stat()
    print(f"  [{log.name}] created {st.st_ctime:.0f} -> mtime {st.st_mtime:.0f}")

print(f"\n{'run':<18} {'calls':>6} {'prompt_tok':>12} {'comp_tok(估)':>13} {'墙钟':>9} {'s/题':>7}")
print("-" * 72)
for name, pdir, model, wall in DAIL_RUNS:
    q = json.load(open(pdir / "questions.json", encoding="utf-8"))["questions"]
    ptok = sum(len(x["prompt"]) for x in q) // CHARS_PER_TOKEN  # prompt_tokens 字段恒 0, 用字符估算
    sqls = (pdir / f"RESULTS_MODEL-{model}.txt").read_text(encoding="utf-8")
    ctok = len(sqls) // CHARS_PER_TOKEN
    h, m, s = map(int, wall.split(":"))
    secs = h * 3600 + m * 60 + s
    print(f"  {name:<18} {len(q):>4}/{len(q)} {ptok:>10,} {ctok:>12,} {wall:>9} {secs/len(q):>6.1f}")

for name, log, wall in REFORCE_RUNS:
    txt = log.read_text(encoding="utf-16", errors="replace")  # ReFoRCE run 日志是 UTF-16 LE
    ms = re.findall(r"chat_session len: \{'prompt_len': (\d+), 'response_len': (\d+), 'num_calls': (\d+)\}", txt)
    plen = sum(int(a) for a, _, _ in ms)
    rlen = sum(int(b) for _, b, _ in ms)
    calls = sum(int(c) for _, _, c in ms)
    h, m, s = map(int, wall.split(":"))
    secs = h * 3600 + m * 60 + s
    n = 300
    print(f"  {name:<18} {n:>4}/{n} {plen//CHARS_PER_TOKEN:>10,} {rlen//CHARS_PER_TOKEN:>12,} {wall:>9} {secs/n:>6.1f}  (calls={calls})")

print()
for name, tag, _ in MACSQL_TAGS:
    out_dir = RES / f"macsql_bird300_{tag}"
    trace = out_dir / "api_trace.json"
    ptok = rtok = calls = bad = 0
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
            ptok += o["prompt_token"]
            rtok += o["response_token"]
            calls += 1
        except Exception:
            bad += 1
    s0 = out_dir / "stdout_shard0.log"
    st = s0.stat()
    secs = int(st.st_mtime - st.st_ctime)
    wall = f"{secs//3600}:{secs%3600//60:02d}:{secs%60:02d}"
    print(f"  {name:<18} {'~300':>4}/300 {ptok:>10,} {rtok:>12,} {wall:>9} {secs/300:>6.1f}  (calls={calls}, bad_lines={bad})")

print()
for name, tag in CHESS_TAGS:
    out_dir = RES / f"chess_bird300_{tag}"
    ptok = rtok = calls = bad = 0
    for f in (out_dir / "trace").glob("q*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                ptok += o["prompt_token"]
                rtok += o["response_token"]
                calls += 1
            except Exception:
                bad += 1
    merged = out_dir / "merged.json"
    secs = int(merged.stat().st_mtime - out_dir.stat().st_ctime)
    wall = f"{secs//3600}:{secs%3600//60:02d}:{secs%60:02d}"
    n = len(json.load(open(merged, encoding="utf-8")))
    print(f"  {name:<18} {n:>4}/300 {ptok:>10,} {rtok:>12,} {wall:>9} {secs/max(n,1):>6.1f}  (calls={calls}, bad_lines={bad})")
print("\n注: DAIL/ReFoRCE token = 字符/4 估算; MAC-SQL/CHESS = API 原生 usage (CHESS completion 含 reasoning)")
