"""E-C (T1-3):验证机制横向对比 — SC-vote(4-way self-consistency) vs
CHESS-UT(单元测试候选筛选),在 eA replay 的冻结 Tier-2 运行题集上给出
拦截率 / 假告警率 / token 成本三元组,并与 RGCV V1+V2 做 McNemar 对比。

协议:
  题集   results/eA_tier2_real_verification.json 中两个 _gated_fullschema
         tag 的 traces(kind 字段 = silent|correct,直接采用不重算);
         冻结预测 = tier2 运行 results/<tag>/q_*.json 的 pred_sql
         (silent/correct 题预测必执行成功)。
  共享采样层 每题 4 个候选 SQL:temperature=0.7,seed=10000*qid+i(i=0..3),
         system/user prompt 与 RGCV 生成同构(SYS_GEN + full_schema_result
         全 schema 注入,与 gated_fullschema 冻结预测同口径;Kimi 平台可能
         固定温度 / 忽略 seed,如实记录)。每候选在 sqlite DB 上执行,
         执行失败记 exec-error。
  SC-vote  4 候选按执行结果多重集等价聚类(exec_equal 语义;执行失败者统一
         归入 exec-error 簇);alarm = 簇数 >= 2;交付 = 最大簇代表结果;
         majority_correct = 交付结果 exec_equal gold 结果。成本 = 4 gen。
  CHESS-UT UT 生成 1 调用(T=0):问题+同款 schema → 恰好 5 条"正确结果应
         满足的性质"断言(JSON 数组容错解析;<2 条 → UT 生成失败,该题两
         机制均记 alarm=NA,标记 degraded)。候选集 = 冻结预测 + 4 采样
         (共 5);判别调用按"执行结果多重集哈希"去重(断言为纯结果性质,
         同结果同判定),每个唯一成功结果 1 调用(T=0,输入 = 问题 + 断言 +
         代表候选 SQL + 结果前 20 行 JSON(单元格截断 80 字符)+ 执行失败
         标记,输出长度 N 布尔数组);执行失败候选 pass_rate=0 跳过判别。
         alarm = 冻结预测 pass_rate < max(全部候选 pass_rate);交付 =
         pass_rate 最高候选(平票冻结预测优先);delivered_correct = 交付
         结果 exec_equal gold。成本 = 4 gen + 1 UT-gen + 去重后 judge
         (共享 4 gen 在两机制行分别完整计入)。
  指标   拦截率(silent)/ 假告警率(correct)+ bootstrap 10^4 95% CI
         (percentile,与 eA 同款);修正数;McNemar vs V1+V2(eA traces)。
输出   results/ec_verification_baselines.json;
       中间产物 results/ec_baselines_<slug>/{state.json, responses/q*.json,
       trace/api_trace.jsonl}。

运行(在 paper/github/experiments 下):
  python run_ec_verification_baselines.py --engine deepseek-v4-pro --smoke
  python run_ec_verification_baselines.py --engine deepseek-v4-pro [--ids 10,11]
  python run_ec_verification_baselines.py --engine kimi-k2.6
  python run_ec_verification_baselines.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]            # paper/github
sys.path.insert(0, str(BASE / "experiments"))
sys.path.insert(0, str(BASE / "src"))

from common import RESULTS, bird_catalog, bird_executor  # noqa: E402

# BIRD dev_databases 路径回退:common.py 默认 paper/github/data(若缺失,
# 依次回退 RGCV 根 data/ 与 DAIL-SQL 数据副本;库内容与列描述同源同版)。
import common as _common  # noqa: E402

REPO = BASE.parents[1]                                # RGCV root
if not _common.BIRD_DB.exists():
    for _cand in (REPO / "data" / "bird" / "dev_20240627" / "dev_databases",
                  REPO / "baselines" / "DAIL-SQL" / "dataset" / "bird" / "dev"
                  / "dev_databases"):
        if _cand.exists():
            _common.BIRD_DB = _cand
            _common.BIRD_DIR = _cand.parent
            break
    assert _common.BIRD_DB.exists(), \
        f"BIRD dev_databases not found (tried fallbacks); last={_common.BIRD_DB}"

from exp_tier2_real import boot_ci  # noqa: E402  (与 eA 同款 bootstrap CI)
from rgcv.dbs import DBError  # noqa: E402
from rgcv.evalx import exec_equal, mcnemar_test  # noqa: E402
from rgcv.llm import LLMClient, SYS_GEN, extract_sqls  # noqa: E402
from rgcv.schema_graph import build_graph, full_schema_result  # noqa: E402

EA_FILE = RESULTS / "eA_tier2_real_verification.json"

# 引擎 → (slug, tier2 冻结运行 tag, eA 判定来源 tag)
ENGINES = {
    "deepseek-v4-pro": ("deepseek-v4-pro",
                        "rgcv_bird300_deepseek-v4-pro_gated_fullschema"),
    "kimi-k2.6": ("kimi-k26",
                  "rgcv_bird300_kimi-k26_gated_fullschema"),
}

N_SAMPLES = 4
GEN_TEMPERATURE = 0.7
SEED_BASE = 10_000                      # seed = SEED_BASE * qid + i
# Kimi-k2.6 thinking-disabled 模式平台仅允许 temperature=0.6
# (400: "invalid temperature: only 0.6 is allowed");全部 phase 统一钳制。
PLATFORM_TEMP_CAP = {"kimi-k2.6": 0.6}


def clamp_temp(engine: str, t: float) -> float:
    return PLATFORM_TEMP_CAP.get(engine, t)

JUDGE_MAX_ROWS = 20
JUDGE_CELL_CHARS = 80

# ------------------------------------------------------------------ prompts
SYS_UTGEN = (
    "You are a test designer for TEXT-TO-SQL verification. Given a database "
    "schema and a natural language question, write assertions about what the "
    "CORRECT query result must satisfy. Assertions must be pure properties of "
    "the result (number of rows, column content, value constraints, "
    "ordering); never describe the SQL or the solving process. Reply with a "
    "JSON array of assertion strings only, no other text.")

SYS_UTJUDGE = (
    "You are a strict test judge for TEXT-TO-SQL. You see a question, a "
    "numbered list of assertions describing properties of the correct "
    "result, a candidate SQL and its execution result. Decide for each "
    "assertion whether the execution result satisfies it. Reply with ONLY a "
    "JSON array of booleans (true/false), one per assertion in order, no "
    "other text.")

_z = lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0}


def _uadd(bucket, uin, uout, calls=1):
    bucket["calls"] += calls
    bucket["tokens_in"] += uin
    bucket["tokens_out"] += uout
    return bucket


# ------------------------------------------------------------------ 题集加载
def load_worklist(tag: str) -> list:
    """eA traces(kind 权威)+ tier2 冻结预测(q_*.json)拼装工作集。"""
    ea = json.loads(EA_FILE.read_text(encoding="utf-8"))
    traces = [t for t in ea["traces"]
              if t.get("tag") == tag and t.get("kind") in ("silent", "correct")]
    tier2_dir = RESULTS / tag
    items = []
    for t in sorted(traces, key=lambda x: int(x["question_id"])):
        qid = int(t["question_id"])
        p = tier2_dir / f"q_{qid}.json"
        if not p.exists():
            print(f"[ec] WARN missing frozen record q_{qid}.json, skipped",
                  flush=True)
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("error") or not rec.get("pred_sql"):
            print(f"[ec] WARN q{qid} frozen record has no pred_sql, skipped",
                  flush=True)
            continue
        items.append({
            "question_id": qid, "db_id": rec["db_id"], "kind": t["kind"],
            "question": rec["question"],           # 已含 [Hint] 后缀,同 tier-2
            "pred_sql": rec["pred_sql"], "gold_sql": rec["gold_sql"],
            "v12_flag": bool(t["V1+V2"]["flagged"]),
        })
    return items


# ------------------------------------------------------------------ 执行
_schema_cache: dict = {}
_gold_cache: dict = {}
_gold_lock = threading.Lock()


def schema_prompt(db_id: str) -> str:
    """与 tier-2 gated_fullschema 同口径的全 schema 注入(full_schema_result)。"""
    if db_id not in _schema_cache:
        graph = build_graph(bird_catalog(db_id), use_semantic=False)
        _schema_cache[db_id] = full_schema_result(graph, "").prompt
    return _schema_cache[db_id]


def gold_result(db_id: str, gold_sql: str):
    key = (db_id, gold_sql)
    with _gold_lock:
        if key not in _gold_cache:
            try:
                _gold_cache[key] = \
                    bird_executor(db_id).execute(gold_sql)
            except DBError:
                _gold_cache[key] = None
        return _gold_cache[key]


def exec_safe(db_id: str, sql: str):
    """执行 SQL;返回 ("ok", (cols, rows, n)) 或 ("err", msg)。"""
    if not sql or not sql.strip():
        return ("err", "empty_sql")
    try:
        return ("ok", bird_executor(db_id).execute(sql))
    except DBError as e:
        return ("err", str(e)[:200])
    except Exception as e:
        return ("err", f"{type(e).__name__}: {e}"[:200])


def res_key(res) -> str:
    """执行结果多重集哈希(判别去重用;基于 evalx._res_multiset 归一)。"""
    from rgcv.evalx import _res_multiset
    return repr(tuple(_res_multiset(res[0], res[1])))


def rows_preview(res) -> str:
    cols, rows, n = res[0], res[1], res[2]

    def cv(v):
        if isinstance(v, bytes):
            v = v.decode("utf-8", "ignore")
        return str(v)[:JUDGE_CELL_CHARS]

    return json.dumps({
        "executed_ok": True,
        "columns": [str(c) for c in cols],
        "rows": [[cv(v) for v in r] for r in list(rows)[:JUDGE_MAX_ROWS]],
        "n_rows_returned": n,
    }, ensure_ascii=False)


# ------------------------------------------------------------------ 机制
def sc_vote(sampled_exec: list, gold) -> dict:
    """机制 1:4 采样候选按结果多重集等价聚类(exec_equal;失败统一簇)。"""
    clusters = []                       # [{rep, members, err}]
    for i, (st, r) in enumerate(sampled_exec):
        if st == "err":
            for c in clusters:
                if c["err"]:
                    c["members"].append(i)
                    break
            else:
                clusters.append({"rep": None, "members": [i], "err": True})
            continue
        placed = False
        for c in clusters:
            if not c["err"] and exec_equal(r, c["rep"]):
                c["members"].append(i)
                placed = True
                break
        if not placed:
            clusters.append({"rep": r, "members": [i], "err": False})
    n_clusters = len(clusters)
    alarm = n_clusters >= 2
    largest = max(clusters, key=lambda c: len(c["members"]))   # 平票取先出现
    delivered_ok = not largest["err"]
    majority_correct = bool(delivered_ok and gold is not None and
                            exec_equal(largest["rep"], gold))
    return {"clusters": n_clusters,
            "cluster_sizes": sorted((len(c["members"]) for c in clusters),
                                    reverse=True),
            "alarm": alarm,
            "delivered_all_exec_error": largest["err"],
            "majority_correct": majority_correct}


def parse_bool_list(text: str, n: int):
    """容错解析长度 n 的布尔数组;失败返回 None。"""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None

    def coerce(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes", "pass")
        return False

    bools = [coerce(x) for x in arr]
    if len(bools) != n:
        return None
    return bools


def parse_assertions(text: str) -> list:
    """容错解析 UT 断言 JSON 数组;失败退化为逐行提取。"""
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                out = [str(x).strip() for x in arr if str(x).strip()]
                if out:
                    return out
        except json.JSONDecodeError:
            pass
    out = []
    for line in text.splitlines():
        s = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
        s = s.rstrip(",").strip()
        if len(s) > 8:
            out.append(s)
    return out[:5]


def chess_ut(question: str, schema: str, cand_exec: list, gold,
             client, usage, st: dict) -> dict:
    """机制 2:CHESS 式单元测试候选筛选。cand_exec = [(name, sql, (st, r))]。"""
    # ---- UT 生成(1 调用,T=0)
    ut_user = (f"### Database Schema\n{schema}\n\n### Question\n{question}"
               "\n\nDesign exactly 5 assertions on what the correct query "
               "result must satisfy. Assertions must be pure properties of "
               "the result (row count, column content, value constraints, "
               "ordering); do not describe the solving process. Reply with "
               "ONLY a JSON array of 5 short assertion strings.")
    text, uin, uout = client.chat("ut_gen", SYS_UTGEN, ut_user,
                                  max_tokens=1024,
                                  temperature=clamp_temp(client.engine, 0))
    _uadd(usage["ut_gen"], uin, uout)
    if not text.strip():                       # API 重试耗尽
        st["status"] = "failed"
        st["error"] = "ut_gen_api_exhausted"
        return {}
    assertions = parse_assertions(text)
    n_assert = len(assertions)
    if n_assert < 2:                           # UT 生成失败 → degraded
        st["status"] = "degraded"
        st["degraded_reason"] = f"ut_assertions={n_assert}"
        return {"assertions": assertions, "n": n_assert, "degraded": True}

    # ---- 判别(按结果多重集哈希去重;执行失败候选 pass_rate=0 跳过)
    groups: dict = {}                          # key -> {rep_res, rep_sql, members}
    pass_rate_of_group: dict = {}
    for name, sql, (stx, r) in cand_exec:
        if stx != "ok":
            continue
        k = res_key(r)
        if k not in groups:
            groups[k] = {"rep_res": r, "rep_sql": sql, "members": []}
        groups[k]["members"].append(name)

    judge_parse_fail = 0
    judge_raw = []                             # 原始判别响应(截断,审计用)
    for k, g in groups.items():
        juser = (f"### Question\n{question}\n\n### Assertions\n" +
                 "\n".join(f"{i + 1}. {a}"
                           for i, a in enumerate(assertions)) +
                 f"\n\n### Candidate SQL\n{g['rep_sql']}\n\n"
                 f"### Execution result\n{rows_preview(g['rep_res'])}\n\n"
                 "For each assertion, decide true if the execution result "
                 f"satisfies it, false otherwise. Reply ONLY the JSON array "
                 f"of {n_assert} booleans.")
        jtext, jin, jout = client.chat("ut_judge", SYS_UTJUDGE, juser,
                                       max_tokens=512,
                                       temperature=clamp_temp(client.engine,
                                                              0))
        _uadd(usage["judge"], jin, jout)
        judge_raw.append(jtext[:500])
        if not jtext.strip():                  # API 重试耗尽
            st["status"] = "failed"
            st["error"] = "ut_judge_api_exhausted"
            return {}
        bools = parse_bool_list(jtext, n_assert)
        if bools is None:
            judge_parse_fail += 1
            pass_rate_of_group[k] = 0.0        # 解析失败:无证据,记 0(如实)
        else:
            pass_rate_of_group[k] = round(sum(bools) / n_assert, 4)

    pass_rates = []                            # 顺序 = cand_exec 顺序
    for name, sql, (stx, r) in cand_exec:
        if stx != "ok":
            pass_rates.append(0.0)
        else:
            pass_rates.append(pass_rate_of_group[res_key(r)])
    mx = max(pass_rates)
    frozen_pr = pass_rates[0]                  # cand_exec[0] = 冻结预测
    alarm = frozen_pr < mx
    delivered_idx = pass_rates.index(mx)       # 首个最大者;pred 在前 → 平票冻结优先
    delivered_name, delivered_sql, (dst, dres) = cand_exec[delivered_idx]
    delivered_correct = bool(dst == "ok" and gold is not None and
                             exec_equal(dres, gold))
    return {"assertions": assertions, "n": n_assert, "degraded": False,
            "pass_rates": pass_rates, "candidate_names":
                [c[0] for c in cand_exec],
            "alarm": alarm, "delivered": delivered_name,
            "delivered_correct": delivered_correct,
            "judge_parse_fail": judge_parse_fail,
            "judge_calls": len(groups),
            "judge_raw": judge_raw}


# ------------------------------------------------------------------ 单题
def process_question(item: dict, client: LLMClient, rdir: Path) -> dict:
    qid = item["question_id"]
    t0 = time.time()
    usage = {"gen": _z(), "ut_gen": _z(), "judge": _z()}
    st = {"status": "done", "error": ""}
    schema = schema_prompt(item["db_id"])
    gen_user = (f"### Database Schema\n{schema}\n\n"
                f"### Question\n{item['question']}"
                "\n\nWrite the SQLite query. Output only the SQL.")

    # ---- 共享采样层:4 候选(t=0.7, seed=10000*qid+i)
    sampled_sqls, sampled_exec = [], []
    for i in range(N_SAMPLES):
        text, uin, uout = client.chat("gen_sc", SYS_GEN, gen_user,
                                      temperature=clamp_temp(
                                          client.engine, GEN_TEMPERATURE),
                                      seed=SEED_BASE * qid + i)
        _uadd(usage["gen"], uin, uout)
        if not text.strip():                   # API 重试耗尽 → failed
            st["status"] = "failed"
            st["error"] = f"gen_api_exhausted_sample{i}"
            break
        sqls = extract_sqls(text, 1)
        sql = sqls[0] if sqls else ""
        sampled_sqls.append(sql)
    if st["status"] != "failed":
        if len(sampled_sqls) < N_SAMPLES:
            st["status"] = "failed"
            st["error"] = "gen_samples_incomplete"
        else:
            gold = gold_result(item["db_id"], item["gold_sql"])
            if gold is None:
                st["status"] = "failed"
                st["error"] = "gold_unexecutable"
            else:
                frozen_exec = exec_safe(item["db_id"], item["pred_sql"])
                for sql in sampled_sqls:
                    sampled_exec.append(exec_safe(item["db_id"], sql))

                # ---- 机制 1:SC-vote
                sc = sc_vote(sampled_exec, gold)

                # ---- 机制 2:CHESS-UT(候选 = 冻结预测 + 4 采样)
                cand_exec = ([("pred", item["pred_sql"], frozen_exec)] +
                             [(f"s{i}", sampled_sqls[i], sampled_exec[i])
                              for i in range(N_SAMPLES)])
                ut = chess_ut(item["question"], schema, cand_exec, gold,
                              client, usage, st)

                # degraded 规则:UT 生成失败(<2 断言)→ 该题两机制 alarm=NA
                sc_alarm = sc["alarm"] if not ut.get("degraded") else None
                sc_majority = (sc["majority_correct"]
                               if not ut.get("degraded") else None)

                out = {
                    "question_id": qid, "db_id": item["db_id"],
                    "kind": item["kind"], "status": st["status"],
                    "error": st.get("error", ""),
                    "degraded_reason": st.get("degraded_reason", ""),
                    "pred_sql": item["pred_sql"],
                    "sampled_sqls": sampled_sqls,
                    "frozen_exec_status": frozen_exec[0],
                    "frozen_exec_error": (str(frozen_exec[1])[:200]
                                          if frozen_exec[0] == "err" else ""),
                    "sampled_exec_status": [e[0] for e in sampled_exec],
                    "sc": {"clusters": sc["clusters"],
                           "cluster_sizes": sc["cluster_sizes"],
                           "alarm": sc_alarm,
                           "alarm_raw": sc["alarm"],
                           "delivered_all_exec_error":
                               sc["delivered_all_exec_error"],
                           "majority_correct": sc_majority,
                           "majority_correct_raw": sc["majority_correct"]},
                    "ut": {k: ut.get(k) for k in
                           ("assertions", "n", "degraded", "pass_rates",
                            "candidate_names", "alarm", "delivered",
                            "delivered_correct", "judge_parse_fail",
                            "judge_calls", "judge_raw")},
                    "v12_flag": item["v12_flag"],
                    "usage": usage,
                    "secs": round(time.time() - t0, 1),
                }
                (rdir / "responses").mkdir(exist_ok=True)
                (rdir / "responses" / f"q{qid}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                return out

    # failed 路径:落盘 minimal 记录(state 记 failed,聚合跳过)
    out = {"question_id": qid, "db_id": item["db_id"],
           "kind": item["kind"], "status": "failed",
           "error": st.get("error", ""), "usage": usage,
           "secs": round(time.time() - t0, 1)}
    (rdir / "responses").mkdir(exist_ok=True)
    (rdir / "responses" / f"q{qid}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ 跑批
def run(engine: str, ids=None, smoke=0, workers=4, retry_failed=False):
    slug, tag = ENGINES[engine]
    rdir = RESULTS / f"ec_baselines_{slug}{'_smoke' if smoke else ''}"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "trace").mkdir(exist_ok=True)

    items = load_worklist(tag)
    if smoke:
        sil = [it for it in items if it["kind"] == "silent"][:smoke // 2 + smoke % 2]
        cor = [it for it in items if it["kind"] == "correct"][:smoke // 2]
        items = sil + cor
    if ids:
        idset = {int(x) for x in ids}
        items = [it for it in items if it["question_id"] in idset]

    state_p = rdir / "state.json"
    try:
        state = json.loads(state_p.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    todo = []
    for it in items:
        prev = state.get(str(it["question_id"]))
        if prev is None or (retry_failed and prev.get("status") == "failed"):
            todo.append(it)
    print(f"[ec:{slug}] worklist={len(items)} todo={len(todo)} "
          f"(engine={engine}, workers={workers}, smoke={smoke or '-'})",
          flush=True)
    if not todo:
        return

    client = LLMClient(engine, trace_path=str(rdir / "trace" /
                                             "api_trace.jsonl"))
    lock = threading.Lock()
    t_start = time.time()
    done_n = 0

    def _finish(it, rec):
        nonlocal done_n
        with lock:
            state[str(it["question_id"])] = {
                "status": rec["status"], "secs": rec.get("secs", 0),
                "error": rec.get("error", ""),
                "usage": rec.get("usage", {})}
            state_p.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                               encoding="utf-8")
            done_n += 1
        u = rec.get("usage", {})
        tot = sum(b["tokens_in"] + b["tokens_out"] for b in u.values())
        print(f"[ec:{slug}] q{it['question_id']} {it['kind']} "
              f"{it['db_id']} {rec['status']} "
              f"sc={rec.get('sc', {}).get('alarm', '-')} "
              f"ut={rec.get('ut', {}).get('alarm', '-')} "
              f"tok={tot} {rec.get('secs', 0)}s "
              f"({done_n}/{len(todo)}, {(time.time() - t_start) / 60:.1f}min)",
              flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_question, it, client, rdir): it
                for it in todo}
        n_err = 0
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                rec = fut.result()
                _finish(it, rec)
            except Exception as e:             # 驱动层异常:记 failed 不卡死
                n_err += 1
                with lock:
                    state[str(it["question_id"])] = {
                        "status": "failed",
                        "error": f"driver:{type(e).__name__}: {e}"[:200]}
                    state_p.write_text(
                        json.dumps(state, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                print(f"[ec:{slug}] q{it['question_id']} DRIVER ERROR "
                      f"{type(e).__name__}: {e}", flush=True)
    u = client.total_usage
    print(f"[ec:{slug}] batch done: +{done_n}/{len(todo)} "
          f"driver_errors={n_err} elapsed={(time.time() - t_start) / 60:.1f}min "
          f"api_calls={u['calls']} tok_in={u['tokens_in']} "
          f"tok_out={u['tokens_out']}", flush=True)


# ------------------------------------------------------------------ 汇总
def aggregate():
    ea = json.loads(EA_FILE.read_text(encoding="utf-8"))
    v12_by_tag: dict = {}
    for t in ea["traces"]:
        if t.get("kind") in ("silent", "correct"):
            v12_by_tag[(t["tag"], int(t["question_id"]))] = \
                bool(t["V1+V2"]["flagged"])

    summary, all_traces = {}, []
    for engine, (slug, tag) in ENGINES.items():
        rdir = RESULTS / f"ec_baselines_{slug}"
        if not rdir.exists() or not list(rdir.glob("responses/q*.json")):
            summary[slug] = {"engine": engine, "frozen_tag": tag,
                             "status": "not_run",
                             "reason": "results/ec_baselines_%s 不存在"
                                       "(未运行;脚本:run_ec_verification_"
                                       "baselines.py --engine %s)"
                                       % (slug, engine)}
            continue
        recs = []
        for p in sorted((rdir / "responses").glob("q*.json"),
                        key=lambda x: int(x.stem[1:])):
            try:
                recs.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        if not recs:
            continue
        failed = [r for r in recs if r["status"] == "failed"]
        degraded = [r for r in recs if r["status"] == "degraded"]
        valid = [r for r in recs if r["status"] == "done"]   # 非 degraded
        done_all = [r for r in recs if r["status"] in ("done", "degraded")]

        def side(rows, mech, fix_key):
            n = len(rows)
            flags = [bool(r[mech]["alarm"]) for r in rows]
            fixes = sum(1 for r in rows
                        if r[mech][fix_key] and r["kind"] == "silent")
            return {"n": n, "alarm_n": sum(flags),
                    "rate": round(sum(flags) / n, 4) if n else 0.0,
                    "ci95": boot_ci(flags, seed=0),
                    "corrections_n": fixes}

        sil = [r for r in valid if r["kind"] == "silent"]
        cor = [r for r in valid if r["kind"] == "correct"]
        sc_s = side(sil, "sc", "majority_correct")
        sc_c = side(cor, "sc", "alarm")
        ut_s = side(sil, "ut", "delivered_correct")
        ut_c = side(cor, "ut", "alarm")

        # 成本:SC = 4 gen;UT = 4 gen + ut_gen + judge(共享 4 gen 双计)
        sc_cost, ut_cost = _z(), _z()
        for r in done_all:
            u = r.get("usage", {})
            _uadd(sc_cost, u.get("gen", {}).get("tokens_in", 0),
                  u.get("gen", {}).get("tokens_out", 0),
                  u.get("gen", {}).get("calls", 0))
            for ph in ("gen", "ut_gen", "judge"):
                b = u.get(ph, {})
                _uadd(ut_cost, b.get("tokens_in", 0), b.get("tokens_out", 0),
                      b.get("calls", 0))
        n_cost = len(done_all)

        # McNemar vs V1+V2(silent,alarm 向量对比)
        def mc(rows, mech_key):
            a = [bool(r[mech_key[0]][mech_key[1]]) for r in rows]
            b = [v12_by_tag[(tag, r["question_id"])] for r in rows]
            return mcnemar_test(a, b)

        # 逐题 trace(全部 done/degraded;failed 只在 state)
        for r in recs:
            if r["status"] == "failed":
                continue
            all_traces.append({
                "question_id": r["question_id"], "db_id": r["db_id"],
                "kind": r["kind"], "engine": engine,
                "status": r["status"],
                "sc_alarm": r["sc"]["alarm"],
                "sc_clusters": r["sc"]["clusters"],
                "sc_majority_correct": r["sc"]["majority_correct"],
                "ut_assertions_n": r["ut"]["n"],
                "ut_pass_rates": r["ut"]["pass_rates"],
                "ut_alarm": r["ut"]["alarm"],
                "ut_delivered_correct": r["ut"]["delivered_correct"],
                "v12_flag": r.get("v12_flag"),
                "usage": r.get("usage", {})})

        summary[slug] = {
            "engine": engine, "frozen_tag": tag,
            "n_questions": len(recs), "n_failed": len(failed),
            "n_degraded": len(degraded),
            "failed_errors": {str(r["question_id"]): r.get("error", "")
                              for r in failed},
            "degraded_reasons": [r.get("degraded_reason", "")
                                 for r in degraded],
            "sc_vote": {
                "silent_interception": sc_s, "correct_false_alarm": sc_c,
                "cost_total": sc_cost,
                "cost_per_question": {k: round(sc_cost[k] / max(n_cost, 1), 1)
                                      for k in sc_cost}},
            "chess_ut": {
                "silent_interception": ut_s, "correct_false_alarm": ut_c,
                "cost_total": ut_cost,
                "cost_per_question": {k: round(ut_cost[k] / max(n_cost, 1), 1)
                                      for k in ut_cost}},
            "mcnemar_silent_sc_vs_v12": mc(sil, ("sc", "alarm")),
            "mcnemar_silent_ut_vs_v12": mc(sil, ("ut", "alarm")),
        }

    out = {
        "config": {
            "engines": {e: {"slug": s, "frozen_tag": t}
                        for e, (s, t) in ENGINES.items()},
            "question_set_source":
                "results/eA_tier2_real_verification.json traces "
                "(kind 字段权威,不重算);冻结预测 = tier2 运行 q_*.json pred_sql",
            "shared_sampling":
                f"每题 {N_SAMPLES} 个候选 SQL,temperature={GEN_TEMPERATURE}"
                "(Kimi-k2.6 thinking-disabled 平台仅允许 0.6,全部 phase "
                f"钳制为 0.6: PLATFORM_TEMP_CAP={PLATFORM_TEMP_CAP});"
                f"seed={SEED_BASE}*question_id+i(i=0..{N_SAMPLES - 1});"
                "system=rgcv.llm.SYS_GEN,user=### Database Schema(全 schema "
                "注入,与 gated_fullschema 冻结预测同口径 full_schema_result)"
                "+### Question(含 [Hint],同 tier-2)+单 SQL 指令;"
                "Kimi 平台可能忽略 seed,如实记录",
            "sc_alarm_def":
                "4 采样候选按执行结果多重集等价聚类(exec_equal:多重集+"
                "1e-6 相对浮点容差);执行失败者统一 exec-error 簇;"
                "alarm = 簇数 >= 2;交付 = 最大簇代表结果(平票取先出现)",
            "ut_alarm_def":
                "UT 生成 1 调用(T=0;kimi 平台钳 0.6)→ 5 条正确结果性质断言;<2 条视为 UT "
                "生成失败,该题两机制均记 alarm=NA(degraded,剔除出两侧 "
                "指标分母);候选集 = 冻结预测 + 4 采样共 5;判别调用按执行"
                "结果多重集哈希去重,每唯一成功结果 1 调用(T=0,问题+断言+"
                "代表候选 SQL+结果前 20 行 JSON(单元格截 80 字符)+执行失败"
                "标记 → 长度 N 布尔数组);执行失败候选 pass_rate=0 跳过判别,"
                "判别解析失败记 pass_rate=0 并计数;alarm = 冻结预测 "
                "pass_rate < max(全部候选);交付 = pass_rate 最高(平票冻结"
                "预测优先)",
            "cost_accounting":
                "SC 行 = 4 gen 调用 usage;UT 行 = 4 gen + 1 UT-gen + 去重后 "
                "judge usage(共享 4 gen 在两机制行分别完整计入);分母 = "
                "done+degraded 题(含 degraded 的实际开销)",
            "ex_criteria": "exec_equal:gold 执行结果多重集比较 + 1e-6 相对"
                           "浮点容差(evalx.exec_equal)",
            "ci": "bootstrap 10^4, percentile 95%(exp_tier2_real.boot_ci 同款)",
            "llm_failure_policy":
                "gen/judge 调用重试耗尽(空响应)→ 该题记 failed 跳过;"
                "UT 断言 <2 → degraded;judge 解析失败记 pass_rate=0",
        },
        "summary": summary,
        "traces": all_traces,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ec_verification_baselines.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    print("\n===== E-C verification baselines (SC-vote vs CHESS-UT) =====")
    for slug, s in summary.items():
        if s.get("status") == "not_run":
            print(f"\n[{slug}] NOT RUN: {s['reason']}")
            continue
        print(f"\n[{slug}] n={s['n_questions']} failed={s['n_failed']} "
              f"degraded={s['n_degraded']}")
        for mech in ("sc_vote", "chess_ut"):
            m = s[mech]
            si, fa = m["silent_interception"], m["correct_false_alarm"]
            print(f"  {mech:<9} silent: {si['alarm_n']}/{si['n']} = "
                  f"{si['rate'] * 100:.1f}%"
                  f"[{si['ci95'][0] * 100:.1f},{si['ci95'][1] * 100:.1f}]  "
                  f"fix={si['corrections_n']}  ||  correct FPR: "
                  f"{fa['alarm_n']}/{fa['n']} = {fa['rate'] * 100:.1f}%"
                  f"[{fa['ci95'][0] * 100:.1f},{fa['ci95'][1] * 100:.1f}]  ||"
                  f"  cost {m['cost_total']['calls']}c "
                  f"{m['cost_total']['tokens_in']}+"
                  f"{m['cost_total']['tokens_out']} "
                  f"(avg {m['cost_per_question']['calls']}c/"
                  f"{m['cost_per_question']['tokens_in']}+"
                  f"{m['cost_per_question']['tokens_out']})")
        for key in ("mcnemar_silent_sc_vs_v12", "mcnemar_silent_ut_vs_v12"):
            m = s[key]
            print(f"  {key}: b={m['b']} c={m['c']} p={m['p_value']}")
    print(f"\n[saved] results/ec_verification_baselines.json", flush=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="", choices=["", *ENGINES])
    ap.add_argument("--ids", default="", help="comma-separated question_id")
    ap.add_argument("--smoke", type=int, default=0,
                    help="smoke N 题(静默+正确混合,独立 _smoke 目录)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    if a.aggregate_only:
        aggregate()
        return
    if not a.engine:
        ap.error("--engine required (or --aggregate-only)")
    run(a.engine,
        ids=[int(x) for x in a.ids.split(",") if x.strip()] or None,
        smoke=a.smoke, workers=a.workers, retry_failed=a.retry_failed)


if __name__ == "__main__":
    main()
