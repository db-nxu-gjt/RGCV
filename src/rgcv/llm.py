"""E5 真实 LLM 接入层(原型 LLM 位点 → 双骨干 API)。

y.docx 生成器骨干为 DeepSeek-V4-Pro / Kimi-K2.6;原型以规则代理离线
运行并把 LLM 位点以钩子保留(generator / set_regenerator / llm_judge)。
本模块提供三个钩子的真实 LLM 实现 + 统一客户端:

  - LLMClient:OpenAI 兼容双骨干客户端。全部调用关闭 reasoning
    (deepseek t=0;kimi thinking-disabled t=0.6 平台固定),
    usage 实测记账 + 行级 jsonl trace + 指数退避重试 + 输出容错解析。
  - LLMGenerator:RuleGenerator 同接口生成器(检索注入 prompt → SQL);
    附 regenerate(修复回退钩子, question+bad_sql → 新 SQL)与
    judge(V3 语义核对钩子, question+sql+cols+rows → (verdict, clause))。

成本口径:预算控制器记账用 API usage 实测值(区别于原型的
tiktoken 估计);api_trace.jsonl 行级 append,坏行跳过。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sqlglot  # noqa: F401  (解析校验用)

from openai import OpenAI  # noqa: E402

from .generator import GenerationOutput  # noqa: E402
from .schema_graph import RetrievalResult  # noqa: E402

# API key 经环境变量注入:DEEPSEEK_API_KEY / MOONSHOT_API_KEY(见 README)。
ENGINES: Dict[str, Dict] = {
    "deepseek-v4-pro": {
        "base": "https://api.deepseek.com/v1",
        "key": os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0,            # 非推理,确定性
        "max_tokens": 8192,
        # 关闭 reasoning(实测:默认带 reasoning_content,completion 27→0)
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    # E-H 中等强度点:DeepSeek-V4-Flash(同端点,t=0 与 pro 口径一致)
    "deepseek-v4-flash": {
        "base": "https://api.deepseek.com/v1",
        "key": os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0,
        "max_tokens": 8192,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "kimi-k2.6": {
        "base": "https://api.moonshot.cn/v1",
        "key": os.getenv("MOONSHOT_API_KEY", ""),
        "temperature": 0.6,          # thinking-disabled 模式平台固定 t=0.6
        "max_tokens": 16384,
        "extra_body": {"thinking": {"type": "disabled"}},  # 关闭 reasoning
    },
    # E-H 中等强度点(T1-1):本地 Ollama qwen2.5-coder:7b(OpenAI 兼容端点)。
    # 模型 qwen25c-rgcv = qwen2.5-coder:7b + PARAMETER num_ctx 8192
    # (默认 num_ctx=2048 会截断 schema prompt;Modelfile 见 experiments/)。
    "qwen25c-rgcv": {
        "base": "http://localhost:11434/v1",
        "key": "ollama",             # Ollama 不校验 key,占位非空即可
        "temperature": 0,            # 与 deepseek 两侧点口径一致(确定性)
        "max_tokens": 8192,
        "extra_body": {"options": {"num_ctx": 8192}},   # 双保险:请求级覆盖
    },
}

RETRY_N = 3
RETRY_BACKOFF_S = (2, 5, 10)


class LLMClient:
    """双骨干统一客户端:usage 记账 + jsonl trace + 重试容错。"""

    def __init__(self, engine: str, trace_path: Optional[str] = None):
        assert engine in ENGINES, f"unknown engine {engine}"
        self.engine = engine
        self.cfg = ENGINES[engine]
        if not self.cfg["key"]:
            raise RuntimeError(
                f"API key for {engine} is not set; export "
                f"DEEPSEEK_API_KEY / MOONSHOT_API_KEY (see README)")
        self.cli = OpenAI(api_key=self.cfg["key"], base_url=self.cfg["base"],
                          timeout=180.0, max_retries=0)
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._qid: Optional[str] = None
        self.usage_by_qid: Dict[str, Dict] = {}
        self.total_usage = {"calls": 0, "tokens_in": 0, "tokens_out": 0}

    # ------------------------------------------------------------- 状态
    def set_qid(self, qid) -> None:
        """当前题 id(串行驱动下单题内所有调用共享记账桶)。"""
        self._qid = str(qid)

    def usage_for_qid(self, qid) -> Dict:
        return self.usage_by_qid.get(str(qid),
                                     {"calls": 0, "tokens_in": 0,
                                      "tokens_out": 0})

    # ------------------------------------------------------------- 主入口
    def chat(self, phase: str, system: str, user: str,
             max_tokens: Optional[int] = None,
             temperature: Optional[float] = None,
             seed: Optional[int] = None) -> Tuple[str, int, int]:
        """返回 (text, tokens_in, tokens_out);失败重试,耗尽返回空串。

        temperature/seed 仅在显式传入时覆盖引擎默认(E-C 方差实验:
        修复重生成作为显式随机源用 t=0.6+seed;主生成保持 t=0 不变)。
        """
        last_err = ""
        for attempt in range(RETRY_N):
            t0 = time.perf_counter()
            try:
                req = {"model": self.engine,
                       "messages": [
                           {"role": "system", "content": system},
                           {"role": "user", "content": user}],
                       "temperature": self.cfg["temperature"],
                       "max_tokens": max_tokens or self.cfg["max_tokens"],
                       "extra_body": self.cfg["extra_body"] or None}
                if temperature is not None:
                    req["temperature"] = temperature
                if seed is not None:
                    req["seed"] = seed
                rsp = self.cli.chat.completions.create(**req)
                text = rsp.choices[0].message.content or ""
                u = getattr(rsp, "usage", None)
                uin = getattr(u, "prompt_tokens", 0) or 0
                uout = getattr(u, "completion_tokens", 0) or 0
                self._record(phase, uin, uout,
                             time.perf_counter() - t0, True, "")
                return text, uin, uout
            except Exception as e:            # API/网络/格式错误 → 退避重试
                last_err = f"{type(e).__name__}: {e}"[:200]
                self._record(phase, 0, 0, time.perf_counter() - t0,
                             False, last_err)
                if attempt < RETRY_N - 1:
                    time.sleep(RETRY_BACKOFF_S[min(attempt,
                                                   len(RETRY_BACKOFF_S) - 1)])
        print(f"[llm:{self.engine}] {phase} FAILED after {RETRY_N} tries: "
              f"{last_err}", flush=True)
        return "", 0, 0

    # ------------------------------------------------------------- 记账
    def _record(self, phase: str, uin: int, uout: int, lat: float,
                ok: bool, err: str) -> None:
        qid = self._qid or "-"
        with self._lock:
            if ok:
                u = self.usage_by_qid.setdefault(
                    qid, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
                u["calls"] += 1
                u["tokens_in"] += uin
                u["tokens_out"] += uout
                self.total_usage["calls"] += 1
                self.total_usage["tokens_in"] += uin
                self.total_usage["tokens_out"] += uout
            if self.trace_path:
                row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "engine": self.engine, "phase": phase, "qid": qid,
                       "tokens_in": uin, "tokens_out": uout,
                       "latency_s": round(lat, 3), "ok": ok, "err": err}
                try:
                    with open(self.trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                except OSError:
                    pass


# ------------------------------------------------------------------ 输出解析
_FENCE = re.compile(r"```(?:sql|SQLite)?", re.I)


def _first_sql(text: str) -> Optional[str]:
    """从自由文本中提取第一段 SQL(SELECT/WITH 起,分号或文末止)。"""
    if not text:
        return None
    text = _FENCE.sub("", text)
    m = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if not m:
        return None
    seg = text[m.start():].strip()
    i = seg.find(";")
    if i != -1:
        seg = seg[:i + 1]
    seg = seg.strip()
    # 容错校验:能被 sqlglot 解析才返回(解析失败仍返回,由执行层兜底)
    return seg or None


def extract_sqls(text: str, n: int) -> List[str]:
    """解析模型输出为 n 个候选 SQL;n>1 时按 [SQLk] 编号或 fence 分段。"""
    if not text:
        return []
    if n <= 1:
        s = _first_sql(text)
        return [s] if s else []
    cleaned = _FENCE.sub("", text)
    parts = re.split(r"\[\s*SQL\s*\d+\s*\]", cleaned, flags=re.I)
    sqls: List[str] = []
    for p in parts[1:] if len(parts) > 1 else parts:
        s = _first_sql(p)
        if s:
            sqls.append(s)
    if not sqls:                       # 编号缺失 → 逐个 SELECT 语句切分
        for m in re.finditer(r"\b(?:SELECT|WITH)\b.*?(?:;|$)", cleaned,
                             re.I | re.S):
            s = m.group(0).strip()
            if s.rstrip().endswith(";") or len(sqls) == 0:
                sqls.append(s)
            if len(sqls) >= n:
                break
    return sqls[:n]


# ------------------------------------------------------------------ Prompt
SYS_GEN = (
    "You are an expert SQLite SQL generator. Given the database schema and a "
    "natural language question, write a correct SQLite query. Rules: use only "
    "tables and columns that appear in the schema; respect SQLite dialect; "
    "output ONLY the SQL statement ending with a semicolon; no explanations, "
    "no markdown fences.")

SYS_REGEN = (
    "You are an expert SQLite SQL repairer. The previous SQL failed to "
    "execute or failed semantic verification. Rewrite it so it correctly "
    "answers the question. Output ONLY the corrected SQL statement ending "
    "with a semicolon; no explanations.")

SYS_JUDGE = (
    "You are a strict SQL semantic verifier. You see a question, a candidate "
    "SQL and its execution result. Decide whether the result answers the "
    "question. Reply with a single JSON object only: "
    '{"verdict": "pass|alarm|reject", "clause": "<suspicious SQL clause or '
    'empty string>"}. pass=correct; alarm=possibly wrong, name the clause; '
    "reject=certainly wrong, name the clause.")


class LLMGenerator:
    """RuleGenerator 同接口的真实 LLM 生成器(位点 1/2/3 三合一)。

    generate   — 位点 1:question + 检索注入 → n 个候选 SQL
    regenerate — 位点 2:修复回退钩子 fn(question, bad_sql) -> new_sql|None
    judge      — 位点 3:V3 语义核对 fn(question, sql, cols, rows) -> (verdict, clause)
    """

    def __init__(self, client: LLMClient, repair_seed: Optional[int] = None):
        self.cli = client
        # E-C 方差实验:repair_seed 非 None 时,修复重生成作为显式随机源
        # (t=0.6 + seed);None 时保持引擎默认(DeepSeek t=0,确定性)。
        self.repair_seed = repair_seed

    # ------------------------------- 位点 1:生成
    def generate(self, question: str, retrieval: RetrievalResult,
                 n_candidates: int = 1) -> GenerationOutput:
        schema = getattr(retrieval, "prompt", "") or ""
        base = f"### Database Schema\n{schema}\n\n### Question\n{question}"
        user = base + (
            f"\n\nWrite {n_candidates} DIFFERENT candidate SQLite queries. "
            "Label them [SQL1], [SQL2], ... one per candidate. Output only "
            "the SQL candidates."
            if n_candidates > 1 else
            "\n\nWrite the SQLite query. Output only the SQL.")
        last_u = [0, 0]
        sqls: List[str] = []
        for _ in range(2):              # 语义层重试一次(格式容错)
            text, uin, uout = self.cli.chat("generation", SYS_GEN, user)
            last_u = [uin, uout]
            sqls = extract_sqls(text, n_candidates)
            if sqls:
                break
            user = base + ("\n\nIMPORTANT: output ONLY SQL statement(s) "
                           "starting with SELECT/WITH and ending with ';'. "
                           "No explanations.")
        if not sqls:
            sqls = [""]                 # 空预测,评测计 0
        while len(sqls) < n_candidates:
            sqls.append(sqls[-1])
        return GenerationOutput(
            sql=sqls[0], n_candidates=n_candidates, candidates=sqls,
            tokens_prompt=last_u[0], tokens_out=last_u[1])

    # ------------------------------- 位点 2:修复回退重生成
    def regenerate(self, question: str, bad_sql: str) -> Optional[str]:
        user = (f"### Question\n{question}\n\n### Failed SQL\n{bad_sql}\n\n"
                "Rewrite the SQL so it executes correctly on the schema and "
                "answers the question. Output only the corrected SQL.")
        temp = 0.6 if self.repair_seed is not None else None
        for _ in range(2):
            text, _, _ = self.cli.chat("regen", SYS_REGEN, user,
                                       temperature=temp,
                                       seed=self.repair_seed)
            s = _first_sql(text)
            if s and s != bad_sql:
                return s
            if s:
                return s
        return None

    # ------------------------------- 位点 3:V3 语义核对
    def judge(self, question: str, sql: str, cols, rows) -> Tuple[str, str]:
        def _cell(v):
            return v.decode("utf-8", "ignore") if isinstance(v, bytes) else v
        try:
            rows_txt = json.dumps(
                [[str(_cell(v)) for v in r] for r in list(rows)[:10]],
                ensure_ascii=False)
        except Exception:
            rows_txt = "[]"
        user = (f"### Question\n{question}\n\n### SQL\n{sql}\n\n"
                f"### Result columns\n{list(cols)}\n\n"
                f"### Sample rows (up to 10)\n{rows_txt}\n\n"
                "Judge the result against the question. Reply JSON only.")
        text, _, _ = self.cli.chat("judge", SYS_JUDGE, user, max_tokens=512)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return "pass", ""           # 解析失败:无证据不惩罚,trace 已记录
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("verdict", "pass")).lower()
            if verdict not in ("pass", "alarm", "reject"):
                verdict = "pass"
            clause = str(obj.get("clause", "") or "")[:200]
            return verdict, clause
        except (json.JSONDecodeError, AttributeError):
            return "pass", ""
