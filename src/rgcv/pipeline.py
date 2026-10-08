"""RGCV 端到端闭环流水线(y.docx 3.2 总体架构)。

在线闭环:检索(模块一)→ 生成(LLM/规则代理)→ 修复(模块二)→
验证(模块三),报警回传模块二做差分修复一轮;全流程由成本感知预算
控制器统一记账(式(3))。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .budget import BudgetController
from .corrector import Corrector, RepairOutcome
from .dbs import Catalog, DBError, DBExecutor
from .generator import GenerationOutput, RuleGenerator
from .schema_graph import (RetrievalResult, SchemaGraph, SchemaGraphRAG,
                           full_schema_result)
from .verifier import VerificationReport, Verifier


@dataclass
class RunRecord:
    question: str
    gold_sql: Optional[str]
    retrieved: Dict = field(default_factory=dict)
    candidates: List[str] = field(default_factory=list)
    final_sql: str = ""
    ex: Optional[bool] = None
    executed_ok: bool = False
    repair: Dict = field(default_factory=dict)
    verification: Dict = field(default_factory=dict)
    cost: Dict = field(default_factory=dict)
    latency_s: float = 0.0
    events: List[str] = field(default_factory=list)


class RGCVPipeline:
    def __init__(self, graph: SchemaGraph, executor: DBExecutor,
                 catalog: Catalog, generator,   # RuleGenerator | LLMGenerator
                 budget_factory: Callable[[], BudgetController],
                 cfg: Optional[Dict] = None):
        self.graph = graph
        self.ex = executor
        self.cat = catalog
        self.gen = generator
        self.budget_factory = budget_factory
        self.cfg = dict(
            k1=6, k2_per_table=12, retrieval_budget_tokens=4000,
            n_candidates=1, use_repair=True, signals=("s1", "s3", "s2"),
            use_verifier=True, verify_layers=("v1", "v2", "v3"),
            diff_repair_rounds=1, seed=0, repair_gate="off",
            full_schema=False)
        self.cfg.update(cfg or {})
        self.retriever = SchemaGraphRAG(
            graph, k1=self.cfg["k1"], k2_per_table=self.cfg["k2_per_table"])
        self.verifier = Verifier(executor, catalog, layers=tuple(
            self.cfg["verify_layers"]),
            llm_judge=getattr(generator, "judge", None))
        self.corrector = Corrector(
            executor, catalog, signals=tuple(self.cfg["signals"]),
            gate=self.cfg["repair_gate"])
        regen = getattr(generator, "regenerate", None)
        if regen is not None:            # 真实 LLM 重生成(位点 2)
            self.corrector.set_regenerator(regen)
        else:                            # 规则代理回退:再取一个规则候选
            self.corrector.set_regenerator(
                lambda q, bad: self.gen.generate(
                    q, self._last_retrieval).candidates[
                    min(self._gen_attempts, 2)])

    _last_retrieval: RetrievalResult = None
    _gen_attempts: int = 0

    def run(self, question: str, gold_sql: Optional[str] = None,
            injected_sql: Optional[str] = None) -> RunRecord:
        t0 = time.perf_counter()
        budget = self.budget_factory()
        rec = RunRecord(question, gold_sql)
        self._gen_attempts = 0

        # ---- 模块一:检索 + 注入(full_schema 消融则免检索全量直注)
        if self.cfg["full_schema"]:
            r = full_schema_result(self.graph, question)
        else:
            r = self.retriever.retrieve(question)
            self.retriever.inject(r, self.cfg["retrieval_budget_tokens"])
        self._last_retrieval = r
        budget.spend("retrieval", tokens_in=r.prompt_tokens,
                     latency_s=r.latency_s)
        rec.retrieved = {
            "tables": [t for t, _ in r.tables],
            "table_scores": {t: round(s, 3) for t, s in r.tables},
            "columns": [(h.table, h.column, round(h.score, 3))
                        for h in r.columns],
            "join_paths": r.join_paths,
            "prompt_tokens": r.prompt_tokens,
            "latency_s": r.latency_s,
        }
        rec.events.append(f"retrieve:{len(r.tables)} tables,"
                          f"{len(r.columns)} cols,"
                          f"{r.prompt_tokens} tok")

        # ---- 生成(候选起点:注入 SQL 或生成器输出)
        if injected_sql is not None:
            candidates = [injected_sql]
            budget.spend("generation", tokens_in=r.prompt_tokens,
                         tokens_out=0)
        else:
            go: GenerationOutput = self.gen.generate(
                question, r, n_candidates=self.cfg["n_candidates"])
            candidates = go.candidates
            budget.spend("generation", tokens_in=go.tokens_prompt,
                         tokens_out=go.tokens_out)
        rec.candidates = candidates
        sql = candidates[0]

        # ---- 模块二:修复
        if self.cfg["use_repair"]:
            outcome: RepairOutcome = self.corrector.repair(
                sql, question, reference_sql=gold_sql)
            rec.repair = {
                "n_steps": outcome.n_steps,
                "signals": list(dict.fromkeys(outcome.signals_used)),
                "fallback": outcome.fallback_used,
                "executed": outcome.executed,
            }
            if outcome.sql and outcome.sql != sql:
                rec.events.append(f"repair:{outcome.n_steps} steps")
            sql = outcome.sql
            candidates = list(dict.fromkeys(candidates + [sql]))

        # ---- 执行状态
        try:
            self.ex.execute(sql)
            rec.executed_ok = True
        except DBError:
            rec.executed_ok = False

        # ---- 模块三:验证 + 差分修复回传
        if self.cfg["use_verifier"] and rec.executed_ok:
            vrep: VerificationReport = self.verifier.verify(
                question, sql, candidates)
            rec.verification = {
                "verdict": vrep.verdict,
                "alarms": vrep.alarmed_clauses,
                "layers": vrep.layer_reports,
                "triggered_v3": vrep.triggered_v3,
                "confidence": vrep.confidence,
                "latency_s": vrep.latency_s,
            }
            rec.events.append(f"verify:{vrep.verdict}")
            # 报警 → 模块二差分修复一轮(S4 信号)
            if vrep.verdict in ("alarm", "reject") and \
                    self.cfg["diff_repair_rounds"] > 0 and \
                    self.cfg["use_repair"]:
                c2 = Corrector(self.ex, self.cat,
                               signals=tuple(self.cfg["signals"]) + ("s4",),
                               budget=budget,
                               gate=self.cfg["repair_gate"])
                o2 = c2.repair(sql, question)
                if o2.sql != sql:
                    try:
                        self.ex.execute(o2.sql)
                        rec.executed_ok = True
                        sql = o2.sql
                        rec.events.append("diff-repair:1 round")
                    except DBError:
                        pass

        rec.final_sql = sql
        rec.cost = budget.total_spent()
        rec.latency_s = round(time.perf_counter() - t0, 3)
        return rec
