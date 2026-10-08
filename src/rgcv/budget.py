"""成本感知预算控制器(y.docx 3.1 定义 5 / 4.5)。

统一记账:检索注入、生成、修复搜索、验证的 token 与延迟;
默认按 6 : 2.5 : 1.5 分配给生成 / 修复 / 验证;超限即接受当前最优候选。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

import tiktoken

_enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """o200k_base token 计数(与 TRAE 模型实际 tokenizer 的近似,报告中标注)。"""
    if not text:
        return 0
    return len(_enc.encode(text))


class BudgetExceeded(Exception):
    pass


@dataclass
class CostRecord:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    n_calls: int = 0


@dataclass
class BudgetController:
    total: int = 32_000          # B:每题总 token 预算(y.docx A4 扫描 4k/8k/16k/32k)
    split: Dict[str, float] = field(
        default_factory=lambda: {"generation": 0.6, "repair": 0.25,
                                 "verification": 0.15})
    # 检索注入属于输入侧,计入 generation 分摊
    _spent: Dict[str, CostRecord] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)

    def phase_budget(self, phase: str) -> int:
        if phase == "retrieval":   # 检索 token 计入生成(注入)侧
            phase = "generation"
        return int(self.total * self.split.get(phase, 0.2))

    def spend(self, phase: str, tokens_in: int = 0, tokens_out: int = 0,
              latency_s: float = 0.0):
        key = "generation" if phase == "retrieval" else phase
        rec = self._spent.setdefault(key, CostRecord())
        rec.tokens_in += tokens_in
        rec.tokens_out += tokens_out
        rec.latency_s += latency_s
        rec.n_calls += 1

    def remaining(self, phase: str) -> int:
        key = "generation" if phase == "retrieval" else phase
        rec = self._spent.get(key, CostRecord())
        return self.phase_budget(phase) - (rec.tokens_in + rec.tokens_out)

    def exceeded(self, phase: str) -> bool:
        return self.remaining(phase) <= 0

    def total_spent(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {"tokens_in": v.tokens_in, "tokens_out": v.tokens_out,
                "tokens": v.tokens_in + v.tokens_out,
                "latency_s": round(v.latency_s, 4), "n_calls": v.n_calls}
            for k, v in self._spent.items()
        }

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0
