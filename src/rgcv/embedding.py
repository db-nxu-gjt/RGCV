"""混合检索组件:BM25 + BGE-M3 稠密召回 + BGE-reranker-v2-m3 重排。

y.docx 4.1.2:第一阶段表级粗检采用 BM25 与 BGE-M3 稠密检索的混合召回,
经 BGE-reranker-v2-m3 重排后取 top-k1 张表。
模型均本地部署(RTX A5500 16GB),不依赖闭源服务。
"""
from __future__ import annotations

import functools
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

MODEL_DIR_M3 = None   # 由 config 注入
MODEL_DIR_RR = None


def set_model_dirs(m3: str, reranker: str):
    global MODEL_DIR_M3, MODEL_DIR_RR
    MODEL_DIR_M3, MODEL_DIR_RR = m3, reranker


def _device() -> str:
    """环境实测:本机 torch 为 CPU 版(2.0.1+cpu),GPU 不可用。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@functools.lru_cache(maxsize=1)
def _dense_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_DIR_M3, device=_device())


@functools.lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    # CPU 环境实测:512 截断的重排开销过高;表级文档较短,256 足够
    return CrossEncoder(MODEL_DIR_RR, device=_device(), max_length=256)


_token_re = re.compile(r"[A-Za-z]+|\d+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text or "")]


class HybridIndex:
    """对一组文档(表级或列级)建混合索引,支持 query 混合打分。"""

    def __init__(self, keys: Sequence[str], docs: Sequence[str]):
        from rank_bm25 import BM25Okapi
        self.keys = list(keys)
        self.docs = list(docs)
        self._toks = [tokenize(d) for d in self.docs]
        self._bm25 = BM25Okapi(self._toks) if self._toks else None
        self._emb = None

    def _dense(self):
        if self._emb is None and self.docs:
            self._emb = _dense_model().encode(
                self.docs, normalize_embeddings=True, show_progress_bar=False)
        return self._emb

    def search(self, query: str, use_bm25: bool = True, use_dense: bool = True,
               topk: int = 10) -> List[Tuple[str, float]]:
        scores = np.zeros(len(self.docs), dtype=np.float64)
        if use_bm25 and self._bm25 is not None:
            s = np.array(self._bm25.get_scores(tokenize(query)))
            s = s / (s.max() + 1e-9)          # min-max 归一化到 [0,1]
            scores += s
        if use_dense and self.docs:
            q = _dense_model().encode([query], normalize_embeddings=True,
                                      show_progress_bar=False)
            sim = (self._dense() @ q.T).ravel()
            scores += np.clip(sim, 0, None)
        order = np.argsort(-scores)[:topk]
        return [(self.keys[i], float(scores[i])) for i in order]


@functools.lru_cache(maxsize=8192)
def _rerank_cached(query: str, candidates: Tuple[str, ...]) -> Tuple[
        Tuple[int, float], ...]:
    pairs = [[query, c] for c in candidates]
    scores = _reranker().predict(pairs, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))
    return tuple((int(i), float(scores[i])) for i in order)


def rerank(query: str, candidates: List[str],
           topk: int = 10) -> List[Tuple[int, float]]:
    """cross-encoder 重排,返回 (索引, 分数) 列表。

    同 (query, candidates) 跨配置/重复调用走缓存(CPU 重排是主要开销)。
    """
    if not candidates:
        return []
    full = _rerank_cached(query, tuple(candidates))
    return list(full[:topk])
