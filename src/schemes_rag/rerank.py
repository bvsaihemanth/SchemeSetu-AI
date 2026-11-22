"""Cross-encoder reranking. Off by default, one env var away from on.

Why this exists as its own module with a null path: reranking is the single
biggest retrieval quality win available to you, but it adds a second model
(~2.3 GB) and ~200-400 ms per query on GPU. Build and measure without it,
then switch RERANK_ENABLED=true and compare on your golden set. The seam is
here so that flip never touches retriever.py.

BGE-reranker-v2-m3 pairs with BGE-M3 and handles Indic scripts, which
matters for a corpus with Hindi scheme names transliterated inconsistently.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str, use_fp16: bool) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None

    def _ensure(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("Loading reranker %s on %s", self.model_name, self.device)
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16 and self.device != "cpu",
                devices=self.device,
            )
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._ensure()
        # normalize=True -> sigmoid output in [0, 1], so RETRIEVAL_MIN_SCORE
        # stays on the same scale whether reranking is on or off.
        scores = model.compute_score([[query, p] for p in passages], normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


@lru_cache
def get_reranker() -> CrossEncoderReranker | None:
    s = get_settings()
    if not s.rerank_enabled:
        return None
    return CrossEncoderReranker(s.rerank_model, s.embed_device, s.embed_use_fp16)
