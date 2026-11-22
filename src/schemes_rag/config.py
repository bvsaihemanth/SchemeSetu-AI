"""Central configuration. Everything tunable lives here, sourced from env/.env.

Nothing else in the codebase reads os.environ directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------- Qdrant ----------
    # "server" = real Qdrant over HTTP (Docker or binary). Required for HNSW,
    # payload indexes, and concurrent ingest + query.
    # "local"  = embedded on-disk mode. Single process only: it takes an
    #            exclusive lock on the folder, so ingest and API cannot both
    #            run. Brute-force search, no payload indexes. Dev fallback.
    qdrant_mode: Literal["server", "local"] = "server"
    qdrant_local_path: str = r"C:\Users\bomma\Documents\RAG\myscheme-ai\vector-db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "gov_schemes"
    qdrant_timeout: float = 30.0
    # int8 scalar quantization: ~4x less RAM for the dense index, ~1% recall loss.
    # Harmless at 725 docs, essential at 100k.
    qdrant_quantization: bool = True
    # Keep raw vectors on disk, quantized ones in RAM.
    qdrant_on_disk: bool = True
    # Shard count CANNOT be changed after creation without a full re-ingest.
    # 4 shards on one node costs almost nothing now and lets you add nodes
    # later and rebalance instead of rebuilding the collection.
    qdrant_shard_number: int = 4
    # Raise to 2 only once you run a multi-node cluster.
    qdrant_replication_factor: int = 1

    # ---------- Embeddings (BGE-M3) ----------
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    # Was forced to "cpu" because BGE-M3 (2.3 GB) and Qwen's KV cache could
    # not share a 4 GB card -- they fit, then ran at CPU speed. With
    # generation hosted, the GPU is free: "cuda" cuts query embedding from
    # ~250 ms to ~25 ms. Still CPU-safe if you'd rather keep the card idle.
    embed_device: Literal["cpu", "cuda", "mps"] = "cuda"
    embed_use_fp16: bool = True
    embed_batch_size: int = 8
    embed_max_length: int = 1024
    # BGE-M3 emits dense + sparse (lexical) from one forward pass. We always
    # store both so hybrid search is a config flip, never a re-ingest.
    embed_store_sparse: bool = True
    # Optional: run embeddings as a separate HTTP service instead of in-process.
    # Set this once you have >1 API worker (see docs/scaling).
    embed_service_url: str | None = None

    # ---------- Chunking ----------
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 40

    # ---------- LLM (Gemini via its OpenAI-compatible endpoint) ----------
    # Gemini speaks the OpenAI chat-completions protocol at this base URL, so
    # llm.py stays a plain OpenAI client and switching providers is config,
    # not code. To go back to a local model:
    #   Ollama : http://localhost:11434/v1   (api key ignored)
    #   vLLM   : http://localhost:8000/v1
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Set GEMINI/LLM_API_KEY in .env -- never commit a real key.
    llm_api_key: str = "api-key-here"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.1
    # Thinking tokens are billed against this budget on Gemini, so a limit
    # tight enough for a non-thinking model yields an EMPTY answer: the model
    # spends the allowance deliberating and never reaches the text. llm.py
    # raises ThinkingOnlyResponse rather than returning "" when that happens.
    llm_max_tokens: int = 2048
    # A hosted model that has not answered in 60 s is not slow, it is broken.
    llm_timeout: float = 60.0
    # Was 6000 for a local 4B, which degraded badly past ~8k of stuffed
    # context. Gemini's window is orders of magnitude larger, so the binding
    # constraint is now cost and answer focus rather than model capacity.
    llm_context_budget_tokens: int = 20000
    # Gemini thinks by default. For extractive answering over retrieved text
    # the reasoning is already done -- it is in the sources -- so deliberation
    # buys latency and tokens, not accuracy. Mapped to the OpenAI-compatible
    # `reasoning_effort` field: "none" | "minimal" | "low" | "medium" | "high".
    #
    # "none" is rejected by models that cannot disable thinking at all
    # (Gemini 2.5 Pro and the Gemini 3 family), so "low" is the safe default:
    # accepted everywhere, and cheap. Set to None to omit the field entirely.
    llm_reasoning_effort: str | None = "low"
    # Headroom added to EVERY request's max_tokens to pay for deliberation.
    #
    # This exists because thinking tokens are billed against max_tokens, so a
    # budget sized for a non-thinking model leaves nothing for the answer. The
    # agents ask for what their OUTPUT needs (700 for an eligibility JSON, 256
    # for the router) and should not each have to know the provider thinks --
    # so the allowance is added here, once.
    #
    # It is a ceiling, not a spend: unused tokens cost nothing. Observed
    # failure without it -- the eligibility assessor returned truncated,
    # unparseable JSON on the turn where it had to reconcile five answers
    # against five criteria, which surfaced to the user as "not enough
    # information to decide" about a scheme it had just been told five facts
    # about. Set to 0 when serving a non-thinking model.
    llm_thinking_headroom_tokens: int = 1536
    # Price per 1,000,000 tokens, for cost-per-request reporting. CONFIG, not
    # constants: rates differ per model, change without notice, and a
    # hardcoded number goes stale silently. Set them from your provider's
    # pricing page -- `evaluate_rag.py` prints which rates it used so a cost
    # figure is always traceable to an input you can check.
    #
    # Input and output are separate because they differ several-fold, and in
    # a RAG system the split is lopsided: ~20k tokens of retrieved context in,
    # a few hundred out. A blended rate would misattribute almost all of it.
    llm_price_input_per_mtok: float = 0.10
    llm_price_output_per_mtok: float = 0.40
    # Retries for TRANSIENT failures only: 429 (rate limited) and 5xx (the
    # provider is overloaded). Distinct from the SDK's own max_retries, which
    # stays at 0 -- that retries everything including timeouts, which turns
    # one 60 s wait into three and looks like a hang.
    #
    # These are different: the server is explicitly saying "later", often with
    # a Retry-After telling you how much later. Not retrying them means a
    # batch job dies on the first burst, which is exactly what an 85-question
    # evaluation is.
    llm_max_retries: int = 4
    llm_retry_base_delay: float = 2.0
    #: Never wait longer than this for a single retry, however large a
    #: Retry-After the server sends. A provider asking for 15 minutes should
    #: surface as a failure you can see, not a process that appears hung.
    llm_retry_max_delay: float = 60.0
    # Client-side request ceiling, requests per minute. 0 disables it.
    #
    # Needed because retries alone do not fix a sustained overrun: 85
    # questions times ~10 calls fired back to back will exhaust a free-tier
    # quota no matter how politely each failure is retried. Pacing turns a
    # cascade of 429s into a slower run that finishes.
    llm_max_requests_per_minute: int = 0
    # Qwen3's /no_think soft switch, appended to the system prompt. Meaningless
    # to Gemini -- and prompt noise -- so it is off unless serving Qwen again.
    llm_no_think_suffix: bool = False
    # Extra JSON merged into the request body for provider options the OpenAI
    # schema lacks. Gemini accepts a nested thinking config, e.g.
    #   LLM_EXTRA_BODY={"google": {"thinking_config": {"thinking_budget": 0}}}
    # Ollama >= 0.9: LLM_EXTRA_BODY={"think": false}
    llm_extra_body: dict = Field(default_factory=dict)

    # ---------- Retrieval ----------
    retrieval_top_k: int = 24  # candidates fetched before reranking
    # 6 was a concession to a 4B's attention, not a retrieval judgement. With
    # a hosted model the extra context is affordable and recall is what limits
    # answer quality, so more chunks reach the prompt.
    retrieval_final_k: int = 8  # chunks actually shown to the LLM
    retrieval_hybrid: bool = True
    retrieval_rrf: bool = True
    # If the best candidate scores below this, we answer "not in my sources"
    # WITHOUT calling the LLM. This is the primary hallucination guard.
    retrieval_min_score: float = 0.35
    # ---------- HyDE (Discovery agent) ----------
    # Applied ONLY when scheme resolution finds nothing confident -- see
    # agents/hyde.py for why it would make named-scheme queries worse.
    hyde_enabled: bool = True
    hyde_max_tokens: int = 200

    # ---------- Discovery ----------
    discovery_max_schemes: int = 6
    # Below this many profile slots, suggestions are too broad to be useful
    # and the agent asks a question instead of guessing.
    discovery_min_slots: int = 1

    # ---------- Eligibility ----------
    # ROUNDS, not questions. Each round asks about every unresolved criterion
    # at once; the UI then steps the user through them one at a time with
    # Yes/No buttons and posts the answers back already paired to their
    # questions. Free-text answer parsing was the failure mode that made a
    # third round pointless, so three is now affordable.
    eligibility_max_rounds: int = 3

    rerank_enabled: bool = False  # flip on once BGE-reranker is pulled
    # The Evidence Verification pass is a SECOND full LLM call, so it roughly
    # doubles latency and cost. It was off because a local 4B could not do it:
    # judging faithfulness is harder than answering, and its JSON frequently
    # failed to parse (observed: 116 s spent, output unparseable, no signal).
    # A hosted model can, so this is now a latency/cost choice rather than a
    # capability one -- VERIFY_ENABLED=true in .env is worth trying.
    # The deterministic guards (score floor, citation markers, numeric
    # grounding) run regardless; this only adds the LLM judge on top.
    verify_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---------- API ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_request_timeout: float = 180.0

    # ---------- Observability ----------
    # Three layers, each answering a different question. They overlap less
    # than they look like they do:
    #   Prometheus/Grafana  how is the SERVICE doing -- request rate, p99,
    #                       error rate. Aggregate, cheap, no payloads.
    #   Langfuse            how is this ANSWER doing -- what was retrieved,
    #                       what the model was sent, what it cost, whether it
    #                       was grounded. Per-trace, with payloads.
    #   LangSmith           the same job as Langfuse. Keep one.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "gov-schemes-rag"

    # ---------- Langfuse ----------
    langfuse_enabled: bool = False
    # Project keys from the Langfuse dashboard (Settings -> API keys), or
    # pre-set via LANGFUSE_INIT_PROJECT_* in ops/langfuse/.env when
    # bootstrapping the self-hosted stack.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    # The self-hosted server. Note BASE_URL, not HOST: the SDK renamed it in
    # v5, and most tutorials still show LANGFUSE_HOST.
    langfuse_base_url: str = "http://localhost:3000"
    # Separates local experiments from real traffic in the dashboard, so one
    # does not quietly skew the other's numbers.
    langfuse_environment: str = "development"
    # 1.0 traces everything. Lower it only under load -- at which point
    # sampling costs you the rare failure you most wanted to see.
    langfuse_sample_rate: float = 1.0
    # Redact the user's personal details before they leave the process.
    #
    # Off by default because on a local single-developer setup the profile is
    # the most useful thing in a trace. Turn it on before this serves real
    # people: the eligibility interview collects age, income, caste category
    # and disability status, and an observability backend should not be where
    # that data is least protected.
    langfuse_mask_pii: bool = False

    metrics_enabled: bool = True
    log_level: str = "INFO"

    # ---------- Paths ----------
    # The corpus itself. Overridden by RAW_DIR in .env; this default points
    # at the real folder so `ingest.py` without --corpus still works.
    raw_dir: str = "schemes_txt"
    processed_dir: str = "data/processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
