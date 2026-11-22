# SchemeSetu-AI

An AI assistant for discovering Indian Government schemes and assessing eligibility. It answers
from a corpus of **725 scheme documents** exported from [myScheme.gov.in](https://www.myscheme.gov.in),
runs a multi-agent interview to work out what a person actually qualifies for, and cites the source
document for every claim it makes.

## Stack

| Layer | Choice | Why this one |
|---|---|---|
| Orchestration | LangGraph | Routing and the multi-turn eligibility interview need real state, not a prompt chain |
| LLM | Gemini (`gemini-3.5-flash-lite`) via the OpenAI-compatible endpoint | One protocol, so the model is a config value |
| Embeddings | BGE-M3, local, GPU | Dense **and** sparse vectors from a single pass |
| Vector DB | Qdrant, named vectors | Hybrid search in one query; scalar quantization keeps 725 docs cheap |
| Fusion | Reciprocal Rank Fusion | Combines dense + sparse without tuning a weight |
| API | FastAPI + one static HTML page | No build step, no node_modules |
| Observability | Langfuse (traces, scores, cost) + Prometheus/Grafana | Per-node traces and token cost per request |
| Evaluation | Custom harness, LLM-as-judge | 85 graded questions with reference answers |

## How it works

```
question
   │
   ├─ understand ──→ route + extract slots (age, income, state, category, occupation)
   │                 missing slots? ─→ clarify ─→ ask the user, keep state
   │
   ├─ retrieve ───── HyDE expansion ─→ BGE-M3 (dense+sparse) ─→ Qdrant ─→ RRF ─→ top-8
   │
   ├─ agent ──────── discovery │ eligibility │ comparison │ application │ documents │ general
   │
   └─ evidence ───── every claim tied to a chunk, or it is dropped
```

Four decisions carry most of the quality:

1. **Section-aware chunking.** Scheme documents are structured (Details / Benefits / Eligibility /
   Application Process / Documents Required). Chunks respect those boundaries, so an eligibility
   question retrieves eligibility text instead of a paragraph straddling two sections.
2. **Entity resolution before retrieval.** "PM-KISAN" and "PMEGP" are resolved against a scheme
   index first. Acronyms live in the URL slug, not the title, so pure vector search misses them.
3. **Hybrid retrieval, fused.** Sparse catches exact scheme names and numbers; dense catches intent
   ("help for a widow starting a tailoring business"). RRF merges both rankings.
4. **Three grounding layers.** Retrieval score gate → citation-or-drop in the evidence node →
   an optional verify pass. An unsupported claim should not survive all three.

## The six agents

| Agent | Does |
|---|---|
| `understand` | Classifies the route, extracts eligibility slots, decides if a clarification is needed |
| `discovery` | Finds candidate schemes from a free-text life situation |
| `eligibility` | Runs the Yes/No interview and returns a verdict with the unmet criteria named |
| `comparison` | Puts two or more schemes side by side on benefit, eligibility and process |
| `application`/`documents` | Extracts the steps and the paperwork for one scheme |
| `evidence` | Checks the answer against retrieved chunks and attaches citations |

## Quickstart

```bash
pip install -e ".[dev]"            # install torch for your CUDA build first
cp .env.example .env               # paste your Gemini key into LLM_API_KEY
docker compose up -d qdrant
python scripts/ingest.py           # chunk + embed 725 docs
python scripts/check_llm.py        # preflight: key, model, GPU, tracing
uvicorn schemes_rag.api.app:app --port 8099
```

Then open `http://localhost:8099` — the UI is served from `/`. Port 8080 is inside a Windows
reserved range on some machines (`WinError 10013`); 8099 avoids it.

## Layout

```
src/schemes_rag/
  config.py        every tunable; the ONLY module that reads env
  chunk.py         section-aware splitting
  embeddings.py    BGE-M3 dense + sparse
  vectorstore.py   the ONLY module importing qdrant_client
  retriever.py     hybrid search + RRF
  graph.py         LangGraph: 9 nodes, routing, interview state
  agents/          one file per agent
  evaluation/      gold set, judges, metrics
  tracing.py       the ONLY module importing langfuse
  api/             FastAPI routes + static/index.html
scripts/           ingest, query, enrich, check_llm, smoke_agents, evaluate_rag
eval/              gold.jsonl, report.json, metric reference
tests/             10 suites
```

Single owner per dependency is deliberate: swapping Qwen for Gemini was a three-line `.env` change
because nothing outside `llm.py` knew which provider was in use.

## Evaluation

```bash
python scripts/evaluate_rag.py --rpm 10 --resume
```

Reports recall@k / precision@k over the retrieval set, plus faithfulness, context relevance,
context precision, context recall, completeness and answer correctness — all judged against the
85 reference answers in `tests/`. Cost per request comes from the token ledger, not an estimate.
Rate-limited questions are recorded as errored and excluded, so a quota problem never reads as a
groundedness failure.

## Known gaps

- **Facet enrichment has never run on the full corpus** — the largest unclaimed quality win.
- `RERANK_ENABLED=false`, `VERIFY_ENABLED=false` — both built, neither switched on.
- `RETRIEVAL_MIN_SCORE=0.35` is uncalibrated; now measurable via the `retrieval_gate_score` metric.
- The gold set needs one human review pass; 15 labels were drafted from reference answers.
- `LANGFUSE_MASK_PII=false` by default. **Turn it on before serving real users** — the eligibility
  interview collects age, income, caste category and disability status.

Full setup, ingest and query internals, monitoring, and the reasoning behind the immutable Qdrant
choices are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
