# Indian Government Schemes RAG

Grounded question answering over myScheme scheme documents. BGE-M3 hybrid
retrieval (local), Qdrant, Gemini for generation, LangGraph orchestration,
FastAPI with a built-in web UI.

Embeddings stay local and generation is hosted, on purpose. BGE-M3 gives dense
*and* sparse vectors from one pass, and the sparse half is what matches
acronyms like "PM-KISAN" that a dense model paraphrases away — a hosted
embedding API would be dense-only, so hybrid search would quietly degrade to
single-vector search. Generation has the opposite trade: it was the one part
a 4 GB card could not do well.

## What works right now

Phase 1 is complete and tested: parse → chunk → embed → index → retrieve →
generate → cite, with intent routing and a verification pass.

| Component | Status |
|---|---|
| `scheme_index.py` — scheme entity resolution | tested, 14/14 name forms |
| `extract.py` — myScheme `.txt` parser | tested on real corpus |
| `chunk.py` — section-aware chunking, atomic FAQs | tested, 16 chunks/doc |
| `embeddings.py` — BGE-M3 dense + sparse | code complete |
| `vectorstore.py` — Qdrant, named vectors, RRF hybrid | tested in-memory |
| `retriever.py` — hybrid + gate + budget | tested |
| `graph.py` — LangGraph, intent routing | tested |
| `agents/understand.py` — agent 1 | tested |
| `api/` — `/query`, `/search`, `/health`, `/stats`, `/metrics` | tested |
| Agents 2–6 as dedicated nodes | retrieval policies in place, nodes pending |
| Ragas eval harness | pending |
| Grafana dashboards | Prometheus wired, dashboards pending |

---

## Setup

### 1. Install

```powershell
cd C:\Users\bomma\Documents\RAG\myscheme-ai
python -m venv .venv
.venv\Scripts\activate

# GPU torch FIRST, or pip will pull the CPU build as a dependency
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify the GPU is actually visible — BGE-M3 on CPU takes about an hour for
725 documents, on GPU a couple of minutes:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Qdrant

```powershell
docker compose up -d qdrant
```

Dashboard: http://localhost:6333/dashboard

No Docker? Set `QDRANT_MODE=local` in `.env` and read the warning in
`.env.example` first — it means ingest and the API cannot run at the same
time, and every query becomes a brute-force scan.

### 3. Configure

```powershell
copy .env.example .env
```

Then edit `.env`:

- `LLM_API_KEY` — paste your Gemini key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)).
  It ships as `api-key-here`, and preflight fails on that value by name rather
  than letting you debug an authentication error.
- `RAW_DIR` — your corpus folder.

`.env` is gitignored. Keep it that way; never commit the key.

### 4. Confirm the model name

```powershell
python scripts\check_llm.py --list-models
```

A wrong `LLM_MODEL` returns a 404 that names no alternative, so this prints
what your key can actually call and suggests the closest match to whatever
you have set. Correct the one line in `.env` if needed.

**Use the bare name**, e.g. `gemini-3.5-flash-lite`. The `/models` listing
returns ids as `models/gemini-3.5-flash-lite`, but the chat endpoint wants it
without the prefix. `llm.py` recovers by retrying with the prefix toggled and
remembering which form worked, so either spelling functions — but the
prefixed one wastes a failed request per process.

### 5. Verify every agent against the real API

Run this *after* Ingest below — it queries the real collection, so it needs
one. (It says so and exits rather than failing obscurely if the collection is
empty.)

```powershell
python scripts\smoke_agents.py
```

The unit tests drive the graph with a scripted LLM, which cannot tell you
whether *this* provider, model, key and index work together. This runs one
representative query through all six agents against real Gemini and real
Qdrant — including the two-turn eligibility interview, where it answers every
question and requires a verdict rather than a re-ask — and exits non-zero if
any agent fails. Costs about a dozen Gemini calls. Scheme names come from your
own index, so it works on any corpus.

Nothing else changes with the provider: `llm.py` is a plain OpenAI-protocol
client, and Gemini serves that protocol, so swapping backends is three lines
of config. To go back to a local model, point `LLM_BASE_URL` at
`http://localhost:11434/v1`, set `LLM_NO_THINK_SUFFIX=true`, and build the
Modelfile kept in `unused/` — but read the notes in that file first, because
a 4B model is what the eligibility interview could not handle.

---

## Ingest

Always dry-run first. It parses and chunks without embedding anything, and
reports any heading it could not map:

```powershell
python scripts\ingest.py --corpus "C:\Users\bomma\Documents\RAG\myscheme-ai\schemes_txt" --dry-run
```

If the `UNMAPPED HEADINGS` section lists anything, add those headings to
`SECTION_ALIASES` in `extract.py` before the real run — an unmapped heading
becomes `section=other` and drops out of every agent's section filter.

Then:

```powershell
python scripts\ingest.py --corpus "...\schemes_txt" --recreate
```

Measured on the real corpus: **725 documents → 12,811 chunks**, 17.7 per
doc, zero parse errors, every heading mapped. Chunk mix is roughly 62% FAQ
pairs and 38% narrative sections — the FAQ pairs are the bulk of the index
and the highest-precision retrieval targets in it.

Coverage gaps worth knowing: 1 document each lacks benefits, eligibility and
application_process; 3 lack documents_required. The ingest report now names
them under `DOCUMENTS WITH MISSING SECTIONS`. A scheme with no eligibility
section is invisible to the eligibility route, so it is worth checking
whether those are genuine source gaps or truncated files.

Embedding 12,811 chunks with BGE-M3 on GPU at batch 32 takes a few minutes.

Re-runs skip unchanged files via a content-hash manifest, so fixing three
documents costs three documents. `--force` re-embeds everything.

---

## Query

`scripts\query.py` now prints the scheme-resolution decision before the
results. When an answer is about the wrong scheme, that line tells you whether
resolution picked the wrong document or found nothing and fell back to
unscoped search. `/search` returns the same under `resolved_schemes` and
`scoped_to_schemes`.

```powershell
# retrieval only -- no LLM. Start here when an answer looks wrong.
python scripts\query.py "Who can apply for PM-KISAN?" --search-only

# full pipeline
python scripts\query.py "Who can apply for Advance Authorisation?"

# interactive
python scripts\query.py
```

## API

```powershell
uvicorn schemes_rag.api.app:app --app-dir src --host 0.0.0.0 --port 8080
```

Then open **http://localhost:8080/** — that is the UI. One static page served
by this same process: a prompt box, answers with their sources, and the
eligibility interview.

The interview is the part worth knowing about. When the Eligibility agent
needs facts it returns every outstanding question at once; the page shows them
**one at a time with Yes / No buttons** and posts the answers back already
paired to their questions.

That pairing is the design. Previously the user typed `1)yes,2)yes,3)no` as
one string and the model had to work out which answer belonged to which of its
criteria — it could not, and reported "not enough information" about a scheme
it had just been told six facts about. A button press cannot be mispaired with
the question above it. If the agent slips and asks something open-ended
anyway, the page shows a text box instead of buttons, because "yes" to "How
much land do you hold?" is not an answer, it is a corrupt fact.

API docs at http://localhost:8080/docs

```powershell
curl -X POST http://localhost:8080/query `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"Who can apply for Advance Authorisation?\"}'
```

`/search` returns retrieval results with no generation — it tells you whether
a bad answer is a retrieval problem or a prompting problem. Use it first.

## Monitoring

```powershell
docker compose up -d prometheus grafana
```

Grafana at **http://localhost:3001** (admin / `GRAFANA_PASSWORD`), Prometheus
at **http://localhost:9090**, pre-wired as a scrape target.

Only Grafana moved off its default, because Langfuse's dashboard needs 3000
(its `NEXTAUTH_URL` has to match the port the browser uses). Langfuse's other
eight ports are dealt with in `ops/langfuse/docker-compose.override.yml` —
see below.

Metrics worth a panel each:

- `rag_queries_total{outcome}` — the share answered vs refused vs clarified.
  A rising `insufficient` rate means either retrieval regressed or users are
  asking about schemes you have not indexed.
- `rag_top_score` — the relevance gate distribution. This is how you
  calibrate `RETRIEVAL_MIN_SCORE` with evidence instead of guessing.
- `rag_stage_duration_seconds{stage}` — where latency actually goes.
  Expect generation to dominate by an order of magnitude.
- `rag_citations_per_answer` — a spike at 0 means the model is ignoring the
  citation instruction, so answers are unverifiable.

### Langfuse — per-answer tracing

Prometheus tells you how the *service* is doing. Langfuse tells you how one
*answer* was built: what was retrieved and with what scores, what the model
was sent, what it cost, whether it was grounded. The two answer different
questions, which is why both are here.

```powershell
.\ops\langfuse\up.ps1          # starts the stack from the Langfuse repo
```

Then in `.env`:

```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...        # already matches langfuse.env; see below
LANGFUSE_SECRET_KEY=...
```

```powershell
python scripts\check_llm.py --check-tracing
```

Dashboard at **http://localhost:3000**. Sign in with the
`LANGFUSE_INIT_USER_*` values in `ops\langfuse\langfuse.env`.

That file pre-seeds the org, project **and API keys** on first start, so the
keys in `.env` are valid before you open the dashboard at all. Start the stack
without `--env-file` and no project is bootstrapped, so those keys do not
exist and the SDK gets a 401 that does not explain why — `up.ps1` exists to
make that mistake hard.

**Why `up.ps1` rather than `docker compose up`.** It passes two files, and
both matter:

`ops/langfuse/docker-compose.override.yml` stops Langfuse publishing eight of
its nine host ports. Its postgres, clickhouse, redis and worker are reached
over the compose network by name, so publishing them buys nothing and costs:
postgres tried to bind 5432, failed (on Windows that port is often inside a
Hyper-V reserved range — `netsh interface ipv4 show excludedportrange
protocol=tcp`), and the failure cascaded — clickhouse never went healthy, web
and worker never started, and the application reported "nothing listening on
3000" with no mention of a port conflict anywhere. The override also clears
the MinIO/Prometheus clash on 9090, which is why Prometheus is back on its
default.

MinIO is *moved* to 9190 rather than hidden: two Langfuse settings
(`LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT` and
`LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT`) generate presigned URLs the
**browser** fetches, so they cannot use the internal name. Hiding MinIO would
silently break trace exports — the route for pulling traces out into a Ragas
eval set. Those two settings are pinned to 9190 in `langfuse.env`; change them
together or exports fail.

The override uses `!reset` / `!override`, which need Compose ≥ 2.24. Compose
*concatenates* `ports` across files, so simply listing fewer ports would add
nothing and remove nothing — the tags are the documented way to clear a
merged list.

`up.ps1` also validates `langfuse.env` before starting anything, because
Langfuse validates its environment with zod and on failure prints
`Invalid environment variables` **naming no field** — then every endpoint
returns 500, so you get a stack that looks healthy and answers nothing. The
two fields with validation stricter than "is a string" are the two it checks:

- `LANGFUSE_INIT_USER_EMAIL` must be a real email with a TLD.
  `z.union([z.email(), z.string().length(0)])` — so `user@local` is rejected.
- `ENCRYPTION_KEY` must be exactly 64 hex characters (`openssl rand -hex 32`),
  and cannot be changed once data exists without making stored API keys
  undecryptable.

**What a trace looks like.** One per turn, grouped into a session by
`thread_id`:

```
query (chain)                      question in, answer out
├── understand (agent)             route, resolved scheme, confidence
│   ├── understand.router          generation: tokens, cost
│   └── eligibility.slots          generation
├── eligibility (agent)
│   ├── retrieve (retriever)       query, gate score, chunks WITH scores
│   └── eligibility.assess         generation
└── evidence (evaluator)           citation + numeric grounding checks
```

Span types are deliberate — `retriever`, `agent`, `evaluator` and `generation`
each render and filter differently in Langfuse, so the shape of the pipeline is
visible rather than a flat list of nine identical spans.

**Scores**, which are the part worth trending (chartable and filterable,
unlike metadata):

| Score | Why you look at it |
|---|---|
| `retrieval_gate_score` | Calibrate `RETRIEVAL_MIN_SCORE` from real traffic instead of the 0.35 it was guessed at. |
| `route` / `outcome` | Which agents get used, and the answered/refused/clarified split. |
| `scheme_resolution` | `exact`/`strong`/`weak` from the entity resolver — the component behind the original "PM-KISAN returns four wrong schemes" bug. |
| `insufficient_evidence` | Refusal rate. Rising means retrieval regressed or the corpus has gaps. |
| `citations` | A cluster at 0 means answers are unverifiable. |
| `eligibility_outcome`, `assessment_failed` | Whether a verdict was a real verdict or a failed call wearing one. |
| `user_feedback` | Thumbs up/down from the UI. The only score automation cannot produce. |

**Cost per agent** comes free: `llm.py` uses the instrumented OpenAI client, so
every call is labelled (`generate.documents`, `eligibility.assess`, …) and the
dashboard attributes tokens and cost to the agent that spent them. This works
only because `llm.py` was written against the OpenAI protocol rather than a
vendor SDK — the same decision that made the move to Gemini three lines of
config.

**Privacy.** `LANGFUSE_MASK_PII=false` by default, because locally the user
profile is the most useful thing in a trace. Turn it on before this serves
real people: the eligibility interview collects age, income, caste category
and disability status, and an observability backend should not be where that
data is least protected. `ops\langfuse\langfuse.env` holds real credentials
for the local instance and is gitignored.

**Not used, on purpose.** Langfuse can also host your prompts. Moving them out
of `prompts.py` would add a network dependency to every request and separate
the prompts from the code review that changes them — and these prompts encode
hard-won rules (see the refusal-rule history in `prompts.py`). They stay in
version control.

**Turning it off** is `LANGFUSE_ENABLED=false`, or uninstalling the package:
`tracing.py` degrades to no-ops and nothing else imports langfuse.

## Tests

No services needed — these use an in-memory Qdrant and a scripted LLM.

```powershell
$env:PYTHONPATH="src"; $env:PYTHONHASHSEED="0"
python -m tests.test_vectorstore_smoke    # Qdrant round trip, hybrid search
python -m tests.test_pipeline_e2e         # ingest -> retrieve -> answer
python -m tests.test_api_graph            # endpoints, UI contract, feedback
python -m tests.test_scheme_resolution    # entity resolution over 725 names
python -m tests.test_llm_client           # thinking, 400/404 recovery
python -m tests.test_check_llm            # all 14 preflight branches
python -m tests.test_agents               # the six agents, incl. the interview
python -m tests.test_smoke_agents         # that the smoke test itself works
python -m tests.test_tracing              # spans, scores, masking, isolation
```

`test_tracing` runs against the **real** Langfuse SDK with an in-memory span
exporter and a mocked transport, so it exercises the code that runs in
production without anything leaving the machine. Mocking the SDK would only
prove the mock matches my assumptions, which is the thing most likely to be
wrong given how often this SDK's API has changed.

For a check against the real Gemini API and your real index, see
`scripts\smoke_agents.py` above.

---

## How it is put together

**Ingest and query never share a process.** `scripts/ingest.py` writes;
the API only reads. So re-embedding the corpus does not interrupt query
traffic, and the API can be replicated without coordination.

**Section-aware chunking is the load-bearing decision.** Every chunk carries
a `section` tag, so the eligibility route searches eligibility text instead
of hoping cosine similarity separates it from benefits prose. This is what
makes specialised agents meaningfully different rather than seven prompts
over the same undifferentiated search. FAQ pairs become atomic chunks, which
are the highest-precision targets in the corpus — user questions resemble
FAQ questions far more than they resemble policy prose.

**Hybrid retrieval, stored up front.** BGE-M3 emits dense and sparse vectors
in one forward pass, and both are stored at ingest time. Dense-only retrieval
fails on scheme acronyms, rupee amounts and category codes — exactly what
users type. Fusion is RRF, because cosine and lexical scores are on different
scales and rank fusion needs no hand-tuned weights.

**Grounding has three layers.** A cosine relevance gate that refuses without
calling the LLM at all; mandatory `[n]` citation markers resolved against
real chunks, with invented markers stripped; and an evidence-verification
pass on high-stakes routes. The gate judges the user's original question even
when the router rewrote the search query, so a rewrite cannot smuggle an
off-topic question past it.

**One LangGraph, agents as nodes.** Intent selects a retrieval policy —
section filter, `k`, whether to verify. Agents 2–6 attach to that table.

### Immutable choices, made deliberately

Three Qdrant settings cannot be changed without a full re-ingest, so they are
set now: named vectors (an unnamed vector cannot gain a sparse sibling
later), `shard_number=4` (free on one node, lets you add nodes and rebalance
instead of rebuilding), and int8 scalar quantization with originals on disk.
At 725 documents none of this matters. At 100,000 all of it does.

---

## Scaling path

| Trigger | Change |
|---|---|
| More than 1 API worker | `EMBED_SERVICE_URL` → separate embedding service. BGE-M3 is ~2.3 GB per worker. |
| Retrieval quality plateau | `RERANK_ENABLED=true`. Biggest single quality win; costs ~300 ms/query. |
| More than ~50k chunks | Add Qdrant nodes; the 4 shards rebalance. |
| Concurrent users | Already handled: a hosted endpoint batches server-side. Watch the rate limit, not the GPU. |
| Generation cost matters | Lower `LLM_CONTEXT_BUDGET_TOKENS` and `RETRIEVAL_FINAL_K` — prompt tokens dominate the bill. |
| More than 1 API replica | Replace `MemorySaver` with a Postgres/Redis checkpointer, or clarification context breaks across replicas. |
| Ingest becomes slow | Queue + worker. The ingest/query split means this changes nothing in the API. |

## Known gaps

- **`state` detection is heuristic.** The sample has `Ministry: Not specified`
  and no state field, so state is inferred from the scheme title against a
  closed list. Run the dry-run report to see how many documents land as
  `level=unknown`; if it is most of them, this needs a real source.
- **Ragas eval is not built yet** (`eval/` is empty). Judging faithfulness is
  harder than answering, which is why this was not attempted against a local
  4B; a hosted model makes it viable, so this is now worth doing. Hand-label
  ~50 questions as the reference set.
- **`VERIFY_ENABLED=false`** for the same reason, and the same reasoning now
  applies: the LLM judge is a second full call (~2x latency and cost) but is
  no longer beyond the model. Worth switching on and measuring. The
  deterministic guards — score floor, citation markers, numeric grounding —
  run either way.
- **Facet enrichment has never been run on the full corpus.**
  `scripts/enrich.py` has only been dry-run on 20 schemes, so `facets` are
  absent from all 12,811 points and profile-based Discovery still falls back
  to unfiltered retrieval. This is the largest unclaimed quality win left.
- **`RETRIEVAL_MIN_SCORE=0.35` is a starting guess**, not a calibrated value.
  Tune it against the `rag_top_score` histogram once real queries flow.
