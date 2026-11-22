# SchemeSetu-AI
An Intelligent AI Assistant for Discovering Indian Government Schemes and Assessing Eligibility

# Environment for the self-hosted Langfuse stack.
#
# Start it with this file rather than editing the Langfuse repo, so an update
# to that repo does not wipe your configuration:
#
#   docker compose -f C:\Users\bomma\Documents\RAG\langfuse\docker-compose.yml \
#                  --env-file ops\langfuse\langfuse.env up -d
#
# or just run:  .\ops\langfuse\up.ps1
#
# ---------------------------------------------------------------------------
# LOCAL DEVELOPMENT CREDENTIALS. Generated for a localhost instance, and
# gitignored. They are real secrets for that instance: regenerate them with
# `openssl rand -hex 32` before this is reachable by anyone else.
# ---------------------------------------------------------------------------

# Required by the stack. SALT and ENCRYPTION_KEY cannot be changed after data
# exists -- changing ENCRYPTION_KEY makes stored API keys undecryptable.
SALT=ce5faee847e7a2709a833be7bb5549d813e867419a656384ed5785dec3258e93
ENCRYPTION_KEY=05c59edde4d0e08d9b1a89f768ce6c2094076f608511c1632ff35a6a44f9dd6d
NEXTAUTH_SECRET=23511bcc145c21eeb1b1da3382e5e3d265e8709b8bc7113a2796c3e7b5001443

# Must match the URL you open in the browser, port included. This is why
# Langfuse keeps 3000 and Grafana moved to 3001: changing this means changing
# two things that have to agree, and a mismatch breaks sign-in with an error
# that does not mention the port.
NEXTAUTH_URL=http://localhost:3000

# ---------------------------------------------------------------------------
# Bootstrap. These create the org, project and user on FIRST START, including
# the API keys -- so the keys in your .env are valid before you have opened
# the dashboard even once. Without this you would have to start the stack,
# sign up, create a project, copy two keys and restart the API.
#
# Applied only when the database is empty. To change them later, change them
# in the dashboard instead.
# ---------------------------------------------------------------------------
LANGFUSE_INIT_ORG_ID=myscheme
LANGFUSE_INIT_ORG_NAME=myScheme
LANGFUSE_INIT_PROJECT_ID=schemes-rag
LANGFUSE_INIT_PROJECT_NAME=Government Schemes RAG
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-ef34f367e91da92adc4bf7cd44ab3572
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-8ae91ed2573b98f8a76c90bd81699fcc

# Sign in to the dashboard with these.
# Must be a VALID email -- Langfuse validates it as
#   z.union([z.email(), z.string().length(0)]).optional()
# so a domain with no TLD is rejected. "bomma@local" failed here, and the only
# symptom was the web container dying with "Invalid environment variables",
# naming no field, which then surfaced as HTTP 500 on every endpoint.
# example.com is reserved by IANA for exactly this.
LANGFUSE_INIT_USER_EMAIL=bomma@example.com
LANGFUSE_INIT_USER_NAME=Bomma
LANGFUSE_INIT_USER_PASSWORD=AgT-lqE8PZF5ileR

# ---------------------------------------------------------------------------
# MinIO's browser-facing endpoints.
#
# These two generate presigned URLs the BROWSER fetches, so they need the
# host-published port, not the internal `minio:9000` name. They default to
# localhost:9090 -- which is our Prometheus -- so the override moves MinIO to
# 9190 and these must agree with it. Change them together or trace exports
# fail with a connection error that does not mention a port.
# ---------------------------------------------------------------------------
LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT=http://localhost:9190
LANGFUSE_S3_BATCH_EXPORT_EXTERNAL_ENDPOINT=http://localhost:9190

# Keep telemetry to Langfuse's servers off for a local instance.
TELEMETRY_ENABLED=false
LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
