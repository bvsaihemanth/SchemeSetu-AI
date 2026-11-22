# Start (or stop) the self-hosted Langfuse stack.
#
#   .\ops\langfuse\up.ps1            # start, wait for the dashboard
#   .\ops\langfuse\up.ps1 -Down      # stop, keep the data
#   .\ops\langfuse\up.ps1 -Down -Wipe  # stop and delete the data
#
# Exists because the full command names three paths and is easy to get subtly
# wrong -- running `docker compose up` from inside the Langfuse repo would
# silently ignore ops\langfuse\langfuse.env and bootstrap no project, so the
# API keys in .env would be rejected with no obvious reason.

param(
    [string]$LangfuseRepo = "C:\Users\bomma\Documents\RAG\langfuse",
    [switch]$Down,
    [switch]$Wipe,
    # Truncated pulls from this registry are common enough to retry by
    # default rather than make the user notice and re-run.
    [int]$PullRetries = 4,
    # Seconds to wait for the dashboard. The default covers a first boot's
    # ClickHouse migration; pass something small for a restart.
    [int]$FirstBootTimeout = 600
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile  = Join-Path $here "langfuse.env"
$override = Join-Path $here "docker-compose.override.yml"
$compose  = Join-Path $LangfuseRepo "docker-compose.yml"

# The override stops Langfuse publishing its postgres, clickhouse, redis
# and worker ports to the host. Without it, postgres tries to bind 5432
# and -- on Windows, where that port is often inside a Hyper-V reserved
# range -- fails, which cascades: clickhouse never goes healthy, web and
# worker never start, and the app reports "nothing listening on 3000".
# It also clears the MinIO/Prometheus clash on 9090.
$composeArgs = @("-f", $compose, "-f", $override, "--env-file", $envFile)

foreach ($p in @($compose, $envFile, $override)) {
    if (-not (Test-Path $p)) { throw "Not found: $p" }
}

# Validate the env file BEFORE starting anything.
#
# Langfuse validates its environment with zod, and on failure the web
# container prints "Invalid environment variables" and a stack trace, naming
# no field. Every endpoint then returns 500, so the application reports a
# healthy-looking stack that answers nothing. A single bad value costs an
# hour, and the two fields below are the only ones here with validation
# stricter than "is a string" -- so they are the two worth checking.
$envText = Get-Content $envFile -Raw
$envProblems = @()

# z.union([z.email(), z.string().length(0)]) -- a domain without a TLD fails.
if ($envText -match '(?m)^\s*LANGFUSE_INIT_USER_EMAIL\s*=\s*(.+?)\s*$') {
    $email = $Matches[1]
    if ($email -and $email -notmatch '^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$') {
        $envProblems += "LANGFUSE_INIT_USER_EMAIL='$email' is not a valid email. " +
                        "Langfuse requires a real domain with a TLD " +
                        "(e.g. bomma@example.com); 'user@local' is rejected."
    }
}

# z.string().length(64) -- generate with: openssl rand -hex 32
if ($envText -match '(?m)^\s*ENCRYPTION_KEY\s*=\s*(.+?)\s*$') {
    $key = $Matches[1]
    if ($key.Length -ne 64 -or $key -notmatch '^[0-9a-fA-F]{64}$') {
        $envProblems += "ENCRYPTION_KEY must be exactly 64 hex characters " +
                        "(got $($key.Length)). Generate: openssl rand -hex 32"
    }
}

if ($envProblems.Count -gt 0) {
    Write-Host "Invalid values in $envFile :" -ForegroundColor Red
    foreach ($p in $envProblems) { Write-Host "  - $p" -ForegroundColor Red }
    Write-Host "`nFix these first. Langfuse would reject them with " -NoNewline
    Write-Host '"Invalid environment variables" and no field name.'
    exit 1
}

if ($Down) {
    $args = @("compose") + $composeArgs + @("down")
    if ($Wipe) {
        Write-Host "Stopping Langfuse AND DELETING ALL TRACES." -ForegroundColor Red
        $args += "-v"
    } else {
        Write-Host "Stopping Langfuse (traces kept)." -ForegroundColor Yellow
    }
    & docker @args
    exit $LASTEXITCODE
}

# The stack is postgres + clickhouse + redis + minio + worker + web, so first
# start pulls several GB and takes a few minutes before the dashboard answers.
Write-Host "Starting Langfuse from $LangfuseRepo" -ForegroundColor Cyan
Write-Host "First run pulls ~6 images; give it a few minutes." -ForegroundColor DarkGray

# Pull explicitly, with retries, BEFORE `up`.
#
# The langfuse web image is ~235 MB and that registry drops long transfers
# often enough that a failed first pull is routine, not exceptional:
#
#   short read: expected 228857902 bytes but got 166033248: unexpected EOF
#
# `up` treats that as fatal and stops, which then presents as the web
# container never starting -- and the application reports it as a connection
# refused on port 3000, three steps removed from the actual cause. Docker
# keeps whole layers it already has, so each retry resumes rather than
# restarting, and the pull gets further every time.
$pulled = $false
for ($attempt = 1; $attempt -le $PullRetries; $attempt++) {
    & docker compose @composeArgs pull
    if ($LASTEXITCODE -eq 0) { $pulled = $true; break }
    if ($attempt -lt $PullRetries) {
        Write-Host "`nPull incomplete (attempt $attempt of $PullRetries). " -ForegroundColor Yellow -NoNewline
        Write-Host "Retrying; completed layers are kept." -ForegroundColor DarkGray
        Start-Sleep -Seconds 5
    }
}
if (-not $pulled) {
    Write-Host "`nStill incomplete after $PullRetries attempts." -ForegroundColor Red
    Write-Host "Several large layers at once can starve a slow connection. Cap it:"
    Write-Host "  Docker Desktop -> Settings -> Docker Engine, add"
    Write-Host '      "max-concurrent-downloads": 2'
    Write-Host "  then Apply & Restart, and run this script again."
    exit 1
}

& docker compose @composeArgs up -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Generous, because the first boot is not like later ones: the worker applies
# the ClickHouse schema before the web container can answer anything that
# touches it, and that takes minutes rather than seconds. Waiting ~2 minutes
# and declaring failure was mistaking a slow migration for a broken stack.
$timeoutSeconds = if ($FirstBootTimeout) { $FirstBootTimeout } else { 600 }
$deadline = (Get-Date).AddSeconds($timeoutSeconds)

Write-Host "`nWaiting for the dashboard on http://localhost:3000"
Write-Host "First boot migrates ClickHouse, so this can take several minutes." -ForegroundColor DarkGray
Write-Host -NoNewline "  "

$lastStatus = "no response yet"
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:3000/api/public/health" `
                               -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) {
            Write-Host " ready." -ForegroundColor Green
            Write-Host "`nDashboard : http://localhost:3000"
            Write-Host "Sign in   : see LANGFUSE_INIT_USER_* in ops\langfuse\langfuse.env"
            Write-Host "`nThen in your .env:  LANGFUSE_ENABLED=true"
            Write-Host "Verify with      :  python scripts\check_llm.py --check-tracing"
            exit 0
        }
        $lastStatus = "HTTP $($r.StatusCode)"
    } catch {
        # A 500 here is progress, not failure: it means the web container is
        # serving but a dependency is not ready. Distinguish it from "nothing
        # is listening", because they look the same as a failed request and
        # mean opposite things.
        $code = $null
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        if ($code) {
            $lastStatus = "HTTP $code (server up, dependency not ready)"
            Write-Host -NoNewline "o"
        } else {
            $lastStatus = "no response (not listening yet)"
            Write-Host -NoNewline "."
        }
    }
    Start-Sleep -Seconds 4
}

Write-Host "`n`nGave up after $timeoutSeconds seconds. Last seen: $lastStatus" -ForegroundColor Yellow

# Show the logs rather than printing a command to go and run. The reason is
# always in them, and a three-path docker compose invocation is tedious enough
# to retype that it tends not to get run.
Write-Host "`n--- langfuse-worker (applies the migrations) ---" -ForegroundColor Cyan
& docker compose @composeArgs logs --tail 25 langfuse-worker
Write-Host "`n--- langfuse-web ---" -ForegroundColor Cyan
& docker compose @composeArgs logs --tail 25 langfuse-web

Write-Host "`nIf those show migrations still running, just wait and re-run:"
Write-Host "  python scripts\check_llm.py --check-tracing"
Write-Host "To follow along:"
Write-Host "  docker compose -f `"$compose`" -f `"$override`" --env-file `"$envFile`" logs -f langfuse-worker"
exit 1
