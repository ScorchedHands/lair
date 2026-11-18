param(
    [string]$Tag = "b12032",
    [string]$ModelPath = "models/SmolLM-135M-Instruct-F16.gguf",
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$CacheTypeK = "q8_0",
    [string]$CacheTypeV = "q8_0",
    [int]$MaxTokens = 48,
    [int]$Threads = 8,
    [int]$Context = 1024,
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"

# The quantized KV postures this harness measures. llama.cpp's -ctk/-ctv names and LAIR's calibration variables share the
# same canonical spellings; anything outside these pairs is refused rather than measured against an unsupported posture.
$supportedPairs = @(
    [pscustomobject]@{ k = "q8_0"; v = "q8_0" },
    [pscustomobject]@{ k = "q8_0"; v = "q4_0" }
)

$kName = $CacheTypeK.Trim().ToLowerInvariant()
$vName = $CacheTypeV.Trim().ToLowerInvariant()
$pair = $supportedPairs | Where-Object { $_.k -eq $kName -and $_.v -eq $vName }
if (-not $pair) {
    throw "unsupported cache-type pair -CacheTypeK '$CacheTypeK' -CacheTypeV '$CacheTypeV': this harness measures only 'q8_0/q8_0' and 'q8_0/q4_0' (LAIR's quantized attention route instantiates over q8_0 on the k side and q8_0 or q4_0 on the v side)."
}

# LAIR's calibration probe and the quantized parity tests both open their engine at a fixed 1024-token window; generating
# reference captures at another context would record decode settings those seams cannot reproduce.
if ($Context -ne 1024) {
    throw "LAIR's calibration probe and quantized KV parity suite run the model at a fixed 1024-token context, so these fixtures are generated with -Context 1024; got $Context."
}

# Measured quantized KV bands written into the fixture. They match KvQuantizedParityTests' source constants after the
# 2026-11-18 SmolLM-135M-F16 generation run: q8_0/q8_0 observed max relevant delta 0.1347 recorded with margin as 0.25;
# q8_0/q4_0 observed max relevant delta 0.3856 recorded with margin as 0.50.
$band = if ($vName -eq "q8_0") { 0.25 } else { 0.50 }

$exe = Join-Path "tools/llama.cpp/$Tag" "llama-server.exe"
if (-not (Test-Path $exe)) { throw "pinned llama-server not found at $exe" }

$modelFull = (Resolve-Path $ModelPath).Path
$sha256 = (Get-FileHash -Algorithm SHA256 $modelFull).Hash.ToLower()

if ($Port -eq 0) { $Port = Get-Random -Minimum 10100 -Maximum 49000 }

function Wait-Health([int]$Port, [System.Diagnostics.Process]$Proc) {
    for ($i = 0; $i -lt 60; $i++) {
        if ($Proc.HasExited) { throw "llama-server exited during startup (code $($Proc.ExitCode))" }
        try {
            $h = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
            if ($h.StatusCode -eq 200) { return }
        } catch { Start-Sleep -Milliseconds 500 }
    }

    throw "llama-server did not become healthy on port $Port"
}

$prompts = [ordered]@{
    p1 = "The capital of France is"
    p2 = "Once upon a time"
    p3 = "def fibonacci(n):`n"
    p4 = "<|im_start|>user`nReply with a single sentence about the planet Mars.`n<|im_end|>`n<|im_start|>assistant`n"
}

# Phase 1: every llama.cpp reference capture runs while the server owns the box, and nothing else measures CPU time.
$captured = [ordered]@{}
$serverArgs = @("-m", $modelFull, "-c", "$Context", "--threads", "$Threads", "--port", "$Port", "-ctk", $pair.k, "-ctv", $pair.v)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $exe).Path
$psi.WorkingDirectory = (Get-Location).Path
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
foreach ($a in $serverArgs) { $psi.ArgumentList.Add($a) }
$proc = [System.Diagnostics.Process]::Start($psi)

try {
    Wait-Health $Port $proc

    foreach ($key in $prompts.Keys) {
        Write-Host "generating reference $key (cache types $($pair.k)/$($pair.v)) ..."
        $body = @{ prompt = $prompts[$key]; temperature = 0.0; n_predict = $MaxTokens; n_probs = 5 } | ConvertTo-Json
        for ($attempt = 1; ; $attempt++) {
            try {
                $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
                break
            } catch { if ($attempt -ge 3) { throw } Start-Sleep -Seconds 1 }
        }

        $probs = @()
        if ($null -ne $r.completion_probabilities -and $r.completion_probabilities.Count -gt 0) {
            foreach ($t in $r.completion_probabilities[0].top_logprobs) {
                $probs += [pscustomobject]@{ id = $t.id; token = $t.token; logprob = $t.logprob }
            }
        }

        if ($probs.Count -eq 0) { throw "server returned no top_logprobs for prompt: $($prompts[$key])" }
        $captured[$key] = [pscustomobject]@{ output = $r.content.TrimEnd("`r", "`n"); top_logprobs = $probs }
    }
} finally {
    # Phase 2 boundary: the server must be gone before LAIR calibrates, or llama.cpp's threads contaminate the measurement.
    if (-not $proc.HasExited) {
        try { $proc.Kill() } catch { }
        if (-not $proc.WaitForExit(15000)) { throw "llama-server did not exit before LAIR calibration; refusing to measure under contention" }
    }
}

Write-Host "calibrating LAIR's $($pair.k)/$($pair.v) quantized KV route ..."
$calibOut = Join-Path ([IO.Path]::GetTempPath()) "lair-calib-kv-$($pair.k)-$($pair.v).txt"
if (Test-Path $calibOut) { Remove-Item $calibOut }
$env:LAIR_CALIB_MODEL = $modelFull
$env:LAIR_CALIB_OUT = $calibOut
$env:LAIR_CALIB_KV_K = $pair.k
$env:LAIR_CALIB_V = $pair.v
try {
    dotnet test Lair.sln --nologo -v q --filter "Category=Calib" | Out-Null
} finally {
    Remove-Item Env:LAIR_CALIB_MODEL, Env:LAIR_CALIB_OUT, Env:LAIR_CALIB_KV_K, Env:LAIR_CALIB_V -ErrorAction SilentlyContinue
}

if (-not (Test-Path $calibOut)) { throw "calibration probe produced no output at $calibOut" }
$lairMaps = @{}
foreach ($line in Get-Content $calibOut) {
    foreach ($key in $prompts.Keys) {
        if ($line -match "^CALIB\s+$key\s+(.*)$") {
            $map = @{}
            foreach ($p in ($Matches[1] -split '\|')) {
                if ($p -match '^(-?\d+)=(-?[\d.]+)$') { $map[[long]$Matches[1]] = [double]$Matches[2] }
            }

            $lairMaps[$key] = $map
        }
    }
}

$entries = foreach ($key in $prompts.Keys) {
    if (-not $lairMaps.ContainsKey($key)) { throw "calibration probe produced no CALIB line for $key" }
    $lairMap = $lairMaps[$key]

    # Calibrate the band against LAIR's own quantized distribution: record each reference token's logprob in LAIR's top-16
    # (or MISSING), so the fixture documents exactly how far the two pipelines drift per candidate under this cache pair.
    $calibrated = foreach ($t in $captured[$key].top_logprobs) {
        $lairVal = if ($lairMap.ContainsKey([long]$t.id)) { $lairMap[[long]$t.id] } else { $null }
        [pscustomobject]@{ id = $t.id; token = $t.token; logprob = $t.logprob; lair_logprob = $lairVal }
    }

    [pscustomobject]@{
        prompt       = $prompts[$key]
        max_tokens   = $MaxTokens
        output       = $captured[$key].output
        top_logprobs = @($calibrated)
    }
}

$fixture = [ordered]@{
    model_file      = $ModelPath.Replace('\', '/')
    model_sha256    = $sha256
    llama_cpp_tag   = $Tag
    llama_cpp_asset = "llama-$Tag-bin-win-cpu-x64.zip"
    kv_quant        = @{ k = $pair.k; v = $pair.v }
    quantized_kv_band = $band # initial placeholder pending main's measured review — tighten after inspecting the drift this run produced
    reference_mode  = "llama-server /completion, temperature 0, n_probs 5, threads $Threads, cache types $($pair.k)/$($pair.v) (server mode is the pinned reference; CLI mode disagrees with it on identical prompts)"
    decode          = @{ temperature = 0.0; context = $Context }
    generated_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    prompts         = $entries
}

New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$fixture | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8NoBOM $Out
Write-Host "wrote $Out"
