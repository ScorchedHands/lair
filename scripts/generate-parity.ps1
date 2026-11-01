param(
    [string]$Tag = "b12032",
    [string]$ModelPath = "models/SmolLM-135M-Instruct-F16.gguf",
    [string]$Out = "tests/fixtures/parity-smollm-f16.json",
    [int]$MaxTokens = 48,
    [int]$Threads = 8,
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$exe = Join-Path "tools/llama.cpp/$Tag" "llama-server.exe"
if (-not (Test-Path $exe)) { throw "pinned llama-server not found at $exe" }

$modelFull = (Resolve-Path $ModelPath).Path
$sha256 = (Get-FileHash -Algorithm SHA256 $modelFull).Hash.ToLower()

if ($Port -eq 0) { $Port = Get-Random -Minimum 10100 -Maximum 49000 }

$serverArgs = @("-m", $modelFull, "-c", "1024", "--threads", "$Threads", "--port", "$Port")
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $exe).Path
$psi.WorkingDirectory = (Get-Location).Path
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
foreach ($a in $serverArgs) { $psi.ArgumentList.Add($a) }
$proc = [System.Diagnostics.Process]::Start($psi)

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

try {
    Wait-Health $Port $proc

    function Invoke-Greedy([string]$Prompt, [int]$MaxTokens) {
        $body = @{ prompt = $Prompt; temperature = 0.0; n_predict = $MaxTokens; n_probs = 5 } | ConvertTo-Json
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

        if ($probs.Count -eq 0) { throw "server returned no top_logprobs for prompt: $Prompt" }
        return [pscustomobject]@{ output = $r.content.TrimEnd("`r", "`n"); top_logprobs = $probs }
    }

    $prompts = [ordered]@{
        p1 = "The capital of France is"
        p2 = "Once upon a time"
        p3 = "def fibonacci(n):`n"
        p4 = "<|im_start|>user`nHello`n<|im_end|>`n<|im_start|>assistant`n"
    }

    $entries = foreach ($key in $prompts.Keys) {
        Write-Host "generating $key ..."
        $gen = Invoke-Greedy $prompts[$key] $MaxTokens

        # Calibrate the tolerance against LAIR's own distribution: record each reference token's logprob in LAIR's
        # top-16 (or MISSING), so the fixture documents exactly how far the two pipelines drift per candidate.
        $calibOut = Join-Path ([IO.Path]::GetTempPath()) "lair-calib.txt"
        if (Test-Path $calibOut) { Remove-Item $calibOut }
        $env:LAIR_CALIB_MODEL = $modelFull
        $env:LAIR_CALIB_OUT = $calibOut
        try {
            dotnet test Lair.sln --nologo -v q --filter "Category=Calib" | Out-Null
        } finally {
            Remove-Item Env:LAIR_CALIB_MODEL, Env:LAIR_CALIB_OUT -ErrorAction SilentlyContinue
        }

        if (-not (Test-Path $calibOut)) { throw "calibration probe produced no output at $calibOut" }
        $lairMap = @{}
        foreach ($line in Get-Content $calibOut) {
            if ($line -match "^CALIB\s+$key\s+(.*)$") {
                foreach ($pair in ($Matches[1] -split '\|')) {
                    if ($pair -match '^(-?\d+)=(-?[\d.]+)$') { $lairMap[[long]$Matches[1]] = [double]$Matches[2] }
                }
            }
        }

        $calibrated = foreach ($t in $gen.top_logprobs) {
            $lairVal = if ($lairMap.ContainsKey([long]$t.id)) { $lairMap[[long]$t.id] } else { $null }
            [pscustomobject]@{ id = $t.id; token = $t.token; logprob = $t.logprob; lair_logprob = $lairVal }
        }

        [pscustomobject]@{
            prompt       = $prompts[$key]
            max_tokens   = $MaxTokens
            output       = $gen.output
            top_logprobs = @($calibrated)
        }
    }

    $fixture = [ordered]@{
        model_file      = $ModelPath.Replace('\', '/')
        model_sha256    = $sha256
        llama_cpp_tag   = $Tag
        llama_cpp_asset = "llama-$Tag-bin-win-cpu-x64.zip"
        reference_mode  = "llama-server /completion, temperature 0, n_probs 5, threads $Threads (server mode is the pinned reference; CLI mode disagrees with it on identical prompts)"
        decode          = @{ temperature = 0.0; context = 1024 }
        generated_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        prompts         = $entries
    }

    New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
    $fixture | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8NoBOM $Out
    Write-Host "wrote $Out"
} finally {
    if (-not $proc.HasExited) {
        $proc.Kill()
        $proc.WaitForExit(5000) | Out-Null
    }
}

