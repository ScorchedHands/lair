#Requires -Version 7.0

[CmdletBinding()]
param(
    [switch]$DryRun,
    [int]$Rounds = 3,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ModelFile = 'C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf',
    [string]$LlamaToolDir = 'C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\tools\llama-b12359-bin-win-cpu-x64'
)

$ErrorActionPreference = 'Stop'

$modelPath = if ([IO.Path]::IsPathRooted($ModelFile)) { $ModelFile } else { Join-Path $RepoRoot $ModelFile }
if (-not (Test-Path -LiteralPath $modelPath)) { throw "missing model file: $modelPath" }

$llamaExe = Join-Path $LlamaToolDir 'llama-bench.exe'
if (-not (Test-Path -LiteralPath $llamaExe)) { throw "missing llama binary: $llamaExe" }

$stampUtcDir = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH-mm-ssZ')
$stampUtcJson = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$reportName = "$stampUtcDir-optimize-prefill-vs-llama"
$repoReportsDir = Join-Path $RepoRoot 'reports'
$reportRelDir = "reports/$reportName"
$reportAbsDir = Join-Path $repoReportsDir $reportName

$solutionPath = Join-Path $RepoRoot 'Lair.sln'
$benchProject = Join-Path $RepoRoot 'bench/Lair.Bench'

function Write-TextNoBom([string]$path, [string]$content) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $path) | Out-Null
    $utf8NoBom = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($path, $content, $utf8NoBom)
}

function Get-Median([double[]]$values) {
    if (-not $values -or @($values).Count -eq 0) { return $null }
    $sorted = @($values | Sort-Object)
    if ($sorted.Count % 2 -eq 1) { return $sorted[[int](($sorted.Count - 1) / 2)] }
    $mid = [int]($sorted.Count / 2)
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2.0
}

function ConvertTo-SafeHtml([string]$text) {
    if ([string]::IsNullOrEmpty($text)) { return '' }
    $text.Replace('&', '&amp;').Replace('<', '&lt;').Replace('>', '&gt;').Replace('"', '&quot;')
}

function Invoke-Dotnet([object[]]$arguments, [string]$label) {
    Write-Host "RUN: dotnet $($arguments -join ' ') [$label]"
    & dotnet @($arguments)
    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    if ($exitCode -ne 0) { throw "$label failed with exit code $exitCode" }
}

function Add-Sample([hashtable]$samplesByGroup, [string]$key, [double]$value) {
    if (-not $samplesByGroup.ContainsKey($key)) { $samplesByGroup[$key] = New-Object Collections.Generic.List[object] }
    [void]$samplesByGroup[$key].Add($value)
}

function Parse-LairThroughputLog([string]$filePath, [hashtable]$samplesByGroup) {
    $currentKernel = $null
    foreach ($line in (Get-Content -LiteralPath $filePath)) {
        $kernelMatch = [regex]::Match($line, 'prefill-profile kernel=(\w+)')
        if ($kernelMatch.Success) {
            $currentKernel = $kernelMatch.Groups[1].Value
            continue
        }

        if (-not $currentKernel) { continue }

        $rowMatch = [regex]::Match($line.TrimStart(), '^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)')
        if (-not $rowMatch.Success) { continue }

        $length = [int]$rowMatch.Groups[1].Value
        Add-Sample $samplesByGroup "lair|$currentKernel|$length|tok_s" ([double]$rowMatch.Groups[4].Value)
    }
}

function Parse-LlamaLog([string]$filePath, [hashtable]$samplesByGroup, [ref]$modelFacts) {
    foreach ($line in (Get-Content -LiteralPath $filePath)) {
        if (-not $modelFacts.Value) {
            $rowMatch = [regex]::Match($line, '\|\s*([^|]+?)\s*\|\s*([^|]+?GiB)\s*\|\s*([^|]+?)\s*\|\s*(CPU[^|]*?)\s*\|\s*(\d+)\s*\|\s*pp\d+\s*\|')
            if ($rowMatch.Success) {
                $modelFacts.Value = [ordered]@{
                    model  = $rowMatch.Groups[1].Value.Trim()
                    size   = $rowMatch.Groups[2].Value.Trim()
                    params = $rowMatch.Groups[3].Value.Trim()
                    backend = $rowMatch.Groups[4].Value.Trim()
                    threads = [int]$rowMatch.Groups[5].Value
                }
            }
        }

        $match = [regex]::Match($line, '\|\s*(pp\d+)\s*\|\s*([\d.]+)\s+±\s+([\d.]+)')
        if (-not $match.Success) { continue }

        Add-Sample $samplesByGroup "llama|$($match.Groups[1].Value)|tok_s" ([double]$match.Groups[2].Value)
    }
}

function Build-Row([hashtable]$lookup, [string]$key, [string]$arm, [string]$metric) {
    if (-not $lookup.ContainsKey($key)) { throw "missing expected samples for key '$key'" }
    $values = @($lookup[$key] | ForEach-Object { [double]$_ })
    if (@($values).Count -eq 0) { throw "empty samples for key '$key'" }

    $sorted = @($values | Sort-Object)
    return @{
        arm     = $arm
        metric  = $metric
        samples = @($values | ForEach-Object { $_.ToString('F3') })
        median  = Get-Median $values
        min     = $sorted[0]
        max     = $sorted[-1]
    }
}

function Require-Key([hashtable]$lookup, [string]$key) {
    if (-not $lookup.ContainsKey($key)) { throw "missing expected samples for key '$key'" }
}

$plannedItems = New-Object Collections.Generic.List[object]
for ($roundIndex = 0; $roundIndex -lt [int]$Rounds; $roundIndex++) {
    [void]$plannedItems.Add([pscustomobject]@{
        kind       = 'lair'
        roundIndex = $roundIndex
        outPath    = Join-Path $reportAbsDir "lair-throughput-round$roundIndex.txt"
    })
    [void]$plannedItems.Add([pscustomobject]@{
        kind       = 'llama'
        roundIndex = $roundIndex
        outPath    = Join-Path $reportAbsDir "llama-bench-round$roundIndex.txt"
    })
}

if ($DryRun) {
    Write-Host "DRY: planned $($plannedItems.Count) benchmark runs under report directory placeholder; no commands executed."
    return
}

Invoke-Dotnet @('build', $solutionPath, '-c', 'Release', '--no-restore') 'dotnet-build'
New-Item -ItemType Directory -Force (Split-Path -Parent $reportAbsDir) | Out-Null

foreach ($item in $plannedItems) {
    if ($item.kind -eq 'lair') {
        $arguments = @(
            'run', '--project', $benchProject, '-c', 'Release', '--no-build', '--',
            'prefill-profile', $modelPath,
            '--threads', '32',
            '--lengths', '128,256',
            '--widths', '64',
            '--kernel', 'gemv,tiled',
            '--reps', '1',
            '--attention-batch-dots', 'on',
            '--matmul-route', 'on',
            '--out', $item.outPath
        )
        Invoke-Dotnet $arguments "bench:lair:round$($item.roundIndex)"
    } else {
        Write-Host "RUN: llama-bench round $($item.roundIndex) [$llamaExe]"
        $process = Start-Process -FilePath $llamaExe `
            -WorkingDirectory $LlamaToolDir `
            -ArgumentList @('-m', $modelPath, '-ngl', '0', '-t', '32', '-p', '128,256', '-r', '1') `
            -RedirectStandardOutput $item.outPath `
            -NoNewWindow `
            -Wait `
            -PassThru
        if ($process.ExitCode -ne 0) { throw "llama-bench failed with exit code $($process.ExitCode)" }
    }
}

$samplesByGroup = @{}
$modelFacts = $null
foreach ($item in $plannedItems) {
    if (-not (Test-Path -LiteralPath $item.outPath)) { throw "missing expected raw log: $($item.outPath)" }

    if ($item.kind -eq 'lair') {
        Parse-LairThroughputLog $item.outPath $samplesByGroup
    } else {
        Parse-LlamaLog $item.outPath $samplesByGroup ([ref]$modelFacts)
    }
}

foreach ($kernel in @('gemv', 'tiled')) {
    foreach ($length in @(128, 256)) { Require-Key $samplesByGroup "lair|$kernel|$length|tok_s" }
}
foreach ($test in @('pp128', 'pp256')) { Require-Key $samplesByGroup "llama|$test|tok_s" }

$rows = @(
    (Build-Row $samplesByGroup 'lair|gemv|128|tok_s' 'LAIR gemv 128' 'batchedTok/s'),
    (Build-Row $samplesByGroup 'lair|tiled|128|tok_s' 'LAIR tiled 128' 'batchedTok/s'),
    (Build-Row $samplesByGroup 'llama|pp128|tok_s' 'llama.cpp pp128' 'tok/s'),
    (Build-Row $samplesByGroup 'lair|gemv|256|tok_s' 'LAIR gemv 256' 'batchedTok/s'),
    (Build-Row $samplesByGroup 'lair|tiled|256|tok_s' 'LAIR tiled 256' 'batchedTok/s'),
    (Build-Row $samplesByGroup 'llama|pp256|tok_s' 'llama.cpp pp256' 'tok/s')
)

$rowLookup = @{}
foreach ($row in $rows) { $rowLookup[$row.arm] = $row }

$dataObject = [ordered]@{
    generated_utc   = $stampUtcJson
    model           = $modelPath
    llama_binary    = $llamaExe
    llama_version_hint = 'b12359'
    host_posture    = 'threads=32 rowWidth=64 lengths 128,256 kernels gemv,tiled; LAIR flags --attention-batch-dots on --matmul-route on'
    method          = "Alternating same-session rounds: each round runs one LAIR prefill-profile reps=1 command followed by one llama.cpp pp128/pp256 reps=1 run. Medians/min/max are parsed from raw logs in this directory."
    host_noise_note = 'Host noise for cross-process CPU benchmarks is high; ratios below use medians only and should be read with the +/-10-35% caveat recorded in benchmark history.'
    rounds          = [int]$Rounds
    model_facts     = $modelFacts
    rows            = @($rows)
}

$dataJson = ConvertTo-Json -InputObject $dataObject -Depth 8
Write-TextNoBom (Join-Path $reportAbsDir 'data.json') ($dataJson + "`n")

$md = [Text.StringBuilder]::new()
[void]$md.AppendLine("# Prefill LAIR vs llama.cpp comparison $stampUtcDir")
[void]$md.AppendLine()
[void]$md.AppendLine("Model: ``$modelPath``.")
[void]$md.AppendLine("llama binary: ``$llamaExe`` (expected build b12359).")
[void]$md.AppendLine("Host posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled.")
[void]$md.AppendLine("LAIR flags: ``--attention-batch-dots on --matmul-route on``; llama.cpp flags: ``-ngl 0 -t 32 -p 128,256 -r 1``.")
[void]$md.AppendLine("Method: alternating same-session rounds with reps=1 per command. Medians/min/max are parsed from raw logs in this directory.")
[void]$md.AppendLine()

if ($modelFacts) {
    [void]$md.AppendLine("| model | size | params | backend | threads |")
    [void]$md.AppendLine('| --- | ---: | ---: | --- | ---: |')
    [void]$md.AppendLine(("| {0} | {1} | {2} | {3} | {4} |" -f $modelFacts.model, $modelFacts.size, $modelFacts.params, $modelFacts.backend, $modelFacts.threads))
    [void]$md.AppendLine()
}

[void]$md.AppendLine('| harness | metric samples | median tok/s | min | max |')
[void]$md.AppendLine('| --- | ---: | ---: | ---: | ---: |')
foreach ($row in $rows) {
    [void]$md.AppendLine(("| {0} | {1} | {2:F3} | {3:F3} | {4:F3} |" -f $row.arm, (($row.samples -join ', ')), $row.median, $row.min, $row.max))
}

[void]$md.AppendLine()
[void]$md.AppendLine('| length | LAIR gemv vs llama pp | LAIR tiled vs llama pp |')
[void]$md.AppendLine('| ---: | ---: | ---: |')
foreach ($length in @(128, 256)) {
    $llamaRow = $rowLookup["llama.cpp pp$length"]
    $gemvRow = $rowLookup["LAIR gemv $length"]
    $tiledRow = $rowLookup["LAIR tiled $length"]
    [void]$md.AppendLine(("| {0} | {1:F2}% | {2:F2}% |" -f $length, (100.0 * $gemvRow.median / $llamaRow.median), (100.0 * $tiledRow.median / $llamaRow.median)))
}

[void]$md.AppendLine()
[void]$md.AppendLine("Interpretation: LAIR's current optimized CPU prefill posture is still below the llama.cpp pp band on this host; ratios are medians only and must be read against same-session noise.")

Write-TextNoBom (Join-Path $reportAbsDir 'summary.md') $md.ToString()

$html = [Text.StringBuilder]::new()
[void]$html.AppendLine('<!DOCTYPE html>')
[void]$html.AppendLine('<html lang="en">')
[void]$html.AppendLine('<head>')
[void]$html.AppendLine('<meta charset="utf-8" />')
[void]$html.AppendLine(("<title>{0} - LAIR vs llama.cpp prefill</title>" -f (ConvertTo-SafeHtml $reportName)))
[void]$html.AppendLine('<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>')
[void]$html.AppendLine(@'
<style>
  body { font-family: "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1080px; color: #1c1c1c; }
  h1 { font-size: 1.45rem; } h2 { font-size: 1.1rem; margin-top: 2.2rem; }
  .note { color: #444; font-size: 0.9rem; max-width: 76ch; }
  canvas { max-width: 1020px; }
</style>
'@)
[void]$html.AppendLine('</head>')
[void]$html.AppendLine('<body>')

[void]$html.AppendLine(("<h1>{0}</h1>" -f (ConvertTo-SafeHtml 'Prefill LAIR vs llama.cpp comparison')))
[void]$html.AppendLine(("<p class='note'>Generated at {0}. Model: <code>{1}</code>. Posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled. Rounds alternate LAIR and llama.cpp commands.</p>" -f (ConvertTo-SafeHtml $stampUtcJson), (ConvertTo-SafeHtml $modelPath)))
[void]$html.AppendLine("<p class='note'>Bar chart compares median batched tok/s for LAIR optimized prefill against llama.cpp pp medians. Host noise is high; ratios are medians only.</p>")

$labels = @('128', '256')
$config = [ordered]@{
    type    = 'bar'
    data    = [ordered]@{
        labels   = $labels
        datasets = @(
            @{ label = 'LAIR gemv'; data = @($rowLookup['LAIR gemv 128'].median, $rowLookup['LAIR gemv 256'].median); backgroundColor = '#3b6fb6' },
            @{ label = 'LAIR tiled'; data = @($rowLookup['LAIR tiled 128'].median, $rowLookup['LAIR tiled 256'].median); backgroundColor = '#c0603f' },
            @{ label = 'llama.cpp pp'; data = @($rowLookup['llama.cpp pp128'].median, $rowLookup['llama.cpp pp256'].median); backgroundColor = '#9aa5b1' }
        )
    }
    options = [ordered]@{
        scales  = [ordered]@{ y = [ordered]@{ title = [ordered]@{ display = $true; text = 'tokens / second' }; beginAtZero = $true } }
        plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } }
    }
}

$jsConfig = ($config | ConvertTo-Json -Depth 8 -Compress)
[void]$html.AppendLine('<h2>Median prefill throughput</h2>')
[void]$html.AppendLine("<canvas id=`"prefillComparisonBar`" height=`"380`"></canvas>")
[void]$html.AppendLine('<script>')
[void]$html.AppendLine("new Chart(document.getElementById('prefillComparisonBar'), $jsConfig);")
[void]$html.AppendLine('</script>')

[void]$html.AppendLine("<h2>Text summary</h2><p class='note'>See <a href='summary.md'><code>summary.md</code></a> for median tables and ratios. Machine-readable rows are in <a href='data.json'><code>data.json</code></a>.</p>")
[void]$html.AppendLine('</body>')
[void]$html.AppendLine('</html>')

Write-TextNoBom (Join-Path $reportAbsDir 'report.html') $html.ToString()

Write-Host "wrote report directory $reportRelDir ($($rows.Count) parsed metric rows, 1 chart)"
