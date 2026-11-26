#Requires -Version 7.0

[CmdletBinding()]
param(
    [switch]$DryRun,
    [int]$Rounds = 3,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ModelFile = 'models/qwen2.5-3b-instruct-q4_k_m.gguf',
    [string]$ProcessName = 'optimize-prefill-chain-group-widths'
)

$ErrorActionPreference = 'Stop'

$modelPath = if ([IO.Path]::IsPathRooted($ModelFile)) { $ModelFile } else { Join-Path $RepoRoot $ModelFile }
if (-not (Test-Path -LiteralPath $modelPath)) { throw "missing model file: $modelPath" }

$stampUtcDir = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH-mm-ssZ')
$stampUtcJson = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$reportName = "$stampUtcDir-$ProcessName"
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

function Get-BenchArguments([string]$axisName, [object[]]$extraArgs, [string]$outPath) {
    $arguments = @(
        'run', '--project', $benchProject, '-c', 'Release', '--no-build', '--',
        'prefill-profile', $modelPath,
        '--threads', '32',
        '--lengths', '128,256',
        '--widths', '64',
        '--kernel', 'gemv,tiled'
    )

    if ($axisName -eq 'stage') {
        $arguments += @('--reps', '1', '--stage-profile')
    } else {
        $arguments += @('--reps', '1')
    }

    $arguments += @($extraArgs)
    $arguments += @('--out', $outPath)
    return ,$arguments
}

function Add-Sample([hashtable]$samplesByGroup, [string]$key, [double]$value) {
    if (-not $samplesByGroup.ContainsKey($key)) { $samplesByGroup[$key] = New-Object Collections.Generic.List[object] }
    [void]$samplesByGroup[$key].Add($value)
}

function Parse-StageLog([string]$filePath, [string]$armKey, [hashtable]$samplesByGroup) {
    foreach ($line in (Get-Content -LiteralPath $filePath)) {
        $match = [regex]::Match(
            $line,
            'prefill-profile stage window=window kernel=(\w+) rowWidth=\d+ length=(\d+) forwardCalls=\d+.*attention_kernel_ms=([\d.]+).*feed_forward_gate_up_ms=([\d.]+).*feed_forward_down_ms=([\d.]+).*total_ms=([\d.]+)'
        )
        if (-not $match.Success) { continue }

        $kernel = $match.Groups[1].Value
        $length = [int]$match.Groups[2].Value
        Add-Sample $samplesByGroup "stage|$armKey|$kernel|$length|attention_kernel_ms" ([double]$match.Groups[3].Value)
        Add-Sample $samplesByGroup "stage|$armKey|$kernel|$length|feed_forward_gate_up_ms" ([double]$match.Groups[4].Value)
        Add-Sample $samplesByGroup "stage|$armKey|$kernel|$length|feed_forward_down_ms" ([double]$match.Groups[5].Value)
        Add-Sample $samplesByGroup "stage|$armKey|$kernel|$length|total_ms" ([double]$match.Groups[6].Value)
    }
}

function Parse-ThroughputLog([string]$filePath, [string]$armKey, [hashtable]$samplesByGroup) {
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
        Add-Sample $samplesByGroup "throughput|$armKey|$currentKernel|$length|batched_ms" ([double]$rowMatch.Groups[3].Value)
        Add-Sample $samplesByGroup "throughput|$armKey|$currentKernel|$length|tok_s" ([double]$rowMatch.Groups[4].Value)
    }
}

function Build-Rows([hashtable]$samplesByGroup) {
    foreach ($entry in $samplesByGroup.GetEnumerator()) {
        $parts = $entry.Key.Split('|')
        if ($parts.Count -ne 5) { continue }
        $values = @($entry.Value | ForEach-Object { [double]$_ })
        if (@($values).Count -eq 0) { continue }

        $sorted = @($values | Sort-Object)
        [pscustomobject]@{
            axis   = $parts[0]
            arm    = $parts[1]
            kernel = $parts[2]
            length = [int]$parts[3]
            metric = $parts[4]
            samples = @($values | ForEach-Object { $_.ToString('F3') })
            median = Get-Median $values
            min    = $sorted[0]
            max    = $sorted[-1]
        }
    }
}

function Require-Arms([string[]]$arms, [hashtable]$samplesByGroup) {
    foreach ($arm in $arms) {
        if (-not @($samplesByGroup.Keys | Where-Object { $_.Split('|')[1] -eq $arm }).Count) {
            throw "no samples captured for arm '$arm'"
        }
    }
}

function Get-MedianValue([hashtable]$lookup, [string]$axisName, [string]$armKey, [string]$kernel, [int]$length, [string]$metric) {
    $key = "$axisName|$armKey|$kernel|$length|$metric"
    if ($lookup.ContainsKey($key)) { return $lookup[$key] }
    throw "missing expected median for key '$key'"
}

function New-BarChart(
    [string]$canvasId,
    [string]$titleText,
    [string[]]$labels,
    [hashtable]$lookup,
    [object[]]$armsSpec,
    [string]$axisName,
    [string]$metricKey,
    [string]$yAxisTitle
) {
    $datasets = foreach ($arm in $armsSpec) {
        $values = @()
        foreach ($shape in $labels) {
            $parts = $shape.Split(' ')
            $kernel = $parts[0]
            $length = [int]$parts[1]
            $values += Get-MedianValue $lookup $axisName $arm.key $kernel $length $metricKey
        }

        [ordered]@{
            label           = $arm.label
            data            = $values
            backgroundColor = $arm.color
        }
    }

    return [pscustomobject]@{
        id     = $canvasId
        title  = $titleText
        config = [ordered]@{
            type    = 'bar'
            data    = [ordered]@{ labels = @($labels); datasets = @($datasets) }
            options = [ordered]@{
                scales  = [ordered]@{ y = [ordered]@{ title = [ordered]@{ display = $true; text = $yAxisTitle }; beginAtZero = $true } }
                plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } }
            }
        }
    }
}

function Append-Table(
    [Text.StringBuilder]$sb,
    [string]$heading,
    [object[]]$armsSpec,
    [hashtable]$lookup,
    [string]$axisName,
    [string]$metricKey
) {
    [void]$sb.AppendLine("## $heading")
    [void]$sb.AppendLine('| arm | gemv 128 | gemv 256 | tiled 128 | tiled 256 |')
    [void]$sb.AppendLine('| --- | ---: | ---: | ---: | ---: |')

    foreach ($arm in $armsSpec) {
        $values = @(
            (Get-MedianValue $lookup $axisName $arm.key 'gemv' 128 $metricKey),
            (Get-MedianValue $lookup $axisName $arm.key 'gemv' 256 $metricKey),
            (Get-MedianValue $lookup $axisName $arm.key 'tiled' 128 $metricKey),
            (Get-MedianValue $lookup $axisName $arm.key 'tiled' 256 $metricKey)
        )

        [void]$sb.AppendLine(('| {0} | {1:F3} | {2:F3} | {3:F3} | {4:F3} |' -f $arm.label, $values[0], $values[1], $values[2], $values[3]))
    }

    [void]$sb.AppendLine()
}

function ConvertTo-JsLiteral($value) { ($value | ConvertTo-Json -Depth 8 -Compress) }

$chainArms = @(
    @{ key = 'chain-default'; label = 'default auto8'; benchArgs = @(); color = '#3b6fb6' },
    @{ key = 'chain-off'; label = 'off individual dots'; benchArgs = @('--attention-batch-dots', 'off'); color = '#9aa5b1' },
    @{ key = 'chain-2'; label = 'pinned 2'; benchArgs = @('--attention-batch-dots', '2'); color = '#c0603f' },
    @{ key = 'chain-4'; label = 'pinned 4'; benchArgs = @('--attention-batch-dots', '4'); color = '#a34c8f' },
    @{ key = 'chain-on'; label = 'on pinned 8'; benchArgs = @('--attention-batch-dots', 'on'); color = '#2a7f62' }
)

$matmulArms = @(
    @{ key = 'matmul-default'; label = 'default auto route8 attention on'; benchArgs = @('--attention-batch-dots', 'on'); color = '#3b6fb6' },
    @{ key = 'matmul-off'; label = 'route off attention on'; benchArgs = @('--attention-batch-dots', 'on', '--matmul-route', 'off'); color = '#9aa5b1' },
    @{ key = 'matmul-4'; label = 'route 4 attention on'; benchArgs = @('--attention-batch-dots', 'on', '--matmul-route', '4'); color = '#c0603f' },
    @{ key = 'matmul-on'; label = 'route pinned 8 attention on'; benchArgs = @('--attention-batch-dots', 'on', '--matmul-route', 'on'); color = '#2a7f62' },
    @{ key = 'matmul-16'; label = 'route pinned 16 attention on'; benchArgs = @('--attention-batch-dots', 'on', '--matmul-route', '16'); color = '#8a5b00' }
)

$plannedItems = New-Object Collections.Generic.List[object]
for ($roundIndex = 0; $roundIndex -lt [int]$Rounds; $roundIndex++) {
    foreach ($arm in @($chainArms + $matmulArms)) {
        foreach ($axisName in @('stage', 'throughput')) {
            $fileName = "$axisName-round$roundIndex-$($arm.key).txt"
            [void]$plannedItems.Add([pscustomobject]@{
                axisName   = $axisName
                armKey     = $arm.key
                roundIndex = $roundIndex
                outPath    = Join-Path $reportAbsDir $fileName
            })
        }
    }
}

if ($DryRun) {
    Write-Host "DRY: planned $($plannedItems.Count) bench runs under report directory placeholder; no dotnet commands executed."
    return
}

Invoke-Dotnet @('build', $solutionPath, '-c', 'Release', '--no-restore') 'dotnet-build'

New-Item -ItemType Directory -Force (Split-Path -Parent $reportAbsDir) | Out-Null

$armByKey = @{}
foreach ($arm in @($chainArms + $matmulArms)) { $armByKey[$arm.key] = $arm }

foreach ($item in $plannedItems) {
    Invoke-Dotnet (Get-BenchArguments $item.axisName $armByKey[$item.armKey].benchArgs $item.outPath) "bench:$($item.axisName):round$($item.roundIndex):$($item.armKey)"
}

$samplesByGroup = @{}
foreach ($item in $plannedItems) {
    if (-not (Test-Path -LiteralPath $item.outPath)) { throw "missing expected raw log: $($item.outPath)" }

    if ($item.axisName -eq 'stage') {
        Parse-StageLog $item.outPath $item.armKey $samplesByGroup
    } else {
        Parse-ThroughputLog $item.outPath $item.armKey $samplesByGroup
    }
}

$requiredArmKeys = @(foreach ($arm in @($chainArms + $matmulArms)) { $arm.key })
Require-Arms $requiredArmKeys $samplesByGroup

$rows = @(Build-Rows $samplesByGroup)
if (@($rows).Count -eq 0) { throw 'no benchmark rows parsed from raw logs' }

$lookup = @{}
foreach ($row in $rows) { $lookup["$($row.axis)|$($row.arm)|$($row.kernel)|$($row.length)|$($row.metric)"] = $row.median }

$shapeLabels = @('gemv 128', 'gemv 256', 'tiled 128', 'tiled 256')
$chartConfigs = @(
    (New-BarChart -canvasId 'chainAttentionKernelBar' -titleText 'Shared-dot chain-group attention kernel medians (stage-profile ms)' -labels $shapeLabels -lookup $lookup -armsSpec $chainArms -axisName 'stage' -metricKey 'attention_kernel_ms' -yAxisTitle 'ms'),
    (New-BarChart -canvasId 'chainThroughputBar' -titleText 'Shared-dot chain-group throughput medians (batchedTok/s)' -labels $shapeLabels -lookup $lookup -armsSpec $chainArms -axisName 'throughput' -metricKey 'tok_s' -yAxisTitle 'tok/s'),
    (New-BarChart -canvasId 'matmulAttentionKernelBar' -titleText 'Matmul-route attention kernel medians with shared dots on (stage-profile ms)' -labels $shapeLabels -lookup $lookup -armsSpec $matmulArms -axisName 'stage' -metricKey 'attention_kernel_ms' -yAxisTitle 'ms'),
    (New-BarChart -canvasId 'matmulThroughputBar' -titleText 'Matmul-route throughput medians with shared dots on (batchedTok/s)' -labels $shapeLabels -lookup $lookup -armsSpec $matmulArms -axisName 'throughput' -metricKey 'tok_s' -yAxisTitle 'tok/s')
)

$dataObject = [ordered]@{
    generated_utc     = $stampUtcJson
    model             = $ModelFile
    generation_note   = "Alternating same-session arms with reps=1 per command. Stage and throughput samples are separated by round/arm alternation; medians/min/max are computed from the raw logs in this directory."
    rounds            = [int]$Rounds
    host_posture      = 'threads=32 rowWidth=64 lengths 128,256 kernels gemv,tiled'
    arms              = @(foreach ($arm in @($chainArms + $matmulArms)) { @{ key = $arm.key; label = $arm.label; benchArgs = @($arm.benchArgs) } })
    rows              = @($rows)
}

$dataJson = ConvertTo-Json -InputObject $dataObject -Depth 8
Write-TextNoBom (Join-Path $reportAbsDir 'data.json') ($dataJson + "`n")

$md = [Text.StringBuilder]::new()
[void]$md.AppendLine("# Prefill chain-group and matmul-route tuning $stampUtcDir")
[void]$md.AppendLine()
[void]$md.AppendLine("Model: ``$ModelFile`` (absolute benchmark path: ``$modelPath``).")
[void]$md.AppendLine("Host posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled.")
[void]$md.AppendLine("Method: alternating same-session arms with reps=1 per command for stage-profile and throughput; medians/min/max computed from raw logs. No production constants are changed by this script.")
[void]$md.AppendLine()

foreach ($arm in @($chainArms + $matmulArms)) {
    $benchArgs = if (@($arm.benchArgs).Count) { ($arm.benchArgs -join ' ') } else { '(default posture)' }
    [void]$md.AppendLine("- ``$($arm.key)``: $($arm.label); bench args: ``$benchArgs``")
}

[void]$md.AppendLine()
Append-Table $md 'Shared-dot chain-group attention_kernel_ms medians' $chainArms $lookup 'stage' 'attention_kernel_ms'
Append-Table $md 'Shared-dot chain-group total_ms medians' $chainArms $lookup 'stage' 'total_ms'
Append-Table $md 'Shared-dot chain-group throughput tok/s medians' $chainArms $lookup 'throughput' 'tok_s'
Append-Table $md 'Matmul-route attention_kernel_ms medians (shared dots on)' $matmulArms $lookup 'stage' 'attention_kernel_ms'
Append-Table $md 'Matmul-route total_ms medians (shared dots on)' $matmulArms $lookup 'stage' 'total_ms'
Append-Table $md 'Matmul-route throughput tok/s medians (shared dots on)' $matmulArms $lookup 'throughput' 'tok_s'

Write-TextNoBom (Join-Path $reportAbsDir 'summary.md') $md.ToString()

$html = [Text.StringBuilder]::new()
[void]$html.AppendLine('<!DOCTYPE html>')
[void]$html.AppendLine('<html lang="en">')
[void]$html.AppendLine('<head>')
[void]$html.AppendLine('<meta charset="utf-8" />')
[void]$html.AppendLine(("<title>{0} - LAIR prefill tuning</title>" -f (ConvertTo-SafeHtml $reportName)))
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

[void]$html.AppendLine(("<h1>{0}</h1>" -f (ConvertTo-SafeHtml 'Prefill chain-group and matmul-route tuning')))
[void]$html.AppendLine(("<p class='note'>Generated at {0}. Model: <code>{1}</code>. Posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled. Arms were alternated within each round; medians/min/max are parsed from raw logs.</p>" -f (ConvertTo-SafeHtml $stampUtcJson), (ConvertTo-SafeHtml $ModelFile)))
[void]$html.AppendLine("<p class='note'>Bar charts compare current auto posture against pinned route/chain widths. Host noise is high; medians alone do not prove a constant change unless samples separate across arms.</p>")

foreach ($chart in $chartConfigs) {
    [void]$html.AppendLine(("<h2>{0}</h2>" -f (ConvertTo-SafeHtml $chart.title)))
    [void]$html.AppendLine("<canvas id=`"$($chart.id)`" height=`"380`"></canvas>")
}

[void]$html.AppendLine('<script>')
foreach ($chart in $chartConfigs) {
    [void]$html.AppendLine("new Chart(document.getElementById('$($chart.id)'), $(ConvertTo-JsLiteral $chart.config));")
}
[void]$html.AppendLine('</script>')

[void]$html.AppendLine("<h2>Text summary</h2><p class='note'>See <a href='summary.md'><code>summary.md</code></a> for arm definitions and median tables. Machine-readable rows are in <a href='data.json'><code>data.json</code></a>.</p>")
[void]$html.AppendLine('</body>')
[void]$html.AppendLine('</html>')

Write-TextNoBom (Join-Path $reportAbsDir 'report.html') $html.ToString()

Write-Host "wrote report directory $reportRelDir ($($rows.Count) parsed metric rows, $($chartConfigs.Count) charts)"
