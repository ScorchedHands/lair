#Requires -Version 7.0

[CmdletBinding()]
param(
    [string]$SourceDir = 'reports/2026-11-26T03-12-26Z-optimize-prefill-chain-group-widths',
    [switch]$InPlace,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ProcessName = 'optimize-prefill-chain-group-tuning-analysis'
)

$ErrorActionPreference = 'Stop'

function Write-TextNoBom([string]$path, [string]$content) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $path) | Out-Null
    $utf8NoBom = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($path, $content, $utf8NoBom)
}

function ConvertTo-SafeHtml([string]$text) {
    if ([string]::IsNullOrEmpty($text)) { return '' }
    $text.Replace('&', '&amp;').Replace('<', '&lt;').Replace('>', '&gt;').Replace('"', '&quot;')
}

function Get-Label([string]$axis, [string]$arm) {
    if ($axis -eq 'stage') {
        switch ($arm) {
            'default' { return 'auto8' }
            'off'     { return 'individual dots off' }
            'g2'      { return 'pinned 2' }
            'g4'      { return 'pinned 4' }
            'on'      { return 'pinned 8' }
        }
    }

    if ($axis -eq 'throughput') {
        switch ($arm) {
            'default' { return 'auto8' }
            'off'     { return 'individual dots off' }
            'g2'      { return 'pinned 2' }
            'g4'      { return 'pinned 4' }
            'on'      { return 'pinned 8' }
        }
    }

    switch ($arm) {
        'auto'   { return 'route auto8' }
        'off'    { return 'route off' }
        'w4'     { return 'route pinned 4' }
        'on'     { return 'route pinned 8' }
        'w16'    { return 'route pinned 16' }
    }

    if ($axis -eq 'matmul_stage') { return "route $arm" }
    return $arm
}

function Get-ArmOrder([string]$axis) {
    switch ($axis) {
        'stage'      { return @('default', 'off', 'g2', 'g4', 'on') }
        'throughput' { return @('default', 'off', 'g2', 'g4', 'on') }
        'matmul_stage' { return @('auto', 'off', 'w4', 'on', 'w16') }
    }

    throw "unknown axis '$axis'"
}

function Get-ReferenceArm([string]$axis) {
    switch ($axis) {
        'stage'      { return 'default' }
        'throughput' { return 'default' }
        'matmul_stage' { return 'auto' }
    }

    throw "unknown axis '$axis'"
}

function Get-Lookup([object[]]$rows) {
    $lookup = @{}
    foreach ($row in $rows) {
        $key = "$($row.axis)|$($row.arm)|$($row.kernel)|$($row.length)|$($row.metric)"
        $lookup[$key] = $row.median
    }

    return $lookup
}

function Format-Range([object]$row) {
    if (-not $row) { return '' }
    '{0:F2} [{1:F2},{2:F2}]' -f $row.median, $row.min, $row.max
}

function Get-MetricRows([object[]]$rows, [string]$axis, [string]$metric) {
    return @($rows | Where-Object { $_.axis -eq $axis -and $_.metric -eq $metric })
}

function Append-ArmTable(
    [Text.StringBuilder]$sb,
    [object[]]$rows,
    [string]$heading,
    [string]$axis,
    [string]$metricKey
) {
    [void]$sb.AppendLine("## $heading")
    [void]$sb.AppendLine('| shape | arm | median [min,max] |')
    [void]$sb.AppendLine('| --- | --- | ---: |')

    foreach ($shape in @('gemv 128', 'gemv 256', 'tiled 128', 'tiled 256')) {
        $parts = $shape.Split(' ')
        $kernel = $parts[0]
        $length = [int]$parts[1]

        foreach ($arm in (Get-ArmOrder $axis)) {
            $row = @($rows | Where-Object { $_.axis -eq $axis -and $_.arm -eq $arm -and $_.kernel -eq $kernel -and $_.length -eq $length -and $_.metric -eq $metricKey } | Select-Object -First 1)
            if (-not $row) { continue }
            [void]$sb.AppendLine(('| {0} | {1} | {2} |' -f $shape, (Get-Label $axis $arm), (Format-Range $row)))
        }

        [void]$sb.AppendLine()
    }
}

function ConvertTo-JsLiteral($value) { ($value | ConvertTo-Json -Depth 8 -Compress) }

function New-AvailableBarChart(
    [string]$canvasId,
    [string]$titleText,
    [object[]]$rows,
    [string]$axisName,
    [string]$metricKey,
    [string]$yAxisTitle,
    [bool]$higherIsBetter = $false
) {
    $shapeLabels = @('gemv 128', 'gemv 256', 'tiled 128', 'tiled 256')
    $lookup = Get-Lookup $rows

    $datasets = foreach ($arm in (Get-ArmOrder $axisName)) {
        $values = @()
        $hasAllShapes = $true
        foreach ($shape in $shapeLabels) {
            $parts = $shape.Split(' ')
            $kernel = $parts[0]
            $length = [int]$parts[1]
            $key = "$axisName|$arm|$kernel|$length|$metricKey"
            if (-not $lookup.ContainsKey($key)) {
                $hasAllShapes = $false
                break
            }

            $values += $lookup[$key]
        }

        if (-not $hasAllShapes) { continue }

        [ordered]@{
            label           = Get-Label $axisName $arm
            data            = @($values)
            backgroundColor = switch ($arm) {
                'default' { '#3b6fb6' }
                'auto'    { '#3b6fb6' }
                'off'     { '#9aa5b1' }
                'g2'      { '#c0603f' }
                'w4'      { '#c0603f' }
                'g4'      { '#a34c8f' }
                'on'      { '#2a7f62' }
                'w16'     { '#8a5b00' }
            }
        }
    }

    if (-not $datasets) { return $null }

    return [pscustomobject]@{
        id     = $canvasId
        title  = $titleText
        config = [ordered]@{
            type    = 'bar'
            data    = [ordered]@{ labels = @($shapeLabels); datasets = @($datasets) }
            options = [ordered]@{
                scales  = [ordered]@{ y = [ordered]@{ title = [ordered]@{ display = $true; text = $yAxisTitle }; beginAtZero = $true } }
                plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } }
            }
        }
    }
}

$sourceAbsDir = if ([IO.Path]::IsPathRooted($SourceDir)) { $SourceDir } else { Join-Path $RepoRoot $SourceDir }
if (-not (Test-Path -LiteralPath $sourceAbsDir)) { throw "missing source report directory: $sourceAbsDir" }

$chainStagePath = Join-Path $sourceAbsDir 'parsed-stage.json'
$matmulStagePath = Join-Path $sourceAbsDir 'parsed-matmul_stage.json'
$throughputPath = Join-Path $sourceAbsDir 'parsed-throughput.json'
foreach ($path in @($chainStagePath, $matmulStagePath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "missing parsed tuning JSON: $path" }
}

$rows = New-Object Collections.Generic.List[object]
foreach ($path in @($chainStagePath, $matmulStagePath, $throughputPath)) {
    if (-not (Test-Path -LiteralPath $path)) { continue }
    foreach ($row in @(Get-Content -LiteralPath $path -Raw | ConvertFrom-Json)) {
        [void]$rows.Add([pscustomobject]@{
            axis   = $row.axis
            arm    = $row.arm
            label  = Get-Label ([string]$row.axis) ([string]$row.arm)
            kernel = $row.kernel
            length = [int]$row.length
            metric = $row.metric
            samples = @($row.samples | ForEach-Object { [double]$_ })
            median = [double]$row.median
            min    = [double]$row.min
            max    = [double]$row.max
        })
    }
}

if ($rows.Count -eq 0) { throw 'no parsed tuning rows found' }

$stampUtcDir = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH-mm-ssZ')
$stampUtcJson = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$outRelDir = if ($InPlace) { $SourceDir } else { "reports/$($stampUtcDir)-$ProcessName" }
$outAbsDir = if ([IO.Path]::IsPathRooted($outRelDir)) { $outRelDir } else { Join-Path $RepoRoot $outRelDir }

$jsonRows = foreach ($row in $rows) {
    @{
        axis   = [string]$row.axis
        arm    = [string]$row.arm
        label  = [string]$row.label
        kernel = [string]$row.kernel
        length = [int]$row.length
        metric = [string]$row.metric
        samples = @($row.samples | ForEach-Object { [double]$_ })
        median = [double]$row.median
        min    = [double]$row.min
        max    = [double]$row.max
    }
}

$dataObject = [ordered]@{
    generated_utc   = $stampUtcJson
    analysis_note   = 'Analysis of completed same-session stage samples from the interrupted chain-group/matmul-route sweep. Matmul throughput arms were not regenerated because pinned widths above zero did not separate cleanly across measured shapes.'
    source_dir      = [string]$SourceDir
    host_posture    = 'threads=32 rowWidth=64 lengths 128,256 kernels gemv,tiled'
    rows            = @($jsonRows)
}

$dataJson = ConvertTo-Json -InputObject $dataObject -Depth 8
Write-TextNoBom (Join-Path $outAbsDir 'data.json') ($dataJson + "`n")

$md = [Text.StringBuilder]::new()
[void]$md.AppendLine("# Prefill chain-group and matmul-route tuning analysis $stampUtcDir")
[void]$md.AppendLine()
[void]$md.AppendLine("Source sweep: ``$SourceDir``.")
[void]$md.AppendLine('Host posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled.')
[void]$md.AppendLine('Method: medians/min/max are parsed from completed same-session stage samples and available chain throughput samples. The interrupted matmul-throughput tail was not used for a constant change because route widths above zero were already indistinguishable in the measured dense-stage arms.')
[void]$md.AppendLine()

Append-ArmTable $md $rows 'Shared-dot chain-group attention_kernel_ms medians' 'stage' 'attention_kernel_ms'
Append-ArmTable $md $rows 'Shared-dot chain-group total_ms medians' 'stage' 'total_ms'
Append-ArmTable $md $rows 'Shared-dot chain-group throughput tok_s medians' 'throughput' 'tok_s'
Append-ArmTable $md $rows 'Matmul-route gate_up_ms medians (shared dots on)' 'matmul_stage' 'gate_up_ms'
Append-ArmTable $md $rows 'Matmul-route ffn_down_ms medians (shared dots on)' 'matmul_stage' 'ffn_down_ms'
Append-ArmTable $md $rows 'Matmul-route total_ms medians (shared dots on)' 'matmul_stage' 'total_ms'

[void]$md.AppendLine('## Decision')
[void]$md.AppendLine()
[void]$md.AppendLine('- Keep `SharedInputDotChainGroup = 8` with vector posture. Individual-dot off and narrower pinned groups do not beat the current auto value cleanly across all measured shapes; the clean signal is batched shared-K dots versus individual dots, already captured by the completed attention batch-dot item.')
[void]$md.AppendLine('- Keep `MatmulRouteBatchWidth = 8` with vector posture. The hybrid route remains clearly better than disabled for dense stage substages, but pinned widths 4/8/16 do not separate cleanly enough across gemv/tiled length128/256 windows to justify changing the auto constant.')
[void]$md.AppendLine('- No production constants or YAML wiring were changed by this analysis.')

Write-TextNoBom (Join-Path $outAbsDir 'summary.md') $md.ToString()

$chartConfigs = @(
    (New-AvailableBarChart -canvasId 'chainAttentionKernelBar' -titleText 'Shared-dot chain-group attention kernel medians (stage-profile ms)' -rows $rows -axisName 'stage' -metricKey 'attention_kernel_ms' -yAxisTitle 'ms'),
    (New-AvailableBarChart -canvasId 'chainThroughputBar' -titleText 'Shared-dot chain-group throughput medians (batchedTok/s)' -rows $rows -axisName 'throughput' -metricKey 'tok_s' -yAxisTitle 'tok/s'),
    (New-AvailableBarChart -canvasId 'matmulGateUpBar' -titleText 'Matmul-route gate/up stage medians with shared dots on (ms)' -rows $rows -axisName 'matmul_stage' -metricKey 'gate_up_ms' -yAxisTitle 'ms'),
    (New-AvailableBarChart -canvasId 'matmulDownBar' -titleText 'Matmul-route ffn_down stage medians with shared dots on (ms)' -rows $rows -axisName 'matmul_stage' -metricKey 'ffn_down_ms' -yAxisTitle 'ms'),
    (New-AvailableBarChart -canvasId 'matmulTotalBar' -titleText 'Matmul-route total window stage medians with shared dots on (ms)' -rows $rows -axisName 'matmul_stage' -metricKey 'total_ms' -yAxisTitle 'ms')
) | Where-Object { $_ }

$html = [Text.StringBuilder]::new()
[void]$html.AppendLine('<!DOCTYPE html>')
[void]$html.AppendLine('<html lang="en">')
[void]$html.AppendLine('<head>')
[void]$html.AppendLine('<meta charset="utf-8" />')
[void]$html.AppendLine(("<title>{0} - LAIR prefill tuning analysis</title>" -f (ConvertTo-SafeHtml $outRelDir)))
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

[void]$html.AppendLine("<h1>Prefill chain-group and matmul-route tuning analysis</h1>")
[void]$html.AppendLine(("<p class='note'>Generated at {0}. Source sweep: <code>{1}</code>. Posture: threads=32, rowWidth=64, lengths 128/256, kernels gemv/tiled.</p>" -f (ConvertTo-SafeHtml $stampUtcJson), (ConvertTo-SafeHtml $SourceDir)))
[void]$html.AppendLine("<p class='note'>This analysis records that the current auto chain group and matmul-route widths remain justified; it does not change production constants. Bar charts compare completed same-session stage arms.</p>")

foreach ($chart in $chartConfigs) {
    [void]$html.AppendLine(("<h2>{0}</h2>" -f (ConvertTo-SafeHtml $chart.title)))
    [void]$html.AppendLine("<canvas id=`"$($chart.id)`" height=`"380`"></canvas>")
}

[void]$html.AppendLine('<script>')
foreach ($chart in $chartConfigs) {
    [void]$html.AppendLine("new Chart(document.getElementById('$($chart.id)'), $(ConvertTo-JsLiteral $chart.config));")
}
[void]$html.AppendLine('</script>')

[void]$html.AppendLine("<h2>Text summary</h2><p class='note'>See <a href='summary.md'><code>summary.md</code></a> for median tables and the decision. Machine-readable rows are in <a href='data.json'><code>data.json</code></a>.</p>")
[void]$html.AppendLine('</body>')
[void]$html.AppendLine('</html>')

Write-TextNoBom (Join-Path $outAbsDir 'report.html') $html.ToString()

Write-Host "wrote tuning analysis report directory $outRelDir ($($rows.Count) parsed metric rows, $($chartConfigs.Count) charts)"