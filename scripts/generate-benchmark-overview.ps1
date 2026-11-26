#Requires -Version 7.0
<#
.SYNOPSIS
  Generates the root benchmark overview page reports/index.html for LAIR CPU performance history.
.DESCRIPTION
  Reads curated machine-readable facts from reports/benchmark-history.json (milestones, llama.cpp bands,
  stage shares), scans committed report directories under reports/ to build a newest-first link table and
  audit which directories follow the current naming convention
  (YYYY-MM-DDTHH-mm-ssZ-optimize-<process>), and reads live per-report data files where practical
  (currently the Q6_K item's data.json) so fresh measurements replace curated fallbacks.
  Emits a Chart.js v4 page consistent with individual report.html pages: line chart for prefill throughput
  history by kernel/length with llama.cpp reference bands, bar charts for the Q6_K down-substage A/B and the
  LAIR-vs-llama decode band comparison, and a pie chart of current stage shares. Idempotent: safe to re-run;
  output is deterministic apart from the generated-at timestamp.
.PARAMETER RepoRoot
  Repository root. Defaults to the parent directory of this script.
.PARAMETER ReportsDir
  Report directory relative to RepoRoot. Default "reports".
.PARAMETER HistoryFile
  Curated history JSON relative to RepoRoot. Default "reports/benchmark-history.json".
.PARAMETER Out
  Output HTML relative to RepoRoot. Default "reports/index.html".
.EXAMPLE
  pwsh -File scripts/generate-benchmark-overview.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ReportsDir = 'reports',
    [string]$HistoryFile = 'reports/benchmark-history.json',
    [string]$Out = 'reports/index.html'
)

$ErrorActionPreference = 'Stop'

function Resolve-RepoPath([string]$relativeOrAbsolute) {
    if ([IO.Path]::IsPathRooted($relativeOrAbsolute)) { return $relativeOrAbsolute }
    Join-Path $RepoRoot $relativeOrAbsolute
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

$historyPath = Resolve-RepoPath $HistoryFile
$outPath = Resolve-RepoPath $Out
if (-not (Test-Path -LiteralPath $historyPath)) { throw "curated history file not found: $historyPath" }
$history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json

# ---------------------------------------------------------------------------
# 1. Scan report directories for links + naming-convention audit
# ---------------------------------------------------------------------------
$reportsRoot = Resolve-RepoPath $ReportsDir
if (-not (Test-Path -LiteralPath $reportsRoot)) { throw "reports directory not found: $reportsRoot" }

$fullTimestampRegex = [regex]'(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z'
$newConventionRegex = [regex]'^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z-[a-z0-9]'

$reportEntries = foreach ($dir in (Get-ChildItem -LiteralPath $reportsRoot -Directory | Sort-Object Name)) {
    if ($dir.Name.StartsWith('.')) { continue }
    $fullMatch = $fullTimestampRegex.Match($dir.Name)
    $dateOnlyMatch = ([regex]'(\d{4}-\d{2}-\d{2})').Match($dir.Name)

    $timestampUtc = $null
    if ($fullMatch.Success) {
        $timestampUtc = '{0}T{1}:{2}:{3}Z' -f $fullMatch.Groups[1].Value, $fullMatch.Groups[2].Value, $fullMatch.Groups[3].Value, $fullMatch.Groups[4].Value
    } elseif ($dateOnlyMatch.Success) {
        $timestampUtc = '{0}T00:00:00Z' -f $dateOnlyMatch.Groups[1].Value
    }

    # Family slug = directory name with timestamp tokens removed.
    $family = $fullTimestampRegex.Replace($dir.Name, '')
    $family = ([regex]'(\d{4}-\d{2}-\d{2})').Replace($family, '')
    $family = $family.Trim('-').Trim()

    # Prefer an HTML page, then a markdown summary.
    $href = $null; $kind = 'directory'
    if (Test-Path -LiteralPath (Join-Path $dir.FullName 'report.html')) { $href = "$($dir.Name)/report.html"; $kind = 'html' }
    elseif (Test-Path -LiteralPath (Join-Path $dir.FullName 'summary.md')) { $href = "$($dir.Name)/summary.md"; $kind = 'markdown' }

    # Manifest lane where a structured report carries one.
    $lane = ''
    $manifestPath = Join-Path $dir.FullName 'manifest.json'
    if (Test-Path -LiteralPath $manifestPath) {
        try {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            if ($manifest.lane) { $lane = [string]$manifest.lane }
        } catch { }
    }

    [pscustomobject]@{
        Name           = $dir.Name
        TimestampUtc   = $timestampUtc
        SortKey        = if ($timestampUtc) { $timestampUtc } else { '0000-00-00T00:00:00Z' }
        Family         = $family
        Href           = $href
        Kind           = $kind
        Lane           = $lane
        NewConvention  = $newConventionRegex.IsMatch($dir.Name)
    }
}

$sortedEntries = @($reportEntries | Sort-Object @{ Expression = 'SortKey'; Descending = $true }, Name)
$newConventionCount = @($sortedEntries | Where-Object NewConvention).Count

# ---------------------------------------------------------------------------
# 2. Chart data (curated history, with live per-report overrides where available)
# ---------------------------------------------------------------------------
$charts = $history.charts
$lineCfgSpec = $charts.prefillThroughputLine

$milestoneLabels = @($lineCfgSpec.milestones | ForEach-Object {
    $raw = $_.dateUtc   # ConvertFrom-Json may hydrate ISO strings to DateTime/DateTimeOffset
    if ($raw -is [datetime]) {
        $u = ([datetime]$raw).ToUniversalTime()
        '{0} {1}' -f $u.ToString('MM-dd'), $u.ToString('HH:mm')
    } elseif ($raw -is [datetimeoffset]) {
        $dt = (([datetimeoffset]$raw).UtcDateTime)
        '{0} {1}' -f $dt.ToString('MM-dd'), $dt.ToString('HH:mm')
    } else {
        $s = [string]$raw
        '{0} {1}' -f $s.Substring(5, 5), $s.Substring(11, 5)
    }
})

$lairColors = [ordered]@{ gemv128 = '#3b6fb6'; gemv256 = '#a34c8f'; tiled128 = '#2a7f62'; tiled256 = '#c0603f' }
$lineDatasets = @(foreach ($key in $lairColors.Keys) {
    [ordered]@{
        label       = if ($key.StartsWith('gemv')) { 'LAIR gemv {0} tok/s' -f $key.Substring(4) } else { 'LAIR tiled {0} tok/s' -f $key.Substring(5) }
        data        = @($lineCfgSpec.milestones | ForEach-Object { $_.lair_medians_tokPerSec.$key })
        borderColor = $lairColors[$key]
        tension     = 0.2
    }
})

$referenceDatasets = @(foreach ($band in $lineCfgSpec.llamaReferenceBands) {
    [ordered]@{
        label       = '{0} ~{1}' -f $band.datasetLabel, $band.value
        data        = @($milestoneLabels | ForEach-Object { $band.value })
        borderColor = if ($band.pm -gt 50) { '#8a8f96' } else { '#444444' }
        borderDash  = @(6, 4)
        pointRadius = 3
        tension     = 0
    }
})

# Q6_K down-substage bar: prefer the live data.json of that report when present.
$downBarSpec = $charts.q6kDownSubstageBar
$downValuesByArm = [ordered]@{}
foreach ($ds in $downBarSpec.datasets) { if (-not $downValuesByArm.Contains($ds.armKey)) { $downValuesByArm[$ds.armKey] = @() } }

$dataJsonPath = Resolve-RepoPath $downBarSpec.preferredLiveSourceFile
$barValueSource = 'curated fallback'
if (Test-Path -LiteralPath $dataJsonPath) {
    try {
        $q6kData = Get-Content -LiteralPath $dataJsonPath -Raw | ConvertFrom-Json
        foreach ($ds in $downBarSpec.datasets) {
            $armValues = @($downBarSpec.labels | ForEach-Object {
                $shapeKey = $_.Replace(' ', '_').ToLowerInvariant()   # "gemv 128" -> gemv_128
                $meds = $q6kData.($downBarSpec.fallbackDataJsonPath)
                if ($null -eq $meds) { throw 'data.json missing stage-profile medians node' }
                $shapeMeds = $meds.$shapeKey
                if ($null -eq $shapeMeds) { throw "data.json missing shape $shapeKey" }
                $val = $shapeMeds.($ds.armKey)
                if ($null -eq $val) { throw "data.json missing arm $($ds.armKey) at $shapeKey" }
                [double]$val
            })
            $downValuesByArm[$ds.armKey] = $armValues
        }
        $barValueSource = 'live data.json'
    } catch {
        Write-Warning "falling back to curated Q6_K bar values: $($_.Exception.Message)"
    }
}
foreach ($ds in $downBarSpec.datasets) {
    if (@($downValuesByArm[$ds.armKey]).Count -ne @($downBarSpec.labels).Count) {
        $downValuesByArm[$ds.armKey] = @($downBarSpec.fallbackValues.($ds.armKey))
    }
}

$decodeMilestones = $charts.decodeBar.milestones
$decodeDatasets = @(
    [ordered]@{
        label           = 'LAIR decode median (tok/s)'
        data            = @($decodeMilestones | ForEach-Object { Get-Median ([double[]]$_.lair_medians_tokPerSec) })
        backgroundColor = '#2a7f62'
    },
    [ordered]@{
        label           = 'llama.cpp tg32 (tok/s)'
        data            = @($decodeMilestones | ForEach-Object { $_.llama_tg32_tokPerSec.value })
        backgroundColor = '#9aa5b1'
    }
)

$pieSpec = $charts.stageSharePie

function ConvertTo-JsLiteral($value) { ($value | ConvertTo-Json -Depth 8 -Compress) }

$lineChartConfig = [ordered]@{
    type    = 'line'
    data    = [ordered]@{ labels = $milestoneLabels; datasets = @(@($lineDatasets) + @($referenceDatasets)) }
    options = [ordered]@{
        scales  = [ordered]@{ y = [ordered]@{ title = [ordered]@{ display = $true; text = $lineCfgSpec.yAxisTitle }; beginAtZero = $true } }
        plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } }
    }
}

$downBarConfig = [ordered]@{
    type    = 'bar'
    data    = [ordered]@{ labels = @($downBarSpec.labels); datasets = @(foreach ($ds in $downBarSpec.datasets) { [ordered]@{ label = $ds.label; data = $downValuesByArm[$ds.armKey]; backgroundColor = if ($ds.armKey -eq 'route_off') { '#9aa5b1' } else { '#2a7f62' } } }) }
    options = [ordered]@{
        scales  = [ordered]@{ y = [ordered]@{ title = [ordered]@{ display = $true; text = $downBarSpec.yAxisTitle }; beginAtZero = $true } }
        plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } }
    }
}

$decodeBarConfig = [ordered]@{
    type    = 'bar'
    data    = [ordered]@{ labels = @($decodeMilestones | ForEach-Object { $_.label }); datasets = $decodeDatasets }
    options = [ordered]@{
        scales  = [ordered]@{ y = [ordered]@{ title = [ordered]@{ display = $true; text = $charts.decodeBar.yAxisTitle }; beginAtZero = $true } }
        plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } }
    }
}

$pieConfig = [ordered]@{
    type    = 'pie'
    data    = [ordered]@{ labels = @($pieSpec.slices | ForEach-Object { $_.label }); datasets = @([ordered]@{ data = @($pieSpec.slices | ForEach-Object { $_.value }); backgroundColor = $pieSpec.backgroundColor }) }
    options = [ordered]@{ plugins = [ordered]@{ legend = [ordered]@{ position = 'bottom' } } }
}

$chartSections = @(
    @{ id = $lineCfgSpec.canvasId; title = $lineCfgSpec.title; note = ('LAIR points are committed-report medians per milestone (new arm for A/B items). Dashed lines: llama.cpp b12359 pp bands measured in {0} and carried across milestones because later sessions did not re-run llama-bench on this box.' -f $lineCfgSpec.llamaReferenceBands[0].measuredInReport); config = $lineChartConfig },
    @{ id = $downBarSpec.canvasId; title = $downBarSpec.title; note = ('{0} Values: {1}' -f $downBarSpec.note, $(if ($barValueSource -eq 'live data.json') { "read live from $($downBarSpec.preferredLiveSourceFile)" } else { 'curated fallbacks in benchmark-history.json' })); config = $downBarConfig },
    @{ id = $charts.decodeBar.canvasId; title = $charts.decodeBar.title; note = 'LAIR bars are the median of each report''s per-run decode-window medians; llama.cpp tg32 bands measured or carried as recorded by each source report.'; config = $decodeBarConfig },
    @{ id = $pieSpec.canvasId; title = $pieSpec.title; note = ('{0} Source: {1}' -f $pieSpec.note, $pieSpec.sourceReport); config = $pieConfig }
)

# ---------------------------------------------------------------------------
# 3. Emit HTML
# ---------------------------------------------------------------------------
$sb = [Text.StringBuilder]::new()
[void]$sb.AppendLine('<!DOCTYPE html>')
[void]$sb.AppendLine('<html lang="en">')
[void]$sb.AppendLine('<head>')
[void]$sb.AppendLine('<meta charset="utf-8" />')
[void]$sb.AppendLine(("<title>{0} - LAIR benchmark overview</title>" -f (ConvertTo-SafeHtml $history.title)))
[void]$sb.AppendLine('<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>')
[void]$sb.AppendLine(@'
<style>
  body { font-family: "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1080px; color: #1c1c1c; }
  h1 { font-size: 1.45rem; } h2 { font-size: 1.1rem; margin-top: 2.2rem; }
  .note { color: #444; font-size: 0.9rem; max-width: 76ch; }
  canvas { max-width: 1020px; }
  table { border-collapse: collapse; font-size: 0.85rem; margin-top: 1rem; width: 100%; }
  th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; vertical-align: top; }
  code { background: #f4f4f4; padding: 0 .3rem; }
  .badge-new { color: #2a7f62; font-weight: 600; }
  .badge-legacy { color: #8a5b00; }
</style>
'@)
[void]$sb.AppendLine('</head>')
[void]$sb.AppendLine('<body>')

$generatedUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
[void]$sb.AppendLine(("<h1>{0}</h1>" -f (ConvertTo-SafeHtml $history.title)))
[void]$sb.AppendLine(("<p class='note'>Generated by <code>scripts/generate-benchmark-overview.ps1</code> at {0} from <code>{1}</code>. Model: {2}. Host lane: {3}; standard posture: {4}.</p>" -f $generatedUtc, (ConvertTo-SafeHtml $HistoryFile), (ConvertTo-SafeHtml ($history.modelFamily.file + ' (' + $history.modelFamily.quant + ')')), (ConvertTo-SafeHtml ($history.hostLane.os + ', ' + $history.hostLane.cpuClass)), (ConvertTo-SafeHtml $history.hostLane.standardPosture)))
[void]$sb.AppendLine(("<p class='note'>Report directory naming going forward: <code>{0}</code> - the left-most UTC timestamp is the sort key. Legacy directories retain their original names and remain linked below. {1} of {2} committed report directories currently follow this convention.</p>" -f (ConvertTo-SafeHtml $history.namingConvention.current), $newConventionCount, $sortedEntries.Count))
[void]$sb.AppendLine("<p class='note'>{0}</p>" -f (ConvertTo-SafeHtml $history.hostLane.noiseNote))

foreach ($cfg in $chartSections) {
    [void]$sb.AppendLine(("<h2>{0}</h2>" -f (ConvertTo-SafeHtml $cfg.title)))
    [void]$sb.AppendLine("<canvas id=`"$($cfg.id)`" height=`"380`"></canvas>")
    [void]$sb.AppendLine(("<p class='note'>{0}</p>" -f (ConvertTo-SafeHtml $cfg.note)))
}

[void]$sb.AppendLine('<script>')
foreach ($cfg in $chartSections) {
    [void]$sb.AppendLine("new Chart(document.getElementById(`"$($cfg.id)`"), $(ConvertTo-JsLiteral $cfg.config));")
}
[void]$sb.AppendLine('</script>')

# Methodology notes + curated tables -------------------------------------------------
[void]$sb.AppendLine('<h2>Methodology</h2><ul class="note">')
foreach ($n in $history.methodologyNotes) { [void]$sb.AppendLine(("<li>{0}</li>" -f (ConvertTo-SafeHtml $n))) }
[void]$sb.AppendLine('</ul>')

$baselineArms = @($lineCfgSpec.milestones | Where-Object { $null -ne $_.baseline_arm_medians_tokPerSec })
if ($baselineArms.Count -gt 0) {
    $armText = ($baselineArms | ForEach-Object { '{0}: gemv128={1} gemv256={2} tiled128={3} tiled256={4}' -f $_.id, $_.baseline_arm_medians_tokPerSec.gemv128, $_.baseline_arm_medians_tokPerSec.gemv256, $_.baseline_arm_medians_tokPerSec.tiled128, $_.baseline_arm_medians_tokPerSec.tiled256 }) -join ' | '
    [void]$sb.AppendLine(("<p class='note'>Same-session pre-item baseline arms (batchedTok/s medians): {0}</p>" -f (ConvertTo-SafeHtml $armText)))
}

$q6kMilestone = @($lineCfgSpec.milestones | Where-Object id -eq 'q6k-hybrid-route')
if ($q6kMilestone.Count -gt 0 -and $null -ne $q6kMilestone[0].default_arm_medians_tokPerSec) {
    $d = $q6kMilestone[0].default_arm_medians_tokPerSec
    [void]$sb.AppendLine(("<p class='note'>Current Q6_K default-posture medians (posture-resolved width 8, plotted as the hybrid-on points on the history line): gemv128={0} gemv256={1} tiled128={2} tiled256={3}</p>" -f $d.gemv128, $d.gemv256, $d.tiled128, $d.tiled256))
}

[void]$sb.AppendLine('<h2>llama.cpp baselines recorded in committed reports</h2>')
[void]$sb.AppendLine('<table><tr><th>report</th><th>threads</th><th>pp128 tok/s</th><th>pp256 tok/s</th><th>tg32 tok/s</th><th>tg64 tok/s</th></tr>')
foreach ($row in $history.llamaCppBaselineTable) {
    [void]$sb.AppendLine(('<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td></tr>' -f (ConvertTo-SafeHtml $row.report), $row.threads, ($null -ne $row.pp128 ? $row.pp128 : '-'), ($null -ne $row.pp256 ? $row.pp256 : '-'), ($null -ne $row.tg32 ? $row.tg32 : '-'), ($null -ne $row.tg64 ? $row.tg64 : '-')))
}
[void]$sb.AppendLine('</table>')

[void]$sb.AppendLine('<h2>Early threads=8 LAIR prefill baselines</h2>')
[void]$sb.AppendLine('<table><tr><th>report</th><th>posture</th><th>gemv128</th><th>gemv256</th><th>tiled128</th><th>tiled256</th></tr>')
foreach ($row in $history.earlyThreads8LairPrefillTable) {
    [void]$sb.AppendLine(('<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td></tr>' -f (ConvertTo-SafeHtml $row.report), (ConvertTo-SafeHtml $row.posture), $row.gemv128, $row.gemv256, $row.tiled128, $row.tiled256))
}
[void]$sb.AppendLine('</table>')

# Report directory links ------------------------------------------------------------
[void]$sb.AppendLine(("<h2>Individual reports ({0})</h2>" -f $sortedEntries.Count))
[void]$sb.AppendLine('<table><tr><th>report directory</th><th>timestamp (UTC)</th><th>naming</th><th>lane / note</th></tr>')
foreach ($entry in $sortedEntries) {
    $link = if ($entry.Href) { "<a href='$($entry.Href)'>$(ConvertTo-SafeHtml $entry.Name)</a>" } else { ConvertTo-SafeHtml $entry.Name }
    $badge = if ($entry.NewConvention) { '<span class="badge-new">new convention</span>' } else { '<span class="badge-legacy">legacy name</span>' }
    [void]$sb.AppendLine(('<tr><td>{0} <code>[{1}]</code></td><td>{2}</td><td>{3}</td><td>{4}</td></tr>' -f $link, $entry.Kind, $(if ($entry.TimestampUtc) { $entry.TimestampUtc } else { '(no timestamp in name)' }), $badge, (ConvertTo-SafeHtml $entry.Lane)))
}
[void]$sb.AppendLine('</table>')

[void]$sb.AppendLine('</body>')
[void]$sb.AppendLine('</html>')

New-Item -ItemType Directory -Force (Split-Path -Parent $outPath) | Out-Null
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($outPath, $sb.ToString(), $utf8NoBom)

Write-Host "wrote $Out ($($sortedEntries.Count) report directories linked; Q6_K bar source: $barValueSource)"
