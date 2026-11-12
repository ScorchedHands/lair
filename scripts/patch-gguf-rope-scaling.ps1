param(
    [string]$InputPath = "models/Qwen3-0.6B-Q8_0.gguf",
    [string]$OutputPath = "models/Qwen3-0.6B-Q8_0-YaRN.gguf"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- rope-scaling declarations to insert

# Exactly the keys the YaRN gate declares; every other rope.scaling operand stays absent so both engines apply
# their shared reference defaults (attn 1, ext 1, beta_fast 32, beta_slow 1) against identical declared values.
$Insert = [ordered]@{
    "qwen3.rope.scaling.type"                    = @{ type = 8; string = "yarn" }  # STRING
    "qwen3.rope.scaling.factor"                  = @{ type = 6; single = 2.0 }     # FLOAT32
    "qwen3.rope.scaling.original_context_length" = @{ type = 4; uint32 = 512 }     # UINT32 (the pinned reference rejects i32 for this key)
}

# ---------------------------------------------------------------- GGUF v3 primitives

function Read-GgufStringBytes([System.IO.BinaryReader]$br) {
    $len = [long]$br.ReadUInt64()
    if ($len -lt 0 -or $len -gt ([int]::MaxValue)) { throw "GGUF string length $len out of range" }

    return $br.ReadBytes([int]$len)
}

# Walks one value and returns its raw byte span (element type + count for arrays), so preserved values are copied
# verbatim instead of re-encoded, keeping float bit patterns exactly as the source carries them.
function Read-GgufValueSpan([System.IO.BinaryReader]$br, [uint32]$type) {
    switch ($type) {
        0 { $br.ReadBytes(1) }         # UINT8
        1 { $br.ReadBytes(1) }         # INT8
        2 { $br.ReadBytes(2) }         # UINT16
        3 { $br.ReadBytes(2) }         # INT16
        4 { $br.ReadBytes(4) }         # UINT32
        5 { $br.ReadBytes(4) }         # INT32
        6 { $br.ReadBytes(4) }         # FLOAT32
        7 { $br.ReadBytes(1) }         # BOOL (stored as int8)
        8 {                            # STRING: u64 length + bytes, span includes the prefix itself
            $len = [long]$br.ReadUInt64()
            if ($len -lt 0 -or $len -gt ([int]::MaxValue)) { throw "GGUF string length $len out of range" }

            $bytes = $br.ReadBytes([int]$len)
            return @([BitConverter]::GetBytes($len)) + $bytes
        }
        9 {                            # ARRAY: u32 element type + u64 count + elements
            $et = [uint32]$br.ReadUInt32()
            if ($et -gt 12) { throw "unsupported GGUF array element type $et" }

            $n = [long]$br.ReadUInt64()

            $ms = New-Object IO.MemoryStream
            $w = New-Object IO.BinaryWriter($ms, [Text.Encoding]::UTF8, $true)
            $w.Write([uint32]$et)
            $w.Write([UInt64]$n)
            for ($i = 0; $i -lt $n; $i++) {
                $el = Read-GgufValueSpan $br $et
                $ms.Write($el, 0, $el.Length)
            }

            return $ms.ToArray()
        }
        10 { $br.ReadBytes(8) }        # UINT64
        11 { $br.ReadBytes(8) }        # INT64
        12 { $br.ReadBytes(8) }        # FLOAT64
        default { throw "unsupported GGUF value type $type" }
    }
}

function Write-GgufString([IO.BinaryWriter]$w, [string]$value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($value)
    $w.Write([UInt64]$bytes.Length)
    $w.Write($bytes, 0, $bytes.Length)
}

# The encoded KV for one declaration in $Insert: key field, u32 type id, value bytes.
function Encode-Declaration($key, $spec) {
    $ms = New-Object IO.MemoryStream
    $w = New-Object IO.BinaryWriter($ms, [Text.Encoding]::UTF8, $true)

    Write-GgufString $w ([string]$key)
    $w.Write([uint32]$spec.type)

    switch ($spec.type) {
        8 { Write-GgufString $w ([string]$spec.string) }
        6 { $w.Write([single]$spec.single) }
        4 { $w.Write([uint32]$spec.uint32) }
        5 { $w.Write([int32]$spec.int32) }
        default { throw "unsupported insertion type id $($spec.type)" }
    }

    return $ms.ToArray()
}

function AlignUp([long]$value, [long]$alignment) {
    # Integer-only ceiling: PowerShell's / operator is double division with no truncation.
    $rem = $value % $alignment
    if ($rem -eq 0) { return $value }

    return $value + ($alignment - $rem)
}

# Copies exactly one byte range from an open FileStream into the output stream.
function Copy-Range([IO.FileStream]$src, [IO.Stream]$dst, [long]$start, [long]$length) {
    if ($length -le 0) { return }

    $src.Seek($start, [IO.SeekOrigin]::Begin) | Out-Null
    $buffer = New-Object byte[] (1MB)
    $remaining = $length
    while ($remaining -gt 0) {
        $chunk = if ($remaining -lt $buffer.Length) { [int]$remaining } else { $buffer.Length }
        $read = $src.Read($buffer, 0, $chunk)
        if ($read -le 0) { throw "unexpected end of source at offset $start" }

        $dst.Write($buffer, 0, $read)
        $remaining -= $read
    }
}

# Compares one byte range in two files chunk by chunk; returns the first mismatch description or null.
function Compare-FileRanges([string]$pathA, [long]$startA, [string]$pathB, [long]$startB, [long]$length) {
    $fsA = [IO.File]::OpenRead($pathA); $fsB = [IO.File]::OpenRead($pathB)
    try {
        $fsA.Seek($startA, [IO.SeekOrigin]::Begin) | Out-Null
        $fsB.Seek($startB, [IO.SeekOrigin]::Begin) | Out-Null

        $bufA = New-Object byte[] (1MB); $bufB = New-Object byte[] (1MB)
        $done = 0
        while ($done -lt $length) {
            $chunk = if (($length - $done) -lt $bufA.Length) { [int]($length - $done) } else { $bufA.Length }
            $readA = $fsA.Read($bufA, 0, $chunk)
            $readB = $fsB.Read($bufB, 0, $chunk)
            if ($readA -ne $readB) { return "short read at offset $($startA + $done): $readA vs $readB" }

            for ($i = 0; $i -lt $readA; $i++) {
                if ($bufA[$i] -ne $bufB[$i]) { return "byte mismatch at data-relative offset $($done + $i)" }
            }

            $done += $readA
        }

        return $null
    } finally {
        $fsA.Close(); $fsB.Close()
    }
}

# The bytes at one absolute position, as hex — used for the data-section head/tail spot check.
function Get-HexAt([string]$path, [long]$start, [int]$count) {
    $fs = [IO.File]::OpenRead($path)
    try {
        if ($start -lt 0) { throw "negative read position $start" }

        $fs.Seek($start, [IO.SeekOrigin]::Begin) | Out-Null
        $bytes = New-Object byte[] $count
        $read = $fs.Read($bytes, 0, $count)
        if ($read -ne $count) { throw "short read of $count bytes at $start (got $read)" }

        return (@($bytes) | ForEach-Object { $_.ToString("x2") }) -join ""
    } finally { $fs.Close() }
}

# ---------------------------------------------------------------- parse the source header

$inputFull = (Resolve-Path -LiteralPath $InputPath).Path
$outputFull = if ([IO.Path]::IsPathRooted($OutputPath)) { [IO.Path]::GetFullPath($OutputPath) } else { [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $OutputPath)) }
if ($outputFull -eq $inputFull) { throw "refusing to overwrite the source model: input and output resolve to the same path" }

$size = (Get-Item -LiteralPath $inputFull).Length

$fsIn = [IO.File]::Open($inputFull, 'Open', 'Read', 'ReadWrite')  # ReadWrite share so a concurrent reader can inspect it
try {
    $br = New-Object IO.BinaryReader($fsIn)                        # the outer finally closes the stream itself

    $magic = [uint32]$br.ReadUInt32()
    if ($magic -ne 0x46554747) { throw "not a GGUF file (magic $magic): $inputFull" }

    $version = [uint32]$br.ReadUInt32()
    if ($version -ne 3) { throw "unsupported GGUF version $version (this patcher writes v3 headers)" }

    $nTensors = [long]$br.ReadUInt64()
    $nKv = [long]$br.ReadUInt64()

    Write-Host "source=$InputPath size=$size version=$version tensors=$nTensors kv=$nKv"

    # One pass over the KV section: every entry is recorded as its verbatim byte span, in original key order.
    $order = New-Object 'System.Collections.Generic.List[string]'
    $spans = @{}   # key -> @{ start; length } absolute offsets covering the whole encoded entry
    for ($i = 0; $i -lt $nKv; $i++) {
        $start = $fsIn.Position
        $keyBytes = Read-GgufStringBytes $br                       # its own u64 prefix is inside the span below
        $key = [Text.Encoding]::UTF8.GetString($keyBytes)

        $type = [uint32]$br.ReadUInt32()
        if ($type -gt 12) { throw "GGUF key '$key' has unsupported value type $type" }

        $valueSpan = Read-GgufValueSpan $br $type

        # The recorded span is what the reader actually consumed; the arithmetic cross-check makes any value-type
        # capture mistake fail here instead of silently shortening a preserved entry.
        $spanLength = [long]($fsIn.Position - $start)
        if ($spanLength -ne 8 + $keyBytes.Length + 4 + $valueSpan.Length) {
            throw "GGUF key '$key' span mismatch: reader consumed $spanLength bytes, capture accounted $($valueSpan.Length) value bytes"
        }

        if ($spans.Contains($key)) { throw "GGUF key '$key' appears twice in the source header" }

        $order.Add($key)
        $spans[$key] = @{ start = $start; length = $spanLength }
    }

    # general.alignment governs where tensor data starts (default 32 when absent).
    $alignment = [long]32
    if ($spans.Contains("general.alignment")) {
        $fsIn.Seek($spans["general.alignment"].start, [IO.SeekOrigin]::Begin) | Out-Null
        Read-GgufStringBytes $br | Out-Null                        # skip the key field again

        switch ([uint32]$br.ReadUInt32()) {
            0 { $alignment = [long]$br.ReadByte() }
            1 { $alignment = [long]$br.ReadSByte() }
            2 { $alignment = [long]$br.ReadUInt16() }
            3 { $alignment = [long]$br.ReadInt16() }
            4 { $alignment = [long]$br.ReadUInt32() }
            5 { $alignment = [long]$br.ReadInt32() }
            default { throw "general.alignment must be an integer value" }
        }

        if ($alignment -le 0) { throw "general.alignment must be positive, got $alignment" }
    }

    # Tensor info follows the KV section contiguously; walk it for names/offsets only — its bytes copy verbatim.
    $tensorInfoStart = $fsIn.Position
    $tensors = @()
    for ($i = 0; $i -lt $nTensors; $i++) {
        $nameBytes = Read-GgufStringBytes $br

        $ndims = [long]$br.ReadUInt32()
        if ($ndims -gt 4) { throw "tensor '$([Text.Encoding]::UTF8.GetString($nameBytes))' has $ndims dims" }

        for ($d = 0; $d -lt $ndims; $d++) { [long]$br.ReadUInt64() | Out-Null }
        $ggmlType = [uint32]$br.ReadUInt32()
        $offset = [long]$br.ReadUInt64()

        $tensors += @{ name = [Text.Encoding]::UTF8.GetString($nameBytes); type = $ggmlType; offset = $offset }
    }

    $headerEnd = $fsIn.Position
} finally {
    $fsIn.Close()
}

$dataStartOld = AlignUp $headerEnd $alignment
if ($dataStartOld -gt $size) { throw "computed data start $dataStartOld exceeds file size $size" }

$replaced = @($Insert.Keys | Where-Object { $spans.Contains($_) })
$inserted = @($Insert.Keys | Where-Object { -not $spans.Contains($_) })

Write-Host "alignment=$alignment headerEnd=$headerEnd dataStartOld=$dataStartOld"
if ($replaced.Length) { Write-Host ("replace=" + ($replaced -join ",")) }
if ($inserted.Length) { Write-Host ("insert=" + ($inserted -join ",")) }

# ---------------------------------------------------------------- write the patched file

New-Item -ItemType Directory -Force (Split-Path $outputFull) | Out-Null

$fsIn2 = [IO.File]::Open($inputFull, 'Open', 'Read', 'ReadWrite')  # read handle for span copies
$fsOut = [IO.File]::Open($outputFull, 'Create', 'Write')           # truncate-and-write: same inputs yield identical bytes
try {
    # Replaced keys stay in the header, so only genuinely new keys change the count.
    $newKv = [long]$nKv + $inserted.Length

    $headerMs = New-Object IO.MemoryStream
    $hw = New-Object IO.BinaryWriter($headerMs, [Text.Encoding]::UTF8, $true)
    $hw.Write([uint32]0x46554747)          # "GGUF"
    $hw.Write([uint32]3)                   # version 3
    $hw.Write([UInt64]$nTensors)           # tensor count unchanged
    $hw.Write([UInt64]$newKv)
    $headerBytes = $headerMs.ToArray()
    $fsOut.Write($headerBytes, 0, $headerBytes.Length)

    $written = 0
    foreach ($key in $order) {
        if (-not $Insert.Contains($key)) {
            Copy-Range $fsIn2 $fsOut $spans[$key].start $spans[$key].length
            $written++
            continue
        }

        # A declared key is replaced in place, keeping the source's KV ordering.
        $bytes = Encode-Declaration $key $Insert[$key]
        $fsOut.Write($bytes, 0, $bytes.Length)
        $written++
    }

    foreach ($key in $inserted) {
        $bytes = Encode-Declaration $key $Insert[$key]
        $fsOut.Write($bytes, 0, $bytes.Length)
        $written++
    }

    if ($written -ne $newKv) { throw "declared kv=$newKv but wrote $written KV entries" }

    # Tensor info copied verbatim from its original span.
    Copy-Range $fsIn2 $fsOut $tensorInfoStart ($headerEnd - $tensorInfoStart)

    # Padding is recomputed after the new header so tensor data lands on an aligned boundary unchanged.
    $newHeaderEnd = $fsOut.Position
    $dataStartNew = AlignUp $newHeaderEnd $alignment
    for ($p = $newHeaderEnd; $p -lt $dataStartNew; $p++) { $fsOut.WriteByte(0) }

    Copy-Range $fsIn2 $fsOut $dataStartOld ($size - $dataStartOld)
    $newSize = $fsOut.Position
} finally {
    $fsIn2.Close()
    $fsOut.Close()
}

Write-Host "kvNew=$newKv headerEndNew=$newHeaderEnd dataStartNew=$dataStartNew sizeNew=$newSize"

# ---------------------------------------------------------------- verify the patch preserved every tensor byte

$tailLen = $size - $dataStartOld
if ($newSize - $dataStartNew -ne $tailLen) { throw "tensor-data length changed: $tailLen -> $($newSize - $dataStartNew)" }

$mismatch = Compare-FileRanges $inputFull $dataStartOld $outputFull $dataStartNew $tailLen
if ($mismatch) { throw "tensor data differs after the patch: $mismatch" }

# Head and tail bytes of a tensor's payload region, at unchanged data-relative offsets.
$firstTensor = if ($tensors.Count -gt 0) { $tensors[0] } else { throw "source carries no tensors" }
foreach ($candidate in $tensors) { if ($candidate.offset -eq 0) { $firstTensor = $candidate; break } }

$headOld = Get-HexAt $inputFull $dataStartOld 16
$headNew = Get-HexAt $outputFull $dataStartNew 16
if ($headOld -ne $headNew) { throw "tensor data head differs: $headOld vs $headNew" }

$tailOld = Get-HexAt $inputFull ($size - 16) 16
$tailNew = Get-HexAt $outputFull ($newSize - 16) 16
if ($tailOld -ne $tailNew) { throw "tensor data tail differs: $tailOld vs $tailNew" }

Write-Host ("first tensor '{0}' offset={1}: head bytes identical ({2}); last data bytes identical ({3})" -f `
    $firstTensor.name, $firstTensor.offset, $headOld, $tailOld)
Write-Host "wrote $OutputPath (kv replaced=$($replaced.Length) inserted=$($inserted.Length); tensor info and all $tailLen data bytes byte-identical)"
