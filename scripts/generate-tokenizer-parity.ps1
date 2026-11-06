param(
    [string]$Tag = "b12032",
    [string]$ModelPath = "models/SmolLM-135M-Instruct-F16.gguf",
    [string]$Out = "tests/fixtures/parity-tokenizer-smollm.json",
    [int]$Threads = 4,
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$exe = Join-Path "tools/llama.cpp/$Tag" "llama-server.exe"
if (-not (Test-Path $exe)) { throw "pinned llama-server not found at $exe" }

$modelFull = (Resolve-Path $ModelPath).Path
$sha256 = (Get-FileHash -Algorithm SHA256 $modelFull).Hash.ToLower()

# ---------------------------------------------------------------- GGUF metadata reader

function Read-GgufString([System.IO.BinaryReader]$br) {
    $len = $br.ReadUInt64()
    return [Text.Encoding]::UTF8.GetString($br.ReadBytes([int]$len))
}

function Skip-GgufValue([System.IO.BinaryReader]$br, [uint32]$type) {
    switch ($type) {
        0 { $br.ReadByte() | Out-Null }    # UINT8
        1 { $br.ReadSByte() | Out-Null }   # INT8
        2 { $br.ReadUInt16() | Out-Null }  # UINT16
        3 { $br.ReadInt16() | Out-Null }   # INT32
        4 { $br.ReadUInt32() | Out-Null }  # UINT32
        5 { $br.ReadInt32() | Out-Null }   # INT32
        6 { $br.ReadSingle() | Out-Null }  # FLOAT32
        7 { $br.ReadBoolean() | Out-Null } # BOOL
        8 { Read-GgufString $br | Out-Null } # STRING
        9 {                                            # ARRAY
            $et = $br.ReadUInt32()
            $n = $br.ReadUInt64()
            for ($i = 0; $i -lt $n; $i++) { Skip-GgufValue $br $et }
        }
        default { throw "unsupported GGUF value type $type" }
    }
}

function Read-GgufKv([string]$path) {
    $fs = [IO.File]::OpenRead($path)
    $br = New-Object IO.BinaryReader($fs)
    try {
        $magic = $br.ReadUInt32(); $ver = $br.ReadUInt32()
        if ($magic -ne 0x46554747) { throw "not a GGUF file: $path" }
        $nTensors = $br.ReadUInt64(); $nKv = $br.ReadUInt64()
        $kv = [ordered]@{}
        for ($i = 0; $i -lt $nKv; $i++) {
            $key = Read-GgufString $br
            $type = $br.ReadUInt32()
            $value = switch ($type) {
                0 { $br.ReadByte() }      # UINT8
                1 { $br.ReadSByte() }     # INT8
                2 { $br.ReadUInt16() }    # UINT16
                3 { $br.ReadInt16() }     # INT16
                4 { $br.ReadUInt32() }    # UINT32
                5 { $br.ReadInt32() }     # INT32
                6 { $br.ReadSingle() }    # FLOAT32
                7 { $br.ReadBoolean() }   # BOOL
                8 { Read-GgufString $br } # STRING
                9 {                       # ARRAY
                    $et = $br.ReadUInt32()
                    $n = $br.ReadUInt64()
                    # List, not += : the tokenizer.ggml.tokens array is ~128K entries and += would be quadratic.
                    $items = New-Object 'System.Collections.Generic.List[object]' $n
                    for ($j = 0; $j -lt $n; $j++) {
                        if ($et -eq 8) { $items.Add((Read-GgufString $br)) } else { Skip-GgufValue $br $et }
                    }
                    $items
                }
                default { Skip-GgufValue $br $type; $null }
            }
            $kv[$key] = $value
        }
        return $kv
    } finally {
        $br.Close(); $fs.Close()
    }
}

$kv = Read-GgufKv $modelFull
$pre = if ($kv.Contains("tokenizer.ggml.pre")) { $kv["tokenizer.ggml.pre"] } else { "<absent:llama-bpe>" }
$addBos = if ($kv.Contains("tokenizer.ggml.add_bos_token")) { [bool]$kv["tokenizer.ggml.add_bos_token"] } else { $true }
$spacePrefix = if ($kv.Contains("tokenizer.ggml.add_space_prefix")) { [bool]$kv["tokenizer.ggml.add_space_prefix"] } else { $true }

Write-Host "model=$ModelPath pre='$pre' add_bos=$addBos add_space_prefix=$spacePrefix sha256=$sha256"

# ---------------------------------------------------------------- capture cases

# Plain-text cases only: parse_special=false and add_special=false keep the reference on the same
# surface LAIR's Encode exposes (no BOS, no special-token parsing), so a mismatch is a tokenizer
# mismatch. Cases cover ASCII words/punctuation, folded leading spaces, digits, CRLF/tab, multibyte
# UTF-8, astral-plane emoji, and a long realistic prompt.
$cases = [ordered]@{
    hello_world        = @{ content = "Hello, world!" }
    leading_space      = @{ content = " world" }
    leading_space_punct = @{ content = " world!" }
    ascii_sentence     = @{ content = "The quick brown fox jumps over the lazy dog." }
    digits             = @{ content = "1234567890 42 3.14159" }
    newline_tab        = @{ content = "line1`nline2`ttabbed`r`ncrlf" }
    multibyte          = @{ content = "café ☕ 你好 — über" }
    emoji              = @{ content = "emoji 😀🎉 test" }
    repeated_words     = @{ content = "the the the the the the the the the the" }
    long_prompt        = @{ content = "You are a helpful assistant. Answer the question concisely.`nQuestion: What is the capital of France?`nAnswer:" }
    empty              = @{ content = "" }
    single_space       = @{ content = " " }
}

# ---------------------------------------------------------------- pinned server

if ($Port -eq 0) { $Port = Get-Random -Minimum 10100 -Maximum 49000 }

# llama-server logs heavily on startup; capture its streams to files instead of in-memory pipes so a
# full pipe buffer can never block the server mid-capture.
$serverOut = Join-Path ([IO.Path]::GetTempPath()) "lair-tokenizer-parity-server.out.log"
$serverErr = Join-Path ([IO.Path]::GetTempPath()) "lair-tokenizer-parity-server.err.log"
$proc = Start-Process -FilePath (Resolve-Path $exe).Path `
    -ArgumentList @("-m", $modelFull, "-c", "512", "--threads", "$Threads", "--port", "$Port") `
    -WorkingDirectory (Get-Location).Path -NoNewWindow -PassThru `
    -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr

function Wait-Health([int]$Port, [System.Diagnostics.Process]$Proc) {
    for ($i = 0; $i -lt 120; $i++) {
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

    $entries = foreach ($name in $cases.Keys) {
        $content = $cases[$name].content
        $body = @{ content = $content; add_special = $false; parse_special = $false } | ConvertTo-Json -Compress
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 120 -UseBasicParsing -SkipHttpErrorCheck
        if ($r.StatusCode -ne 200) { throw "/tokenize failed for case '$name' ($($r.StatusCode)): $($r.Content)" }
        $json = $r.Content | ConvertFrom-Json

        Write-Host "  case $name -> $($json.tokens -join ',')"
        [ordered]@{
            name        = $name
            content     = $content
            tokens      = @($json.tokens)
            tokens_str  = @($json.tokens_str)
        }
    }

    $fixture = [ordered]@{
        model_file        = $ModelPath.Replace('\', '/')
        model_sha256      = $sha256
        pre               = $pre
        add_bos           = $addBos
        add_space_prefix  = $spacePrefix
        llama_cpp_tag     = $Tag
        llama_cpp_asset   = "llama-$Tag-bin-win-cpu-x64.zip"
        reference_mode    = "llama-server /tokenize with add_special=false and parse_special=false — the same " +
                            "surface LAIR's GgufTokenizer.Encode exposes (no BOS, no special-token parsing). " +
                            "tokens are the reference id sequence and tokens_str the pieces the reference " +
                            "resolved them to; the parity tests assert LAIR's Encode matches id-for-id. " +
                            "Cases are plain text so a mismatch is a tokenizer mismatch, not a special-token one."
        generated_utc     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        cases             = @($entries)
    }

    New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
    $fixture | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8NoBOM $Out
    Write-Host "wrote $Out"
} finally {
    if (-not $proc.HasExited) {
        $proc.Kill()
        $proc.WaitForExit(5000) | Out-Null
    }
}
