param(
    [string]$Tag = "b12032",
    [string]$ModelPath = "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    [string]$Out = "tests/fixtures/parity-templates-llama32.json",
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
        3 { $br.ReadInt16() | Out-Null }   # INT16
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
if (-not $kv.Contains("tokenizer.chat_template")) { throw "model has no tokenizer.chat_template: $ModelPath" }
$chatTemplate = $kv["tokenizer.chat_template"]
$utf8 = [Text.Encoding]::UTF8.GetBytes($chatTemplate)
$hasher = [Security.Cryptography.SHA256]::Create()
$templateSha256 = ([BitConverter]::ToString($hasher.ComputeHash($utf8)) -replace '-', '').ToLower()

$tokens = $kv["tokenizer.ggml.tokens"]
$bosId = [int]$kv["tokenizer.ggml.bos_token_id"]
# eos_token_id may be a single id or an array of ids; the first is the model EOS.
$eosRaw = $kv["tokenizer.ggml.eos_token_id"]
$eosId = if ($eosRaw -is [System.Collections.IEnumerable] -and $eosRaw -isnot [string]) { [int]($eosRaw[0]) } else { [int]$eosRaw }
$bosToken = $tokens[$bosId]
$eosToken = $tokens[$eosId]
$addBos = if ($kv.Contains("tokenizer.ggml.add_bos_token")) { [bool]$kv["tokenizer.ggml.add_bos_token"] } else { $true }

Write-Host "chat_template sha256=$templateSha256 bos='$bosToken' eos='$eosToken' add_bos=$addBos"

# ---------------------------------------------------------------- pinned server

if ($Port -eq 0) { $Port = Get-Random -Minimum 10100 -Maximum 49000 }

# llama-server logs heavily on startup; capture its streams to files instead of in-memory pipes so a
# full pipe buffer can never block the server mid-capture.
$serverOut = Join-Path ([IO.Path]::GetTempPath()) "lair-template-parity-server.out.log"
$serverErr = Join-Path ([IO.Path]::GetTempPath()) "lair-template-parity-server.err.log"
$proc = Start-Process -FilePath (Resolve-Path $exe).Path `
    -ArgumentList @("-m", $modelFull, "-c", "2048", "--threads", "$Threads", "--port", "$Port") `
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

    function Invoke-ApplyTemplate([string]$Body) {
        for ($attempt = 1; ; $attempt++) {
            # -SkipHttpErrorCheck keeps the 4xx response readable: a refusal from the reference is a
            # documented fact to record, and its body is already disposed by the time a catch block runs.
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/apply-template" -Method Post -ContentType "application/json" -Body $Body -TimeoutSec 120 -UseBasicParsing -SkipHttpErrorCheck
            if ($r.StatusCode -eq 200) {
                $json = $r.Content | ConvertFrom-Json
                return [pscustomobject]@{ Prompt = $json.prompt; Refusal = $null }
            }
            if ([int]$r.StatusCode -eq 400) {
                return [pscustomobject]@{ Prompt = $null; Refusal = $r.Content }
            }
            if ($attempt -ge 3) { throw "apply-template failed with HTTP $([int]$r.StatusCode): $($r.Content)" }
            Start-Sleep -Seconds 1
        }
    }

    # The template's date_string comes from llama.cpp's strftime_now (the capture clock). Probe an empty
    # conversation to read the exact date string the reference used, so the fixture pins it and the tests can
    # re-render deterministically without llama.cpp.
    $dateProbe = (Invoke-ApplyTemplate (@{ messages = @(); add_generation_prompt = $false } | ConvertTo-Json)).Prompt
    if ($null -eq $dateProbe) { throw "date probe was refused by the reference" }
    if ($dateProbe -notmatch 'Today Date: (.+?)\n') { throw "date probe did not expose a Today Date line: $dateProbe" }
    $dateString = $Matches[1]
    Write-Host "reference date_string='$dateString'"

    $toolDefinition = [ordered]@{
        type     = "function"
        function = [ordered]@{
            name        = "calculator"
            description = "Perform arithmetic on a numeric expression"
            parameters  = [ordered]@{
                type       = "object"
                properties = [ordered]@{
                    expression = [ordered]@{
                        type        = "string"
                        description = "Arithmetic expression to evaluate"
                    }
                }
                required   = @("expression")
            }
        }
    }

    $calculatorCall = [ordered]@{
        id       = "call_01"
        type     = "function"
        function = [ordered]@{
            name       = "calculator"
            arguments  = '{"expression": "2+2"}'
        }
    }

    $sysUserMessages = @(
        [ordered]@{ role = "system"; content = "You are a helpful assistant." },
        [ordered]@{ role = "user"; content = "What is the capital of France?" }
    )

    $multiTurnMessages = @(
        [ordered]@{ role = "system"; content = "You are a helpful assistant." },
        [ordered]@{ role = "user"; content = "What is 2+2?" },
        [ordered]@{ role = "assistant"; content = "2+2 is 4." },
        [ordered]@{ role = "user"; content = "And 3+3?" }
    )

    $toolCallTrailingMessages = @(
        [ordered]@{ role = "system"; content = "You are a helpful assistant." },
        [ordered]@{ role = "user"; content = "What is 2+2?" },
        [ordered]@{ role = "assistant"; content = ""; tool_calls = @($calculatorCall) }
    )

    $toolCallWithResponseMessages = @(
        [ordered]@{ role = "system"; content = "You are a helpful assistant." },
        [ordered]@{ role = "user"; content = "What is 2+2?" },
        [ordered]@{ role = "assistant"; content = ""; tool_calls = @($calculatorCall) },
        [ordered]@{ role = "tool"; tool_call_id = "call_01"; content = "4" }
    )

    $toolResponseMessages = @(
        [ordered]@{ role = "system"; content = "You are a helpful assistant." },
        [ordered]@{ role = "user"; content = "What is 2+2?" },
        [ordered]@{ role = "assistant"; content = ""; tool_calls = @($calculatorCall) },
        [ordered]@{ role = "tool"; tool_call_id = "call_01"; content = "4" },
        [ordered]@{ role = "assistant"; content = "The answer is 4." },
        [ordered]@{ role = "user"; content = "Thanks!" }
    )

    $cases = [ordered]@{
        sys_user                = @{ messages = $sysUserMessages; tools = $null }
        multi_turn              = @{ messages = $multiTurnMessages; tools = $null }
        tool_call_trailing      = @{ messages = $toolCallTrailingMessages; tools = $null }
        tool_call_with_response = @{ messages = $toolCallWithResponseMessages; tools = $null }
        tool_response           = @{ messages = $toolResponseMessages; tools = $null }
        tools_declared          = @{ messages = $sysUserMessages; tools = @($toolDefinition) }
        tools_tool_response     = @{ messages = $toolResponseMessages; tools = @($toolDefinition) }
    }

    $entries = foreach ($name in $cases.Keys) {
        foreach ($addGenerationPrompt in @($true, $false)) {
            Write-Host "capturing $name add_generation_prompt=$addGenerationPrompt ..."

            $bodyMap = [ordered]@{
                messages              = $cases[$name].messages
                add_generation_prompt = $addGenerationPrompt
            }
            if ($null -ne $cases[$name].tools) { $bodyMap["tools"] = $cases[$name].tools }
            $body = $bodyMap | ConvertTo-Json -Depth 12 -Compress

            $result = Invoke-ApplyTemplate $body

            $entry = [ordered]@{
                name                  = "$name" + $(if ($addGenerationPrompt) { "_gen" } else { "_no_gen" })
                messages              = $cases[$name].messages
                add_generation_prompt = $addGenerationPrompt
            }
            if ($null -ne $cases[$name].tools) { $entry["tools"] = $cases[$name].tools }

            if ($null -ne $result.Refusal) {
                Write-Host "  reference refused (400): $($result.Refusal)"
                $entry["reference_refusal"] = $result.Refusal
            } else {
                $entry["expected_prompt"] = $result.Prompt
            }

            $entry
        }
    }

    $fixture = [ordered]@{
        model_file        = $ModelPath.Replace('\', '/')
        model_sha256      = $sha256
        chat_template     = $chatTemplate
        chat_template_sha256 = $templateSha256
        bos_token         = $bosToken
        eos_token         = $eosToken
        add_bos           = $addBos
        date_string       = $dateString
        llama_cpp_tag     = $Tag
        llama_cpp_asset   = "llama-$Tag-bin-win-cpu-x64.zip"
        reference_mode    = "llama-server /apply-template (OpenAI chat body: messages, optional tools, add_generation_prompt). " +
                            "The reference strips a template-emitted leading BOS from the rendered text when the model's " +
                            "add_bos_token policy is true (llama.cpp re-adds it at tokenization), so expected_prompt has no " +
                            "leading bos_token; the parity tests add it back when comparing the adapter render, which binds " +
                            "the model BOS into the template context. Cases the reference refuses with 400 (e.g. " +
                            "add_generation_prompt=true on a conversation ending in an assistant tool-call message) record a " +
                            "reference_refusal instead of a prompt. date_string is pinned from the reference capture clock."
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
