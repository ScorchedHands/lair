# LAIR configuration reference

LAIR reads a strict YAML file at startup and re-validates it during hot reload. Unknown keys, duplicate keys, malformed scalars, out-of-range numbers, and unsupported placement choices fail with an actionable error naming the qualified key and value. The current schema has no compatibility aliases for older capacity or tier spellings; use only the names documented here.

## Top-level shape

| Key | Required? | Default / behavior |
|---|---:|---|
| `server` | No | Defaults to bind `127.0.0.1`, port `8080`. |
| `engine` | No | Defaults to automatic thread selection, fixed block size 16, max batch width 512, and the default prefill kernel/route posture. |
| `models` | Yes | Must contain exactly one model entry in this build. |
| `ui` | No | Controls Web UI recent-request ring size; defaults to 25. |
| `host` | No | Declares operator-supplied capacity posture and NUMA node overrides. Omit the whole block to declare none. A present `host:` block must state at least one of `dram_mb`, `vram_mb`, `cpu_cores`, `disk_path`, or `numa.nodes`. |

## Server settings

| Key | Accepted values / range | Default |
|---|---|---|
| `server.bind` | Non-empty address string | `127.0.0.1` |
| `server.port` | Integer 1–65535 | `8080` |

Server listener settings are established when the server starts; edits to them require a restart.

## Engine settings

| Key | Accepted values / range | Default | Hot-reload class |
|---|---|---:|---|
| `engine.threads` | Integer ≥ 0; 0 selects automatically | 0 | Restart-only |
| `engine.block_size` | Fixed to 16 in this build | 16 | Cannot be changed to a valid value while serving |
| `engine.max_batch_tokens` | Integer ≥ 1 | 512 | Live |
| `engine.prefill_step_budget_ms` | Optional integer ≥ 1; absent means no measured step-time budget | unset | Live |
| `engine.prefill_kernel` | `gemv`, `tiled`, or `gemm`; model open may refuse a kernel whose dense weight formats it cannot serve | `gemv` | New requests only |

## Host capacity posture

The top-level host block states what the operator declares about this machine. It does not require exact agreement with reported topology for system-memory totals; engine startup warn-and-accepts declared DRAM totals against OS topology when present, while accelerator totals are operator-declared ceilings rather than driver-detected budgets.

```yaml
host:
  dram_mb: 65536       # optional system memory total in MB; omit to declare no byte total
  vram_mb: 8192        # optional accelerator ceiling declared by the operator; omit to declare none
  cpu_cores: 32        # optional logical-processor total for compute grants and placement reporting
  disk_path: D:\       # presence-only storage marker; host.disk_path takes no byte ceiling here
  numa:
    nodes:             # one override per reported node position, keyed by index from 0
      - index: 0
        mb: 32768
        cores: 16
```

Rules:

- `dram_mb`, `vram_mb`, and per-node `mb` values are whole-megabyte integers ≥ 1 when present.
- `cpu_cores` and per-node `cores` values are logical processor counts ≥ 1 when present.
- `disk_path` is a non-empty path marker only; no byte ceiling belongs at that host position. Model budgets use their own tier cap keys instead.
- Each NUMA node override names one reported node index (`index` ≥ 0, unique) and states `mb`, `cores`, or both. An entry with neither value is refused.
- Host capacity totals, disk-path posture, and per-node overrides bind accounting/placement decisions when the engine opens; edits are restart-only.

## Model entries

Each model entry accepts exactly these keys:

```yaml
models:
  - name: smollm-135m-instruct-q8_0   # required
    path: models/smollm-135m-instruct-q8_0.gguf  # required; must exist unless a source pin can fetch it
    type: chat                        # optional; only `chat` is supported in this build
    max_context_tokens: 4096          # integer ≥ 128, default 4096
    kv_blocks: 256                    # integer ≥ 8 when declared; budgets can carry the same ceiling instead
    force_fenced_tools: false         # optional bool, default false
    tool_parser: auto                 # `auto`, `template`, `fenced`, or `none`
    reasoning: keep                   # `keep` or `strip`
    max_reasoning_tokens: 256         # optional integer ≥ 1; requires tokenizer marker support at model open
    max_response_tokens: 1024         # optional integer ≥ 1
    max_concurrent_requests: 8        # optional integer ≥ 1; absent is uncapped
    budgets: ...                      # optional per-resource residency ceilings, see below
    source: ...                       # optional published-source pin, see below
    placement: ...                    # optional placement posture, see below
```

`models[0].kv_blocks` and `models[0].budgets.kv.caps.dram_blocks` spell the same system-memory KV ceiling. If both are present, they must state the same number; if only one is present, that spelling sizes the reservation. Neither declaration keeps today's default of 256 unless neither key is declared.

## Model budgets

Budget chains declare per-resource residency ceilings for three resources: `weights`, `kv`, and `prefill_buffers`. A resource omitted from `budgets` declares no chain. Every present chain requires an ordered tier list and may optionally declare ceiling keys under `caps`.

Tier vocabulary, highest-priority residency first:

```text
dram -> vram -> disk
```

A chain must name at least one tier, cannot repeat a tier, and must keep the priority order above while skipping tiers it does not use. Cap key names are built from the tier plus that resource's unit:

| Resource | Unit | Accepted cap keys |
|---|---|---|
| `weights` | megabytes of working copies | `dram_mb`, `vram_mb`, `disk_mb` (schema name; weight disk ceilings are refused because weights already read their complete set from storage) |
| `kv` | KV blocks | `dram_blocks`, `vram_blocks`, `disk_blocks` |
| `prefill_buffers` | megabytes of working copies | `dram_mb`, `vram_mb`, `disk_mb` |

Rules:

- A ceiling binds only a tier its own chain names. For example, `caps.vram_mb` is invalid unless the same resource's `order` includes `vram`.
- System-memory KV ceilings (`kv.dram_blocks`) reserve physical blocks exactly like legacy `kv_blocks`, so they must be ≥ 8 and within an integer block count.
- Other declared ceilings must be ≥ 1; omitting a cap at a named tier means "no model-level ceiling there," not unlimited capacity.
- Weight disk positions are read-side origin-store positions, not working-copy claims, so weight `disk_mb` ceilings are refused. Write-side resources (`kv`, `prefill_buffers`) can name the disk tier in schema, but model open resolves chains against actually opened pools and refuses claims that cannot be seated there.

Example:

```yaml
models:
  - name: example-model
    path: models/example.gguf
    budgets:
      weights:
        order: [dram]
        caps: { dram_mb: 4096 }
      kv:
        order: [dram, vram]
        caps: { dram_blocks: 256 }
      prefill_buffers:
        order: [vram]
        caps: { vram_mb: 128 }
```

Budget chain orders are structural and restart-only. System-memory KV ceilings behave live like the legacy `kv_blocks` ceiling; device-backed KV ceilings are restart-only because reseating an opened block pool is not a live operation; weight ceilings are restart-only; prefill-buffer ceilings apply to new requests even when no route currently clamps against them.

## Placement posture

Placement keys decide where model execution, prompt windows, and KV blocks run or sit. The current build resolves neutral `auto` postures onto the backends it can actually open today: whole-model device/prefill defaults resolve CPU-only behavior, while accelerator-backed split spans are expressed through `gpu_layers` when a usable CUDA seam is opened.

```yaml
placement:
  device: auto           # omitted or `auto` resolves to CPU-only behavior; explicit accelerator posture is refused until a backend reports it
  kv_residency: dram     # optional; `auto`, `dram`, `vram`, or `split`; absent defaults host-resident today
  prefill: auto          # omitted or `auto` resolves to CPU-only behavior; explicit accelerator stage is refused until a backend reports it
  gpu_layers: 0          # optional integer ≥ 0; count of transformer layers execution opens on device `cuda`, counted from layer 0
  prefill_lane: shared   # optional; `shared` or `dedicated`
  core_set: "0-15"       # optional pinned logical cores, e.g. `"0-15"` or `"0,2,8-11"`
  thread_quota: 8        # optional integer ≥ 1 cap for this model's workers
  core_overcommit: false # optional bool; relaxes disjoint core-set ownership for this model only
  numa_spread: auto      # optional; `auto`, `local`, or `interleave`
```

Placement rules and hot-reload classes:

| Key | Accepted values / default | Hot-reload class | Notes |
|---|---|---|---|
| `placement.device` | `auto` (default when omitted) or `cpu`; accelerator posture refused until a compute backend reports it in the CUDA seam | Restart-only | Resolves once at load onto opened whole-model compute behavior. |
| `placement.kv_residency` | `auto`, `dram`, `vram`, `split`; absent defaults host-resident today | Restart-only | Model open decides whether device-backed block pools can be seated and enforces finite ceilings for those claims. |
| `placement.prefill` | `auto` (default when omitted) or `cpu`; accelerator stage refused until a compute backend reports it in the CUDA seam | Restart-only | Resolves once at load onto opened prefill behavior. |
| `placement.gpu_layers` | Integer ≥ 0, default 0; names layers execution opens on device `cuda`, counted from layer 0 | Restart-only | Model open may refuse if the file lacks enough layers, no usable CUDA/capacity seam can back the split, or an opened span uses tensor formats outside the current PTX kernel coverage (`F32`, `F16`, `BF16`, `Q4_K`, `Q6_K`). |
| `placement.prefill_lane` | `shared` (default) or `dedicated` | Restart-only | Dedicated lane sub-allocates a prefill worker share and activation buffers from this model's claim. |
| `placement.core_set` | Optional comma-separated core ids/ranges, e.g. `"0-15"`; omitted leaves the model unpinned | Restart-only | Pinned models may not share cores unless that model opts into overcommit. |
| `placement.thread_quota` | Optional integer ≥ 1 | Live | Caps this model's worker count regardless of engine thread selection. |
| `placement.core_overcommit` | Boolean, default false | Restart-only | Relaxes disjoint core-set ownership for this model only; changes contention established at open. |
| `placement.numa_spread` | `auto`, `local`, or `interleave`; default auto | New requests only | Records preference later placement decisions read; opened worker bindings are not re-bound live. |

## Model source pins

A model may declare a published-source pin instead of relying on an already-present local file:

```yaml
source:
  repo: QuantFactory/SmolLM-135M-Instruct-GGUF   # required non-empty Hugging Face repo slug
  revision: main                                  # required commit sha or branch; stored as declared until fetch resolves branches to shas
  file: SmolLM-135M-Instruct.Q8_0.gguf           # exactly one of `file` or `files` is required
```

A source block may use either a single `file:` value or a non-empty `files:` list. Each file entry can be a plain path string, or a mapping with `path`, optional lowercase 64-character hex `sha256`, and optional non-negative integer `size`. Source pins are structural loadability posture; edits require restart because manifest/fetch behavior is established when the model opens. Gated repos authenticate through `LAIR_HF_TOKEN` (or fallback `HUGGING_FACE_HUB_TOKEN`) without logging or storing tokens in manifests/errors.

## Hot-reload classes

Live changes apply inside the running engine at step boundaries:

- `engine.max_batch_tokens`
- `engine.prefill_step_budget_ms`
- system-memory KV ceilings through legacy `models[0].kv_blocks` when both configurations use that spelling, or through `models[0].budgets.kv.caps.dram_blocks`
- `models[0].placement.thread_quota`
- `ui.recent_requests`

New-request-only changes apply to requests admitted after the edit:

- `engine.prefill_kernel`
- `models[0].max_context_tokens`
- tool/reasoning policy keys (`force_fenced_tools`, `tool_parser`, `reasoning`)
- token budget ceilings (`max_reasoning_tokens`, `max_response_tokens`)
- admission concurrency (`max_concurrent_requests`)
- `models[0].placement.numa_spread`
- prefill-buffer ceiling edits

Restart-only changes are refused with a restart message because they establish opened engine state, pins, backend routes, capacity accounting, or structural chains:

- server bind/port and engine thread sizing
- model identity/path/type/source pins
- host capacity totals (`host.dram_mb`, `host.vram_mb`, `host.cpu_cores`), disk-path posture (`host.disk_path`), and NUMA node overrides (`host.numa.nodes`)
- whole-model placement postures (`device`, `prefill`), KV residency mode, GPU layer split count, prefill lane, core pinning/overcommit settings, source pins, budget chain orders, weight ceilings, device-backed KV ceilings, and any other key not classified live or new-request-only
