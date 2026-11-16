# `lair.yaml` Configuration Reference

Every key the configuration loader accepts, what it controls, and how a live edit of it behaves. `lair.example.yaml` in the repository root is a complete copy-and-edit starting point.

## How the file is read

- One YAML file per host, named by `serve -c <path>` / `chat -c <path>` / `model … -c <path>`. There is no default path and no search.
- Validation is strict. An unknown key at any level is rejected with a message naming the key, the location, and the allowed keys at that level; a duplicate key is rejected outright. Nothing is silently ignored.
- A value of the wrong type is rejected as `… must be an integer value.` / `… must be a true or false value.` / `… must be a text value.`
- A rejected value never partially applies: the whole file fails to load and the process exits `1`.
- Top-level keys: `server`, `engine`, `models`, `ui`, `tiers`. `models` is the only required one.

## Mutation classes (hot reload)

While `serve` runs, the watched config file is re-read and each difference is routed by class. `chat` and `monitor` never watch anything.

| Class | Meaning |
|---|---|
| live | Takes effect inside the running engine, at the next step boundary. |
| new-requests | Changes admission/validation for requests admitted after the edit. Sequences already in flight are untouched. |
| restart-only | Refused with an actionable log line; the old value stays in force. |

The full classification table is at the end of this document. A refused edit is logged as `config_reload field=… class=… outcome=refused …` and the running engine keeps every current value; an edit that fails validation is dropped whole.

---

## `server`

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `server.bind` | string, non-empty | `127.0.0.1` | The address the HTTP listener binds. Loopback by default; anything else exposes the model to that network. |
| `server.port` | integer, `1`–`65535` | `8080` | The HTTP port. |

An out-of-range port is refused naming the value. The listener binds once at startup, so both keys are restart-only.

---

## `engine`

Engine-wide knobs shared by the served model.

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `engine.threads` | integer ≥ 0 | `0` | Worker threads. `0` means choose automatically: one worker per **physical** core, SMT siblings excluded (decode is memory-bandwidth-bound, so a sibling thread mostly adds contention). An explicit non-zero value wins over auto-sizing. |
| `engine.block_size` | integer, fixed | `16` | KV block size. Any value other than `16` is refused: `block size is fixed to 16 in this build`. |
| `engine.max_batch_tokens` | integer ≥ 1 | `512` | The largest prompt chunk the scheduler prefills in one step. It is a ceiling, not a fixed chunk size. |
| `engine.prefill_step_budget_ms` | integer ≥ 1, optional | omitted | A wall-clock budget each prefill step aims at. Omitted means chunk width comes from `max_batch_tokens` alone. Present lets the scheduler size each future chunk from measured step cost toward this budget, still capped by `max_batch_tokens` and never below one token. |
| `engine.prefill_kernel` | `gemv` \| `tiled` \| `gemm` | `gemv` | Which attention/weight path the prefill route selects. `gemv` is the batched row window, `tiled` adds row-tiled causal attention, `gemm` traverses dense weights by blocked matmul. |

`prefill_kernel` accepts all three values at load, but a selection this engine cannot run for the served file is refused **at model open**, naming that file's own weight formats. Today the blocked-matmul path exists only for `Q8_0`, `Q4_K`, and `IQ4_XS`: a file with any other dense format is refused rather than falling back.

Chunk width and row width are separate numbers: `max_batch_tokens` bounds how many prompt tokens a step forwards, while the engine's prompt windows run in sub-batches of a resolved **row** count (measured default 32, engine ceiling 64) that no configuration key sets on this build. A declared `prefill_buffers` cap can narrow that effective row width; see budgets below.

---

## `tiers` (optional)

Declared resident capacity for this host. These are declarations, not measurements: what the host actually carries is what its reported topology says, and a declaration that disagrees is **accepted with a warning** (`tiers_capacity … accepted against reported topology`), never a startup failure. The loader never consults topology itself.

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `tiers.dram_mb` | integer ≥ 1, optional | omitted (declares no byte total) | Declared system-memory total in megabytes. |
| `tiers.dram_cores` | integer ≥ 1, optional | omitted (declares no core total) | Declared logical-processor total. Worker claims are first-come against the host's posture, so an over-subscribed later claim is granted partially rather than refused; a per-node override below is the escape hatch when topology detection is wrong. |
| `tiers.nvme_path` | string, non-empty, optional | omitted | Declares only **that** this host reads from a disk tier, and which path. A byte ceiling does not belong here: the model file already is the complete set. |
| `tiers.nodes` | list of overrides, optional | omitted (no overrides) | Per-node overrides keyed by reported node position. |

Each `tiers.nodes` entry:

| Key | Type / allowed values | Meaning |
|---|---|---|
| `index` | integer ≥ 0, required, unique | The node position this override names, counting the nodes the host reports from 0. |
| `mb` | integer ≥ 1, optional | Declared megabytes for that node. |
| `cores` | integer ≥ 1, optional | Declared logical processors for that node. |

An override never replaces the reported posture; it states a disagreement with it, and the disagreement is logged. An entry that states neither `mb` nor `cores` is refused, as is a duplicate `index`. A `tiers:` block that states nothing at all is refused — omit the block entirely to declare no capacity.

Worker claims seat against the posture the host reports (a `nodes:` override corrects detection when it is wrong): an earlier claim is never preempted or displaced, and a later claim that cannot be satisfied in full is granted what its predecessors left rather than being refused. The `compute_grant` load line shows requested beside granted.

```yaml
tiers:
  dram_mb: 65536
  dram_cores: 32
  nodes:
    - index: 0
      mb: 32768
      cores: 16
```

---

## `models`

A list that must contain **exactly one** entry: this build serves a single model, and zero or several entries are refused (`must contain exactly one model … (found N)`).

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `models[].name` | string, required | — | The model id the API serves and every log line names. |
| `models[].path` | string, non-empty, required | — | The GGUF file to open. It must exist on disk **unless** the entry declares `source:`, in which case it may name a file that a fetch will create. |
| `models[].type` | `chat` only | `chat` | Model kind. Any other value is refused: `embed models are not supported in this build`. |
| `models[].max_context_tokens` | integer ≥ 128 | `4096` | The admission window: requests longer than this are refused with `context_length_exceeded`, and it is what `/v1/models` advertises as `context_length`. |
| `models[].kv_blocks` | integer ≥ 8 | `256` | The model's live KV block cap. It is a soft cap inside the physical reservation sized when the engine opens, so raising it above that reservation is clamped, not honoured. Equivalent to `budgets.kv.caps.dram_blocks` — see the budgets note. |
| `models[].force_fenced_tools` | boolean | `false` | When true, chat requests that declare tools always get LAIR's fenced tool block, even if the model's chat template renders tools itself, and the declared tools are kept out of the template's own `tools` context so exactly one tool surface reaches the model. |
| `models[].tool_parser` | `auto` \| `template` \| `fenced` \| `none` | `auto` | Which tool output parser the model resolves to at load. `auto`: the loaded template's detected capabilities decide, with LAIR's fenced block as fallback. `template`: the model's own chat-template rendering is authoritative and no fenced injection happens. `fenced`: LAIR's fenced block for every declared-tool request regardless of the template. `none`: inert — no tool surface reaches the model, content stays byte-for-byte as generated, and no `tool_calls` are reported. |
| `models[].reasoning` | `keep` \| `strip` | `keep` | How a recognized reasoning region reaches the wire. `keep`: it streams through as ordinary visible content, byte-for-byte. `strip`: it is withheld from visible content while its boundaries stay recorded. Recognition keys on the markup a generation actually contains, not on this key: with neither `<think>` nor `</think>` present, either value leaves output untouched. |
| `models[].max_reasoning_tokens` | integer ≥ 1, optional | omitted (unbudgeted) | How many tokens a reasoning region may spend before it is closed. **Hard constraint:** it binds only a model whose tokenizer carries `<think>` and `</think>` as whole tokens. A model declaring it on a vocabulary that cannot express either marker is refused at model open naming the model, the key, and the marker. |
| `models[].max_response_tokens` | integer ≥ 1, optional | omitted (unbudgeted) | How many tokens of visible output the model may generate. Omitted leaves request-level token limits at their existing semantics. |
| `models[].max_concurrent_requests` | integer ≥ 1, optional | omitted (uncapped) | Hard cap on how many requests this model holds admitted at once. A submission that would exceed it is refused naming the model and its quota slot rather than queued. Omitted bounds nothing. |
| `models[].placement` | mapping | see below | Where the model runs. |
| `models[].source` | mapping | omitted | Published-source pin. See below. |
| `models[].budgets` | mapping | omitted | Per-resource tier ceilings. See below. |

When both `max_reasoning_tokens` and `max_response_tokens` are set, they are separate budgets: a generation can exceed one bound and still produce an answer. Whichever ceiling ends a run is reported as `budget_exhausted` on the response body, on the closing chunk of a stream, and on the monitor's request frames.

---

## `models[].placement`

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `device` | `cpu` only | `cpu` | Compute device. Any other value is refused, naming the accelerator seam (`lair.backends.cuda`) that has to report the device before it becomes placeable. |
| `kv_residency` | `dram` only | `dram` | Where KV blocks live. Same accelerator-seam refusal for any other value. |
| `prefill` | `cpu` only | `cpu` | Where prompt forwards run. Same refusal for any other value. |
| `core_set` | string, optional | omitted (unpinned) | Pinned logical core ids as a comma-separated list of ids and inclusive ranges: `"0-15"`, `"0,2,8-11"`. Empty/blank means unpinned. Ids must be non-negative, ranges must be `low-high` with `low <= high`. |
| `thread_quota` | integer ≥ 1, optional | omitted (uncapped) | Hard cap on this model's worker threads, regardless of `engine.threads`. |
| `prefill_lane` | `shared` \| `dedicated` | `shared` | `shared`: prompt forwards ride the model's single execution backend. `dedicated`: prompt forwards run on a second CPU execution backend capped to its own share of this model's worker claim (floor of a quarter of the claim, with at least one worker left for decode) and its own activation buffers. |
| `core_overcommit` | boolean | `false` | Whether this model's pinned core set may overlap cores another model already claims. Default keeps pinned core sets disjoint. True relaxes the rule for this model's own claim, which then shares held cores under the compute-grant rules. |
| `numa_spread` | `auto` \| `local` \| `interleave` | `auto` | NUMA spread posture. `auto`: placement chooses the least-loaded reported node, lowest node index as tiebreak; a pinned core set resolves to the node carrying it. `local`: one node carries the model. `interleave`: a spread posture across the reported nodes. |

Constraints enforced beyond the value lists:

- Pinned core sets must partition the machine: two models pinning the same core id are refused naming both, unless the later model sets `core_overcommit: true`.
- A core id outside `[0, 63]` is refused at engine open: an affinity mask cannot express a wider processor group.
- Core pinning is applied on Windows only. On other platforms a configured `core_set` is a documented no-op that warns once at startup rather than being approximated.
- A non-empty `core_set` beside `numa_spread: interleave` is refused naming the model and the conflict; a pinned set spanning reported nodes is refused for `auto` and `local`.
- `numa_spread` is a preference resolved against reported topology, never a promise that the operating system moved anything.

---

## `models[].source` (optional)

Pins where the model's files come from. With `source:` present, `path` may name a file that does not exist yet: load fetches exactly the pinned revision, verifies the bytes, and only then opens the GGUF.

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `source.repo` | string, non-empty, required | — | The Hugging Face repo slug. |
| `source.revision` | string, non-empty, required | — | The commit sha or branch name pinned **as declared**. A branch becomes a commit sha when fetch resolves it into the manifest; load never re-resolves it. |
| `source.file` | a path string, or a mapping with `path` | required (one of `file`/`files`) | A single model file. |
| `source.files` | non-empty list of path strings or mappings | required (one of `file`/`files`) | A list spelling, for shard sets. |

Declaring both `file` and `files` is refused. Each file entry may carry:

| Key | Type / allowed values | Meaning |
|---|---|---|
| `path` | string, non-empty, required | The file's path/name within the repo, kept as a readable name beside the pack directory. Absolute paths and `..` escapes are refused by the fetch path. |
| `sha256` | string, exactly 64 lowercase hex characters | The pinned hash. Omitted pins no hash yet; the manifest records the hash LAIR computes after download. |
| `size` | integer ≥ 0 | The declared byte size. Omitted declares no size. |

Load-time behaviour: the pack directory beside `path` holds a `manifest.json`. A present artifact is verified with no network call at all; a disagreement refuses naming repo, revision, file, expected, and actual. An absent pinned file is fetched at exactly the revision the manifest already resolved. A pack with a `source:` block but no manifest entry yet takes the config as the pin, fetches if absent, and records the resolved sha. A transport failure while a fetch is required refuses naming what was being fetched — never a substitute revision, never an unverified fallback. A model with no `source:` is served from its local path with no provenance check at all.

Gated repositories authenticate through `LAIR_HF_TOKEN` (fallback `HUGGING_FACE_HUB_TOKEN`). A token is never accepted in this file — see the CLI reference.

---

## `models[].budgets` (optional)

Per-resource ceilings: each resource the entry states anything about gets its own ordered tier chain and its own caps. An exceeded ceiling moves **that resource** down its own chain rather than relocating the whole model.

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `budgets.weights` | mapping (`order`, `caps`) | omitted (declares nothing) | The weight bytes a forward reads. |
| `budgets.kv` | mapping (`order`, `caps`) | omitted | The paged KV blocks sequences write. |
| `budgets.prefill_buffers` | mapping (`order`, `caps`) | omitted | The activation buffers a prompt window works in. |
| `<resource>.order` | non-empty list of `dram` \| `vram` \| `nvme`, required | — | Tiers this resource demotes through, highest-priority residency first: `dram`, then `vram`, then `nvme`. Any tier may be skipped; no tier may appear twice; a chain must start at the tier its copies live on. |
| `<resource>.caps` | mapping of ceiling keys | omitted (no ceiling anywhere) | One ceiling per tier, and only for tiers that resource's own `order` names. |

Ceiling keys are a tier name joined to the unit that resource counts in:

| Resource | Ceiling keys | Unit |
|---|---|---|
| `weights`, `prefill_buffers` | `dram_mb`, `vram_mb`, `nvme_mb` | megabytes |
| `kv` | `dram_blocks`, `vram_blocks`, `nvme_blocks` | KV blocks |

Rules the loader enforces:

- A ceiling binds only a tier its own `order` names; naming one that is not in the chain is refused.
- `kv` at `dram` carries `kv_blocks`' floor of 8 (it reserves physical blocks exactly as `kv_blocks` does) and must fit in an `int`. Every other ceiling must be 1 or greater. Omitting a ceiling is "no model-level ceiling here", which is a different answer from zero.
- `weights` may name `nvme` as the **terminal** position of its chain (it is the origin store that already holds the complete set), but a ceiling there is refused: `weights read their complete set from disk storage already, so no model-level ceiling describes that position`.
- A read-side origin tier must be last in its chain, never mid-chain.

`kv_blocks` and `budgets.kv.caps.dram_blocks` spell the same ceiling. If both appear they must state the same number, or load fails naming both paths and values; if they agree, startup sizing is unchanged and one deprecation line says budgets are canonical for what LAIR holds at system memory. If only the budget spells it, that value sizes the reservation. If neither appears, today's default of 256 applies.

What actually binds on this build:

- `budgets.kv.caps.dram_blocks` is the live KV cap, the same seam `kv_blocks` uses.
- `budgets.prefill_buffers` caps narrow the effective prompt-window row width at engine open, and the engine reports requested versus effective row width.
- `budgets.weights` caps record the load-time residency decision.
- A chain naming a tier no opened pool backs is refused **at model open** naming the model, the resource, the tier, and what is missing (`accelerator-resident working copies run in 'lair.backends.cuda', which this build does not carry` for a `vram` claim; a disk-backed pool for a write-side `nvme` claim). Vocabulary is accepted; capability is reported, never silently dropped.

```yaml
models:
  - name: smollm-135m-instruct-q8_0
    path: models/smollm-135m-instruct-q8_0.gguf
    budgets:
      weights:
        order: [dram, nvme]      # nvme terminal, uncapped: it is the origin store
      kv:
        order: [dram]
        caps: { dram_blocks: 256 }
      prefill_buffers:
        order: [dram]
        caps: { dram_mb: 512 }
```

---

## `ui`

| Key | Type / allowed values | Default | Controls |
|---|---|---|---|
| `ui.recent_requests` | integer `1`–`1000` | `25` | How many completed requests the monitor's recent-request ring keeps, newest first. Shrinking it drops the oldest entries immediately. |

---

## Mutation class for every key

Taken from the change-application table, not from prose. Anything not listed as live or new-requests is restart-only, so no edit can fall through unclassified.

| Key path | Class | What a live edit actually does |
|---|---|---|
| `engine.max_batch_tokens` | live | Prefill chunk width changes at the next step boundary; sequences mid-chunk finish at the old width. |
| `engine.prefill_step_budget_ms` | live | Future chunks are sized from measured step cost from the next chunk issue; a forward already running finishes at the width chosen before the edit. Removing the key returns width to the configured ceiling. |
| `models[].kv_blocks` | live | Soft cap flips and, when lowered, drains least-recently-used cached prefix blocks. Clamped to the physical reservation opened at startup; active sequences are never touched. |
| `models[].budgets.kv.caps.dram_blocks` | live | Same seam as `kv_blocks`. Refused if the chain this ceiling binds does not hold working copies at `dram`. |
| `models[].placement.thread_quota` | live | Effective worker count changes from the next kernel call; lowered workers idle. With a dedicated prefill lane open it maps onto the decode side only and is refused when the new quota cannot cover both lanes. |
| `ui.recent_requests` | live | Ring capacity changes immediately; entries beyond the new cap are dropped oldest-first. |
| `engine.prefill_kernel` | new-requests | Re-points the prompt-window route from its next window boundary; a selection the served file's formats cannot run is refused naming them. |
| `models[].max_context_tokens` | new-requests | Admission validates against the new window; in-flight sequences are unaffected. |
| `models[].force_fenced_tools` | new-requests | Later chat requests re-resolve their tool surface; in-flight sequences keep their resolved limits. |
| `models[].tool_parser` | new-requests | Later chat requests resolve to the new parser. |
| `models[].reasoning` | new-requests | Later chat requests carry recognized reasoning regions under the new policy. |
| `models[].max_reasoning_tokens` | new-requests | Carried for later requests. Refused if the tokenizer cannot carry the reasoning markers the budget would bound. |
| `models[].max_response_tokens` | new-requests | Carried for later requests. |
| `models[].max_concurrent_requests` | new-requests | Submissions from now on are refused once every slot of the new cap is held; already-admitted sequences keep their slots and finish. |
| `models[].placement.numa_spread` | new-requests | Records the preference for later placement decisions. The node the engine opened on, its worker masks, and its weight first-touch posture stay as the open resolved them. |
| `models[].budgets.prefill_buffers.caps.*` | new-requests | Classified for later requests, but a live edit is **refused**: the physical clamp is established when the engine opens its row window, and a live edit cannot re-size it. Nothing in force changes. |
| `server.bind`, `server.port` | restart-only | The listener binds once at startup. |
| `engine.threads` | restart-only | Established when the engine opens. |
| `engine.block_size` | restart-only | KV block geometry is fixed for the engine's lifetime. |
| `models[].name`, `models[].path`, `models[].type` | restart-only | Server-side model reload is not implemented; restart to serve a different model. |
| `models[].source` | restart-only | Loadability depends on the manifest and fetch behaviour established when the model opened; a live edit cannot move a pin. The whole block reports as one field. |
| `models[].budgets.weights.caps.*` | restart-only | A weight ceiling binds what every forward reads; retargeting it means re-reading slabs from the model file, which is not implemented in this build. |
| `models[].budgets.<resource>.order` | restart-only | Chain shape was resolved when the engine opened. |
| `models[].placement.device`, `kv_residency`, `prefill` | restart-only | |
| `models[].placement.core_set` | restart-only | Pinning is established for the lifetime of the pool workers. |
| `models[].placement.prefill_lane` | restart-only | Pools, masks, activation buffers, and collision posture are load-time facts. |
| `models[].placement.core_overcommit` | restart-only | Changes what a running engine's workers contend for. |
| `tiers.dram_mb`, `tiers.dram_cores`, `tiers.nvme_path`, `tiers.nodes` | restart-only | Declared capacity binds accounting at open. `tiers.nodes` reports as one field carrying the whole declared posture. |

---

## Planned configuration axes

These keys are **not accepted by this build**. They are named in the project plan for unshipped work and are listed here so a reader cannot mistake them for available options. Anything outside this list that the loader rejects today stays rejected.

| Key path | Status | What it is meant to become |
|---|---|---|
| `placement.device: cuda` / `hybrid` | planned | An accelerator becomes placeable only when a compute backend reports it. Today any other value is refused naming `lair.backends.cuda`. |
| `placement.gpu_layers` (layer split point) | planned | Contiguous split: first N transformer layers resident in accelerator memory, remainder on CPU. Not a key this build parses. |
| `placement.kv_residency: vram` / `split` | planned | Device-backed KV block pools; `split` means device-tagged blocks read in two passes. |
| `placement.prefill: gpu` | planned | Prompt processing on an accelerator, with produced KV landing per `kv_residency`. |
| `budgets.*.caps.vram_mb`, `budgets.kv.caps.vram_blocks` | planned | The `vram` tier of an existing budget chain becomes a real ceiling rather than a refusal naming the missing pool. No new top-level keys are intended. |
| `budgets.kv` write-side `nvme` caps, `budgets.prefill_buffers` disk caps | planned | Write-side bytes reach storage only where something writes them; today that machinery does not exist. |
| Dense weight demotion to `nvme` while serving, and live weight-cap edits moving from restart-only to adjustable | planned | Needs refcounted, double-buffered slab migration. Until then weight ceilings are refused live. |
| `budgets.ple` (a per-resource chain for an n-gram/PLE table, with its own order and caps) | planned | A read-side resource like weights: origin shard uncapped, hot working-set cap per tier. |
| `models:` accepting more than one entry, and hot add/remove of models through a config edit | planned | Today exactly one entry is accepted and changing a model is a restart. |
| `models[].type: embed` | planned | Encoder-only models and an embeddings endpoint. |
| Runtime-adjustable per-resource tier caps (budget rebalancing while serving) | planned | Physical re-reservation at step boundaries. |
| Vision input configuration (a vision encoder beside a chat model) | planned, design tentative | No key names have been settled. |
| `sm_share`-shaped accelerator compute ceilings | not accepted, deliberately | A ceiling that names a partition nothing enforces would make a measured claim impossible to honour; route segments report stream-slot posture only. |
| An `auto` device fallback (silent CPU fallback when a configured accelerator is absent) | backlog idea, deliberately not a default | Fail-fast on a declared device is the posture. |
| An admin API or CLI mutation surface | explicit non-goal | Configuration-file hot reload is the only write path. |
