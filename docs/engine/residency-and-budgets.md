Scope: This page covers how LAIR accounts capacity today — the compute-capacity ledger and the device (tier/resource) ledger, budget chains per model resource and what each tier actually backs in this build, weight arenas and mmap placement of model files, host posture detection with declared-override semantics, NUMA spread preferences, CPU worker placement on Linux physical cores with auto worker sizing, prefill-buffer row-width clamps, PLE row residency gating, and `budget_exhausted` refusal semantics. As of 2026-11-27 this reflects implemented behavior under `src/` only; planned placement capabilities are stated as refused vocabulary, never as working posture.

# Capacity accounting: ledgers, budget chains, weight arenas, host posture

## What this subsystem is for

Two questions have to answer before a model serves and while it runs: *can this admission fit what the machine actually holds* (ledgers), and *what may move where when its declared ceilings bind* (budget chains). LAIR splits these deliberately: ledgers count real capacity claims against real pools; budget chains are per-resource declarations of residency priority with optional ceilings. Neither relocates anything silently — every unresolvable posture refuses, naming the model, resource, tier, and what this host lacks.

## The two ledgers

```mermaid
classDiagram
    class ComputeCapacityLedger {
        +int RemainingCores
    }
    class ComputeGrant {
        +int RequestedCores
        +int GrantedCores
        +int ExclusiveCores
        +bool SharesHeldCores
    }
    class DeviceLedger {
        +ClaimsAt(tier, resource)
        +SeamFor(tier, resource)
    }
    class CapacitySeam {
        <<record>>
    }
    ComputeCapacityLedger --> ComputeGrant : seats claims in registration order
    ComputeGrant ..> DeviceLedger : one admission consults both ledgers
    DeviceLedger o-- CapacitySeam : keyed by tier and resource, only opened positions exist
```

Paths: `src/Lair.Models/ComputeCapacityLedger.cs`, `src/Lair.Models/DeviceLedger.cs`. Semantics the diagram sketches (full detail below): a grant records what was asked against what runs; ledger seats exist only for tier/resource positions an open actually backs.

- **Compute capacity** is core quota, not bytes. A model's grant states requested cores (`RequestedCores` — its engine thread choice capped by `thread_quota` where declared), granted cores (`GrantedCores`, never zero: "a claim that finds nothing left runs beside standing grants rather than being refused"), and exclusive cores (`ExclusiveCores`, clamped so the total across claims never exceeds the reported host logical-processor count). A later oversubscribed claim is *clamped to what remains* — a partial grant naming requested against granted (`GrantVersusRequest()`, e.g. `8/3`) rather than an admission failure; fairness degradation stays visible through `SharesHeldCores` and the grant's diagnostic text ("requested 8 cores / granted 5 (2 exclusive, shared with earlier grants)" shape). Overlap beyond exclusivity happens only where `models[].placement.core_overcommit` opts in, keeping the whole requested share while recording the overlap — "records the overlap instead of hiding it". Claims seat on the nodes their placement chose: capacity is tracked per reported node, a claim fills its chosen nodes in node order clamping to each node's unclaimed remainder, and a pinned posture's node is authoritative (the seam never moves a pinned claim elsewhere). A claim registered without an explicit node choice spreads across every reported node in node order. `RemainingCores` answers what no seated grant has taken as its own; shared overlap never consumes it.
- **Device capacity** claims seat at a `[tier, resource]` position through a `CapacitySeam` record (`src/Lair.Models/DeviceLedger.cs`) — the ledger is built from exactly the seam set the opened posture reports, so an unopened tier has no seatable position rather than an unlimited one. Claims arrive in registration order; first-come-first-served is load-bearing: an earlier claim is never revoked, preempted, or displaced by a later one — when room is gone, *later* admissions answer refusal (or deferral for device fair-share candidates, which hold their slot and ride a later boundary — scheduling.md owns that split). `ClaimsAt(tier, resource)` gives the operator view of contested holders at a tier; `SeamFor(tier, resource)` resolves one position's seam. Read-side origin-store positions (a model file on disk) deliberately hold no ledger capacity: nothing is claimed against storage that already holds the complete set.

Which tier positions exist to seat against comes from the opened posture itself (`CapacitySeam` / `ReportOpenedTier`, feeding `GenerationService.AcceleratorAccountingOf` and weight-tier reporting as cuda-backend.md describes): a CPU-only open opens DRAM positions only; an accelerator-backed open adds device positions for whatever its backend actually backs.

## Budget chains: resources, tiers, units

Each model may declare per-resource budget chains (`src/Lair.Models/ModelBudgets.cs`). The vocabulary is fixed and small:

| Resource key | What it bounds | Ceiling unit | Read/write side |
|---|---|---|---|
| `weights` | weight bytes a forward reads | `_mb` | read-side — complete set already exists in the model file (origin store) |
| `kv` | paged KV blocks sequences write | `_blocks` | write-side — resident where its pool opened |
| `prefill_buffers` | transient activation working sets of a prompt window | `_mb` | write-side, "never disk material today" by design comment |
| `ple` | PLE n-gram table rows admitted into residency | `_mb` | read-side origin store; only resident rows are bounded |

Tiers run in one fixed priority order — **`dram`, then `vram`, then `disk`** — any tier skippable, none repeated (`BudgetTierNames.DeclaredNames`). Cap keys are tier name joined to the resource's unit: `budgets.kv.caps.dram_blocks`, `budgets.weights.caps.vram_mb`, and so on; a key built from one resource's unit never parses as another's (megabytes are not how KV counts anything, blocks are not how weights count theirs — refusal wording names the accepted vocabulary).

Config shape (`src/Lair.Http/Config/YamlConfigLoader.cs`): `models[].budgets.<resource>.order` is a list of tier names highest-priority first; `.caps` maps qualified ceiling keys. A cap binds only a tier its own chain's order names — declaring `vram_mb` beside an order that skips `vram` refuses at load. Structural edits (`order`, the whole block) are restart-only by design: they carry no single canonical scalar to hot-compare, and semantic changes surface through full-config diff instead (ConfigReload skip rule). Individual cap scalars hot-edit per key — see "Knobs" below for which ones apply live today.

Resolution lands each declared position against what the host opened (`ModelBudgets.Resolve`): a working-copy claim at a tier no opened pool backs refuses naming model, resource, tier, and the missing capability — weights-in-VRAM names the compute backend seam that would need device-resident weight copies; KV-in-VRAM names the device-backed block pool; write-side disk claims name the writing machinery this build does not carry (`DiskWriteSupport` is false everywhere here). A read-side resource's `disk` position resolves as its **origin store** — legal only as the chain's final position, and never under a ceiling of ours because that storage already holds the complete set.

```mermaid
flowchart TD
    D[declared chains: order + caps] --> V{structural pass:<br/>known tiers, priority order,<br/>no repeats, cap on named tier only}
    V -->|malformed| R1[refused at config load<br/>naming qualified path and value]
    V -->|well-formed| P{posture pass against opened pools:<br/>OpenedPoolTiers + OpenedKvPoolTiers,<br/>DiskWriteSupport=false today}
    P -->|read-side disk position, last in chain| O[lands as origin store<br/>no ceiling, no ledger claim]
    P -->|tier with an opened pool| C[resolved working-copy position<br/>with declared cap or none]
    P -->|tier nothing backs| R2[refused at model open:<br/>names model + resource + tier + missing capability]
    C --> L[claims seat in DeviceLedger,<br/>registration order, FCFS]
```

**What each tier actually backs today:** `dram` backs everything this engine opens — weight arenas, KV pools, activation buffers. `vram` seats claims only where an open reports positions: device-resident weight copies behind a CUDA split posture, and *KV* specifically through device-backed block ranges (mirror accounting; `budgets.kv.caps.vram_blocks` must state a finite ceiling or the residency refuses — refusal wording quoted in "Refusals"). Quantized KV pairs refuse any non-host residency outright (kv-cache.md). `disk` is real only as the read-side origin store of model files; no write-side bytes reach storage anywhere in this build, and declaring that posture answers with what is missing rather than pretending.

## `budget_exhausted` semantics

Two distinct refusal families ride on top of admission, both named end to end:

- **Declared-ceiling breach.** When a claim cannot fit its chain's declared cap at the tier it resolved to, resolution/admission raises the budget exception naming *model, resource, exhausted tier*, and the ledger breakdown behind it (`BudgetExhaustedException`, `src/Lair.Models/`). Serving reports it as the budget-exhausted marker on the result path (request-lifecycle.md).
- **Pool capacity exhaustion.** KV admission that eviction cannot free refuses as capacity exhaustion — *before* advancement consumes anything, never admitted-then-tripped mid-run (`GenerationScheduler.cs`; scheduling.md owns the refusal surface).

FCFS is what makes these deterministic: because earlier claims are never displaced, a refused admission describes exactly the room already held. Nothing in this build preempts a running sequence to reclaim bytes; hot KV ceiling edits trim *free* capacity and force prefix eviction first (active pipelines keep their blocks — kv-cache.md).

```mermaid
sequenceDiagram
    participant S as Scheduler admission
    participant C as ComputeCapacityLedger
    participant D as DeviceLedger
    participant P as Pool + PrefixCache
    S->>C: seat worker claim on chosen nodes
    Note over C: oversubscribed remainder clamps to a partial grant,<br/>never refuses, never displaces an earlier seat
    S->>D: seat tier/resource claims in registration order
    alt room exists at the position
        D-->>S: claim seated — FCFS holds it for this sequence's lifetime
    else declared chain cap cannot fit
        D-->>S: BudgetExhaustedException naming model, resource,<br/>exhausted tier, and the ledger breakdown behind it
    else device fair-share room is short but standing grants hold slots
        D-->>S: deferral — candidate keeps its slot and rides a later boundary<br/>(DeviceCoScheduler posture, not an admission refusal)
    end
    S->>P: KV reservation sized for prompt + generation ceiling
    P-->>S: evict cached prefixes first; if still short,<br/>capacity-exhaustion refusal before advancement consumes anything
```

## Weight arenas and mmap placement

Default weight residency is **the model file itself, memory-mapped read-only**: `GgufMmapWeightSource` (`src/Lair.Loader/GgufMmapWeightSource.cs`) opens one whole-file mapping for a single-file GGUF (or index-aligned per-shard mappings for split models) and resolves every logical tensor as a span over the mapped bytes — no arena copy of weights happens in this posture, so DRAM occupancy follows pages actually touched rather than file size claimed upfront.

Views layered on that backing stay copies-free: `ExpertSlabWeightSource` exposes MoE tensors as per-expert slabs at their contiguous byte ranges for router-pointer gathers while "the backing store stays exactly what it was — the mmap arena today, an arena-copied source, or a residency tier later" (its own documentation; no expert tensor is ever copied there). The CUDA `DeviceCopy` posture caches a one-time weight upload per tensor instead of mapping, and its accounting deliberately excludes those uploads from compute-call transfer tallies while a mapped posture adds no persistent device bytes at all (cuda-backend.md owns the kernel side).

PLE rows map exactly the shard that owns them (`PleRowSource`, `src/Lair.Loader/PleRowSource.cs`): one whole-physical-shard mapping, touched row pages only, with prefetch hints naming ranges inside the view's own extent. Hot-row residency (`PleHotRowProvider`) admits copies parity-checked against their source spans before counting as resident; on Linux the hot arena is an anonymous `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` range madvise'd `MADV_HUGEPAGE`, and on every other host it is plain committed aligned memory — **the requested backing reports as not applied rather than emulated**. Hits/misses are tallied so measured hit rates describe what actually ran.

PLE residency ceilings gate whole rows, never guessed bytes: `PleResidencyGate` (`src/Lair.Models/PleResidencyGate.cs`) divides declared `ple.caps.dram_mb` against each table's own row stride (same ceiling division the provider applies at admission — "admission and budget gating can never disagree about how many rows fit"), and a selection the clamp cuts is reported as refused rather than silently narrowed.

## Prefill-buffer clamps: ceilings that narrow work, not refuse it

A declared `prefill_buffers` chain sizes prompt windows instead of claiming standing bytes (`src/Lair.Models/PrefillBufferClamp.cs`). At engine open one sub-batch row's activation cost is measured against every copy-holding position in the chain; **the binding tier allowing the fewest rows wins**, producing a `PrefillRowWidthClamp` (tier, cap, requested width beside allowed width) resolved once like every other forward-shape value. No arithmetic in any forward reads it — prompt routes split at the clamped effective width (`_effectivePrefillRowWidth`, `TransformerModel.cs`). A clamp that allows zero rows still reports with `AllowsPositiveWidth` false: "the posture reads off diagnostics rather than vanishing into a silently narrowed window", and the scheduler exposes it as internal diagnostic ("reports why rows narrowed and never moves a token" — `GenerationScheduler.ResolvedPrefillRowClamp`). With no declared cap nothing binds, which is today's default behavior.

## Host posture detection and declared overrides

`HostTopology.cs` (`src/Lair.Models/`) probes what the machine reports — nodes with per-node byte/core totals (Windows path `ProbeWindowsNodes`; Linux interop via `src/Lair.Loader/LinuxPlacementInterop.cs`). The honesty rule is explicit: **a capacity a host does not report reads as absent/null, never zero**, so absence and emptiness stay distinguishable.

Operators may declare over it (`host.dram_mb`, `host.vram_mb`, `host.cpu_cores`, `host.disk_path`, plus `host.numa.nodes` per-node overrides — the loader's `HostPostureKeys`). Each must state a positive number where it states anything; `disk_path` declares only *that this host reads from disk* ("no ceiling belongs at that position, so nothing here reads a number beside it"). A declared total disagreeing with probed topology is **accepted with a warning naming both**: `host_capacity model='…' … — accepted against reported topology` (`GenerationService.ReportDeclaredCapacity`). Ledger seats follow the *effective* posture after acceptance. The monitor publishes the resolved reading as `compute_topology model=…` rows (monitoring-and-metrics.md owns the stream).

NUMA placement is a preference, never a promise: `models[].placement.numa_spread` accepts `auto`, `local`, or `interleave`; what lands on this host's node layout is decided at open per OS path (`NumaSpreadPreference` in the loader/placement seams), and no refusal claims cross-node movement that the OS declines to give.

### CPU worker placement: physical cores, auto sizing, and Linux affinity

An unpinned model's compute threads land wherever the OS puts them — for DRAM-bound weight streaming that drift is measurable cost, not noise. On the lab Xeon lane (dual E5-2683 v4, AVX2+FMA only, governor performance), tasksetting a prefill process to node0 *physical* cores lifted implied Q4_K weight bandwidth from 0.415 GB/s (two unpinned runs, spread ~7%) to 0.54 GB/s and end-to-end prefill throughput by roughly +7% at gemv/length-128 through +14% at tiled/length-256; pinning the same process across node0's HT siblings instead *dropped* implied bandwidth to 0.36 GB/s — below unpinned. Those arms, their posture table, and the raw logs live in `reports/2026-11-27T14-14-58Z-optimize-q4k-widebatch-xeon-pinning` (Lair.Bench prefill-profile + q4k-widebatch, Qwen2.5-3B Instruct Q4_K_M, threads=8). The same report's threads=32 auto arms carry a recorded 84% run-to-run dispersion signature consistent with pool/dispatch overhead rather than kernel limits — placement and worker-count posture are the live levers there, not ISA paths.

What this build does today follows from that measurement:

- **Auto worker sizing** (`CpuBackend.AutoWorkerCount`, `src/Lair.Backends.Cpu/CpuBackend.cs`) sizes an auto open by *detected physical cores*, never by logical processor count, on both Windows (the topology API) and Linux. On Linux the detection is sysfs-derived: distinct SMT sibling groups read from per-CPU `thread_siblings_list` files (`WorkerPlacement.BuildCores`, `src/Lair.Backends.Cpu/WorkerPlacement.cs`), restricted to the processor set sched_getaffinity reports for this process — a container confined to part of the machine detects only its own physical cores. On the lab host that resolves an auto open to 32 workers (its distinct-core capacity) instead of the 64 logical processors `Environment.ProcessorCount` would report. An explicit nonzero `engine.threads` still wins over auto-sizing, and a per-model `thread_quota` caps either choice; placement follows the chosen count and never changes it.
- **Worker-thread affinity application** now exists on Windows (`SetThreadAffinityMask`) *and* Linux (libc `sched_setaffinity`, pid 0 — the calling thread exactly as sched_setaffinity(2) documents), all through one seam: `src/Lair.Backends.Cpu/ThreadAffinity.cs`. Explicitly configured core sets keep their documented contract — ids inside `[0, 63]`, refused at open outside it (`BuildCoreMask`); on any other platform a configured core set remains the documented once-warned no-op. Multi-word processor masks (word *w* covering ids `[w·64, w·64+63)`) carry auto placement past one group; Windows refuses anything wider rather than pinning through the wrong group, and Linux hosts with more than 1024 named processors report unappliable masks instead of truncating.
- **Auto placement of unpinned models on Linux** (`WorkerPlacement.Select`, applied in the `CpuBackend` open): candidates are confined to the allowed processor mask first — a restricted cpuset can only narrow placement, never widen it; cores with no member inside the mask drop out and cores whose primary is outside participate through their lowest allowed sibling. Candidates then sort node-local (lowest NUMA index, then package id, then primary id), so a quota that fits one socket never spans another; each worker gets one distinct physical core while capacity lasts; only when quota exceeds the number of usable distinct cores does placement borrow HT siblings, round-robin across already-chosen cores rather than stacking a pair. On this host with an 8-worker auto posture, selection resolves primaries like `0-7` — exactly the measured-winner cpu list shape. The derived mask covers the engine's uncapped thread choice for the model's whole lifetime; live quota edits change participation counts, not placement (scheduling.md).
- **Fallback semantics are silent by design.** Unreadable sysfs (`DetectLinuxPosture()` returns null), a topology that cannot express the requested quota with zero usable cores, or an all-zero affinity read leaves today's unpinned posture standing — no new failure mode, no partial pin invented. A placement shorter than quota (a cpuset holding four primaries but none of their siblings) pins those four and lets the remaining workers run unpinned-by-necessity rather than escaping the mask. The internal `autoPlaceWorkers` open parameter exists so measurement/test opens can disable derivation; it is not a host knob and changes no worker count.

```mermaid
flowchart TD
    Q[open with threads=0 auto or explicit quota] --> S{AutoWorkerCount:<br/>detected physical cores?}
    S -->|Windows API or Linux sysfs groups<br/>&#47; affinity-mask restricted| A[sized by distinct cores,<br/>capped by logical count]
    S -->|detection unavailable| B[fallback: Environment.ProcessorCount]
    Q2[core set configured?] -->|yes, ids in 0..63| P1[pin workers to it:<br/>Windows SetThreadAffinityMask<br/>Linux sched_setaffinity]
    Q2 -->|no + Linux host| D{sysfs posture readable?}
    D -->|null: unreadable sysfs| U[unpinned posture, silently]
    D -->|readable| E[confine cores to allowed mask<br/>never widen a cpuset]
    E --> F[node-local order: node index,<br/>package id, primary id]
    F --> G{quota ≤ usable distinct cores?}
    G -->|yes| H[pin primaries only:<br/>one core per worker]
    G -->|no| I[borrow HT siblings round-robin<br/>across already-chosen cores]
    I --> J{selection empty?}
    J -->|empty: no usable core in mask| U
```

Measured posture behind these choices (from the report directory cited above; gemv/tiled e2e prefill arms, threads=8, lengths 128/256):

| arm | cpu list applied | implied weight GB/s (gemv) | tok/s medians (len 128 / 256) |
|---|---|---:|---|
| unpinned | none — OS placement | 0.415 (runs 0.43, 0.40) | gemv 13.42 / 11.58 · tiled 11.59 / 10.53 |
| physical-pinned | node0 primaries `0-15` | 0.54 | gemv 14.37 / 10.99 · tiled 12.73 / 12.01 |
| HT-pinned | node0 with siblings `0-15,32-47` | 0.36 | gemv 12.13 / 10.93 · tiled 11.46 / 10.69 |

The engine documents what the build does: physical-preference placement is landed for Linux opens; node-derived full-cpulist masks remain a Windows-only derivation (`CpuBackend.SupportsCorePinning`) precisely because HT-wide masks measured worse than distinct-physical-core selection on this host, and nothing in this build derives such a mask on Linux. No public knob selects or disables auto placement — `engine.threads`, `placement.core_set`, and `placement.thread_quota` behave exactly as documented in the user reference; there is no YAML key for core lists wider than group zero, no per-model placement override beyond what already exists, and macOS/other platforms keep their once-warned no-op rather than a faked affinity.

What is *not* yet claimed: the landed derivation has not separated from noise as an end-to-end win on its own. An alternating-build A/B (`scripts/run-xeon-alternating-build-ab.py`, `reports/2026-11-27T23-07-15Z-optimize-affinity-selfplacement-xeon-t8`: same host/model, explicit threads=8, length 256, kernels gemv/tiled, 3 warmups + 13 interleaved rotated rounds) compares the pre-placement published build against a clean worktree publish of this placement commit — totals move `gemv +3.07%` and `tiled −2.07%`, every metric inside its own ±12–29% band, nothing separating. Worker-thread affinity alone therefore does not reproduce the whole-process `taskset -c 0-15` posture above: that manual arm pinned the entire process — including model load/first-touch of weight pages and every engine thread — while this build places worker threads at open and leaves load-time residency where the loader happens to run. The placement machinery is landed and documented as implemented behavior; calling it a throughput improvement waits for a whole-process posture test (loader/main-thread participation in the placed core set, or manual taskset against both builds) that puts a window total outside its own band.

## Knobs this page owns

| Key | Live class today | What it decides | Bounds/refusals |
|---|---|---|---|
| `models[].budgets.<resource>.order` (+ whole budgets block) | RestartOnly (structural list — hot-compare has no canonical scalar; semantic edits surface via full diff, `ConfigReload.cs`) | chain tiers and priority for that resource | config-load structural refusals above; posture refusal at open |
| `models[].budgets.kv.caps.dram_blocks` | **Live** when the model's resolved chain carries a system-memory KV tier (maps to canonical block count — `GenerationService.cs`) | standing KV reservation/cap in blocks | floor 8 like legacy `kv_blocks`; hot edit trims free side and can force prefix eviction; refused live if no such tier resolves |
| `models[].budgets.kv.caps.vram_blocks` | RestartOnly (explicit, with residency keys) | finite device-resident KV ceiling | required by any non-`dram` `kv_residency`; refusal wording quoted below |
| `models[].budgets.weights.caps.*`, `.ple…`, `.prefill_buffers…` caps | **Refused mid-run** — RestartOnly (`GenerationService.cs`) | declared ceilings for weight claims, PLE rows, prefill clamps | positive-only at load; enforcement as described per section above |
| `models[].placement.kv_residency` + `budgets.kv.caps.vram_blocks` + `host.vram_mb` | RestartOnly | which block ranges seat device-side | non-`dram` without finite ceiling refuses; quantized KV beside non-host residency refuses; no usable CUDA behind the storage seam refuses (also cuda-backend.md) |
| `models[].placement.core_set`, `.thread_quota`, `.core_overcommit` | per reload table (scheduling.md owns live classes) | core-grant shape in the compute ledger | oversubscription clamps to remainder, not refusal; overlap only where opted in and recorded |
| `host.*` posture keys + `host.numa.nodes` | RestartOnly | effective capacity totals ledgers seat against | positive-number validation at load; declared-vs-probed disagreement accepted with warning line |

## Refusals today (concrete wording families)

- Chain shape: unknown tier, repeated tier, out-of-priority order (`dram → vram → disk` is the only legal direction), a cap on a tier its chain's `order` does not name, non-positive caps ("ceilings must be 1 or greater — omit the cap to declare no ceiling there"), and any ceiling on a read-side origin position ("that storage already holds the complete set").
- Working-copy claim at an unopened residency: weights/PLE in VRAM name the compute-backend capability; KV in VRAM names the device-backed pool posture; write-side disk claims state "write-side bytes reach storage only where something writes them" — declared vocabulary, never a silent drop to DRAM.
- Device-backed KV residency without a finite ceiling: `device-backed KV blocks are admitted only against a finite ceiling… Declare 'models[].budgets.kv.caps.vram_blocks' … or resolve the residency to 'dram', which needs no ceiling because it opens no device pool` (`src/Lair.Http/GenerationService.cs`).
- Legacy-vs-canonical disagreement: `kv_blocks` and `budgets.kv.caps.dram_blocks` "spell the same system-memory KV ceiling, so they must state the same number" (loader refusal); a legacy key alone still works but logs the deprecated-key notice pointing at budgets.
- Mid-run edits to weight/PLE/prefill-buffer caps: refused rather than half-applied; only the canonical KV DRAM ceiling hot-edits when its tier resolves.

Cross-links: block pools, residency modes as they affect cache reads, and quantized-pair refusals in [kv-cache.md](kv-cache.md); admission ordering, deferral vs refusal, and reload classes in [scheduling.md](scheduling.md); result-side budget reporting in [request-lifecycle.md](request-lifecycle.md); opened-tier accounting and monitor publishing of device posture in [cuda-backend.md](cuda-backend.md) and [monitoring-and-metrics.md](monitoring-and-metrics.md); row-width machinery these clamps bind against in [cpu-kernels.md](cpu-kernels.md); profile-driven window values feeding KV windows in [model-profiles.md](model-profiles.md).
