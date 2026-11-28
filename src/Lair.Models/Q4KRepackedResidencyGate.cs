using Lair.Backends.Cpu;

namespace Lair.Models;

/// <summary>
/// The clamp one model's resolved weights ceilings apply to a resident repacked copy of its dense Q4_K rows: the binding ceiling,
/// the packed row cost the file already holds, the repacked row cost such a copy adds beside it, the row count asked for, and how
/// many whole rows those two allow side by side. A position with no declared ceiling bounds nothing; a chain whose capped positions
/// all fit the request produces no clamp at all — today's shape unchanged. The binding position is the one allowing the fewest rows,
/// exactly as every other residency gate decides per its own declaration, and the weights chain's origin-store position (the file
/// holding the complete packed set) bounds nothing of ours: a repacked copy lives beside that store in system memory, so only
/// copy-holding positions cap it. A binding that allows fewer than one row states what no residency can admit — the gate still
/// reports this unfit clamp rather than letting a caller read past a declared ceiling quietly.
/// </summary>
public sealed record Q4KRepackedRowClamp
{
    /// <summary>The tier whose declared ceiling binds repacked weight rows.</summary>
    public required BudgetTier Tier { get; init; }

    /// <summary>The binding ceiling, in megabytes of resident weights at that position.</summary>
    public required long CapMegabytes { get; init; }

    /// <summary>Packed bytes one dense Q4_K row already costs the model file's own encoding.</summary>
    public required long PackedRowBytes { get; init; }

    /// <summary>Repacked bytes one resident copy of that row adds beside it (<see cref="Q4KResidencyLayout.RepackedBlockBytes"/> per super-block).</summary>
    public required long RepackedRowBytes { get; init; }

    /// <summary>The row count the residency asked to hold before this clamp — requested, distinct from allowed.</summary>
    public required long RequestedRows { get; init; }

    /// <summary>How many whole repacked rows fit beside no other claim against the binding ceiling. May state zero when even one row's bytes exceed it.</summary>
    public required long AllowedRows { get; init; }

    /// <summary>Whether this clamp admits a positive number of resident repacked rows at all.</summary>
    public bool AllowsPositiveRows => AllowedRows >= 1;

    /// <summary>The clamp as the text a diagnostic reports: binding ceiling, both per-row costs in bytes, and requested against allowed in whole rows.</summary>
    public string Describe() =>
        $"'{BudgetTierNames.Of(Tier)}' caps {BudgetResourceNames.Weights} residency at {CapMegabytes} megabytes — " +
        $"a dense Q4_K row costs {PackedRowBytes} packed bytes and a repacked copy adds {RepackedRowBytes} resident bytes there, so " +
        $"repacked rows narrow from {RequestedRows} to {AllowedRows}";
}

/// <summary>
/// Computes the clamp a resolved weights chain applies to holding repacked copies of dense Q4_K weight rows. The arithmetic is the
/// same ceiling division <see cref="PleResidencyGate"/> applies to declared budgets and the same row-count shape residency reports, so
/// budget gating and any admission beside it can never disagree about how many rows fit — this gate answers what the model's own
/// ceilings allow against a repacked row's byte cost, while what currently available capacity leaves beside other models' seated claims
/// stays where ledger observations decide. A chain with no capped working-copy position clamps nothing and produces null: today's shape
/// exactly. Like every residency gate here it bounds whole rows or refuses — Q4_K is a fixed-stride format, so its per-row packed cost
/// (<see cref="Q4KResidencyLayout.PackedRowBytes"/> for the column count) and repacked copy cost
/// (<see cref="Q4KResidencyLayout.RepackedRowBytes"/>) are exact numbers this build can name rather than estimates. A chain that names a
/// resource other than weights refuses naming the accepted vocabulary: repacked weight copies belong to exactly one budgetable resource,
/// and borrowing another resource's ceiling would be parallel accounting under a different name.
/// </summary>
public static class Q4KRepackedResidencyGate
{
    /// <summary>
    /// The clamp this chain's declared ceilings apply to a requested resident-repacked-row count: the working-copy position allowing the
    /// fewest rows binds, and no clamp is produced when nothing caps anything or what does cap already fits the request. A binding that
    /// allows zero rows still reports the clamp — with <see cref="Q4KRepackedRowClamp.AllowsPositiveRows"/> false — so the posture reads off
    /// diagnostics rather than vanishing into a silently narrowed residency.
    /// </summary>
    public static Q4KRepackedRowClamp? Compute(ResolvedBudgetChain chain, int cols, long requestedRows)
    {
        ArgumentNullException.ThrowIfNull(chain);

        if (chain.Resource != BudgetResource.Weights)
        {
            throw new ModelException(
                $"'{BudgetResourceNames.Of(chain.Resource)}' is not '{BudgetResourceNames.Weights}': repacked Q4_K weight copies bound against the " +
                $"'{BudgetResourceNames.Weights}' chain only ({string.Join(", ", BudgetResourceNames.DeclaredKeys)})");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(requestedRows, 0L);

        long packedRowBytes = Q4KResidencyLayout.PackedRowBytes(cols);
        long repackedRowBytes = Q4KResidencyLayout.RepackedRowBytes(cols);

        Q4KRepackedRowClamp? binding = null;
        foreach (ResolvedBudgetTier position in chain.Tiers)
        {
            if (position.IsOriginStore || position.Cap is null)
            {
                continue; // the file holding the packed set bounds nothing of ours, and a working-copy tier with no ceiling states "no model-level limit here"
            }

            long allowed = PleResidencyGate.FitsRows(position.Cap.Value, repackedRowBytes);
            if (binding is null || allowed < binding.AllowedRows)
            {
                binding = new Q4KRepackedRowClamp
                {
                    Tier = position.Tier,
                    CapMegabytes = position.Cap.Value,
                    PackedRowBytes = packedRowBytes,
                    RepackedRowBytes = repackedRowBytes,
                    RequestedRows = requestedRows,
                    AllowedRows = allowed,
                };
            }
        }

        return binding is null || binding.AllowedRows >= requestedRows ? null : binding;
    }
}
