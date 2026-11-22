namespace Lair.Models;

/// <summary>
/// The clamp one model's resolved PLE ceilings apply to gathered-row residency: the binding ceiling, the packed bytes every admitted row
/// costs on this table's quant format, the row count a gather asked for, and how many rows those two allow side by side. A position with no
/// declared ceiling bounds nothing; a chain whose capped positions all fit the request produces no clamp at all — today's shape unchanged. The
/// binding position is the one allowing the fewest rows: every working-copy tier decides for itself from its own declaration, exactly as claims
/// walk their chain, and the table's origin-store position (the file holding the whole table) bounds nothing of ours. A binding that allows fewer
/// than one row states what no gather can admit — residency still reports this unfit clamp so capacity accounting names why rather than reading
/// past a declared ceiling quietly.
/// </summary>
public sealed record PleRowResidencyClamp
{
    /// <summary>The tier whose declared ceiling binds gathered-row residency.</summary>
    public required BudgetTier Tier { get; init; }

    /// <summary>The binding ceiling, in megabytes of resident PLE rows.</summary>
    public required long CapMegabytes { get; init; }

    /// <summary>Packed bytes one admitted row costs on this table's quant format.</summary>
    public required long RowBytes { get; init; }

    /// <summary>The row count the gather asked for before this clamp — requested, distinct from allowed.</summary>
    public required long RequestedRows { get; init; }

    /// <summary>How many resident PLE rows fit beside no other claim against the binding ceiling. May state zero when even one row's bytes exceed it.</summary>
    public required long AllowedRows { get; init; }

    /// <summary>Whether this clamp admits a positive number of resident rows at all.</summary>
    public bool AllowsPositiveRows => AllowedRows >= 1;

    /// <summary>The clamp as the text a diagnostic reports: binding ceiling, per-row packed bytes, and requested against allowed in rows.</summary>
    public string Describe() =>
        $"'{BudgetTierNames.Of(Tier)}' caps {BudgetResourceNames.Ple} residency at {CapMegabytes} megabytes — " +
        $"an admitted row costs {RowBytes} packed bytes there, so resident rows narrow from {RequestedRows} to {AllowedRows}";
}

/// <summary>
/// Computes the clamp a resolved PLE chain applies to how many gathered rows residency admits. The calculation is deliberately free of ledger
/// occupancy and pool totals — it answers what this model's own declared ceilings allow against its per-row packed cost, which is half of what lands
/// on an opened gather; what currently available capacity leaves beside other models' seated claims is where admission observations decide, in each
/// resource's native accounting. A chain with no capped working-copy position clamps nothing and produces null: today's shape exactly. The PLE row
/// cost is a packed row of the table's own quant format — variable-length encodings have no per-row byte cost to bound residency against, so this
/// seam takes whole rows or refuses rather than guessing at bytes it cannot count.
/// </summary>
public static class PleResidencyGate
{
    /// <summary>The megabyte unit PLE ceilings count resident rows against.</summary>
    public const long Megabytes = 1024L * 1024;

    /// <summary>How many whole packed rows of this per-row cost a ceiling stated in megabytes allows, saturating at the widest row count residency can express.</summary>
    public static long FitsRows(long capMegabytes, long rowBytes)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capMegabytes, 0L);

        if (rowBytes <= 0)
        {
            throw new ArgumentException(
                $"a PLE gather admits whole packed rows, so a per-row cost of {rowBytes} bytes bounds nothing", nameof(rowBytes));
        }

        // A ceiling large enough to overflow the byte product states "no row count this seam can express is excluded by it": saturate rather than wrap.
        long capBytes = capMegabytes > long.MaxValue / Megabytes ? long.MaxValue : checked(capMegabytes * Megabytes);
        return Math.Min(capBytes / rowBytes, long.MaxValue);
    }

    /// <summary>
    /// The clamp this chain's declared ceilings apply to a requested resident-row count: the working-copy position allowing the fewest rows binds, and no
    /// clamp is produced when nothing caps anything or what does cap already fits the request. A binding that allows zero rows still reports the clamp — with
    /// <see cref="PleRowResidencyClamp.AllowsPositiveRows"/> false — so the posture reads off diagnostics rather than vanishing into a silently narrowed gather.
    /// </summary>
    public static PleRowResidencyClamp? Compute(ResolvedBudgetChain chain, long rowBytes, long requestedRows)
    {
        ArgumentNullException.ThrowIfNull(chain);

        if (chain.Resource != BudgetResource.Ple)
        {
            throw new ArgumentException(
                $"'{BudgetResourceNames.Of(chain.Resource)}' is not '{BudgetResourceNames.Ple}': PLE residency clamps only the chain that bounds gathered table rows", nameof(chain));
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(requestedRows, 0L);

        PleRowResidencyClamp? binding = null;
        foreach (ResolvedBudgetTier position in chain.Tiers)
        {
            if (position.IsOriginStore || position.Cap is null)
            {
                continue; // the file holding the whole table bounds nothing of ours, and a working-copy tier with no ceiling states "no model-level limit here"
            }

            long allowed = FitsRows(position.Cap.Value, rowBytes);
            if (binding is null || allowed < binding.AllowedRows)
            {
                binding = new PleRowResidencyClamp
                {
                    Tier = position.Tier,
                    CapMegabytes = position.Cap.Value,
                    RowBytes = rowBytes,
                    RequestedRows = requestedRows,
                    AllowedRows = allowed,
                };
            }
        }

        return binding is null || binding.AllowedRows >= requestedRows ? null : binding;
    }
}
