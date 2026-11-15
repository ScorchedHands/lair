namespace Lair.Models;

/// <summary>
/// The clamp one model's resolved prefill-buffer ceilings apply to the engine's row window: the binding ceiling, the activation bytes every
/// sub-batch row carries on this engine's prompt route, the width the plan requested, and the widest row count those two allow. A position with
/// no declared ceiling bounds nothing; a chain whose positions all fit the requested width produces no clamp at all — today's shape unchanged.
/// The binding position is the one allowing the fewest rows: every copy-holding position its own declaration holds decides for itself, exactly as
/// claims walk their chain. A binding that allows fewer than one row states what no window can run — the engine still opens on a single-row
/// window and reports this unfit clamp so capacity accounting names why rather than running over the declared ceiling quietly.
/// </summary>
public sealed record PrefillRowWidthClamp
{
    /// <summary>The tier whose declared ceiling binds the row window.</summary>
    public required BudgetTier Tier { get; init; }

    /// <summary>The binding ceiling, in this resource's megabyte unit.</summary>
    public required long CapMegabytes { get; init; }

    /// <summary>Total mutable activation bytes one sub-batch row carries on the engine's prompt route.</summary>
    public required long RowActivationBytes { get; init; }

    /// <summary>The row width the plan resolved before this clamp — requested, distinct from allowed.</summary>
    public required int RequestedRowWidth { get; init; }

    /// <summary>The widest sub-batch whose activation bytes fit beside no other claim against the binding ceiling. May state zero when even one row's bytes exceed it.</summary>
    public required int AllowedRowWidth { get; init; }

    /// <summary>Whether this clamp allows a positive row window at all.</summary>
    public bool AllowsPositiveWidth => AllowedRowWidth >= 1;

    /// <summary>The clamp as the text a diagnostic reports: binding ceiling, per-row bytes, and requested against allowed in rows.</summary>
    public string Describe() =>
        $"'{BudgetTierNames.Of(Tier)}' caps prefill working buffers at {CapMegabytes} megabytes — " +
        $"a sub-batch row costs {RowActivationBytes} activation bytes there, so the window narrows from {RequestedRowWidth} rows to {AllowedRowWidth}";
}

/// <summary>
/// Computes the clamp a resolved prefill-buffer chain applies to an engine's row window. The calculation is deliberately free of ledger
/// occupancy and pool totals — it answers what this model's own declared ceilings allow against its per-row cost, which is half of what lands
/// on the opened engine; what currently available capacity leaves beside other models' seated claims is where admission observations decide,
/// in each resource's native accounting. A chain with no capped copy-holding position clamps nothing and produces null: today's shape exactly.
/// </summary>
public static class PrefillBufferClamp
{
    /// <summary>The megabyte unit prefill-buffer ceilings count rows against.</summary>
    public const long Megabytes = 1024L * 1024;

    /// <summary>How many whole sub-batch rows of this per-row cost a ceiling stated in megabytes allows.</summary>
    public static int FitsRows(long capMegabytes, long rowActivationBytes) => (int)Math.Min(capMegabytes * Megabytes / Math.Max(1L, rowActivationBytes), int.MaxValue);

    /// <summary>
    /// The clamp this chain's declared ceilings apply to a requested row width: the copy-holding position allowing the fewest rows binds, and no
    /// clamp is produced when nothing caps anything or what does cap already fits the request. A binding that allows zero rows still reports the
    /// clamp — with <see cref="PrefillRowWidthClamp.AllowsPositiveWidth"/> false — so the posture reads off diagnostics rather than vanishing into
    /// a silently narrowed window.
    /// </summary>
    public static PrefillRowWidthClamp? Compute(ResolvedBudgetChain chain, long rowActivationBytes, int requestedRowWidth)
    {
        ArgumentNullException.ThrowIfNull(chain);

        PrefillRowWidthClamp? binding = null;
        foreach (ResolvedBudgetTier position in chain.Tiers)
        {
            if (position.IsOriginStore || position.Cap is null)
            {
                continue; // a copy-holding position with no declared ceiling states "no model-level limit here", and an origin store declares none of ours
            }

            int allowed = FitsRows(position.Cap.Value, rowActivationBytes);
            if (binding is null || allowed < binding.AllowedRowWidth)
            {
                binding = new PrefillRowWidthClamp
                {
                    Tier = position.Tier,
                    CapMegabytes = position.Cap.Value,
                    RowActivationBytes = rowActivationBytes,
                    RequestedRowWidth = requestedRowWidth,
                    AllowedRowWidth = allowed,
                };
            }
        }

        return binding is null || binding.AllowedRowWidth >= requestedRowWidth ? null : binding;
    }
}
