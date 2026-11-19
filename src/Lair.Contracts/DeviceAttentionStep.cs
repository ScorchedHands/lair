namespace Lair.Contracts;

/// <summary>
/// The composed attention window a device-resident prefill posture hands one layer of one prompt sub-batch to: the
/// staged query rows and new key/value rows cross once, the causal <c>softmax(QK^T)V</c> walk runs against storage this
/// step owns — its KV write included — and only the attention result crosses back. Scores, softmax state, and every K/V
/// read of the walk stay on the far side of the seam. Rows arrive with the engine's own rotation already applied: the
/// composition carries the attention family (the KV write and the causal walk), not the rotary pass, so a host that
/// rotates at its own resolved tables, bases, and pairings hands those bytes here unchanged.
/// <para>
/// The geometry facts answer what one opened step can compose for which model shape: positions and layers its storage
/// holds, the head layout it walks, and the widest window its staging buffers hold. A posture that composes a whole
/// engine's prompt windows against one such step checks those facts against the plan at open and refuses naming both
/// sides on any mismatch rather than running some other composition behind it.
/// </para>
/// <para>
/// Ownership follows the alternate-weight-source rule: whoever opens a step owns its lifetime, and an engine that runs
/// windows through one never disposes it — the caller's teardown releases it after the last forward over it ends.
/// </para>
/// </summary>
public interface IDeviceAttentionStep : IDisposable
{
    /// <summary>Positions this step's KV storage holds; a composed window may only read and write positions below this.</summary>
    int MaxPositions { get; }

    /// <summary>Layers this step's KV storage holds a key/value region for.</summary>
    int LayerCount { get; }

    /// <summary>Query heads one row of the model layout carries, as the composing engine sees them — whatever internal head staging the walk needs to cover them.</summary>
    int HeadCount { get; }

    /// <summary>Key/value heads one row of the model layout carries.</summary>
    int KvHeadCount { get; }

    /// <summary>Dimensions of one head region.</summary>
    int HeadDim { get; }

    /// <summary>The widest sub-batch this step stages at once — rows, in the composing engine's row width.</summary>
    int MaxWindowRows { get; }

    /// <summary>Whether this step's KV storage is a pool's resident device store rather than a private arena it allocated itself.</summary>
    bool KvStoreBacked { get; }

    /// <summary>
    /// Composes one layer of one window: stage the rows, write their key/value bytes into this step's storage at these
    /// positions, and walk causal attention over that storage for every row — each row attending exactly the positions
    /// at or before its own under the same positive-window narrowing a host route applies. The rows must already carry
    /// the engine's rotation; <paramref name="output"/> receives them in the model layout: one row per staged row,
    /// <c>HeadCount * HeadDim</c> elements each.
    /// </summary>
    void RunAttentionWindow(
        int layer,
        int positionBase,
        ReadOnlySpan<float> qRows,
        ReadOnlySpan<float> kRows,
        ReadOnlySpan<float> vRows,
        float attnScale,
        float softcap,
        int window,
        Span<float> output);
}
