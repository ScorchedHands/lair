using Lair.Http;

namespace Lair.Tests;

/// <summary>
/// Reasoning-region recognition and the visibility policy that acts on it. Recognition keys on the markup a generation
/// contains — an region opens at <c>&lt;think&gt;</c> and closes at the first <c>&lt;/think&gt;</c> after it, regions do not
/// nest, a close with no open is ordinary text, and an open that never closes runs to the end of what was generated. The
/// recognizer works over fully decoded text and over the same generation as it streams piece by piece: a boundary marker
/// split across pieces stays pending until the next piece settles it, so region content reaches neither the wire nor a
/// false boundary. A keep policy never touches visible content; a strip policy removes recognized spans while recording them.
/// </summary>
[Trait("Category", "ToolCalls")]
public sealed class ReasoningRegionTests
{
    private const string Open = "<think>";
    private const string Close = "</think>";

    // ---------------------------------------------------------------- recognition over fully decoded text

    [Fact]
    public void Recognition_RecognizedRegionSpansAreHalfOpenAndCoverTheMarkup()
    {
        string text = $"Answer {Open}why two plus two{Close} done";

        IReadOnlyList<ReasoningRegion> regions = ReasoningRecognition.FindRegions(text);

        ReasoningRegion region = Assert.Single(regions);
        Assert.Equal(text.IndexOf(Open, StringComparison.Ordinal), region.Start);
        Assert.Equal(text.IndexOf(Close, StringComparison.Ordinal) + Close.Length, region.End);
        Assert.Equal($"{Open}why two plus two{Close}", text[region.Start..region.End]);
    }

    [Fact]
    public void Recognition_MultipleRegions_ReportInDocumentOrder()
    {
        string text = $"a{Open}one{Close} b{Open}two{Close} c";

        IReadOnlyList<ReasoningRegion> regions = ReasoningRecognition.FindRegions(text);

        Assert.Equal(2, regions.Count);
        Assert.True(regions[0].Start < regions[1].Start);
        foreach (ReasoningRegion region in regions)
        {
            Assert.StartsWith(Open, text[region.Start..region.End]);
            Assert.EndsWith(Close, text[region.Start..region.End]);
        }
    }

    [Fact]
    public void Recognition_MarkupInsideARegionIsContent_NotANestedOpen()
    {
        string text = $"{Open}outer {Open} inner{Close}";

        IReadOnlyList<ReasoningRegion> regions = ReasoningRecognition.FindRegions(text);

        Assert.Single(regions);
        Assert.Equal(0, regions[0].Start);
        Assert.Equal(text.Length, regions[0].End); // the first close closes the region; markup inside it is its content
    }

    [Fact]
    public void Recognition_CloseWithoutOpen_IsOrdinaryText_AndNothingIsRecognized()
    {
        string text = $"no reasoning here {Close} trailing";

        Assert.Empty(ReasoningRecognition.FindRegions(text));
        Assert.Equal(text, ReasoningRecognition.Strip(text).VisibleText);
        Assert.Empty(ReasoningRecognition.Strip(text).Regions);
    }

    [Fact]
    public void Recognition_OpenThatNeverCloses_RunsToEndOfGeneration()
    {
        string text = $"answer {Open}still reasoning when the budget stops";

        IReadOnlyList<ReasoningRegion> regions = ReasoningRecognition.FindRegions(text);

        ReasoningRegion region = Assert.Single(regions);
        Assert.Equal(text.IndexOf(Open, StringComparison.Ordinal), region.Start);
        Assert.Equal(text.Length, region.End);
    }

    // ---------------------------------------------------------------- strip policy over fully decoded text

    [Fact]
    public void Strip_RemovesEachRecognizedSpanAndRecordsItsBoundaries()
    {
        string text = $"a{Open}hidden a{Close} b{Open}hidden b{Close} c";

        ReasoningDetection detection = ReasoningRecognition.Strip(text);

        Assert.Equal(2, detection.Regions.Count);
        Assert.Equal("a b c", detection.VisibleText); // the spans come out verbatim: only recognized markup and content vanishes, surrounding bytes keep their own spacing
    }

    [Fact]
    public void Strip_GenerationWithoutTheMarkup_KeepsContentByteIdenticalAndRecordsNothing()
    {
        string text = "plain output with no reasoning region at all";

        ReasoningDetection detection = ReasoningRecognition.Strip(text);

        Assert.Equal(text, detection.VisibleText);
        Assert.Empty(detection.Regions);
    }

    [Fact]
    public void Strip_UnderKeepPolicy_TheRecognizedBoundariesAreRecordedButContentIsUntouched()
    {
        string text = $"answer {Open}private{Close} visible";

        // The keep path records what recognition found and hands back the generated text as it arrived.
        ReasoningDetection kept = new(ReasoningRecognition.FindRegions(text), text);

        Assert.Single(kept.Regions);
        Assert.Equal(text, kept.VisibleText);
    }

    // ---------------------------------------------------------------- streaming recognition, piece by piece

    [Fact]
    public void Streaming_BoundaryMarkersSplitAcrossPieces_AreSettledByTheNextPieceAndNeverLeak()
    {
        const string whole = "Answer <think>hidden</think>done";
        string[] pieces = ["Answer ", "<th", "ink>hidden<", "/think>", "done"];

        ReasoningRegionFilter filter = new();
        List<string> visible = [];
        foreach (string piece in pieces)
        {
            visible.AddRange(filter.Feed(piece));
        }

        visible.AddRange(filter.Finish());

        Assert.Equal("Answer done", string.Concat(visible)); // region markup and content withheld; surrounding text intact, in order
        ReasoningRegion region = Assert.Single(filter.Regions);
        Assert.Equal(whole.IndexOf("<think>", StringComparison.Ordinal), region.Start);
        Assert.Equal(whole.IndexOf("</think>", StringComparison.Ordinal) + Close.Length, region.End);
    }

    [Fact]
    public void Streaming_RegionOpenedAndClosedAcrossManyPieces_IsRecognizedOnce()
    {
        string[] pieces = ["x", Open[..3], Open[3..], "secret", "<", "/think>", "y"];

        ReasoningRegionFilter filter = new();
        List<string> visible = [];
        foreach (string piece in pieces)
        {
            visible.AddRange(filter.Feed(piece));
        }

        visible.AddRange(filter.Finish());

        Assert.Equal("xy", string.Concat(visible)); // every withheld character settles as region content, never as a boundary
        Assert.Single(filter.Regions);
    }

    [Fact]
    public void Streaming_OpenThatTheStreamNeverCloses_RecordsToTheEndAndWithholdsWhatFollows()
    {
        const string whole = "visible <think>still thinking";

        ReasoningRegionFilter filter = new();
        List<string> visible = [];
        foreach (string piece in new string[] { "visible ", Open, "still thinking" })
        {
            visible.AddRange(filter.Feed(piece));
        }

        visible.AddRange(filter.Finish());

        Assert.Equal("visible ", string.Concat(visible)); // reasoning cut off by the end of generation is still withheld
        ReasoningRegion region = Assert.Single(filter.Regions);
        Assert.Equal(whole.IndexOf(Open, StringComparison.Ordinal), region.Start);
        Assert.Equal(whole.Length, region.End);
    }

    [Fact]
    public void Streaming_GenerationWithoutTheMarkup_ReachesTheWireByteIdenticalInPieceOrder()
    {
        const string whole = "no reasoning in this generation";
        ReasoningRegionFilter filter = new();
        List<string> visible = [];
        foreach (string piece in new string[] { "no reas", "oning in ", "this gen", "eration" })
        {
            visible.AddRange(filter.Feed(piece));
        }

        visible.AddRange(filter.Finish());

        Assert.Equal(whole, string.Concat(visible)); // a pending tail that settles as ordinary text still belongs on the wire
        Assert.Empty(filter.Regions);
    }

    [Fact]
    public void Streaming_CloseMarkupWithNoOpenAhead_StaysOrdinaryTextByteForByte()
    {
        const string whole = $"trailing {Close} text";
        ReasoningRegionFilter filter = new();
        List<string> visible = [];
        foreach (string piece in new string[] { "trailing ", "</th", "ink>", " text" })
        {
            visible.AddRange(filter.Feed(piece));
        }

        visible.AddRange(filter.Finish());

        Assert.Equal(whole, string.Concat(visible));
        Assert.Empty(filter.Regions);
    }

    // ---------------------------------------------------------------- resolved policy per model

    [Fact]
    public void Plan_ReasoningKeyResolvesToTheConfiguredPolicy()
    {
        Assert.Equal(ReasoningPolicy.Strip, OutputParserRegistry.Resolve("none", false, capabilities: null, "strip").Reasoning);
        Assert.Equal(ReasoningPolicy.Keep, OutputParserRegistry.Resolve("none", false, capabilities: null, "keep").Reasoning);

        // Absent key and the loader default both land on keep, which is what every model serves today.
        Assert.Equal(ReasoningPolicy.Keep, OutputParserRegistry.Resolve("none", false, capabilities: null).Reasoning);
    }

    [Fact]
    public void Plan_ParserChoiceAndReasoningPolicyResolveIndependently()
    {
        // A tool surface and a reasoning policy are separate decisions on the same plan; neither keys on the other.
        OutputParserPlan fencedStrip = OutputParserRegistry.Resolve("fenced", false, capabilities: null, "strip");

        Assert.Equal(ToolParser.Fenced, fencedStrip.ToolParser);
        Assert.IsType<FencedJsonToolOutputParser>(fencedStrip.Parser);
        Assert.Equal(ReasoningPolicy.Strip, fencedStrip.Reasoning);
    }
}
