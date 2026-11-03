using Lair.Http;

namespace Lair.Tests;

/// <summary>
/// Parser-level tests for the tool pipeline over synthetic model outputs: the injected prompt block's shape, and the
/// fenced-JSON detector rules. Accepted calls are stripped from visible text with their arguments compacted; unknown
/// names, malformed JSON, non-json fences, extra properties, and unterminated fences all stay ordinary text verbatim.
/// No engine or fixture is involved; these hold regardless of model compliance.
/// </summary>
[Trait("Category", "Api")]
public class ToolPipelineTests
{
    private static readonly ToolDefinition WeatherTool = new(
        "get_weather",
        "Fetch current weather for a city.",
        "{" + "\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]" + "}");

    private static readonly ToolDefinition EmailTool = new(
        "send_email",
        null,
        "{" + "\"type\":\"object\",\"properties\":{\"to\":{\"type\":\"string\"},\"body\":{\"type\":\"string\"}}" + "}");

    [Fact]
    public void Compose_AppendsInstructionLinePerToolSchemaFencesAndReplyContract()
    {
        const string rendered = "USER-PROMPT-TEXT";
        IReadOnlyList<ToolDefinition> tools = [WeatherTool, EmailTool];

        string composed = ToolPromptComposer.Compose(rendered, tools);

        Assert.StartsWith(rendered, composed); // the rendered prompt is untouched ahead of the block
        Assert.Contains("\nYou can call these tools:\n", composed);

        Assert.Equal(2, CountOccurrences(composed, "```json\n"));
        Assert.Contains("\"name\":\"get_weather\"", composed);
        Assert.Contains("\"description\":\"Fetch current weather for a city.\"", composed);
        Assert.Contains(WeatherTool.ParametersSchema, composed); // the parameter schema rides in verbatim

        Assert.Contains("\"name\":\"send_email\"", composed);
        Assert.Contains(EmailTool.ParametersSchema, composed);
        Assert.DoesNotContain(@"{""name"":""send_email"",""description", composed); // no description key for a tool without one

        Assert.True(composed.EndsWith(ToolPromptComposer.ContractLine), "the reply contract line must close the block");
    }

    [Fact]
    public void Detect_SingleValidCall_ReportsTheCallAndStripsOnlyTheFence()
    {
        string text = "Before the call\n```json\n" + Call("get_weather", "{\"city\": \"Oslo\"}") + "\n```\nafter the call";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Single(detection.Calls);
        Assert.Equal("get_weather", detection.Calls[0].Name);
        Assert.Equal("{\"city\":\"Oslo\"}", detection.Calls[0].Arguments); // re-encoded compactly on the way in

        string visible = detection.VisibleText;
        Assert.StartsWith("Before the call", visible);
        Assert.EndsWith("after the call", visible);
        Assert.DoesNotContain("```", visible);
        Assert.DoesNotContain("get_weather", visible);
    }

    [Fact]
    public void Detect_TwoValidCalls_ReportsBothInDocumentOrder()
    {
        string text = "Opening.\n```json\n" + Call("send_email", "{\"to\": \"a@b.c\"}") + "\n```\nbetween the calls\n"
            + "```json\n" + Call("get_weather", "{\"city\":\"Oslo\"}") + "\n```\nclosing.";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather", "send_email"]);

        Assert.Equal(2, detection.Calls.Count);
        Assert.Equal("send_email", detection.Calls[0].Name);
        Assert.Equal("{\"to\":\"a@b.c\"}", detection.Calls[0].Arguments);
        Assert.Equal("get_weather", detection.Calls[1].Name);

        string visible = detection.VisibleText;
        Assert.StartsWith("Opening.", visible);
        Assert.Contains("between the calls", visible);
        Assert.EndsWith("closing.", visible);
        Assert.DoesNotContain("```", visible);
    }

    [Fact]
    public void Detect_UnknownToolName_StaysOrdinaryTextByteForByte()
    {
        string text = "Answer.\n```json\n" + Call("lookup_address", "{\"street\":\"Elm\"}") + "\n```\nThe end.";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Empty(detection.Calls);
        Assert.Equal(text, detection.VisibleText); // nothing was accepted: the whole output survives verbatim
    }

    [Theory]
    [InlineData("{\"name\":\"get_weather\",\"arguments\":")]         // truncated object
    [InlineData("{\"name\":\"get_weather\"}")]                        // no arguments property
    [InlineData("{\"arguments\":{\"city\":\"Oslo\"},\"name\":42}")]   // name is not a string
    [InlineData("not json at all")]                                   // no JSON in the fence
    public void Detect_MalformedCallBlock_StaysOrdinaryTextByteForByte(string body)
    {
        string text = "Answer.\n```json\n" + body + "\n```\nThe end.";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Empty(detection.Calls);
        Assert.Equal(text, detection.VisibleText);
    }

    [Fact]
    public void Detect_ExtraPropertiesInCallBlock_StaysOrdinaryText()
    {
        string body = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"},\"id\":7}";
        string text = "Answer.\n```json\n" + body + "\n```\nThe end.";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Empty(detection.Calls); // only the exact two-property shape is a call
        Assert.Equal(text, detection.VisibleText);
    }

    [Fact]
    public void Detect_TrailingProseAfterAcceptedCall_KeepsTheProseAndStripsTheFence()
    {
        string text = "Thinking.\n```json\n" + Call("get_weather", "{\"city\":\"Oslo\"}")
            + "\n```\nThat is the call I want to make; here is why it fits.";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Single(detection.Calls);
        string visible = detection.VisibleText;
        Assert.StartsWith("Thinking.", visible);
        Assert.Contains("That is the call I want to make", visible);
        Assert.EndsWith("; here is why it fits.", visible);
        Assert.DoesNotContain("```", visible);
    }

    [Fact]
    public void Detect_NonJsonLanguageTag_IsNeverACallEvenWithAValidCallInside()
    {
        string text = "Answer.\n```python\n" + Call("get_weather", "{\"city\":\"Oslo\"}") + "\n```\nThe end.";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Empty(detection.Calls); // only a bare json tag opens a candidate block
        Assert.Equal(text, detection.VisibleText);
    }

    [Fact]
    public void Detect_UnterminatedJsonFence_StaysOrdinaryText()
    {
        string text = "Answer.\n```json\n" + Call("get_weather", "{\"city\":\"Oslo\"}"); // no closing fence

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Empty(detection.Calls);
        Assert.Equal(text, detection.VisibleText);
    }

    [Fact]
    public void Detect_ForcedChoice_AcceptsOnlyTheForcedNameAmongDeclaredTools()
    {
        string text = "Opening.\n```json\n" + Call("send_email", "{\"to\":\"a@b.c\"}") + "\n```\nmiddle prose\n"
            + "```json\n" + Call("get_weather", "{\"city\":\"Oslo\"}") + "\n```\nclosing.";

        // A forced choice narrows the accepted-name set to that one member of the declared tools.
        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Single(detection.Calls);
        Assert.Equal("get_weather", detection.Calls[0].Name);

        string visible = detection.VisibleText;
        Assert.Contains("\"name\":\"send_email\"", visible); // the non-forced call stays ordinary text, fence included
        Assert.DoesNotContain("\"name\":\"get_weather\"", visible); // only the forced name was stripped
    }

    [Fact]
    public void Detect_PrettyPrintedArguments_ArriveCompactlyOnTheCall()
    {
        string body = "{\"name\":\"get_weather\",\"arguments\":{\"city\": \"Oslo\", \"days\": 2}}";
        string text = "```json\n" + body + "\n```";

        ToolDetection detection = ToolCallDetector.Detect(text, ["get_weather"]);

        Assert.Single(detection.Calls);
        Assert.Equal("{\"city\":\"Oslo\",\"days\":2}", detection.Calls[0].Arguments);
    }

    // One candidate call body: the exact two-property shape the detector accepts.
    private static string Call(string name, string argumentsJson) =>
        "{\"name\":\"" + name + "\",\"arguments\":" + argumentsJson + "}";

    private static int CountOccurrences(string text, string needle)
    {
        int count = 0;
        for (int index = text.IndexOf(needle, StringComparison.Ordinal); index >= 0;
             index = text.IndexOf(needle, index + needle.Length, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }
}

