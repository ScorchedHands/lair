using Lair.Models.Scheduling;

namespace Lair.Tests;

/// <summary>
/// The host device co-scheduling window itself: a declared compute share resolves to its floor-bounded slice of the window's step slots,
/// a grant leaves room only for what this model has not already launched, launched slots return exactly when their forwards complete, and
/// a model resident without a declared share is bounded by nothing the seam does. Every refusal the seam states names the accepted form of
/// a share. Grants and releases decide which candidates a boundary may launch — nothing here cancels, displaces, or reorders a step already
/// running, and these gates read that posture straight off the seam's own occupancy.
/// </summary>
[Trait("Category", "Scheduling")]
public sealed class DeviceCoSchedulerTests
{
    [Fact]
    public void GrantedCap_ResolvesEachShareToItsSliceOfTheWindow_FlooredAtOneSlot()
    {
        Assert.Equal(1, DeviceCoScheduler.GrantedCap(1));     // a 1% share still holds a slot: a share can never round a resident model off the device
        Assert.Equal(1, DeviceCoScheduler.GrantedCap(12));    // 8 * 12 / 100 = 0.96 floors to the one-slot minimum
        Assert.Equal(2, DeviceCoScheduler.GrantedCap(30));    // 8 * 30 / 100 = 2.4 truncates to 2
        Assert.Equal(4, DeviceCoScheduler.GrantedCap(50));
        Assert.Equal(8, DeviceCoScheduler.GrantedCap(100));   // a full-share model may hold the whole window

        ArgumentOutOfRangeException outOfRange = Assert.Throws<ArgumentOutOfRangeException>(() => DeviceCoScheduler.GrantedCap(0));
        Assert.Contains("1 to 100", outOfRange.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => DeviceCoScheduler.GrantedCap(101));
    }

    [Fact]
    public void RequestStepSlots_GradesAgainstWhatThisModelAlreadyLaunched_AndReleasesOnlyOnCompletion()
    {
        DeviceCoScheduler seam = new("cuda");
        seam.RegisterModel("heavy", 30); // cap 2 of the window's 8 slots

        DeviceCoScheduler.CoScheduleGrant first = seam.RequestStepSlots("heavy", requested: 5);
        Assert.Equal(2, first.GrantedSlots);      // the cap bounds this model's own device occupancy
        Assert.True(first.ShareBound);
        Assert.Equal(30, first.SharePercent);
        Assert.Equal(2, first.CapSlots);
        Assert.Equal(0, seam.OutstandingStepSlots("heavy")); // a grant reserves nothing until work launches

        seam.ReserveStepSlots("heavy", launched: first.GrantedSlots); // the boundary's launches enter the window before their forwards run
        Assert.Equal(2, seam.OutstandingStepSlots("heavy")); // launched-but-unfinished work holds its slots — this is what a later boundary defers against

        DeviceCoScheduler.CoScheduleGrant whileBusy = seam.RequestStepSlots("heavy", requested: 5);
        Assert.Equal(0, whileBusy.GrantedSlots);  // every slot this model may hold is already held by its in-flight forwards
        Assert.True(whileBusy.ShareBound);

        seam.CompleteStepSlots("heavy", launched: 2);
        Assert.Equal(0, seam.OutstandingStepSlots("heavy")); // slots return exactly when the device work that held them is done

        DeviceCoScheduler.CoScheduleGrant afterRelease = seam.RequestStepSlots("heavy", requested: 5);
        Assert.Equal(2, afterRelease.GrantedSlots); // the same cap admits again once the model's own occupancy drained
    }

    [Fact]
    public void RequestStepSlots_LeavesAnUnseatedOrSharelessModelUnbounded_AndNeverReleasesBelowZero()
    {
        DeviceCoScheduler seam = new("cuda");
        seam.RegisterModel("resident", null); // resident on the device with no declared share: today's posture, bounded by its own caps alone

        Assert.Equal(7, seam.RequestStepSlots("resident", requested: 7).GrantedSlots);
        Assert.False(seam.RequestStepSlots("resident", requested: 7).ShareBound);

        Assert.Equal(3, seam.RequestStepSlots("never-registered", requested: 3).GrantedSlots); // a model this seam never seated is bounded by nothing here

        seam.CompleteStepSlots("resident", launched: 9); // a release past what was ever launched clamps at zero rather than going negative
        seam.CompleteStepSlots("never-registered", launched: 4); // a completion for an unseated model is a no-op
        Assert.Equal(0, seam.OutstandingStepSlots("resident"));
    }

    [Fact]
    public void UpdateComputeShare_ReResolvesFromTheNextGrant_WithoutDisturbingHeldSlots_AndRefusesUnseatedModels()
    {
        DeviceCoScheduler seam = new("cuda");
        seam.RegisterModel("model", 1); // cap 1
        seam.ReserveStepSlots("model", launched: 1); // one slot held by in-flight work

        Assert.True(seam.UpdateComputeShare("model", 100)); // a live raise takes effect from the next grant on
        Assert.Equal(100, seam.ShareFor("model"));

        DeviceCoScheduler.CoScheduleGrant raised = seam.RequestStepSlots("model", requested: 8);
        Assert.Equal(7, raised.GrantedSlots); // the one slot the in-flight sequence holds still counts against the new cap — a re-share never preempts it

        Assert.True(seam.UpdateComputeShare("model", null)); // declaring no share again leaves the model resident but unbounded by fair-share
        Assert.Equal(8, seam.RequestStepSlots("model", requested: 8).GrantedSlots);

        Assert.False(seam.UpdateComputeShare("not-resident", 50)); // re-sharing a model this seam does not seat states that plainly
        ArgumentOutOfRangeException refused = Assert.Throws<ArgumentOutOfRangeException>(() => seam.UpdateComputeShare("model", 200));
        Assert.Contains("1 to 100", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RegisterModel_SeatsEachModelOnce_AndRefusesAShareOutsideTheAcceptedRange_NamingIt()
    {
        DeviceCoScheduler seam = new("cuda");

        seam.RegisterModel("model", 40);
        seam.RegisterModel("model", 60); // a second registration updates the seat rather than opening another
        Assert.Equal(60, seam.ShareFor("model"));

        ArgumentOutOfRangeException refused = Assert.Throws<ArgumentOutOfRangeException>(() => seam.RegisterModel("model", 0));
        Assert.Contains("1 to 100", refused.Message, StringComparison.Ordinal);
        Assert.Contains("cuda", refused.Message, StringComparison.Ordinal); // the refusal names the device the window bounds

        Assert.True(seam.UnregisterModel("model"));
        Assert.False(seam.UnregisterModel("model")); // a seat already gone is a no-op, not a fault
        Assert.Null(seam.ShareFor("model"));
    }

    [Fact]
    public void CapText_NamesTheModelItsShareAndTheSegmentTheWindowBounds()
    {
        string text = DeviceCoScheduler.CapText("heavy-model", 30, "segment 0 ('cuda' at 'Vram')");

        Assert.Contains("model 'heavy-model' compute_share 30%", text, StringComparison.Ordinal);
        Assert.Contains("segment 0 ('cuda' at 'Vram')", text, StringComparison.Ordinal);
        Assert.Contains("2 co-scheduling step slot(s) of this window's 8", text, StringComparison.Ordinal);
    }
}
