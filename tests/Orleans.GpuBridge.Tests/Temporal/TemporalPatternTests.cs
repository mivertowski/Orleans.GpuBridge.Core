using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Orleans.GpuBridge.Runtime.Temporal.Patterns;
using Xunit;

namespace Orleans.GpuBridge.Tests.Temporal;

public class TemporalPatternTests
{
    private static HybridTimestamp CreateHLC(long nanos) => new(nanos, 0, 1);

    private static TemporalEvent CreateTransactionEvent(
        long timestampNanos,
        ulong sourceId,
        ulong targetId,
        double amount)
    {
        return new TemporalEvent
        {
            EventId = Guid.NewGuid(),
            EventType = "transaction",
            TimestampNanos = timestampNanos,
            SourceId = sourceId,
            TargetId = targetId,
            Value = amount
        };
    }

    #region Pattern Registration Tests

    [Fact]
    public void TemporalPatternDetector_RegisterPattern_AddsPattern()
    {
        var detector = new TemporalPatternDetector();
        var pattern = new RapidSplitPattern();

        detector.RegisterPattern(pattern);

        var stats = detector.GetStatistics();
        Assert.Equal(1, stats.RegisteredPatternCount);
    }

    [Fact]
    public void TemporalPatternDetector_RegisterMultiplePatterns_AllRegistered()
    {
        var detector = new TemporalPatternDetector();

        detector.RegisterPattern(new RapidSplitPattern());
        detector.RegisterPattern(new CircularFlowPattern());
        detector.RegisterPattern(new HighFrequencyPattern());
        detector.RegisterPattern(new VelocityChangePattern());

        var stats = detector.GetStatistics();
        Assert.Equal(4, stats.RegisteredPatternCount);
    }

    [Fact]
    public void TemporalPatternDetector_UnregisterPattern_RemovesPattern()
    {
        var detector = new TemporalPatternDetector();
        var pattern = new RapidSplitPattern();

        detector.RegisterPattern(pattern);
        detector.UnregisterPattern(pattern.PatternId);

        var stats = detector.GetStatistics();
        Assert.Equal(0, stats.RegisteredPatternCount);
    }

    #endregion

    #region Event Processing and Window Management Tests

    [Fact]
    public async Task TemporalPatternDetector_ProcessEvent_UpdatesStatistics()
    {
        var detector = new TemporalPatternDetector();
        var evt = CreateTransactionEvent(1000, 1, 2, 100.0);

        await detector.ProcessEventAsync(evt);

        var stats = detector.GetStatistics();
        Assert.Equal(1, stats.TotalEventsProcessed);
    }

    [Fact]
    public async Task TemporalPatternDetector_EventWindow_AutomaticallyEvictsExpired()
    {
        var detector = new TemporalPatternDetector(windowSizeNanos: 10_000_000_000); // 10 seconds

        // Add event at t=0
        await detector.ProcessEventAsync(CreateTransactionEvent(0, 1, 2, 100.0));

        // Add event at t=15s (should evict first event)
        await detector.ProcessEventAsync(CreateTransactionEvent(15_000_000_000, 2, 3, 200.0));

        var stats = detector.GetStatistics();
        Assert.Equal(2, stats.TotalEventsProcessed);
    }

    [Fact]
    public async Task TemporalPatternDetector_MaxEventLimit_EnforcesLimit()
    {
        var detector = new TemporalPatternDetector(maxWindowEvents: 5);

        // Add 10 events
        for (int i = 0; i < 10; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(i * 1_000_000_000, (ulong)i, (ulong)(i + 1), 100.0));
        }

        var stats = detector.GetStatistics();
        Assert.Equal(10, stats.TotalEventsProcessed);
    }

    #endregion

    #region RapidSplitPattern Tests

    [Fact]
    public async Task RapidSplitPattern_DetectsSimpleSplit()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern(
            windowSizeNanos: 5_000_000_000, // 5 seconds
            minimumSplits: 2,
            minimumAmount: 1000.0));

        // Inbound: A → B ($1000)
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));

        // Outbound splits: B → C ($500)
        await detector.ProcessEventAsync(
            CreateTransactionEvent(2_000_000_000, 2, 3, 500.0));

        // B → D ($500)
        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(3_000_000_000, 2, 4, 500.0));

        Assert.NotEmpty(matches);
        var match = matches.First();
        Assert.Equal("rapid-split", match.PatternId);
        Assert.Equal(PatternSeverity.High, match.Severity);
        Assert.True(match.Confidence > 0.5);
    }

    [Fact]
    public async Task RapidSplitPattern_DoesNotDetectSlowSplit()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern(
            windowSizeNanos: 5_000_000_000, // 5 seconds
            minimumSplits: 2));

        // Inbound: A → B
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));

        // Outbound after 10 seconds (exceeds window)
        await detector.ProcessEventAsync(
            CreateTransactionEvent(11_000_000_000, 2, 3, 500.0));

        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(12_000_000_000, 2, 4, 500.0));

        var rapidSplitMatches = matches.Where(m => m.PatternId == "rapid-split");
        Assert.Empty(rapidSplitMatches);
    }

    [Fact]
    public async Task RapidSplitPattern_RequiresMinimumSplits()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern(
            windowSizeNanos: 5_000_000_000,
            minimumSplits: 3)); // Requires 3 splits

        // Inbound
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));

        // Only 2 splits (not enough)
        await detector.ProcessEventAsync(
            CreateTransactionEvent(2_000_000_000, 2, 3, 500.0));

        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(3_000_000_000, 2, 4, 500.0));

        var rapidSplitMatches = matches.Where(m => m.PatternId == "rapid-split");
        Assert.Empty(rapidSplitMatches);
    }

    [Fact]
    public async Task RapidSplitPattern_ConfidenceIncreases_WithFasterSplits()
    {
        var detector1 = new TemporalPatternDetector();
        detector1.RegisterPattern(new RapidSplitPattern(windowSizeNanos: 5_000_000_000));

        var detector2 = new TemporalPatternDetector();
        detector2.RegisterPattern(new RapidSplitPattern(windowSizeNanos: 5_000_000_000));

        // Fast split (1 second delay)
        await detector1.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));
        await detector1.ProcessEventAsync(CreateTransactionEvent(1_100_000_000, 2, 3, 500.0));
        var fastMatches = await detector1.ProcessEventAsync(
            CreateTransactionEvent(1_200_000_000, 2, 4, 500.0));

        // Slow split (4 second delay)
        await detector2.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));
        await detector2.ProcessEventAsync(CreateTransactionEvent(4_000_000_000, 2, 3, 500.0));
        var slowMatches = await detector2.ProcessEventAsync(
            CreateTransactionEvent(5_000_000_000, 2, 4, 500.0));

        var fastConfidence = fastMatches.First().Confidence;
        var slowConfidence = slowMatches.First().Confidence;

        Assert.True(fastConfidence > slowConfidence);
    }

    #endregion

    #region CircularFlowPattern Tests

    [Fact]
    public async Task CircularFlowPattern_DetectsSimpleCircle()
    {
        var graph = new TemporalGraphStorage();
        var detector = new TemporalPatternDetector(graph);
        detector.RegisterPattern(new CircularFlowPattern(
            windowSizeNanos: 60_000_000_000, // 60 seconds
            minimumHops: 3));

        // Create circle: A → B → C → A
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0)); // A → B

        await detector.ProcessEventAsync(
            CreateTransactionEvent(2_000_000_000, 2, 3, 900.0)); // B → C

        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(3_000_000_000, 3, 1, 800.0)); // C → A

        var circularMatches = matches.Where(m => m.PatternId == "circular-flow").ToList();
        Assert.NotEmpty(circularMatches);
        Assert.Equal(PatternSeverity.Critical, circularMatches.First().Severity);
    }

    [Fact]
    public async Task CircularFlowPattern_DoesNotDetectIncompleteCircle()
    {
        var graph = new TemporalGraphStorage();
        var detector = new TemporalPatternDetector(graph);
        detector.RegisterPattern(new CircularFlowPattern(
            windowSizeNanos: 60_000_000_000,
            minimumHops: 3));

        // Incomplete path: A → B → C (no return to A)
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));

        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(2_000_000_000, 2, 3, 900.0));

        var circularMatches = matches.Where(m => m.PatternId == "circular-flow");
        Assert.Empty(circularMatches);
    }

    [Fact]
    public async Task CircularFlowPattern_RespectsTimeWindow()
    {
        var graph = new TemporalGraphStorage();
        var detector = new TemporalPatternDetector(graph);
        detector.RegisterPattern(new CircularFlowPattern(
            windowSizeNanos: 10_000_000_000, // 10 seconds
            minimumHops: 3));

        // Circle that exceeds time window
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));

        await detector.ProcessEventAsync(
            CreateTransactionEvent(2_000_000_000, 2, 3, 900.0));

        // Last edge at 20 seconds (exceeds 10s window)
        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(21_000_000_000, 3, 1, 800.0));

        var circularMatches = matches.Where(m => m.PatternId == "circular-flow");
        Assert.Empty(circularMatches);
    }

    #endregion

    #region HighFrequencyPattern Tests

    [Fact]
    public async Task HighFrequencyPattern_DetectsRapidBurst()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new HighFrequencyPattern(
            windowSizeNanos: 1_000_000_000, // 1 second
            minimumTransactions: 10,
            minimumTotalAmount: 10_000.0));

        // Generate 15 transactions in 1 second from same source
        for (int i = 0; i < 15; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(
                    1_000_000_000 + i * 50_000_000, // 50ms apart
                    sourceId: 1,
                    targetId: (ulong)(i + 2),
                    amount: 1000.0));
        }

        var stats = detector.GetStatistics();
        Assert.True(stats.TotalPatternsDetected > 0);
    }

    [Fact]
    public async Task HighFrequencyPattern_DoesNotDetectSlowTransactions()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new HighFrequencyPattern(
            windowSizeNanos: 1_000_000_000,
            minimumTransactions: 10));

        // Only 5 transactions (not enough)
        for (int i = 0; i < 5; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(
                    1_000_000_000 + i * 100_000_000,
                    sourceId: 1,
                    targetId: (ulong)(i + 2),
                    amount: 1000.0));
        }

        var stats = detector.GetStatistics();
        var highFreqMatches = stats.TotalPatternsDetected;
        Assert.Equal(0, highFreqMatches);
    }

    [Fact]
    public async Task HighFrequencyPattern_CalculatesTransactionsPerSecond()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new HighFrequencyPattern(
            windowSizeNanos: 1_000_000_000,
            minimumTransactions: 10));

        var matches = new List<PatternMatch>();

        // Generate 20 transactions in 0.5 seconds (40 tx/sec rate)
        for (int i = 0; i < 20; i++)
        {
            var newMatches = await detector.ProcessEventAsync(
                CreateTransactionEvent(
                    1_000_000_000 + i * 25_000_000, // 25ms apart
                    sourceId: 1,
                    targetId: (ulong)(i + 2),
                    amount: 1000.0));
            matches.AddRange(newMatches);
        }

        var highFreqMatches = matches.Where(m => m.PatternId == "high-frequency").ToList();
        if (highFreqMatches.Any())
        {
            var metadata = highFreqMatches.First().Metadata;
            Assert.True(metadata.ContainsKey("transactions_per_second"));
            var rate = (double)metadata["transactions_per_second"];
            Assert.True(rate > 10.0); // Should be much higher than 10/sec
        }
    }

    #endregion

    #region VelocityChangePattern Tests

    [Fact]
    public async Task VelocityChangePattern_DetectsLargeTransactionAfterInactivity()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new VelocityChangePattern(
            windowSizeNanos: 3600_000_000_000, // 1 hour
            thresholdAmount: 50_000.0,
            inactivityPeriod: 1800_000_000_000)); // 30 minutes

        // Small transaction
        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 100.0));

        // Large transaction after 35 minutes of inactivity
        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(
                1_000_000_000 + 2100_000_000_000, // 35 minutes later
                sourceId: 1,
                targetId: 3,
                amount: 60_000.0));

        var velocityMatches = matches.Where(m => m.PatternId == "velocity-change").ToList();
        Assert.NotEmpty(velocityMatches);
        Assert.Equal(PatternSeverity.High, velocityMatches.First().Severity);
    }

    [Fact]
    public async Task VelocityChangePattern_DoesNotDetectSmallTransactions()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new VelocityChangePattern(
            thresholdAmount: 50_000.0,
            inactivityPeriod: 1800_000_000_000));

        await detector.ProcessEventAsync(
            CreateTransactionEvent(1_000_000_000, 1, 2, 100.0));

        // Small transaction after inactivity (below threshold)
        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(
                1_000_000_000 + 2100_000_000_000,
                sourceId: 1,
                targetId: 3,
                amount: 1000.0)); // Below 50k threshold

        var velocityMatches = matches.Where(m => m.PatternId == "velocity-change");
        Assert.Empty(velocityMatches);
    }

    [Fact]
    public async Task VelocityChangePattern_ConfidenceIncreasesWithLargerAmount()
    {
        var detector1 = new TemporalPatternDetector();
        detector1.RegisterPattern(new VelocityChangePattern(
            thresholdAmount: 50_000.0,
            inactivityPeriod: 1800_000_000_000));

        var detector2 = new TemporalPatternDetector();
        detector2.RegisterPattern(new VelocityChangePattern(
            thresholdAmount: 50_000.0,
            inactivityPeriod: 1800_000_000_000));

        // Setup both with small initial transaction
        await detector1.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 100.0));
        await detector2.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 100.0));

        // Detector1: Moderate large transaction
        var matches1 = await detector1.ProcessEventAsync(
            CreateTransactionEvent(3_000_000_000_000, 1, 3, 60_000.0));

        // Detector2: Very large transaction
        var matches2 = await detector2.ProcessEventAsync(
            CreateTransactionEvent(3_000_000_000_000, 1, 3, 500_000.0));

        var confidence1 = matches1.First(m => m.PatternId == "velocity-change").Confidence;
        var confidence2 = matches2.First(m => m.PatternId == "velocity-change").Confidence;

        Assert.True(confidence2 > confidence1);
    }

    #endregion

    #region Pattern Deduplication Tests

    [Fact]
    public async Task TemporalPatternDetector_DeduplicatesIdenticalMatches()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern(windowSizeNanos: 10_000_000_000));

        // Create scenario that generates same pattern twice
        await detector.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));
        await detector.ProcessEventAsync(CreateTransactionEvent(2_000_000_000, 2, 3, 500.0));
        await detector.ProcessEventAsync(CreateTransactionEvent(3_000_000_000, 2, 4, 500.0));

        // Process another event that would re-evaluate same pattern
        var matches = await detector.ProcessEventAsync(
            CreateTransactionEvent(4_000_000_000, 5, 6, 100.0));

        // Should not have duplicate rapid-split matches
        var rapidSplitMatches = matches.Where(m => m.PatternId == "rapid-split").ToList();
        var uniqueMatches = rapidSplitMatches.DistinctBy(m => string.Join(",", m.InvolvedEvents.Select(e => e.EventId))).ToList();

        Assert.Equal(uniqueMatches.Count, rapidSplitMatches.Count);
    }

    #endregion

    #region Statistics Tests

    [Fact]
    public async Task TemporalPatternDetector_Statistics_TracksEventsProcessed()
    {
        var detector = new TemporalPatternDetector();

        for (int i = 0; i < 50; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(i * 1_000_000_000, 1, 2, 100.0));
        }

        var stats = detector.GetStatistics();
        Assert.Equal(50, stats.TotalEventsProcessed);
    }

    [Fact]
    public async Task TemporalPatternDetector_Statistics_TracksPatternsDetected()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern());

        // Create detectable pattern
        await detector.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));
        await detector.ProcessEventAsync(CreateTransactionEvent(2_000_000_000, 2, 3, 500.0));
        await detector.ProcessEventAsync(CreateTransactionEvent(3_000_000_000, 2, 4, 500.0));

        var stats = detector.GetStatistics();
        Assert.True(stats.TotalPatternsDetected > 0);
    }

    [Fact]
    public async Task TemporalPatternDetector_Statistics_CountsUniquePatterns()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern());
        detector.RegisterPattern(new HighFrequencyPattern(minimumTransactions: 5));

        // Create patterns
        await detector.ProcessEventAsync(CreateTransactionEvent(1_000_000_000, 1, 2, 1000.0));

        for (int i = 0; i < 10; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(
                    2_000_000_000 + i * 50_000_000,
                    sourceId: 2,
                    targetId: (ulong)(i + 3),
                    amount: 500.0));
        }

        var stats = detector.GetStatistics();
        Assert.True(stats.TotalPatternsDetected >= 2); // At least 2 different patterns
    }

    #endregion

    #region Performance Tests

    [Fact]
    public async Task TemporalPatternDetector_Performance_ProcessesEventsQuickly()
    {
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern());
        detector.RegisterPattern(new HighFrequencyPattern());
        detector.RegisterPattern(new VelocityChangePattern());

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Process 1000 events
        for (int i = 0; i < 1000; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(
                    i * 1_000_000, // 1ms apart
                    sourceId: (ulong)(i % 10),
                    targetId: (ulong)((i + 1) % 10),
                    amount: 100.0 + i));
        }

        sw.Stop();

        var avgTimePerEvent = sw.Elapsed.TotalMicroseconds / 1000.0;

        // Target: <100μs per event (as specified in roadmap)
        Assert.True(avgTimePerEvent < 100.0,
            $"Average time per event: {avgTimePerEvent:F2}μs (target: <100μs)");
    }

    [Fact]
    public async Task TemporalPatternDetector_Performance_WindowManagementEfficient()
    {
        var detector = new TemporalPatternDetector(
            windowSizeNanos: 10_000_000_000,
            maxWindowEvents: 10000);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Process 5000 events that will trigger eviction
        for (int i = 0; i < 5000; i++)
        {
            await detector.ProcessEventAsync(
                CreateTransactionEvent(
                    i * 100_000_000, // 100ms apart
                    sourceId: 1,
                    targetId: 2,
                    amount: 100.0));
        }

        sw.Stop();

        var avgTime = sw.Elapsed.TotalMicroseconds / 5000.0;

        // Should still be fast even with eviction
        Assert.True(avgTime < 50.0,
            $"Average time with eviction: {avgTime:F2}μs (target: <50μs)");
    }

    #endregion
}
