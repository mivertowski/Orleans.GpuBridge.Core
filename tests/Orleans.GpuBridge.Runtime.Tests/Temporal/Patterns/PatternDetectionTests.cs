// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using FluentAssertions;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Orleans.GpuBridge.Runtime.Temporal.Patterns;
using Xunit;

namespace Orleans.GpuBridge.Runtime.Tests.Temporal.Patterns;

/// <summary>
/// Comprehensive tests for temporal pattern detection.
/// </summary>
public class PatternDetectionTests
{
    #region TemporalPatternDetector Tests

    [Fact]
    public void TemporalPatternDetector_RegisterPattern_ShouldSucceed()
    {
        // Arrange
        var detector = new TemporalPatternDetector();
        var pattern = new RapidSplitPattern();

        // Act
        detector.RegisterPattern(pattern);

        // Assert
        detector.RegisteredPatternCount.Should().Be(1);
    }

    [Fact]
    public void TemporalPatternDetector_RegisterDuplicatePattern_ShouldThrow()
    {
        // Arrange
        var detector = new TemporalPatternDetector();
        var pattern = new RapidSplitPattern();
        detector.RegisterPattern(pattern);

        // Act & Assert
        var act = () => detector.RegisterPattern(pattern);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*already registered*");
    }

    [Fact]
    public void TemporalPatternDetector_UnregisterPattern_ShouldSucceed()
    {
        // Arrange
        var detector = new TemporalPatternDetector();
        var pattern = new RapidSplitPattern();
        detector.RegisterPattern(pattern);

        // Act
        var result = detector.UnregisterPattern(pattern.PatternId);

        // Assert
        result.Should().BeTrue();
        detector.RegisteredPatternCount.Should().Be(0);
    }

    [Fact]
    public async Task TemporalPatternDetector_ProcessEvent_ShouldAddToWindow()
    {
        // Arrange
        var detector = new TemporalPatternDetector();
        var evt = CreateTransactionEvent(100, 200, 1000.0);

        // Act
        await detector.ProcessEventAsync(evt);

        // Assert
        detector.WindowEventCount.Should().Be(1);
    }

    [Fact]
    public async Task TemporalPatternDetector_GetStatistics_ShouldReturnValidStats()
    {
        // Arrange
        var detector = new TemporalPatternDetector();
        detector.RegisterPattern(new RapidSplitPattern());

        var events = CreateTransactionSequence(10);
        foreach (var evt in events)
        {
            await detector.ProcessEventAsync(evt);
        }

        // Act
        var stats = detector.GetStatistics();

        // Assert
        stats.TotalEventsProcessed.Should().Be(10);
        stats.RegisteredPatternCount.Should().Be(1);
    }

    #endregion

    #region RapidSplitPattern Tests

    [Fact]
    public async Task RapidSplitPattern_ShouldDetectSplitTransaction()
    {
        // Arrange
        var pattern = new RapidSplitPattern(
            windowSizeNanos: 10_000_000_000, // 10 seconds
            minimumSplits: 2,
            minimumAmount: 500.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>
        {
            // Inbound to account 100
            CreateTransactionEvent(50, 100, 1000.0, now),
            // Outbound splits from account 100
            CreateTransactionEvent(100, 200, 400.0, now + 1_000_000_000),
            CreateTransactionEvent(100, 300, 400.0, now + 2_000_000_000),
            CreateTransactionEvent(100, 400, 200.0, now + 3_000_000_000)
        };

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("rapid-split");
    }

    [Fact]
    public async Task RapidSplitPattern_ShouldNotDetectNormalTransaction()
    {
        // Arrange
        var pattern = new RapidSplitPattern(
            windowSizeNanos: 5_000_000_000,
            minimumSplits: 2,
            minimumAmount: 1000.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>
        {
            CreateTransactionEvent(100, 200, 500.0, now),
            CreateTransactionEvent(300, 400, 600.0, now + 1_000_000_000)
        };

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().BeEmpty();
    }

    #endregion

    #region HighFrequencyPattern Tests

    [Fact]
    public async Task HighFrequencyPattern_ShouldDetectBurstActivity()
    {
        // Arrange
        var pattern = new HighFrequencyPattern(
            windowSizeNanos: 1_000_000_000, // 1 second
            minimumTransactions: 5,
            minimumTotalAmount: 1000.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>();

        // Create burst of transactions from same source
        for (int i = 0; i < 10; i++)
        {
            events.Add(CreateTransactionEvent(
                100, (ulong)(200 + i), 500.0,
                now + i * 50_000_000)); // 50ms apart
        }

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("high-frequency");
    }

    [Fact]
    public async Task HighFrequencyPattern_ShouldNotDetectSlowActivity()
    {
        // Arrange
        var pattern = new HighFrequencyPattern(
            windowSizeNanos: 1_000_000_000,
            minimumTransactions: 5,
            minimumTotalAmount: 1000.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>();

        // Create slow transactions (2 seconds apart)
        for (int i = 0; i < 5; i++)
        {
            events.Add(CreateTransactionEvent(
                100, (ulong)(200 + i), 500.0,
                now + i * 2_000_000_000));
        }

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().BeEmpty();
    }

    #endregion

    #region CircularFlowPattern Tests

    [Fact]
    public async Task CircularFlowPattern_ShouldDetectCircle()
    {
        // Arrange
        var pattern = new CircularFlowPattern(
            windowSizeNanos: 60_000_000_000,
            minimumHops: 3,
            maximumHops: 10);

        var graph = new TemporalGraphStorage();
        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var hlc = new Abstractions.Temporal.HybridTimestamp(now, 0, 1);

        // Create circular path: A -> B -> C -> A
        graph.AddEdge(100, 200, now, now + 10_000_000, hlc, 1000.0, "transaction");
        graph.AddEdge(200, 300, now + 1_000_000_000, now + 1_010_000_000, hlc, 1000.0, "transaction");
        graph.AddEdge(300, 100, now + 2_000_000_000, now + 2_010_000_000, hlc, 1000.0, "transaction");

        var events = new List<TemporalEvent>
        {
            CreateTransactionEvent(100, 200, 1000.0, now),
            CreateTransactionEvent(200, 300, 1000.0, now + 1_000_000_000),
            CreateTransactionEvent(300, 100, 1000.0, now + 2_000_000_000)
        };

        // Act
        var matches = await pattern.MatchAsync(events, graph);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("circular-flow");
    }

    #endregion

    #region TemporalClusterPattern Tests

    [Fact]
    public async Task TemporalClusterPattern_ShouldDetectCluster()
    {
        // Arrange
        var pattern = new TemporalClusterPattern(
            windowSizeNanos: 10_000_000_000,
            clusterSize: 5,
            clusterRatio: 2.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>();

        // Create clustered events (10 events in 100ms)
        for (int i = 0; i < 10; i++)
        {
            events.Add(CreateTransactionEvent(
                (ulong)(100 + i), (ulong)(200 + i), 100.0,
                now + i * 10_000_000)); // 10ms apart
        }

        // Add sparse events (spread over 10 seconds)
        for (int i = 0; i < 5; i++)
        {
            events.Add(CreateTransactionEvent(
                (ulong)(300 + i), (ulong)(400 + i), 100.0,
                now + 500_000_000 + i * 2_000_000_000)); // 2s apart
        }

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("temporal-cluster");
    }

    #endregion

    #region StructuringPattern Tests

    [Fact]
    public async Task StructuringPattern_ShouldDetectStructuring()
    {
        // Arrange
        var pattern = new StructuringPattern(
            windowSizeNanos: 86_400_000_000_000, // 24 hours
            threshold: 10_000.0,
            marginPercent: 10.0,
            minimumCount: 3);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>
        {
            // Transactions just below $10,000 threshold
            CreateTransactionEvent(100, 200, 9500.0, now),
            CreateTransactionEvent(100, 300, 9800.0, now + 1_000_000_000),
            CreateTransactionEvent(100, 400, 9200.0, now + 2_000_000_000),
            CreateTransactionEvent(100, 500, 9600.0, now + 3_000_000_000)
        };

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("structuring");
    }

    #endregion

    #region FanOutPattern Tests

    [Fact]
    public async Task FanOutPattern_ShouldDetectFanOut()
    {
        // Arrange
        var pattern = new FanOutPattern(
            windowSizeNanos: 300_000_000_000,
            minimumTargets: 5,
            minimumTotalAmount: 5000.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>();

        // Single source sending to multiple targets
        for (int i = 0; i < 10; i++)
        {
            events.Add(CreateTransactionEvent(
                100, (ulong)(200 + i), 1000.0,
                now + i * 1_000_000_000));
        }

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("fan-out");
    }

    #endregion

    #region FanInPattern Tests

    [Fact]
    public async Task FanInPattern_ShouldDetectFanIn()
    {
        // Arrange
        var pattern = new FanInPattern(
            windowSizeNanos: 300_000_000_000,
            minimumSources: 5,
            minimumTotalAmount: 5000.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>();

        // Multiple sources sending to single target
        for (int i = 0; i < 10; i++)
        {
            events.Add(CreateTransactionEvent(
                (ulong)(100 + i), 200, 1000.0,
                now + i * 1_000_000_000));
        }

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("fan-in");
    }

    #endregion

    #region RoundTripPattern Tests

    [Fact]
    public async Task RoundTripPattern_ShouldDetectRoundTrip()
    {
        // Arrange
        var pattern = new RoundTripPattern(
            windowSizeNanos: 3600_000_000_000,
            minimumAmount: 1000.0,
            tolerancePercent: 5.0);

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>
        {
            // A sends to B
            CreateTransactionEvent(100, 200, 5000.0, now),
            // B sends back to A (same amount)
            CreateTransactionEvent(200, 100, 5000.0, now + 60_000_000_000) // 1 minute later
        };

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("round-trip");
    }

    #endregion

    #region VelocityChangePattern Tests

    [Fact]
    public async Task VelocityChangePattern_ShouldDetectVelocityChange()
    {
        // Arrange
        var pattern = new VelocityChangePattern(
            windowSizeNanos: 3600_000_000_000,
            thresholdAmount: 10_000.0,
            inactivityPeriod: 1800_000_000_000); // 30 minutes

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>
        {
            // Small transaction
            CreateTransactionEvent(100, 200, 100.0, now),
            // Large transaction after 45 minutes of inactivity
            CreateTransactionEvent(100, 300, 50000.0, now + 2700_000_000_000)
        };

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
        matches.First().PatternId.Should().Be("velocity-change");
    }

    #endregion

    #region GpuPatternMatcher Tests

    [Fact]
    public async Task GpuPatternMatcher_ShouldProcessEvents()
    {
        // Arrange
        var matcher = new GpuPatternMatcher();
        var patterns = new List<ITemporalPattern>
        {
            new RapidSplitPattern(),
            new HighFrequencyPattern()
        };
        var events = CreateTransactionSequence(100);

        // Act
        var matches = await matcher.FindPatternsAsync(events, patterns);

        // Assert
        matches.Should().NotBeNull();
        matcher.Statistics.TotalEventsProcessed.Should().Be(100);
    }

    [Fact]
    public async Task GpuPatternMatcher_ShouldReturnStatistics()
    {
        // Arrange
        var matcher = new GpuPatternMatcher();
        var patterns = new List<ITemporalPattern> { new HighFrequencyPattern() };
        var events = CreateTransactionSequence(50);

        // Act
        await matcher.FindPatternsAsync(events, patterns);
        var stats = matcher.Statistics;

        // Assert
        stats.TotalEventsProcessed.Should().Be(50);
        stats.TotalPatternsChecked.Should().Be(1);
    }

    #endregion

    #region PatternDSL Tests

    [Fact]
    public void PatternBuilder_ShouldCreatePattern()
    {
        // Act
        var pattern = PatternBuilder.Create("test-pattern")
            .Named("Test Pattern")
            .WithDescription("A test pattern")
            .WithWindowSize(TimeSpan.FromSeconds(5))
            .WithSeverity(PatternSeverity.High)
            .ForEventType("transaction")
            .GroupBySource()
            .MatchWhenCount(3)
            .WithConfidence(0.9)
            .Build();

        // Assert
        pattern.PatternId.Should().Be("test-pattern");
        pattern.Name.Should().Be("Test Pattern");
        pattern.WindowSizeNanos.Should().Be(5_000_000_000);
        pattern.Severity.Should().Be(PatternSeverity.High);
    }

    [Fact]
    public async Task PatternBuilder_ShouldMatchEvents()
    {
        // Arrange
        var pattern = PatternBuilder.Create("burst-test")
            .Named("Burst Test")
            .WithWindowSize(TimeSpan.FromSeconds(1))
            .ForEventType("transaction")
            .GroupBySource()
            .MatchWhenCount(3)
            .Build();

        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        var events = new List<TemporalEvent>
        {
            CreateTransactionEvent(100, 200, 100.0, now),
            CreateTransactionEvent(100, 300, 100.0, now + 100_000_000),
            CreateTransactionEvent(100, 400, 100.0, now + 200_000_000),
            CreateTransactionEvent(100, 500, 100.0, now + 300_000_000)
        };

        // Act
        var matches = await pattern.MatchAsync(events, null);

        // Assert
        matches.Should().NotBeEmpty();
    }

    [Fact]
    public void PatternTemplates_BurstActivity_ShouldCreate()
    {
        // Act
        var pattern = PatternTemplates.BurstActivity(5, TimeSpan.FromSeconds(1));

        // Assert
        pattern.PatternId.Should().Be("burst-activity");
        pattern.Name.Should().Be("Burst Activity");
    }

    [Fact]
    public void PatternTemplates_LargeTransaction_ShouldCreate()
    {
        // Act
        var pattern = PatternTemplates.LargeTransaction(10000.0);

        // Assert
        pattern.PatternId.Should().Be("large-transaction");
    }

    #endregion

    #region PatternMatch Tests

    [Fact]
    public void PatternMatch_DurationNanos_ShouldCalculate()
    {
        // Arrange
        var match = new PatternMatch
        {
            PatternId = "test",
            PatternName = "Test",
            DetectionTimeNanos = 1000,
            WindowStartNanos = 100,
            WindowEndNanos = 500,
            InvolvedEvents = new List<TemporalEvent>()
        };

        // Assert
        match.DurationNanos.Should().Be(400);
        match.DurationSeconds.Should().BeApproximately(0.0000004, 0.0000001);
    }

    #endregion

    #region Helper Methods

    private static TemporalEvent CreateTransactionEvent(
        ulong sourceId, ulong targetId, double amount, long? timestampNanos = null)
    {
        return new TemporalEvent
        {
            EventId = Guid.NewGuid(),
            EventType = "transaction",
            TimestampNanos = timestampNanos ?? DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(),
            SourceId = sourceId,
            TargetId = targetId,
            Value = amount
        };
    }

    private static List<TemporalEvent> CreateTransactionSequence(int count)
    {
        var events = new List<TemporalEvent>();
        var now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        for (int i = 0; i < count; i++)
        {
            events.Add(CreateTransactionEvent(
                (ulong)(100 + i % 10),
                (ulong)(200 + i % 20),
                100.0 + i * 10,
                now + i * 100_000_000)); // 100ms apart
        }

        return events;
    }

    #endregion
}
