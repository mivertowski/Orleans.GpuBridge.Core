// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Runtime.Temporal.Patterns;

/// <summary>
/// Detects temporal clustering of events (suspicious activity bursts).
/// </summary>
/// <remarks>
/// <para>
/// Pattern: Events cluster together in time more than expected by chance.
/// Uses statistical analysis to detect abnormal temporal concentrations.
/// </para>
/// <para>
/// Applications:
/// - Coordinated fraud detection
/// - Bot network activity
/// - Market manipulation
/// - DDoS attack patterns
/// </para>
/// </remarks>
public sealed class TemporalClusterPattern : TemporalPatternBase
{
    private readonly int _clusterSize;
    private readonly double _clusterRatio;
    private readonly string? _eventTypeFilter;

    /// <summary>
    /// Gets the unique identifier for the temporal cluster pattern.
    /// </summary>
    public override string PatternId => "temporal-cluster";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Temporal Event Clustering";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects statistically abnormal clustering of events in time";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (High).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.High;

    /// <summary>
    /// Creates a new temporal cluster pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window for clustering analysis (default: 10 seconds)</param>
    /// <param name="clusterSize">Minimum events to form a cluster (default: 5)</param>
    /// <param name="clusterRatio">Ratio of cluster density to baseline (default: 3.0)</param>
    /// <param name="eventTypeFilter">Optional event type filter (null = all types)</param>
    public TemporalClusterPattern(
        long windowSizeNanos = 10_000_000_000,
        int clusterSize = 5,
        double clusterRatio = 3.0,
        string? eventTypeFilter = null)
    {
        if (clusterSize < 2)
            throw new ArgumentException("Cluster size must be at least 2", nameof(clusterSize));
        if (clusterRatio <= 1.0)
            throw new ArgumentException("Cluster ratio must be greater than 1.0", nameof(clusterRatio));

        WindowSizeNanos = windowSizeNanos;
        _clusterSize = clusterSize;
        _clusterRatio = clusterRatio;
        _eventTypeFilter = eventTypeFilter;
    }

    /// <summary>
    /// Asynchronously searches for temporal clustering patterns in the provided events.
    /// </summary>
    /// <param name="events">The events to analyze</param>
    /// <param name="graph">Optional temporal graph (not used by this pattern)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>A collection of pattern matches found</returns>
    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask;

        var matches = new List<PatternMatch>();

        // Filter events if needed
        var filteredEvents = _eventTypeFilter != null
            ? FilterByType(events, _eventTypeFilter).ToList()
            : events.ToList();

        if (filteredEvents.Count < _clusterSize)
            return matches;

        // Sort by timestamp
        var sorted = filteredEvents.OrderBy(e => e.TimestampNanos).ToList();

        // Calculate baseline event rate
        var totalSpan = sorted[^1].TimestampNanos - sorted[0].TimestampNanos;
        if (totalSpan <= 0)
            return matches;

        var baselineRate = sorted.Count / (double)totalSpan;

        // Slide through looking for clusters
        var subWindowSize = WindowSizeNanos / 10; // Use smaller sub-windows

        for (int i = 0; i < sorted.Count; i++)
        {
            var windowStart = sorted[i].TimestampNanos;
            var windowEnd = windowStart + subWindowSize;

            var clusterEvents = sorted
                .Skip(i)
                .TakeWhile(e => e.TimestampNanos <= windowEnd)
                .ToList();

            if (clusterEvents.Count >= _clusterSize)
            {
                // Calculate cluster density
                var clusterSpan = clusterEvents[^1].TimestampNanos - clusterEvents[0].TimestampNanos;
                if (clusterSpan > 0)
                {
                    var clusterRate = clusterEvents.Count / (double)clusterSpan;
                    var ratio = clusterRate / baselineRate;

                    if (ratio >= _clusterRatio)
                    {
                        var metadata = new Dictionary<string, object>
                        {
                            ["cluster_count"] = clusterEvents.Count,
                            ["cluster_span_ns"] = clusterSpan,
                            ["cluster_rate"] = clusterRate,
                            ["baseline_rate"] = baselineRate,
                            ["density_ratio"] = ratio
                        };

                        // Higher confidence for higher ratio
                        var confidence = Math.Min(ratio / (_clusterRatio * 2), 1.0);

                        matches.Add(CreateMatch(clusterEvents, confidence, metadata));

                        // Skip ahead to avoid overlapping clusters
                        i += clusterEvents.Count / 2;
                    }
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects structuring patterns (transactions just below reporting thresholds).
/// </summary>
/// <remarks>
/// <para>
/// Pattern: Multiple transactions from the same source that are suspiciously
/// close to but below regulatory reporting thresholds.
/// </para>
/// <para>
/// Common thresholds:
/// - $10,000 (US Bank Secrecy Act)
/// - €15,000 (EU Anti-Money Laundering)
/// - Various other jurisdiction-specific limits
/// </para>
/// </remarks>
public sealed class StructuringPattern : TemporalPatternBase
{
    private readonly double _threshold;
    private readonly double _marginPercent;
    private readonly int _minimumCount;

    /// <summary>
    /// Gets the unique identifier for the structuring pattern.
    /// </summary>
    public override string PatternId => "structuring";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Transaction Structuring";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects transactions structured just below reporting thresholds";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (Critical).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.Critical;

    /// <summary>
    /// Creates a new structuring pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window (default: 24 hours)</param>
    /// <param name="threshold">Reporting threshold amount (default: $10,000)</param>
    /// <param name="marginPercent">Percent below threshold to flag (default: 10%)</param>
    /// <param name="minimumCount">Minimum structured transactions (default: 3)</param>
    public StructuringPattern(
        long windowSizeNanos = 86_400_000_000_000, // 24 hours
        double threshold = 10_000.0,
        double marginPercent = 10.0,
        int minimumCount = 3)
    {
        WindowSizeNanos = windowSizeNanos;
        _threshold = threshold;
        _marginPercent = marginPercent;
        _minimumCount = minimumCount;
    }

    /// <summary>
    /// Asynchronously searches for transaction structuring patterns in the provided events.
    /// </summary>
    /// <param name="events">The events to analyze</param>
    /// <param name="graph">Optional temporal graph (not used by this pattern)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>A collection of pattern matches found</returns>
    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask;

        var matches = new List<PatternMatch>();
        var transactions = FilterByType(events, "transaction").ToList();

        if (transactions.Count < _minimumCount)
            return matches;

        var lowerBound = _threshold * (1.0 - _marginPercent / 100.0);

        // Group by source
        var bySource = GroupBySource(transactions);

        foreach (var sourceGroup in bySource)
        {
            // Find transactions in the "structuring zone"
            var structuredTxs = sourceGroup
                .Where(t => t.Value.HasValue && t.Value >= lowerBound && t.Value < _threshold)
                .OrderBy(t => t.TimestampNanos)
                .ToList();

            if (structuredTxs.Count >= _minimumCount)
            {
                // Check if they're within the time window
                var first = structuredTxs.First().TimestampNanos;
                var last = structuredTxs.Last().TimestampNanos;

                if (last - first <= WindowSizeNanos)
                {
                    var totalAmount = structuredTxs.Sum(t => t.Value ?? 0);
                    var avgAmount = totalAmount / structuredTxs.Count;
                    var avgProximity = avgAmount / _threshold;

                    var metadata = new Dictionary<string, object>
                    {
                        ["source_account"] = sourceGroup.Key,
                        ["structured_count"] = structuredTxs.Count,
                        ["total_amount"] = totalAmount,
                        ["average_amount"] = avgAmount,
                        ["threshold"] = _threshold,
                        ["average_proximity"] = avgProximity
                    };

                    // Confidence based on count and proximity to threshold
                    var countFactor = Math.Min(structuredTxs.Count / (_minimumCount * 2.0), 0.5);
                    var proximityFactor = avgProximity * 0.5;
                    var confidence = countFactor + proximityFactor;

                    matches.Add(CreateMatch(structuredTxs, confidence, metadata));
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects fan-out patterns (one source to many targets).
/// </summary>
public sealed class FanOutPattern : TemporalPatternBase
{
    private readonly int _minimumTargets;
    private readonly double _minimumTotalAmount;

    /// <summary>
    /// Gets the unique identifier for the fan-out pattern.
    /// </summary>
    public override string PatternId => "fan-out";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Fan-Out Distribution";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects rapid distribution from single source to multiple targets";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (Medium).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.Medium;

    /// <summary>
    /// Creates a new fan-out pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window for fan-out detection (default: 5 minutes)</param>
    /// <param name="minimumTargets">Minimum number of unique targets to trigger detection (default: 5)</param>
    /// <param name="minimumTotalAmount">Minimum total transaction amount to trigger detection (default: $10,000)</param>
    public FanOutPattern(
        long windowSizeNanos = 300_000_000_000, // 5 minutes
        int minimumTargets = 5,
        double minimumTotalAmount = 10_000.0)
    {
        WindowSizeNanos = windowSizeNanos;
        _minimumTargets = minimumTargets;
        _minimumTotalAmount = minimumTotalAmount;
    }

    /// <summary>
    /// Asynchronously searches for fan-out patterns in the provided events.
    /// </summary>
    /// <param name="events">The events to analyze</param>
    /// <param name="graph">Optional temporal graph (not used by this pattern)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>A collection of pattern matches found</returns>
    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask;

        var matches = new List<PatternMatch>();
        var transactions = FilterByType(events, "transaction").ToList();

        if (transactions.Count < _minimumTargets)
            return matches;

        var bySource = GroupBySource(transactions);

        foreach (var sourceGroup in bySource)
        {
            var sorted = sourceGroup.OrderBy(t => t.TimestampNanos).ToList();

            for (int i = 0; i < sorted.Count; i++)
            {
                var windowStart = sorted[i].TimestampNanos;
                var windowEnd = windowStart + WindowSizeNanos;

                var windowTxs = sorted
                    .Skip(i)
                    .TakeWhile(t => t.TimestampNanos <= windowEnd)
                    .ToList();

                // Count unique targets
                var uniqueTargets = windowTxs
                    .Where(t => t.TargetId.HasValue)
                    .Select(t => t.TargetId!.Value)
                    .Distinct()
                    .Count();

                var totalAmount = windowTxs.Sum(t => t.Value ?? 0);

                if (uniqueTargets >= _minimumTargets && totalAmount >= _minimumTotalAmount)
                {
                    var metadata = new Dictionary<string, object>
                    {
                        ["source_account"] = sourceGroup.Key,
                        ["unique_targets"] = uniqueTargets,
                        ["transaction_count"] = windowTxs.Count,
                        ["total_amount"] = totalAmount,
                        ["average_per_target"] = totalAmount / uniqueTargets
                    };

                    var confidence = Math.Min(
                        (uniqueTargets / (double)(_minimumTargets * 2)) * 0.5 +
                        (totalAmount / (_minimumTotalAmount * 2)) * 0.5,
                        1.0);

                    matches.Add(CreateMatch(windowTxs, confidence, metadata));

                    // Skip ahead
                    i += windowTxs.Count / 2;
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects fan-in patterns (many sources to one target).
/// </summary>
public sealed class FanInPattern : TemporalPatternBase
{
    private readonly int _minimumSources;
    private readonly double _minimumTotalAmount;

    /// <summary>
    /// Gets the unique identifier for the fan-in pattern.
    /// </summary>
    public override string PatternId => "fan-in";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Fan-In Aggregation";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects rapid aggregation from multiple sources to single target";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (Medium).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.Medium;

    /// <summary>
    /// Creates a new fan-in pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window for fan-in detection (default: 5 minutes)</param>
    /// <param name="minimumSources">Minimum number of unique sources to trigger detection (default: 5)</param>
    /// <param name="minimumTotalAmount">Minimum total transaction amount to trigger detection (default: $10,000)</param>
    public FanInPattern(
        long windowSizeNanos = 300_000_000_000, // 5 minutes
        int minimumSources = 5,
        double minimumTotalAmount = 10_000.0)
    {
        WindowSizeNanos = windowSizeNanos;
        _minimumSources = minimumSources;
        _minimumTotalAmount = minimumTotalAmount;
    }

    /// <summary>
    /// Asynchronously searches for fan-in patterns in the provided events.
    /// </summary>
    /// <param name="events">The events to analyze</param>
    /// <param name="graph">Optional temporal graph (not used by this pattern)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>A collection of pattern matches found</returns>
    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask;

        var matches = new List<PatternMatch>();
        var transactions = FilterByType(events, "transaction").ToList();

        if (transactions.Count < _minimumSources)
            return matches;

        var byTarget = GroupByTarget(transactions);

        foreach (var targetGroup in byTarget)
        {
            var sorted = targetGroup.OrderBy(t => t.TimestampNanos).ToList();

            for (int i = 0; i < sorted.Count; i++)
            {
                var windowStart = sorted[i].TimestampNanos;
                var windowEnd = windowStart + WindowSizeNanos;

                var windowTxs = sorted
                    .Skip(i)
                    .TakeWhile(t => t.TimestampNanos <= windowEnd)
                    .ToList();

                // Count unique sources
                var uniqueSources = windowTxs
                    .Where(t => t.SourceId.HasValue)
                    .Select(t => t.SourceId!.Value)
                    .Distinct()
                    .Count();

                var totalAmount = windowTxs.Sum(t => t.Value ?? 0);

                if (uniqueSources >= _minimumSources && totalAmount >= _minimumTotalAmount)
                {
                    var metadata = new Dictionary<string, object>
                    {
                        ["target_account"] = targetGroup.Key,
                        ["unique_sources"] = uniqueSources,
                        ["transaction_count"] = windowTxs.Count,
                        ["total_amount"] = totalAmount,
                        ["average_per_source"] = totalAmount / uniqueSources
                    };

                    var confidence = Math.Min(
                        (uniqueSources / (double)(_minimumSources * 2)) * 0.5 +
                        (totalAmount / (_minimumTotalAmount * 2)) * 0.5,
                        1.0);

                    matches.Add(CreateMatch(windowTxs, confidence, metadata));

                    // Skip ahead
                    i += windowTxs.Count / 2;
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects round-trip patterns (money sent and returned within time window).
/// </summary>
public sealed class RoundTripPattern : TemporalPatternBase
{
    private readonly double _minimumAmount;
    private readonly double _tolerancePercent;

    /// <summary>
    /// Gets the unique identifier for the round-trip pattern.
    /// </summary>
    public override string PatternId => "round-trip";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Round-Trip Transaction";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects money sent and returned between same accounts";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (High).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.High;

    /// <summary>
    /// Creates a new round-trip pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window for round-trip detection (default: 1 hour)</param>
    /// <param name="minimumAmount">Minimum transaction amount to trigger detection (default: $5,000)</param>
    /// <param name="tolerancePercent">Percentage tolerance for amount matching (default: 5%)</param>
    public RoundTripPattern(
        long windowSizeNanos = 3600_000_000_000, // 1 hour
        double minimumAmount = 5_000.0,
        double tolerancePercent = 5.0)
    {
        WindowSizeNanos = windowSizeNanos;
        _minimumAmount = minimumAmount;
        _tolerancePercent = tolerancePercent;
    }

    /// <summary>
    /// Asynchronously searches for round-trip patterns in the provided events.
    /// </summary>
    /// <param name="events">The events to analyze</param>
    /// <param name="graph">Optional temporal graph (not used by this pattern)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>A collection of pattern matches found</returns>
    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask;

        var matches = new List<PatternMatch>();
        var transactions = FilterByType(events, "transaction").ToList();

        if (transactions.Count < 2)
            return matches;

        var sorted = transactions.OrderBy(t => t.TimestampNanos).ToList();

        for (int i = 0; i < sorted.Count; i++)
        {
            var outgoing = sorted[i];
            if (!outgoing.SourceId.HasValue || !outgoing.TargetId.HasValue ||
                !outgoing.Value.HasValue || outgoing.Value < _minimumAmount)
                continue;

            var windowEnd = outgoing.TimestampNanos + WindowSizeNanos;

            // Look for return transaction
            var returning = sorted.Skip(i + 1)
                .TakeWhile(t => t.TimestampNanos <= windowEnd)
                .FirstOrDefault(t =>
                    t.SourceId == outgoing.TargetId &&
                    t.TargetId == outgoing.SourceId &&
                    t.Value.HasValue &&
                    Math.Abs(t.Value.Value - outgoing.Value.Value) / outgoing.Value.Value * 100 <= _tolerancePercent);

            if (returning != null)
            {
                var metadata = new Dictionary<string, object>
                {
                    ["source_account"] = outgoing.SourceId.Value,
                    ["target_account"] = outgoing.TargetId.Value,
                    ["outgoing_amount"] = outgoing.Value.Value,
                    ["returning_amount"] = returning.Value!.Value,
                    ["round_trip_time_ns"] = returning.TimestampNanos - outgoing.TimestampNanos,
                    ["amount_difference_percent"] =
                        Math.Abs(returning.Value.Value - outgoing.Value.Value) / outgoing.Value.Value * 100
                };

                // Confidence based on amount match and timing
                var amountMatch = 1.0 - Math.Abs(returning.Value.Value - outgoing.Value.Value) /
                                       outgoing.Value.Value;
                var timingFactor = 1.0 - (returning.TimestampNanos - outgoing.TimestampNanos) /
                                        (double)WindowSizeNanos;
                var confidence = (amountMatch + timingFactor) / 2.0;

                matches.Add(CreateMatch(new[] { outgoing, returning }, confidence, metadata));
            }
        }

        return matches;
    }
}
