using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Runtime.Temporal.Patterns;

/// <summary>
/// Detects rapid transaction splitting pattern (potential money laundering).
/// </summary>
/// <remarks>
/// <para>
/// Pattern: An account receives money and quickly splits it into multiple outbound transactions.
/// </para>
/// <para>
/// Example: A→B ($1000), then B→C ($500) and B→D ($500) within 5 seconds.
/// This is suspicious because normal transactions don't split that quickly.
/// </para>
/// </remarks>
public sealed class RapidSplitPattern : TemporalPatternBase
{
    private readonly int _minimumSplits;
    private readonly double _minimumAmount;

    public override string PatternId => "rapid-split";
    public override string Name => "Rapid Transaction Split";
    public override string Description =>
        "Detects when an account receives money and quickly splits it to multiple recipients";

    public override long WindowSizeNanos { get; }
    public override PatternSeverity Severity => PatternSeverity.High;

    /// <summary>
    /// Creates a new rapid split pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window for detection (default: 5 seconds)</param>
    /// <param name="minimumSplits">Minimum number of outbound transactions to trigger (default: 2)</param>
    /// <param name="minimumAmount">Minimum transaction amount to consider (default: 1000)</param>
    public RapidSplitPattern(
        long windowSizeNanos = 5_000_000_000,
        int minimumSplits = 2,
        double minimumAmount = 1000.0)
    {
        WindowSizeNanos = windowSizeNanos;
        _minimumSplits = minimumSplits;
        _minimumAmount = minimumAmount;
    }

    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask; // No async work, but maintain interface

        var matches = new List<PatternMatch>();

        // Filter to transaction events only
        var transactions = FilterByType(events, "transaction").ToList();
        if (transactions.Count < _minimumSplits + 1)
            return matches;

        // Group by target (inbound transactions)
        var inboundByTarget = GroupByTarget(transactions);

        foreach (var targetGroup in inboundByTarget)
        {
            var targetAccount = targetGroup.Key;

            // Check each inbound transaction
            foreach (var inbound in targetGroup)
            {
                if (!inbound.Value.HasValue || inbound.Value < _minimumAmount)
                    continue;

                var inboundTime = inbound.TimestampNanos;

                // Find outbound transactions from this account within window
                var outbounds = transactions
                    .Where(t => t.SourceId == targetAccount)
                    .Where(t => t.TimestampNanos > inboundTime)
                    .Where(t => t.TimestampNanos - inboundTime <= WindowSizeNanos)
                    .Where(t => t.Value.HasValue && t.Value >= _minimumAmount / 10) // Consider smaller splits
                    .ToList();

                if (outbounds.Count >= _minimumSplits)
                {
                    var involvedEvents = new List<TemporalEvent> { inbound };
                    involvedEvents.AddRange(outbounds);

                    var totalOutbound = outbounds.Sum(t => t.Value ?? 0);
                    var splitRatio = totalOutbound / inbound.Value.Value;

                    // Calculate confidence based on how quickly splits occurred and amount ratio
                    var maxDelay = outbounds.Max(t => t.TimestampNanos - inboundTime);
                    var delayFactor = 1.0 - (maxDelay / (double)WindowSizeNanos);
                    var ratioFactor = Math.Min(splitRatio, 1.0);
                    var confidence = (delayFactor + ratioFactor) / 2.0;

                    var metadata = new Dictionary<string, object>
                    {
                        ["account"] = targetAccount,
                        ["inbound_amount"] = inbound.Value.Value,
                        ["outbound_count"] = outbounds.Count,
                        ["total_outbound"] = totalOutbound,
                        ["split_ratio"] = splitRatio,
                        ["max_delay_ms"] = maxDelay / 1_000_000.0
                    };

                    matches.Add(CreateMatch(involvedEvents, confidence, metadata));
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects circular money flow (potential money laundering).
/// </summary>
/// <remarks>
/// <para>
/// Pattern: Money travels in a circle back to the origin account.
/// Example: A→B→C→A
/// </para>
/// </remarks>
public sealed class CircularFlowPattern : TemporalPatternBase
{
    private readonly int _minimumHops;
    private readonly int _maximumHops;

    /// <summary>
    /// Gets the unique identifier for the circular flow pattern.
    /// </summary>
    public override string PatternId => "circular-flow";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Circular Money Flow";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects when money travels in a circle back to the origin account";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (Critical).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.Critical;

    /// <summary>
    /// Creates a new circular flow pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window (default: 60 seconds)</param>
    /// <param name="minimumHops">Minimum path length (default: 3 for A→B→C→A)</param>
    /// <param name="maximumHops">Maximum path length to search (default: 10)</param>
    public CircularFlowPattern(
        long windowSizeNanos = 60_000_000_000,
        int minimumHops = 3,
        int maximumHops = 10)
    {
        WindowSizeNanos = windowSizeNanos;
        _minimumHops = minimumHops;
        _maximumHops = maximumHops;
    }

    /// <summary>
    /// Asynchronously searches for circular money flow patterns in the provided events.
    /// </summary>
    /// <param name="events">The events to analyze</param>
    /// <param name="graph">Temporal graph for path-based pattern matching</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>A collection of pattern matches found</returns>
    public override async Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default)
    {
        await Task.CompletedTask;

        if (graph == null)
            return Enumerable.Empty<PatternMatch>();

        var matches = new List<PatternMatch>();
        var transactions = FilterByType(events, "transaction").ToList();

        if (transactions.Count < _minimumHops)
            return matches;

        // Get all unique accounts involved
        var accounts = transactions
            .Where(t => t.SourceId.HasValue)
            .Select(t => t.SourceId!.Value)
            .Distinct()
            .ToList();

        // For each account, check if there's a circular path
        foreach (var account in accounts)
        {
            var paths = graph.FindTemporalPaths(
                startNode: account,
                endNode: account,
                maxTimeSpanNanos: WindowSizeNanos,
                maxPathLength: _maximumHops);

            foreach (var path in paths)
            {
                if (path.Length >= _minimumHops)
                {
                    // Find events corresponding to this path
                    var involvedEvents = new List<TemporalEvent>();

                    foreach (var edge in path.Edges)
                    {
                        var evt = transactions.FirstOrDefault(t =>
                            t.SourceId == edge.SourceId &&
                            t.TargetId == edge.TargetId &&
                            Math.Abs(t.TimestampNanos - edge.ValidFrom) < 1_000_000); // 1ms tolerance

                        if (evt != null)
                            involvedEvents.Add(evt);
                    }

                    if (involvedEvents.Count >= _minimumHops)
                    {
                        var metadata = new Dictionary<string, object>
                        {
                            ["origin_account"] = account,
                            ["path_length"] = path.Length,
                            ["total_amount"] = path.TotalWeight,
                            ["path_nodes"] = string.Join("→", path.GetNodes())
                        };

                        // Higher confidence for shorter circles
                        var confidence = 1.0 - (path.Length / (double)_maximumHops) * 0.5;

                        matches.Add(CreateMatch(involvedEvents, confidence, metadata));
                    }
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects high-frequency transaction patterns from a single source.
/// </summary>
/// <remarks>
/// <para>
/// Pattern: Many transactions from the same source in a short time.
/// May indicate automated trading, bot activity, or fraud.
/// </para>
/// </remarks>
public sealed class HighFrequencyPattern : TemporalPatternBase
{
    private readonly int _minimumTransactions;
    private readonly double _minimumTotalAmount;

    /// <summary>
    /// Gets the unique identifier for the high-frequency pattern.
    /// </summary>
    public override string PatternId => "high-frequency";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "High-Frequency Transactions";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects unusually high transaction frequency from a single source";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (Medium).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.Medium;

    /// <summary>
    /// Creates a new high-frequency pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window (default: 1 second)</param>
    /// <param name="minimumTransactions">Minimum transaction count to trigger (default: 10)</param>
    /// <param name="minimumTotalAmount">Minimum total amount (default: 10000)</param>
    public HighFrequencyPattern(
        long windowSizeNanos = 1_000_000_000,
        int minimumTransactions = 10,
        double minimumTotalAmount = 10_000.0)
    {
        WindowSizeNanos = windowSizeNanos;
        _minimumTransactions = minimumTransactions;
        _minimumTotalAmount = minimumTotalAmount;
    }

    /// <summary>
    /// Asynchronously searches for high-frequency transaction patterns in the provided events.
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

        if (transactions.Count < _minimumTransactions)
            return matches;

        // Group by source account
        var bySource = GroupBySource(transactions);

        foreach (var sourceGroup in bySource)
        {
            var sourceAccount = sourceGroup.Key;
            var sourceTransactions = sourceGroup.ToList();

            if (sourceTransactions.Count < _minimumTransactions)
                continue;

            // Use sliding window to find bursts
            var sorted = sourceTransactions.OrderBy(t => t.TimestampNanos).ToList();

            for (int i = 0; i < sorted.Count; i++)
            {
                var windowStart = sorted[i].TimestampNanos;
                var windowEnd = windowStart + WindowSizeNanos;

                var windowTransactions = sorted
                    .Skip(i)
                    .TakeWhile(t => t.TimestampNanos <= windowEnd)
                    .ToList();

                if (windowTransactions.Count >= _minimumTransactions)
                {
                    var totalAmount = windowTransactions.Sum(t => t.Value ?? 0);

                    if (totalAmount >= _minimumTotalAmount)
                    {
                        // Calculate transaction rate
                        var duration = windowTransactions.Last().TimestampNanos -
                                     windowTransactions.First().TimestampNanos;
                        var rate = windowTransactions.Count / (duration / 1_000_000_000.0);

                        var metadata = new Dictionary<string, object>
                        {
                            ["source_account"] = sourceAccount,
                            ["transaction_count"] = windowTransactions.Count,
                            ["total_amount"] = totalAmount,
                            ["transactions_per_second"] = rate,
                            ["average_amount"] = totalAmount / windowTransactions.Count
                        };

                        // Higher confidence for more transactions and higher rate
                        var countFactor = Math.Min(windowTransactions.Count / (double)(_minimumTransactions * 2), 1.0);
                        var rateFactor = Math.Min(rate / 100.0, 1.0);
                        var confidence = (countFactor + rateFactor) / 2.0;

                        matches.Add(CreateMatch(windowTransactions, confidence, metadata));

                        // Skip ahead to avoid overlapping matches
                        i += windowTransactions.Count / 2;
                    }
                }
            }
        }

        return matches;
    }
}

/// <summary>
/// Detects sudden velocity changes (large transaction after period of inactivity).
/// </summary>
public sealed class VelocityChangePattern : TemporalPatternBase
{
    private readonly double _thresholdAmount;
    private readonly long _inactivityPeriod;

    /// <summary>
    /// Gets the unique identifier for the velocity change pattern.
    /// </summary>
    public override string PatternId => "velocity-change";

    /// <summary>
    /// Gets the human-readable name of the pattern.
    /// </summary>
    public override string Name => "Transaction Velocity Change";

    /// <summary>
    /// Gets a description of what this pattern detects.
    /// </summary>
    public override string Description =>
        "Detects sudden large transactions after period of inactivity";

    /// <summary>
    /// Gets the time window size for pattern detection in nanoseconds.
    /// </summary>
    public override long WindowSizeNanos { get; }

    /// <summary>
    /// Gets the severity level of this pattern (High).
    /// </summary>
    public override PatternSeverity Severity => PatternSeverity.High;

    /// <summary>
    /// Creates a new velocity change pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Time window for detection (default: 1 hour)</param>
    /// <param name="thresholdAmount">Minimum transaction amount to trigger (default: 50,000)</param>
    /// <param name="inactivityPeriod">Minimum inactivity period before large transaction (default: 30 minutes)</param>
    public VelocityChangePattern(
        long windowSizeNanos = 3600_000_000_000, // 1 hour
        double thresholdAmount = 50_000.0,
        long inactivityPeriod = 1800_000_000_000) // 30 minutes
    {
        WindowSizeNanos = windowSizeNanos;
        _thresholdAmount = thresholdAmount;
        _inactivityPeriod = inactivityPeriod;
    }

    /// <summary>
    /// Asynchronously searches for velocity change patterns in the provided events.
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
        var transactions = FilterByType(events, "transaction")
            .OrderBy(t => t.TimestampNanos)
            .ToList();

        var bySource = GroupBySource(transactions);

        foreach (var sourceGroup in bySource)
        {
            var sorted = sourceGroup.OrderBy(t => t.TimestampNanos).ToList();

            for (int i = 1; i < sorted.Count; i++)
            {
                var current = sorted[i];
                var previous = sorted[i - 1];

                var timeSinceLast = current.TimestampNanos - previous.TimestampNanos;

                // Check for inactivity period followed by large transaction
                if (timeSinceLast >= _inactivityPeriod &&
                    current.Value >= _thresholdAmount)
                {
                    var metadata = new Dictionary<string, object>
                    {
                        ["source_account"] = current.SourceId!.Value,
                        ["inactivity_seconds"] = timeSinceLast / 1_000_000_000.0,
                        ["transaction_amount"] = current.Value!.Value,
                        ["previous_amount"] = previous.Value ?? 0
                    };

                    var confidence = Math.Min(
                        (timeSinceLast / (double)_inactivityPeriod) * 0.5 +
                        (current.Value!.Value / _thresholdAmount) * 0.5,
                        1.0);

                    matches.Add(CreateMatch(new[] { previous, current }, confidence, metadata));
                }
            }
        }

        return matches;
    }
}
