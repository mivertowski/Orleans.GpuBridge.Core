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
/// Fluent DSL for defining temporal patterns.
/// </summary>
/// <remarks>
/// <para>
/// Provides a builder pattern for creating custom temporal patterns without
/// implementing the full ITemporalPattern interface.
/// </para>
/// <para>
/// Example usage:
/// <code>
/// var pattern = PatternBuilder.Create("my-pattern")
///     .Named("My Custom Pattern")
///     .WithWindowSize(TimeSpan.FromSeconds(5))
///     .WithSeverity(PatternSeverity.High)
///     .FilterEvents(e => e.EventType == "transaction")
///     .GroupBy(e => e.SourceId)
///     .MatchWhen((group, events) => group.Count() >= 3)
///     .WithConfidence((group, events) => 0.9)
///     .Build();
/// </code>
/// </para>
/// </remarks>
public sealed class PatternBuilder
{
    private string _patternId = string.Empty;
    private string _name = string.Empty;
    private string _description = string.Empty;
    private long _windowSizeNanos;
    private PatternSeverity _severity = PatternSeverity.Medium;
    private Func<TemporalEvent, bool>? _eventFilter;
    private Func<TemporalEvent, object>? _groupBySelector;
    private Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, bool>? _matchPredicate;
    private Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, double>? _confidenceCalculator;
    private Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, Dictionary<string, object>>? _metadataBuilder;

    private PatternBuilder() { }

    /// <summary>
    /// Creates a new pattern builder with the specified ID.
    /// </summary>
    public static PatternBuilder Create(string patternId)
    {
        ArgumentException.ThrowIfNullOrEmpty(patternId);
        return new PatternBuilder { _patternId = patternId };
    }

    /// <summary>
    /// Sets the pattern name.
    /// </summary>
    public PatternBuilder Named(string name)
    {
        _name = name ?? throw new ArgumentNullException(nameof(name));
        return this;
    }

    /// <summary>
    /// Sets the pattern description.
    /// </summary>
    public PatternBuilder WithDescription(string description)
    {
        _description = description ?? throw new ArgumentNullException(nameof(description));
        return this;
    }

    /// <summary>
    /// Sets the time window size.
    /// </summary>
    public PatternBuilder WithWindowSize(TimeSpan window)
    {
        if (window <= TimeSpan.Zero)
            throw new ArgumentException("Window must be positive", nameof(window));
        _windowSizeNanos = (long)(window.TotalMilliseconds * 1_000_000);
        return this;
    }

    /// <summary>
    /// Sets the time window size in nanoseconds.
    /// </summary>
    public PatternBuilder WithWindowSizeNanos(long windowSizeNanos)
    {
        if (windowSizeNanos <= 0)
            throw new ArgumentException("Window must be positive", nameof(windowSizeNanos));
        _windowSizeNanos = windowSizeNanos;
        return this;
    }

    /// <summary>
    /// Sets the pattern severity.
    /// </summary>
    public PatternBuilder WithSeverity(PatternSeverity severity)
    {
        _severity = severity;
        return this;
    }

    /// <summary>
    /// Filters events to consider.
    /// </summary>
    public PatternBuilder FilterEvents(Func<TemporalEvent, bool> filter)
    {
        _eventFilter = filter ?? throw new ArgumentNullException(nameof(filter));
        return this;
    }

    /// <summary>
    /// Filters events by type.
    /// </summary>
    public PatternBuilder ForEventType(string eventType)
    {
        _eventFilter = e => e.EventType == eventType;
        return this;
    }

    /// <summary>
    /// Groups events for pattern matching.
    /// </summary>
    public PatternBuilder GroupBy(Func<TemporalEvent, object> selector)
    {
        _groupBySelector = selector ?? throw new ArgumentNullException(nameof(selector));
        return this;
    }

    /// <summary>
    /// Groups events by source ID.
    /// </summary>
    public PatternBuilder GroupBySource()
    {
        _groupBySelector = e => e.SourceId ?? 0UL;
        return this;
    }

    /// <summary>
    /// Groups events by target ID.
    /// </summary>
    public PatternBuilder GroupByTarget()
    {
        _groupBySelector = e => e.TargetId ?? 0UL;
        return this;
    }

    /// <summary>
    /// Sets the match condition.
    /// </summary>
    public PatternBuilder MatchWhen(
        Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, bool> predicate)
    {
        _matchPredicate = predicate ?? throw new ArgumentNullException(nameof(predicate));
        return this;
    }

    /// <summary>
    /// Sets the match condition (simplified version).
    /// </summary>
    public PatternBuilder MatchWhen(Func<IEnumerable<TemporalEvent>, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        _matchPredicate = (group, all) => predicate(group);
        return this;
    }

    /// <summary>
    /// Matches when group has at least the specified number of events.
    /// </summary>
    public PatternBuilder MatchWhenCount(int minimumCount)
    {
        if (minimumCount < 1)
            throw new ArgumentException("Minimum count must be at least 1", nameof(minimumCount));
        _matchPredicate = (group, all) => group.Count() >= minimumCount;
        return this;
    }

    /// <summary>
    /// Sets the confidence calculator.
    /// </summary>
    public PatternBuilder WithConfidence(
        Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, double> calculator)
    {
        _confidenceCalculator = calculator ?? throw new ArgumentNullException(nameof(calculator));
        return this;
    }

    /// <summary>
    /// Sets a fixed confidence level.
    /// </summary>
    public PatternBuilder WithConfidence(double confidence)
    {
        if (confidence < 0 || confidence > 1)
            throw new ArgumentException("Confidence must be between 0 and 1", nameof(confidence));
        _confidenceCalculator = (_, _) => confidence;
        return this;
    }

    /// <summary>
    /// Sets the metadata builder.
    /// </summary>
    public PatternBuilder WithMetadata(
        Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, Dictionary<string, object>> builder)
    {
        _metadataBuilder = builder ?? throw new ArgumentNullException(nameof(builder));
        return this;
    }

    /// <summary>
    /// Builds the pattern.
    /// </summary>
    public ITemporalPattern Build()
    {
        if (string.IsNullOrEmpty(_patternId))
            throw new InvalidOperationException("Pattern ID is required");
        if (string.IsNullOrEmpty(_name))
            _name = _patternId;
        if (_windowSizeNanos <= 0)
            throw new InvalidOperationException("Window size must be set");

        return new DslPattern(
            _patternId,
            _name,
            _description,
            _windowSizeNanos,
            _severity,
            _eventFilter,
            _groupBySelector,
            _matchPredicate ?? ((_, _) => true),
            _confidenceCalculator ?? ((_, _) => 1.0),
            _metadataBuilder);
    }

    /// <summary>
    /// Internal pattern implementation created by the DSL.
    /// </summary>
    private sealed class DslPattern : ITemporalPattern
    {
        private readonly Func<TemporalEvent, bool>? _eventFilter;
        private readonly Func<TemporalEvent, object>? _groupBySelector;
        private readonly Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, bool> _matchPredicate;
        private readonly Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, double> _confidenceCalculator;
        private readonly Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, Dictionary<string, object>>? _metadataBuilder;

        public string PatternId { get; }
        public string Name { get; }
        public string Description { get; }
        public long WindowSizeNanos { get; }
        public PatternSeverity Severity { get; }

        public DslPattern(
            string patternId,
            string name,
            string description,
            long windowSizeNanos,
            PatternSeverity severity,
            Func<TemporalEvent, bool>? eventFilter,
            Func<TemporalEvent, object>? groupBySelector,
            Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, bool> matchPredicate,
            Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, double> confidenceCalculator,
            Func<IGrouping<object, TemporalEvent>, IReadOnlyList<TemporalEvent>, Dictionary<string, object>>? metadataBuilder)
        {
            PatternId = patternId;
            Name = name;
            Description = description;
            WindowSizeNanos = windowSizeNanos;
            Severity = severity;
            _eventFilter = eventFilter;
            _groupBySelector = groupBySelector;
            _matchPredicate = matchPredicate;
            _confidenceCalculator = confidenceCalculator;
            _metadataBuilder = metadataBuilder;
        }

        public async Task<IEnumerable<PatternMatch>> MatchAsync(
            IReadOnlyList<TemporalEvent> events,
            TemporalGraphStorage? graph,
            CancellationToken ct = default)
        {
            await Task.CompletedTask;

            var matches = new List<PatternMatch>();

            // Filter events
            var filtered = _eventFilter != null
                ? events.Where(_eventFilter).ToList()
                : events.ToList();

            if (filtered.Count == 0)
                return matches;

            // Group if selector provided
            if (_groupBySelector != null)
            {
                var groups = filtered.GroupBy(_groupBySelector);

                foreach (var group in groups)
                {
                    if (_matchPredicate(group, events))
                    {
                        var groupEvents = group.ToList();
                        var confidence = _confidenceCalculator(group, events);
                        var metadata = _metadataBuilder?.Invoke(group, events);

                        matches.Add(CreateMatch(groupEvents, confidence, metadata));
                    }
                }
            }
            else
            {
                // No grouping - check all filtered events as one group
                var pseudoGroup = filtered.GroupBy(_ => (object)0UL).First();

                if (_matchPredicate(pseudoGroup, events))
                {
                    var confidence = _confidenceCalculator(pseudoGroup, events);
                    var metadata = _metadataBuilder?.Invoke(pseudoGroup, events);

                    matches.Add(CreateMatch(filtered, confidence, metadata));
                }
            }

            return matches;
        }

        private PatternMatch CreateMatch(
            IReadOnlyList<TemporalEvent> involvedEvents,
            double confidence,
            Dictionary<string, object>? metadata)
        {
            if (involvedEvents.Count == 0)
                throw new ArgumentException("At least one event required");

            var windowStart = involvedEvents.Min(e => e.TimestampNanos);
            var windowEnd = involvedEvents.Max(e => e.TimestampNanos);

            return new PatternMatch
            {
                PatternId = PatternId,
                PatternName = Name,
                DetectionTimeNanos = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds(),
                WindowStartNanos = windowStart,
                WindowEndNanos = windowEnd,
                InvolvedEvents = involvedEvents,
                Confidence = confidence,
                Severity = Severity,
                Metadata = metadata ?? new Dictionary<string, object>()
            };
        }
    }
}

/// <summary>
/// Extension methods for pattern configuration.
/// </summary>
public static class PatternBuilderExtensions
{
    /// <summary>
    /// Creates a pattern that matches high-frequency events.
    /// </summary>
    public static PatternBuilder ForHighFrequency(
        this PatternBuilder builder,
        int minimumCount,
        TimeSpan window)
    {
        return builder
            .WithWindowSize(window)
            .GroupBySource()
            .MatchWhenCount(minimumCount)
            .WithConfidence((group, all) =>
                Math.Min(group.Count() / (minimumCount * 2.0), 1.0));
    }

    /// <summary>
    /// Creates a pattern that matches value thresholds.
    /// </summary>
    public static PatternBuilder WithValueThreshold(
        this PatternBuilder builder,
        double minimumValue)
    {
        return builder.FilterEvents(e => e.Value.HasValue && e.Value >= minimumValue);
    }
}

/// <summary>
/// Pre-built pattern templates.
/// </summary>
public static class PatternTemplates
{
    /// <summary>
    /// Creates a pattern for detecting burst activity.
    /// </summary>
    public static ITemporalPattern BurstActivity(int threshold, TimeSpan window)
    {
        return PatternBuilder.Create("burst-activity")
            .Named("Burst Activity")
            .WithDescription($"Detects {threshold}+ events within {window.TotalSeconds}s")
            .WithWindowSize(window)
            .WithSeverity(PatternSeverity.Medium)
            .GroupBySource()
            .MatchWhenCount(threshold)
            .WithConfidence((group, all) =>
                Math.Min(group.Count() / (threshold * 2.0), 1.0))
            .WithMetadata((group, all) => new Dictionary<string, object>
            {
                ["event_count"] = group.Count(),
                ["source"] = group.Key
            })
            .Build();
    }

    /// <summary>
    /// Creates a pattern for detecting large transactions.
    /// </summary>
    public static ITemporalPattern LargeTransaction(double threshold)
    {
        return PatternBuilder.Create("large-transaction")
            .Named("Large Transaction")
            .WithDescription($"Detects transactions over ${threshold:N0}")
            .WithWindowSizeNanos(1_000_000_000) // 1 second
            .WithSeverity(PatternSeverity.Low)
            .ForEventType("transaction")
            .FilterEvents(e => e.Value.HasValue && e.Value >= threshold)
            .MatchWhen(events => events.Any())
            .WithConfidence((group, all) =>
            {
                var maxValue = group.Max(e => e.Value ?? 0);
                return Math.Min(maxValue / (threshold * 10), 1.0);
            })
            .Build();
    }

    /// <summary>
    /// Creates a pattern for detecting repeated targets.
    /// </summary>
    public static ITemporalPattern RepeatedTarget(int threshold, TimeSpan window)
    {
        return PatternBuilder.Create("repeated-target")
            .Named("Repeated Target")
            .WithDescription($"Detects {threshold}+ sends to same target within {window.TotalSeconds}s")
            .WithWindowSize(window)
            .WithSeverity(PatternSeverity.Medium)
            .ForEventType("transaction")
            .GroupByTarget()
            .MatchWhenCount(threshold)
            .Build();
    }
}
