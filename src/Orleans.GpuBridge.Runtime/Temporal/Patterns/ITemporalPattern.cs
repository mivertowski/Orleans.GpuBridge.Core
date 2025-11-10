using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Runtime.Temporal.Patterns;

/// <summary>
/// Represents a temporal pattern that can be detected in event streams.
/// </summary>
/// <remarks>
/// <para>
/// Temporal patterns define conditions that must be met within a time window.
/// Examples:
/// - Rapid transaction splitting: A→B followed by B→{C,D} within 5 seconds
/// - Circular flow: A→B→C→A (money returns to origin)
/// - High-frequency trading: >10 transactions from same source in 1 second
/// </para>
/// </remarks>
public interface ITemporalPattern
{
    /// <summary>
    /// Unique identifier for this pattern.
    /// </summary>
    string PatternId { get; }

    /// <summary>
    /// Human-readable name of the pattern.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Description of what this pattern detects.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Time window size for pattern matching (in nanoseconds).
    /// </summary>
    long WindowSizeNanos { get; }

    /// <summary>
    /// Severity level of detected patterns.
    /// </summary>
    PatternSeverity Severity { get; }

    /// <summary>
    /// Attempts to match this pattern against a window of events.
    /// </summary>
    /// <param name="events">Events in the current window (time-ordered)</param>
    /// <param name="graph">Temporal graph for path queries (optional)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Match results if pattern detected, empty otherwise</returns>
    Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default);
}

/// <summary>
/// Represents a detected pattern instance.
/// </summary>
public sealed record PatternMatch
{
    /// <summary>
    /// The pattern that was matched.
    /// </summary>
    public required string PatternId { get; init; }

    /// <summary>
    /// Pattern name.
    /// </summary>
    public required string PatternName { get; init; }

    /// <summary>
    /// When the pattern was detected.
    /// </summary>
    public required long DetectionTimeNanos { get; init; }

    /// <summary>
    /// Start time of the pattern window.
    /// </summary>
    public required long WindowStartNanos { get; init; }

    /// <summary>
    /// End time of the pattern window.
    /// </summary>
    public required long WindowEndNanos { get; init; }

    /// <summary>
    /// Events involved in the pattern.
    /// </summary>
    public required IReadOnlyList<TemporalEvent> InvolvedEvents { get; init; }

    /// <summary>
    /// Confidence score (0.0 to 1.0).
    /// </summary>
    public double Confidence { get; init; } = 1.0;

    /// <summary>
    /// Severity level.
    /// </summary>
    public PatternSeverity Severity { get; init; }

    /// <summary>
    /// Additional metadata about the match.
    /// </summary>
    public Dictionary<string, object> Metadata { get; init; } = new();

    /// <summary>
    /// Duration of the pattern in nanoseconds.
    /// </summary>
    public long DurationNanos => WindowEndNanos - WindowStartNanos;

    /// <summary>
    /// Duration in seconds.
    /// </summary>
    public double DurationSeconds => DurationNanos / 1_000_000_000.0;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"PatternMatch({PatternName}, {InvolvedEvents.Count} events, {DurationSeconds:F2}s, confidence={Confidence:F2})";
    }
}

/// <summary>
/// Represents an event in the temporal graph.
/// </summary>
public sealed record TemporalEvent
{
    /// <summary>
    /// Unique event identifier.
    /// </summary>
    public required Guid EventId { get; init; }

    /// <summary>
    /// Event type (e.g., "transaction", "message", "login").
    /// </summary>
    public required string EventType { get; init; }

    /// <summary>
    /// Timestamp when the event occurred.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Source entity (e.g., account ID, user ID).
    /// </summary>
    public ulong? SourceId { get; init; }

    /// <summary>
    /// Target entity (e.g., recipient account, target user).
    /// </summary>
    public ulong? TargetId { get; init; }

    /// <summary>
    /// Event value (e.g., transaction amount).
    /// </summary>
    public double? Value { get; init; }

    /// <summary>
    /// Additional event data.
    /// </summary>
    public Dictionary<string, object> Data { get; init; } = new();

    /// <inheritdoc/>
    public override string ToString()
    {
        var sourceStr = SourceId.HasValue ? $" from {SourceId}" : "";
        var targetStr = TargetId.HasValue ? $" to {TargetId}" : "";
        var valueStr = Value.HasValue ? $" (${Value:F2})" : "";
        return $"{EventType}{sourceStr}{targetStr}{valueStr} at {TimestampNanos}ns";
    }
}

/// <summary>
/// Severity levels for detected patterns.
/// </summary>
public enum PatternSeverity
{
    /// <summary>
    /// Informational - normal behavior, logged for analysis.
    /// </summary>
    Info = 0,

    /// <summary>
    /// Low severity - potentially interesting but not concerning.
    /// </summary>
    Low = 1,

    /// <summary>
    /// Medium severity - unusual behavior, should be reviewed.
    /// </summary>
    Medium = 2,

    /// <summary>
    /// High severity - suspicious behavior, requires investigation.
    /// </summary>
    High = 3,

    /// <summary>
    /// Critical severity - likely fraudulent, immediate action required.
    /// </summary>
    Critical = 4
}

/// <summary>
/// Base class for temporal patterns with common functionality.
/// </summary>
public abstract class TemporalPatternBase : ITemporalPattern
{
    /// <inheritdoc/>
    public abstract string PatternId { get; }

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract string Description { get; }

    /// <inheritdoc/>
    public abstract long WindowSizeNanos { get; }

    /// <inheritdoc/>
    public virtual PatternSeverity Severity => PatternSeverity.Medium;

    /// <inheritdoc/>
    public abstract Task<IEnumerable<PatternMatch>> MatchAsync(
        IReadOnlyList<TemporalEvent> events,
        TemporalGraphStorage? graph,
        CancellationToken ct = default);

    /// <summary>
    /// Helper to create a pattern match result.
    /// </summary>
    protected PatternMatch CreateMatch(
        IReadOnlyList<TemporalEvent> involvedEvents,
        double confidence = 1.0,
        Dictionary<string, object>? metadata = null)
    {
        if (involvedEvents.Count == 0)
            throw new ArgumentException("At least one event required", nameof(involvedEvents));

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

    /// <summary>
    /// Filters events by type.
    /// </summary>
    protected IEnumerable<TemporalEvent> FilterByType(
        IReadOnlyList<TemporalEvent> events,
        string eventType)
    {
        return events.Where(e => e.EventType == eventType);
    }

    /// <summary>
    /// Filters events by time range.
    /// </summary>
    protected IEnumerable<TemporalEvent> FilterByTimeRange(
        IReadOnlyList<TemporalEvent> events,
        long startTimeNanos,
        long endTimeNanos)
    {
        return events.Where(e =>
            e.TimestampNanos >= startTimeNanos &&
            e.TimestampNanos <= endTimeNanos);
    }

    /// <summary>
    /// Groups events by source ID.
    /// </summary>
    protected ILookup<ulong, TemporalEvent> GroupBySource(
        IEnumerable<TemporalEvent> events)
    {
        return events.Where(e => e.SourceId.HasValue)
                     .ToLookup(e => e.SourceId!.Value);
    }

    /// <summary>
    /// Groups events by target ID.
    /// </summary>
    protected ILookup<ulong, TemporalEvent> GroupByTarget(
        IEnumerable<TemporalEvent> events)
    {
        return events.Where(e => e.TargetId.HasValue)
                     .ToLookup(e => e.TargetId!.Value);
    }
}

/// <summary>
/// Extension methods for working with timestamps.
/// </summary>
public static class DateTimeOffsetExtensions
{
    /// <summary>
    /// Converts DateTimeOffset to nanoseconds since Unix epoch.
    /// </summary>
    public static long ToUnixTimeNanoseconds(this DateTimeOffset dateTime)
    {
        var epochTicks = new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero).Ticks;
        var elapsedTicks = dateTime.Ticks - epochTicks;
        return elapsedTicks * 100; // Convert ticks (100ns) to nanoseconds
    }
}
