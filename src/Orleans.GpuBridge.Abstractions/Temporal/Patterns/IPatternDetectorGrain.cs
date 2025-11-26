// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Orleans;

namespace Orleans.GpuBridge.Abstractions.Temporal.Patterns;

/// <summary>
/// Orleans grain interface for distributed temporal pattern detection.
/// </summary>
/// <remarks>
/// <para>
/// This grain provides distributed, stateful pattern detection with:
/// <list type="bullet">
/// <item><description>Sliding time window management</description></item>
/// <item><description>Multiple pattern registration</description></item>
/// <item><description>GPU-accelerated pattern matching (when available)</description></item>
/// <item><description>Pattern match persistence and deduplication</description></item>
/// <item><description>Real-time statistics tracking</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Use Cases:</strong>
/// <list type="bullet">
/// <item><description>Fraud detection in financial transactions</description></item>
/// <item><description>Security monitoring and anomaly detection</description></item>
/// <item><description>IoT sensor pattern analysis</description></item>
/// <item><description>Network traffic pattern detection</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IPatternDetectorGrain : IGrainWithStringKey
{
    /// <summary>
    /// Registers a pattern for detection using pattern configuration.
    /// </summary>
    /// <param name="config">Pattern configuration</param>
    /// <returns>True if registration succeeded</returns>
    Task<bool> RegisterPatternAsync(PatternConfiguration config);

    /// <summary>
    /// Unregisters a pattern.
    /// </summary>
    /// <param name="patternId">Pattern identifier</param>
    /// <returns>True if pattern was removed</returns>
    Task<bool> UnregisterPatternAsync(string patternId);

    /// <summary>
    /// Gets all registered patterns.
    /// </summary>
    /// <returns>List of registered pattern configurations</returns>
    Task<IReadOnlyList<PatternConfiguration>> GetRegisteredPatternsAsync();

    /// <summary>
    /// Processes a single event and returns any triggered matches.
    /// </summary>
    /// <param name="evt">Event to process</param>
    /// <returns>Pattern matches triggered by this event</returns>
    Task<IReadOnlyList<PatternMatchData>> ProcessEventAsync(TemporalEventData evt);

    /// <summary>
    /// Processes a batch of events and returns all triggered matches.
    /// </summary>
    /// <param name="events">Events to process</param>
    /// <returns>Pattern matches triggered by these events</returns>
    Task<PatternProcessingResult> ProcessEventBatchAsync(IReadOnlyList<TemporalEventData> events);

    /// <summary>
    /// Gets recent pattern matches.
    /// </summary>
    /// <param name="maxResults">Maximum results to return</param>
    /// <param name="sinceNanos">Optional: only get matches after this timestamp</param>
    /// <returns>Recent pattern matches</returns>
    Task<IReadOnlyList<PatternMatchData>> GetRecentMatchesAsync(
        int maxResults = 100,
        long? sinceNanos = null);

    /// <summary>
    /// Gets matches for a specific pattern.
    /// </summary>
    /// <param name="patternId">Pattern identifier</param>
    /// <param name="maxResults">Maximum results to return</param>
    /// <returns>Matches for the specified pattern</returns>
    Task<IReadOnlyList<PatternMatchData>> GetMatchesByPatternAsync(
        string patternId,
        int maxResults = 100);

    /// <summary>
    /// Gets detection statistics.
    /// </summary>
    /// <returns>Current detection statistics</returns>
    Task<PatternDetectionStats> GetStatisticsAsync();

    /// <summary>
    /// Clears all events and matches (resets detector state).
    /// </summary>
    Task ClearAsync();

    /// <summary>
    /// Sets the sliding window configuration.
    /// </summary>
    /// <param name="windowSizeNanos">Window size in nanoseconds</param>
    /// <param name="maxEvents">Maximum events to keep in window</param>
    Task ConfigureWindowAsync(long windowSizeNanos, int maxEvents = 10_000);
}

/// <summary>
/// Configuration for a pattern to register.
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record PatternConfiguration
{
    /// <summary>
    /// Unique pattern identifier.
    /// </summary>
    [Id(0)]
    public required string PatternId { get; init; }

    /// <summary>
    /// Pattern type (built-in pattern class name).
    /// </summary>
    [Id(1)]
    public required string PatternType { get; init; }

    /// <summary>
    /// Human-readable pattern name.
    /// </summary>
    [Id(2)]
    public string Name { get; init; } = string.Empty;

    /// <summary>
    /// Pattern description.
    /// </summary>
    [Id(3)]
    public string Description { get; init; } = string.Empty;

    /// <summary>
    /// Time window for pattern detection (nanoseconds).
    /// </summary>
    [Id(4)]
    public long WindowSizeNanos { get; init; }

    /// <summary>
    /// Pattern severity level.
    /// </summary>
    [Id(5)]
    public int Severity { get; init; }

    /// <summary>
    /// Pattern-specific parameters.
    /// </summary>
    [Id(6)]
    public IReadOnlyDictionary<string, string>? Parameters { get; init; }

    /// <summary>
    /// Whether pattern is currently enabled.
    /// </summary>
    [Id(7)]
    public bool Enabled { get; init; } = true;
}

/// <summary>
/// Temporal event data for pattern detection.
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record TemporalEventData
{
    /// <summary>
    /// Unique event identifier.
    /// </summary>
    [Id(0)]
    public required Guid EventId { get; init; }

    /// <summary>
    /// Event type (e.g., "transaction", "login", "message").
    /// </summary>
    [Id(1)]
    public required string EventType { get; init; }

    /// <summary>
    /// Event timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(2)]
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Source entity identifier.
    /// </summary>
    [Id(3)]
    public ulong? SourceId { get; init; }

    /// <summary>
    /// Target entity identifier.
    /// </summary>
    [Id(4)]
    public ulong? TargetId { get; init; }

    /// <summary>
    /// Event value (e.g., transaction amount).
    /// </summary>
    [Id(5)]
    public double? Value { get; init; }

    /// <summary>
    /// Additional event properties.
    /// </summary>
    [Id(6)]
    public IReadOnlyDictionary<string, string>? Properties { get; init; }
}

/// <summary>
/// Pattern match data for Orleans serialization.
/// </summary>
[GenerateSerializer]
[Immutable]
public sealed record PatternMatchData
{
    /// <summary>
    /// Pattern that was matched.
    /// </summary>
    [Id(0)]
    public required string PatternId { get; init; }

    /// <summary>
    /// Pattern name.
    /// </summary>
    [Id(1)]
    public required string PatternName { get; init; }

    /// <summary>
    /// When the pattern was detected.
    /// </summary>
    [Id(2)]
    public required long DetectionTimeNanos { get; init; }

    /// <summary>
    /// Start of the pattern window.
    /// </summary>
    [Id(3)]
    public required long WindowStartNanos { get; init; }

    /// <summary>
    /// End of the pattern window.
    /// </summary>
    [Id(4)]
    public required long WindowEndNanos { get; init; }

    /// <summary>
    /// Number of events involved in the pattern.
    /// </summary>
    [Id(5)]
    public int InvolvedEventCount { get; init; }

    /// <summary>
    /// IDs of events involved in the pattern.
    /// </summary>
    [Id(6)]
    public IReadOnlyList<Guid> InvolvedEventIds { get; init; } = Array.Empty<Guid>();

    /// <summary>
    /// Match confidence (0.0 to 1.0).
    /// </summary>
    [Id(7)]
    public double Confidence { get; init; } = 1.0;

    /// <summary>
    /// Severity level.
    /// </summary>
    [Id(8)]
    public int Severity { get; init; }

    /// <summary>
    /// Additional metadata.
    /// </summary>
    [Id(9)]
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }

    /// <summary>
    /// Duration of the pattern in nanoseconds.
    /// </summary>
    public long DurationNanos => WindowEndNanos - WindowStartNanos;
}

/// <summary>
/// Result of processing a batch of events.
/// </summary>
[GenerateSerializer]
public sealed record PatternProcessingResult
{
    /// <summary>
    /// Pattern matches found.
    /// </summary>
    [Id(0)]
    public IReadOnlyList<PatternMatchData> Matches { get; init; } = Array.Empty<PatternMatchData>();

    /// <summary>
    /// Number of events processed.
    /// </summary>
    [Id(1)]
    public int EventsProcessed { get; init; }

    /// <summary>
    /// Number of patterns checked.
    /// </summary>
    [Id(2)]
    public int PatternsChecked { get; init; }

    /// <summary>
    /// Processing duration in nanoseconds.
    /// </summary>
    [Id(3)]
    public long ProcessingTimeNanos { get; init; }

    /// <summary>
    /// Whether GPU acceleration was used.
    /// </summary>
    [Id(4)]
    public bool UsedGpuAcceleration { get; init; }
}

/// <summary>
/// Pattern detection statistics.
/// </summary>
[GenerateSerializer]
public sealed record PatternDetectionStats
{
    /// <summary>
    /// Total events processed.
    /// </summary>
    [Id(0)]
    public long TotalEventsProcessed { get; init; }

    /// <summary>
    /// Total patterns detected.
    /// </summary>
    [Id(1)]
    public long TotalPatternsDetected { get; init; }

    /// <summary>
    /// Total pattern checks performed.
    /// </summary>
    [Id(2)]
    public long TotalPatternChecks { get; init; }

    /// <summary>
    /// Number of registered patterns.
    /// </summary>
    [Id(3)]
    public int RegisteredPatternCount { get; init; }

    /// <summary>
    /// Current events in sliding window.
    /// </summary>
    [Id(4)]
    public int CurrentWindowSize { get; init; }

    /// <summary>
    /// Recent match count.
    /// </summary>
    [Id(5)]
    public int RecentMatchCount { get; init; }

    /// <summary>
    /// Detection rate (matches per check).
    /// </summary>
    [Id(6)]
    public double DetectionRate { get; init; }

    /// <summary>
    /// Average processing time per event (nanoseconds).
    /// </summary>
    [Id(7)]
    public double AverageProcessingTimeNanos { get; init; }

    /// <summary>
    /// GPU utilization percentage.
    /// </summary>
    [Id(8)]
    public double GpuUtilizationPercent { get; init; }

    /// <summary>
    /// Per-pattern statistics.
    /// </summary>
    [Id(9)]
    public IReadOnlyList<PatternStats> PatternStatistics { get; init; } = Array.Empty<PatternStats>();
}

/// <summary>
/// Statistics for a single pattern.
/// </summary>
[GenerateSerializer]
public sealed record PatternStats
{
    /// <summary>
    /// Pattern identifier.
    /// </summary>
    [Id(0)]
    public required string PatternId { get; init; }

    /// <summary>
    /// Pattern name.
    /// </summary>
    [Id(1)]
    public required string PatternName { get; init; }

    /// <summary>
    /// Number of matches.
    /// </summary>
    [Id(2)]
    public int MatchCount { get; init; }

    /// <summary>
    /// Pattern severity.
    /// </summary>
    [Id(3)]
    public int Severity { get; init; }

    /// <summary>
    /// Whether pattern is enabled.
    /// </summary>
    [Id(4)]
    public bool Enabled { get; init; }
}
