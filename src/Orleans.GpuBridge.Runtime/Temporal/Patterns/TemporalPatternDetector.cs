using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Runtime.Temporal.Patterns;

/// <summary>
/// Detects temporal patterns in event streams using sliding time windows.
/// </summary>
/// <remarks>
/// <para>
/// The pattern detector maintains a sliding window of recent events and
/// continuously checks registered patterns for matches.
/// </para>
/// <para>
/// Features:
/// - Sliding time window (configurable size)
/// - Multiple pattern matching
/// - Automatic event expiration
/// - Pattern match deduplication
/// - Statistics tracking
/// </para>
/// </remarks>
public sealed class TemporalPatternDetector
{
    private readonly List<ITemporalPattern> _patterns = new();
    private readonly List<TemporalEvent> _eventWindow = new();
    private readonly List<PatternMatch> _recentMatches = new();
    private readonly HashSet<string> _matchedEventSets = new(); // Deduplication
    private readonly TemporalGraphStorage? _graph;
    private readonly ILogger? _logger;

    private readonly long _windowSizeNanos;
    private readonly long _slideIntervalNanos;
    private readonly int _maxWindowEvents;

    // Statistics
    private long _totalEventsProcessed;
    private long _totalPatternsDetected;
    private long _totalPatternChecks;

    /// <summary>
    /// Gets the current number of events in the window.
    /// </summary>
    public int WindowEventCount => _eventWindow.Count;

    /// <summary>
    /// Gets the number of registered patterns.
    /// </summary>
    public int RegisteredPatternCount => _patterns.Count;

    /// <summary>
    /// Gets recent pattern matches.
    /// </summary>
    public IReadOnlyList<PatternMatch> RecentMatches => _recentMatches;

    /// <summary>
    /// Gets the window size in nanoseconds.
    /// </summary>
    public long WindowSizeNanos => _windowSizeNanos;

    /// <summary>
    /// Creates a new temporal pattern detector.
    /// </summary>
    /// <param name="windowSizeNanos">Size of the sliding window in nanoseconds</param>
    /// <param name="slideIntervalNanos">How often to slide the window (0 = slide on every event)</param>
    /// <param name="maxWindowEvents">Maximum events to keep in window (prevents memory issues)</param>
    /// <param name="graph">Optional temporal graph for path-based pattern matching</param>
    /// <param name="logger">Optional logger</param>
    public TemporalPatternDetector(
        long windowSizeNanos = 5_000_000_000, // 5 seconds default
        long slideIntervalNanos = 0,           // Slide on every event
        int maxWindowEvents = 10_000,
        TemporalGraphStorage? graph = null,
        ILogger? logger = null)
    {
        if (windowSizeNanos <= 0)
            throw new ArgumentException("Window size must be positive", nameof(windowSizeNanos));

        _windowSizeNanos = windowSizeNanos;
        _slideIntervalNanos = slideIntervalNanos;
        _maxWindowEvents = maxWindowEvents;
        _graph = graph;
        _logger = logger;
    }

    /// <summary>
    /// Registers a pattern for detection.
    /// </summary>
    public void RegisterPattern(ITemporalPattern pattern)
    {
        ArgumentNullException.ThrowIfNull(pattern);

        if (_patterns.Any(p => p.PatternId == pattern.PatternId))
        {
            throw new InvalidOperationException(
                $"Pattern with ID '{pattern.PatternId}' is already registered");
        }

        _patterns.Add(pattern);
        _logger?.LogInformation(
            "Registered pattern: {PatternName} (ID: {PatternId}, Window: {WindowMs}ms)",
            pattern.Name, pattern.PatternId, pattern.WindowSizeNanos / 1_000_000);
    }

    /// <summary>
    /// Unregisters a pattern.
    /// </summary>
    public bool UnregisterPattern(string patternId)
    {
        var pattern = _patterns.FirstOrDefault(p => p.PatternId == patternId);
        if (pattern == null)
            return false;

        _patterns.Remove(pattern);
        _logger?.LogInformation("Unregistered pattern: {PatternId}", patternId);
        return true;
    }

    /// <summary>
    /// Processes a new event and checks for pattern matches.
    /// </summary>
    public async Task<IEnumerable<PatternMatch>> ProcessEventAsync(
        TemporalEvent evt,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(evt);

        _totalEventsProcessed++;

        // Add event to window
        _eventWindow.Add(evt);

        // Add to graph if available
        if (_graph != null && evt.SourceId.HasValue && evt.TargetId.HasValue)
        {
            var hlc = new Abstractions.Temporal.HybridTimestamp(
                evt.TimestampNanos, 0, 1);

            _graph.AddEdge(
                evt.SourceId.Value,
                evt.TargetId.Value,
                evt.TimestampNanos,
                evt.TimestampNanos + 1_000_000, // 1ms validity
                hlc,
                weight: evt.Value ?? 1.0,
                edgeType: evt.EventType);
        }

        // Remove events outside window
        EvictExpiredEvents(evt.TimestampNanos);

        // Limit window size
        if (_eventWindow.Count > _maxWindowEvents)
        {
            var toRemove = _eventWindow.Count - _maxWindowEvents;
            _eventWindow.RemoveRange(0, toRemove);
            _logger?.LogWarning(
                "Window exceeded max size, removed {Count} oldest events",
                toRemove);
        }

        // Check all patterns
        var matches = new List<PatternMatch>();

        foreach (var pattern in _patterns)
        {
            _totalPatternChecks++;

            try
            {
                var patternMatches = await pattern.MatchAsync(
                    _eventWindow, _graph, ct);

                foreach (var match in patternMatches)
                {
                    // Deduplicate matches
                    var matchKey = GetMatchKey(match);
                    if (!_matchedEventSets.Contains(matchKey))
                    {
                        matches.Add(match);
                        _recentMatches.Add(match);
                        _matchedEventSets.Add(matchKey);
                        _totalPatternsDetected++;

                        _logger?.LogWarning(
                            "Pattern detected: {PatternName} (Severity: {Severity}, Confidence: {Confidence:F2})",
                            match.PatternName, match.Severity, match.Confidence);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex,
                    "Error checking pattern {PatternId}: {Message}",
                    pattern.PatternId, ex.Message);
            }
        }

        // Trim recent matches list
        if (_recentMatches.Count > 1000)
        {
            _recentMatches.RemoveRange(0, _recentMatches.Count - 1000);
        }

        return matches;
    }

    /// <summary>
    /// Removes events that are outside the time window.
    /// </summary>
    private void EvictExpiredEvents(long currentTimeNanos)
    {
        var cutoffTime = currentTimeNanos - _windowSizeNanos;
        var expiredCount = _eventWindow.Count(e => e.TimestampNanos < cutoffTime);

        if (expiredCount > 0)
        {
            _eventWindow.RemoveAll(e => e.TimestampNanos < cutoffTime);
            _logger?.LogTrace("Evicted {Count} expired events", expiredCount);
        }
    }

    /// <summary>
    /// Generates a unique key for pattern match deduplication.
    /// </summary>
    private static string GetMatchKey(PatternMatch match)
    {
        // Create key from pattern ID and involved event IDs
        var eventIds = string.Join(",",
            match.InvolvedEvents.Select(e => e.EventId).OrderBy(id => id));
        return $"{match.PatternId}:{eventIds}";
    }

    /// <summary>
    /// Gets statistics about pattern detection.
    /// </summary>
    public PatternDetectionStatistics GetStatistics()
    {
        var patternStats = _patterns.Select(p => new PatternStatistics
        {
            PatternId = p.PatternId,
            PatternName = p.Name,
            MatchCount = _recentMatches.Count(m => m.PatternId == p.PatternId),
            Severity = p.Severity
        }).ToList();

        return new PatternDetectionStatistics
        {
            TotalEventsProcessed = _totalEventsProcessed,
            TotalPatternsDetected = _totalPatternsDetected,
            TotalPatternChecks = _totalPatternChecks,
            RegisteredPatternCount = _patterns.Count,
            CurrentWindowSize = _eventWindow.Count,
            RecentMatchCount = _recentMatches.Count,
            PatternStatistics = patternStats
        };
    }

    /// <summary>
    /// Clears all events and matches.
    /// </summary>
    public void Clear()
    {
        _eventWindow.Clear();
        _recentMatches.Clear();
        _matchedEventSets.Clear();
        _totalEventsProcessed = 0;
        _totalPatternsDetected = 0;
        _totalPatternChecks = 0;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalPatternDetector({_patterns.Count} patterns, " +
               $"{_eventWindow.Count} events in window, " +
               $"{_totalPatternsDetected} detected)";
    }
}

/// <summary>
/// Statistics about pattern detection.
/// </summary>
public sealed record PatternDetectionStatistics
{
    public long TotalEventsProcessed { get; init; }
    public long TotalPatternsDetected { get; init; }
    public long TotalPatternChecks { get; init; }
    public int RegisteredPatternCount { get; init; }
    public int CurrentWindowSize { get; init; }
    public int RecentMatchCount { get; init; }
    public List<PatternStatistics> PatternStatistics { get; init; } = new();

    public double DetectionRate =>
        TotalPatternChecks > 0 ? (double)TotalPatternsDetected / TotalPatternChecks : 0;

    public override string ToString()
    {
        return $"PatternStats(Processed={TotalEventsProcessed}, " +
               $"Detected={TotalPatternsDetected}, " +
               $"DetectionRate={DetectionRate:P2})";
    }
}

/// <summary>
/// Statistics for a specific pattern.
/// </summary>
public sealed record PatternStatistics
{
    public required string PatternId { get; init; }
    public required string PatternName { get; init; }
    public int MatchCount { get; init; }
    public PatternSeverity Severity { get; init; }
}
