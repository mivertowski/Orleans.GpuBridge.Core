using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Orleans.GpuBridge.Resilience.Fallback;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Collects and maintains metrics for fallback chain operations
/// </summary>
public sealed class FallbackMetricsCollector : IDisposable
{
    private readonly ConcurrentDictionary<FallbackLevel, LevelMetrics> _levelMetrics;
    private readonly ConcurrentQueue<MetricEntry> _recentEntries;
    private readonly object _lock = new();
    private readonly TimeSpan _retentionPeriod;
    private DateTimeOffset _lastCleanup;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="FallbackMetricsCollector"/> class.
    /// </summary>
    /// <param name="retentionPeriod">Optional retention period for metrics. Defaults to 1 hour.</param>
    public FallbackMetricsCollector(TimeSpan? retentionPeriod = null)
    {
        _retentionPeriod = retentionPeriod ?? TimeSpan.FromHours(1);
        _levelMetrics = new ConcurrentDictionary<FallbackLevel, LevelMetrics>();
        _recentEntries = new ConcurrentQueue<MetricEntry>();
        _lastCleanup = DateTimeOffset.UtcNow;
        
        // Initialize metrics for all levels
        foreach (FallbackLevel level in Enum.GetValues<FallbackLevel>())
        {
            _levelMetrics[level] = new LevelMetrics();
        }
    }

    /// <summary>
    /// Records a successful execution
    /// </summary>
    public void RecordSuccess(FallbackLevel level, TimeSpan duration)
    {
        if (_disposed) return;
        
        var timestamp = DateTimeOffset.UtcNow;
        var entry = new MetricEntry(timestamp, level, MetricType.Success, duration, null, null);
        
        _recentEntries.Enqueue(entry);
        
        _levelMetrics.AddOrUpdate(level, 
            new LevelMetrics().RecordSuccess(duration),
            (_, existing) => existing.RecordSuccess(duration));
        
        CleanupIfNeeded();
    }

    /// <summary>
    /// Records a failed execution
    /// </summary>
    public void RecordFailure(FallbackLevel level, Exception exception)
    {
        if (_disposed) return;
        
        var timestamp = DateTimeOffset.UtcNow;
        var entry = new MetricEntry(timestamp, level, MetricType.Failure, TimeSpan.Zero, exception.GetType().Name, null);
        
        _recentEntries.Enqueue(entry);
        
        _levelMetrics.AddOrUpdate(level, 
            new LevelMetrics().RecordFailure(exception),
            (_, existing) => existing.RecordFailure(exception));
        
        CleanupIfNeeded();
    }

    /// <summary>
    /// Records a total chain failure
    /// </summary>
    public void RecordTotalFailure(TimeSpan duration)
    {
        if (_disposed) return;
        
        var timestamp = DateTimeOffset.UtcNow;
        var entry = new MetricEntry(timestamp, FallbackLevel.Failed, MetricType.TotalFailure, duration, null, null);
        
        _recentEntries.Enqueue(entry);
        CleanupIfNeeded();
    }

    /// <summary>
    /// Records a degradation event
    /// </summary>
    public void RecordDegradation(FallbackLevel fromLevel, FallbackLevel toLevel, string reason)
    {
        if (_disposed) return;
        
        var timestamp = DateTimeOffset.UtcNow;
        var entry = new MetricEntry(timestamp, fromLevel, MetricType.Degradation, TimeSpan.Zero, reason, toLevel);
        
        _recentEntries.Enqueue(entry);
        CleanupIfNeeded();
    }

    /// <summary>
    /// Records a recovery event
    /// </summary>
    public void RecordRecovery(FallbackLevel fromLevel, FallbackLevel toLevel)
    {
        if (_disposed) return;
        
        var timestamp = DateTimeOffset.UtcNow;
        var entry = new MetricEntry(timestamp, fromLevel, MetricType.Recovery, TimeSpan.Zero, null, toLevel);
        
        _recentEntries.Enqueue(entry);
        CleanupIfNeeded();
    }

    /// <summary>
    /// Gets comprehensive fallback chain metrics
    /// </summary>
    public FallbackChainMetrics GetMetrics(FallbackLevel currentLevel)
    {
        CleanupIfNeeded();
        
        var levelMetrics = new Dictionary<FallbackLevel, FallbackLevelMetrics>();
        
        foreach (var kvp in _levelMetrics)
        {
            var metrics = kvp.Value;
            levelMetrics[kvp.Key] = new FallbackLevelMetrics(
                Level: kvp.Key,
                TotalRequests: metrics.TotalRequests,
                SuccessCount: metrics.SuccessCount,
                FailureCount: metrics.FailureCount,
                ErrorRate: metrics.ErrorRate,
                AverageLatency: metrics.AverageLatency,
                LastUsed: metrics.LastUsed);
        }
        
        var recentEvents = GetRecentEvents(TimeSpan.FromMinutes(30));
        
        return new FallbackChainMetrics(
            CurrentLevel: currentLevel,
            LevelMetrics: levelMetrics,
            RecentEvents: recentEvents,
            TotalDegradations: recentEvents.Count(e => e.Type == MetricType.Degradation),
            TotalRecoveries: recentEvents.Count(e => e.Type == MetricType.Recovery),
            LastCleanup: _lastCleanup);
    }

    /// <summary>
    /// Gets metrics for a specific level
    /// </summary>
    public FallbackLevelMetrics GetLevelMetrics(FallbackLevel level)
    {
        if (!_levelMetrics.TryGetValue(level, out var metrics))
        {
            return new FallbackLevelMetrics(level, 0, 0, 0, 0.0, TimeSpan.Zero, null);
        }
        
        return new FallbackLevelMetrics(
            Level: level,
            TotalRequests: metrics.TotalRequests,
            SuccessCount: metrics.SuccessCount,
            FailureCount: metrics.FailureCount,
            ErrorRate: metrics.ErrorRate,
            AverageLatency: metrics.AverageLatency,
            LastUsed: metrics.LastUsed);
    }

    /// <summary>
    /// Gets recent events within the specified time window
    /// </summary>
    private List<FallbackEvent> GetRecentEvents(TimeSpan timeWindow)
    {
        var cutoff = DateTimeOffset.UtcNow - timeWindow;
        var events = new List<FallbackEvent>();
        
        foreach (var entry in _recentEntries.Where(e => e.Timestamp >= cutoff))
        {
            events.Add(new FallbackEvent(
                Timestamp: entry.Timestamp,
                Level: entry.Level,
                Type: entry.Type,
                Duration: entry.Duration,
                Details: entry.Details,
                TargetLevel: entry.TargetLevel));
        }
        
        return events.OrderByDescending(e => e.Timestamp).ToList();
    }

    /// <summary>
    /// Cleans up old entries if needed
    /// </summary>
    private void CleanupIfNeeded()
    {
        if (DateTimeOffset.UtcNow - _lastCleanup < TimeSpan.FromMinutes(10)) return;
        
        lock (_lock)
        {
            if (DateTimeOffset.UtcNow - _lastCleanup < TimeSpan.FromMinutes(10)) return;
            
            var cutoff = DateTimeOffset.UtcNow - _retentionPeriod;
            
            // Clean up old entries
            var entriesToKeep = new List<MetricEntry>();
            while (_recentEntries.TryDequeue(out var entry))
            {
                if (entry.Timestamp >= cutoff)
                {
                    entriesToKeep.Add(entry);
                }
            }
            
            // Re-enqueue entries to keep
            foreach (var entry in entriesToKeep)
            {
                _recentEntries.Enqueue(entry);
            }
            
            _lastCleanup = DateTimeOffset.UtcNow;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;

        _disposed = true;
        _levelMetrics.Clear();
        
        while (_recentEntries.TryDequeue(out _)) { }
        
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Thread-safe metrics for a specific fallback level
/// </summary>
internal sealed class LevelMetrics
{
    private long _totalRequests;
    private long _successCount;
    private long _failureCount;
    private long _totalLatencyTicks;
    private DateTimeOffset _lastUsed;

    public long TotalRequests => _totalRequests;
    public long SuccessCount => _successCount;
    public long FailureCount => _failureCount;
    public double ErrorRate => TotalRequests == 0 ? 0.0 : (double)FailureCount / TotalRequests;
    public TimeSpan AverageLatency => SuccessCount == 0 ? TimeSpan.Zero : new TimeSpan(_totalLatencyTicks / SuccessCount);
    public DateTimeOffset? LastUsed => _lastUsed == default ? null : _lastUsed;

    public LevelMetrics RecordSuccess(TimeSpan duration)
    {
        Interlocked.Increment(ref _totalRequests);
        Interlocked.Increment(ref _successCount);
        Interlocked.Add(ref _totalLatencyTicks, duration.Ticks);
        _lastUsed = DateTimeOffset.UtcNow;
        return this;
    }

    public LevelMetrics RecordFailure(Exception exception)
    {
        Interlocked.Increment(ref _totalRequests);
        Interlocked.Increment(ref _failureCount);
        _lastUsed = DateTimeOffset.UtcNow;
        return this;
    }
}

/// <summary>
/// Individual metric entry
/// </summary>
internal readonly record struct MetricEntry(
    DateTimeOffset Timestamp,
    FallbackLevel Level,
    MetricType Type,
    TimeSpan Duration,
    string? Details,
    FallbackLevel? TargetLevel);

/// <summary>
/// Types of metrics for fallback operations.
/// </summary>
public enum MetricType
{
    /// <summary>Operation completed successfully.</summary>
    Success,
    /// <summary>Operation failed at current level.</summary>
    Failure,
    /// <summary>Operation failed at all fallback levels.</summary>
    TotalFailure,
    /// <summary>System degraded to a lower fallback level.</summary>
    Degradation,
    /// <summary>System recovered to a higher level.</summary>
    Recovery
}

/// <summary>
/// Comprehensive fallback chain metrics
/// </summary>
public readonly record struct FallbackChainMetrics(
    FallbackLevel CurrentLevel,
    IReadOnlyDictionary<FallbackLevel, FallbackLevelMetrics> LevelMetrics,
    IReadOnlyList<FallbackEvent> RecentEvents,
    int TotalDegradations,
    int TotalRecoveries,
    DateTimeOffset LastCleanup);

/// <summary>
/// Metrics for a specific fallback level
/// </summary>
public readonly record struct FallbackLevelMetrics(
    FallbackLevel Level,
    long TotalRequests,
    long SuccessCount,
    long FailureCount,
    double ErrorRate,
    TimeSpan AverageLatency,
    DateTimeOffset? LastUsed);

/// <summary>
/// Fallback event information
/// </summary>
public readonly record struct FallbackEvent(
    DateTimeOffset Timestamp,
    FallbackLevel Level,
    MetricType Type,
    TimeSpan Duration,
    string? Details,
    FallbackLevel? TargetLevel);