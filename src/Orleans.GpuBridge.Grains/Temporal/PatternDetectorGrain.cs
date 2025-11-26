// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Runtime;
using Orleans.GpuBridge.Abstractions.Temporal.Patterns;
using Orleans.GpuBridge.Runtime.Temporal.Patterns;
using Orleans.GpuBridge.Runtime.Temporal.Graph;

namespace Orleans.GpuBridge.Grains.Temporal;

/// <summary>
/// Orleans grain implementation for distributed temporal pattern detection.
/// </summary>
/// <remarks>
/// <para>
/// This grain maintains a sliding window of events and checks registered
/// patterns for matches. It supports:
/// <list type="bullet">
/// <item><description>Dynamic pattern registration/unregistration</description></item>
/// <item><description>Batch event processing</description></item>
/// <item><description>GPU-accelerated pattern matching (when available)</description></item>
/// <item><description>Persistent state via Orleans storage</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class PatternDetectorGrain : Grain, IPatternDetectorGrain
{
    private readonly ILogger<PatternDetectorGrain> _logger;
    private readonly IPersistentState<PatternDetectorState> _state;

    private TemporalPatternDetector _detector = null!;
    private GpuPatternMatcher? _gpuMatcher;
    private readonly Dictionary<string, ITemporalPattern> _patterns = new();
    private string _detectorId = string.Empty;

    // Default configuration
    private long _windowSizeNanos = 5_000_000_000; // 5 seconds
    private int _maxWindowEvents = 10_000;

    public PatternDetectorGrain(
        ILogger<PatternDetectorGrain> logger,
        [PersistentState("patternDetector", "GpuBridgeStore")]
        IPersistentState<PatternDetectorState> state)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _state = state ?? throw new ArgumentNullException(nameof(state));
    }

    public override async Task OnActivateAsync(CancellationToken cancellationToken)
    {
        _detectorId = this.GetPrimaryKeyString();

        // Initialize detector with default or restored configuration
        if (_state.State.WindowSizeNanos > 0)
            _windowSizeNanos = _state.State.WindowSizeNanos;
        if (_state.State.MaxWindowEvents > 0)
            _maxWindowEvents = _state.State.MaxWindowEvents;

        _detector = new TemporalPatternDetector(
            _windowSizeNanos,
            slideIntervalNanos: 0,
            _maxWindowEvents,
            graph: null,
            logger: _logger);

        // Initialize GPU matcher
        try
        {
            _gpuMatcher = new GpuPatternMatcher(
                new GpuPatternMatcherOptions
                {
                    EnableGpuAcceleration = true,
                    MinEventsForGpu = 1000
                },
                _logger);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to initialize GPU pattern matcher, using CPU only");
        }

        // Restore registered patterns
        if (_state.State.PatternConfigs is { Count: > 0 })
        {
            foreach (var config in _state.State.PatternConfigs)
            {
                if (config.Enabled)
                {
                    try
                    {
                        var pattern = CreatePatternFromConfig(config);
                        if (pattern != null)
                        {
                            _detector.RegisterPattern(pattern);
                            _patterns[config.PatternId] = pattern;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex,
                            "Failed to restore pattern {PatternId}",
                            config.PatternId);
                    }
                }
            }

            _logger.LogInformation(
                "Pattern detector {DetectorId} restored with {PatternCount} patterns",
                _detectorId, _patterns.Count);
        }

        await base.OnActivateAsync(cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<bool> RegisterPatternAsync(PatternConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(config);

        try
        {
            if (_patterns.ContainsKey(config.PatternId))
            {
                _logger.LogWarning(
                    "Pattern {PatternId} already registered",
                    config.PatternId);
                return false;
            }

            var pattern = CreatePatternFromConfig(config);
            if (pattern == null)
            {
                _logger.LogError(
                    "Unknown pattern type: {PatternType}",
                    config.PatternType);
                return false;
            }

            _detector.RegisterPattern(pattern);
            _patterns[config.PatternId] = pattern;

            // Persist configuration
            _state.State.PatternConfigs ??= new List<PatternConfiguration>();
            _state.State.PatternConfigs.Add(config);
            await _state.WriteStateAsync();

            _logger.LogInformation(
                "Registered pattern {PatternName} (ID: {PatternId})",
                config.Name, config.PatternId);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Failed to register pattern {PatternId}",
                config.PatternId);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> UnregisterPatternAsync(string patternId)
    {
        ArgumentException.ThrowIfNullOrEmpty(patternId);

        if (!_patterns.ContainsKey(patternId))
            return false;

        _detector.UnregisterPattern(patternId);
        _patterns.Remove(patternId);

        // Update persisted state
        _state.State.PatternConfigs?.RemoveAll(c => c.PatternId == patternId);
        await _state.WriteStateAsync();

        _logger.LogInformation("Unregistered pattern {PatternId}", patternId);
        return true;
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<PatternConfiguration>> GetRegisteredPatternsAsync()
    {
        var configs = _state.State.PatternConfigs ?? new List<PatternConfiguration>();
        return Task.FromResult<IReadOnlyList<PatternConfiguration>>(configs);
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<PatternMatchData>> ProcessEventAsync(TemporalEventData evt)
    {
        ArgumentNullException.ThrowIfNull(evt);

        var internalEvent = ConvertToInternalEvent(evt);
        var matches = await _detector.ProcessEventAsync(internalEvent);

        _state.State.TotalEventsProcessed++;

        return matches.Select(ConvertToMatchData).ToList();
    }

    /// <inheritdoc/>
    public async Task<PatternProcessingResult> ProcessEventBatchAsync(
        IReadOnlyList<TemporalEventData> events)
    {
        ArgumentNullException.ThrowIfNull(events);

        if (events.Count == 0)
        {
            return new PatternProcessingResult
            {
                EventsProcessed = 0,
                PatternsChecked = _patterns.Count
            };
        }

        var sw = Stopwatch.StartNew();
        var allMatches = new List<PatternMatchData>();
        var usedGpu = false;

        // Convert events
        var internalEvents = events.Select(ConvertToInternalEvent).ToList();

        // Try GPU accelerated matching for large batches
        if (_gpuMatcher != null && events.Count >= 1000 && _patterns.Count > 0)
        {
            try
            {
                var gpuMatches = await _gpuMatcher.FindPatternsAsync(
                    internalEvents,
                    _patterns.Values.ToList(),
                    null);

                allMatches.AddRange(gpuMatches.Select(ConvertToMatchData));
                usedGpu = _gpuMatcher.IsGpuAvailable;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "GPU pattern matching failed, using sequential");
            }
        }
        else
        {
            // Process events sequentially
            foreach (var evt in internalEvents)
            {
                var matches = await _detector.ProcessEventAsync(evt);
                allMatches.AddRange(matches.Select(ConvertToMatchData));
            }
        }

        sw.Stop();

        _state.State.TotalEventsProcessed += events.Count;

        return new PatternProcessingResult
        {
            Matches = allMatches,
            EventsProcessed = events.Count,
            PatternsChecked = _patterns.Count,
            ProcessingTimeNanos = sw.ElapsedTicks * 100,
            UsedGpuAcceleration = usedGpu
        };
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<PatternMatchData>> GetRecentMatchesAsync(
        int maxResults = 100,
        long? sinceNanos = null)
    {
        var matches = _detector.RecentMatches
            .Where(m => !sinceNanos.HasValue || m.DetectionTimeNanos > sinceNanos.Value)
            .OrderByDescending(m => m.DetectionTimeNanos)
            .Take(maxResults)
            .Select(ConvertToMatchData)
            .ToList();

        return Task.FromResult<IReadOnlyList<PatternMatchData>>(matches);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<PatternMatchData>> GetMatchesByPatternAsync(
        string patternId,
        int maxResults = 100)
    {
        var matches = _detector.RecentMatches
            .Where(m => m.PatternId == patternId)
            .OrderByDescending(m => m.DetectionTimeNanos)
            .Take(maxResults)
            .Select(ConvertToMatchData)
            .ToList();

        return Task.FromResult<IReadOnlyList<PatternMatchData>>(matches);
    }

    /// <inheritdoc/>
    public Task<PatternDetectionStats> GetStatisticsAsync()
    {
        var internalStats = _detector.GetStatistics();
        var gpuStats = _gpuMatcher?.Statistics;

        var patternStats = internalStats.PatternStatistics
            .Select(p => new PatternStats
            {
                PatternId = p.PatternId,
                PatternName = p.PatternName,
                MatchCount = p.MatchCount,
                Severity = (int)p.Severity,
                Enabled = _patterns.ContainsKey(p.PatternId)
            })
            .ToList();

        return Task.FromResult(new PatternDetectionStats
        {
            TotalEventsProcessed = internalStats.TotalEventsProcessed,
            TotalPatternsDetected = internalStats.TotalPatternsDetected,
            TotalPatternChecks = internalStats.TotalPatternChecks,
            RegisteredPatternCount = internalStats.RegisteredPatternCount,
            CurrentWindowSize = internalStats.CurrentWindowSize,
            RecentMatchCount = internalStats.RecentMatchCount,
            DetectionRate = internalStats.DetectionRate,
            AverageProcessingTimeNanos = internalStats.TotalEventsProcessed > 0
                ? (double)(gpuStats?.TotalProcessingTimeNanos ?? 0) / internalStats.TotalEventsProcessed
                : 0,
            GpuUtilizationPercent = gpuStats?.GpuUtilizationPercent ?? 0,
            PatternStatistics = patternStats
        });
    }

    /// <inheritdoc/>
    public async Task ClearAsync()
    {
        _detector.Clear();
        _state.State.TotalEventsProcessed = 0;
        await _state.WriteStateAsync();

        _logger.LogInformation("Cleared pattern detector {DetectorId}", _detectorId);
    }

    /// <inheritdoc/>
    public async Task ConfigureWindowAsync(long windowSizeNanos, int maxEvents = 10_000)
    {
        if (windowSizeNanos <= 0)
            throw new ArgumentException("Window size must be positive", nameof(windowSizeNanos));
        if (maxEvents <= 0)
            throw new ArgumentException("Max events must be positive", nameof(maxEvents));

        _windowSizeNanos = windowSizeNanos;
        _maxWindowEvents = maxEvents;

        // Recreate detector with new configuration
        var graph = new TemporalGraphStorage(_logger);
        _detector = new TemporalPatternDetector(
            windowSizeNanos,
            slideIntervalNanos: 0,
            maxEvents,
            graph,
            _logger);

        // Re-register patterns
        foreach (var pattern in _patterns.Values)
        {
            _detector.RegisterPattern(pattern);
        }

        // Persist configuration
        _state.State.WindowSizeNanos = windowSizeNanos;
        _state.State.MaxWindowEvents = maxEvents;
        await _state.WriteStateAsync();

        _logger.LogInformation(
            "Configured window: {WindowMs}ms, max {MaxEvents} events",
            windowSizeNanos / 1_000_000, maxEvents);
    }

    /// <summary>
    /// Creates a pattern instance from configuration.
    /// </summary>
    private ITemporalPattern? CreatePatternFromConfig(PatternConfiguration config)
    {
        return config.PatternType.ToLowerInvariant() switch
        {
            "rapid-split" or "rapidsplit" => new RapidSplitPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 5_000_000_000,
                GetIntParam(config, "minimumSplits", 2),
                GetDoubleParam(config, "minimumAmount", 1000.0)),

            "circular-flow" or "circularflow" => new CircularFlowPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 60_000_000_000,
                GetIntParam(config, "minimumHops", 3),
                GetIntParam(config, "maximumHops", 10)),

            "high-frequency" or "highfrequency" => new HighFrequencyPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 1_000_000_000,
                GetIntParam(config, "minimumTransactions", 10),
                GetDoubleParam(config, "minimumTotalAmount", 10_000.0)),

            "velocity-change" or "velocitychange" => new VelocityChangePattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 3600_000_000_000,
                GetDoubleParam(config, "thresholdAmount", 50_000.0),
                GetLongParam(config, "inactivityPeriod", 1800_000_000_000)),

            "temporal-cluster" or "temporalcluster" => new TemporalClusterPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 10_000_000_000,
                GetIntParam(config, "clusterSize", 5),
                GetDoubleParam(config, "clusterRatio", 3.0)),

            "structuring" => new StructuringPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 86_400_000_000_000,
                GetDoubleParam(config, "threshold", 10_000.0),
                GetDoubleParam(config, "marginPercent", 10.0),
                GetIntParam(config, "minimumCount", 3)),

            "fan-out" or "fanout" => new FanOutPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 300_000_000_000,
                GetIntParam(config, "minimumTargets", 5),
                GetDoubleParam(config, "minimumTotalAmount", 10_000.0)),

            "fan-in" or "fanin" => new FanInPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 300_000_000_000,
                GetIntParam(config, "minimumSources", 5),
                GetDoubleParam(config, "minimumTotalAmount", 10_000.0)),

            "round-trip" or "roundtrip" => new RoundTripPattern(
                config.WindowSizeNanos > 0 ? config.WindowSizeNanos : 3600_000_000_000,
                GetDoubleParam(config, "minimumAmount", 5_000.0),
                GetDoubleParam(config, "tolerancePercent", 5.0)),

            _ => null
        };
    }

    private static int GetIntParam(PatternConfiguration config, string key, int defaultValue)
    {
        if (config.Parameters?.TryGetValue(key, out var value) == true &&
            int.TryParse(value, out var result))
            return result;
        return defaultValue;
    }

    private static long GetLongParam(PatternConfiguration config, string key, long defaultValue)
    {
        if (config.Parameters?.TryGetValue(key, out var value) == true &&
            long.TryParse(value, out var result))
            return result;
        return defaultValue;
    }

    private static double GetDoubleParam(PatternConfiguration config, string key, double defaultValue)
    {
        if (config.Parameters?.TryGetValue(key, out var value) == true &&
            double.TryParse(value, out var result))
            return result;
        return defaultValue;
    }

    private static TemporalEvent ConvertToInternalEvent(TemporalEventData data)
    {
        return new TemporalEvent
        {
            EventId = data.EventId,
            EventType = data.EventType,
            TimestampNanos = data.TimestampNanos,
            SourceId = data.SourceId,
            TargetId = data.TargetId,
            Value = data.Value,
            Data = data.Properties?.ToDictionary(
                kvp => kvp.Key,
                kvp => (object)kvp.Value) ?? new Dictionary<string, object>()
        };
    }

    private static PatternMatchData ConvertToMatchData(PatternMatch match)
    {
        return new PatternMatchData
        {
            PatternId = match.PatternId,
            PatternName = match.PatternName,
            DetectionTimeNanos = match.DetectionTimeNanos,
            WindowStartNanos = match.WindowStartNanos,
            WindowEndNanos = match.WindowEndNanos,
            InvolvedEventCount = match.InvolvedEvents.Count,
            InvolvedEventIds = match.InvolvedEvents.Select(e => e.EventId).ToList(),
            Confidence = match.Confidence,
            Severity = (int)match.Severity,
            Metadata = match.Metadata.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value?.ToString() ?? string.Empty)
        };
    }
}

/// <summary>
/// Persistent state for pattern detector grain.
/// </summary>
[GenerateSerializer]
public sealed class PatternDetectorState
{
    /// <summary>
    /// Registered pattern configurations.
    /// </summary>
    [Id(0)]
    public List<PatternConfiguration>? PatternConfigs { get; set; }

    /// <summary>
    /// Window size configuration.
    /// </summary>
    [Id(1)]
    public long WindowSizeNanos { get; set; }

    /// <summary>
    /// Maximum window events configuration.
    /// </summary>
    [Id(2)]
    public int MaxWindowEvents { get; set; }

    /// <summary>
    /// Total events processed (for statistics).
    /// </summary>
    [Id(3)]
    public long TotalEventsProcessed { get; set; }
}
