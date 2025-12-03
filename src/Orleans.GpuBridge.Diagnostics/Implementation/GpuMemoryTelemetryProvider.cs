// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Diagnostics.Interfaces;

namespace Orleans.GpuBridge.Diagnostics.Implementation;

/// <summary>
/// Implementation of per-grain GPU memory telemetry collection using OpenTelemetry metrics.
/// This service provides detailed tracking of GPU memory allocations at the grain level
/// with support for real-time event streaming and comprehensive statistics.
/// </summary>
/// <remarks>
/// <para>
/// Thread-safe implementation using concurrent collections for high-throughput scenarios.
/// Integrates with OpenTelemetry for enterprise observability dashboards.
/// </para>
/// </remarks>
public sealed class GpuMemoryTelemetryProvider : IGpuMemoryTelemetryProvider, IDisposable
{
    private readonly ILogger<GpuMemoryTelemetryProvider> _logger;
    private readonly Meter _meter;

    // Per-grain memory tracking
    private readonly ConcurrentDictionary<GrainKey, GrainMemoryTracker> _grainTrackers = new();

    // Memory pool statistics by device
    private readonly ConcurrentDictionary<int, MemoryPoolStats> _poolStats = new();

    // Event streaming channel
    private readonly Channel<GpuMemoryEvent> _eventChannel;
    private bool _disposed;

    #region OpenTelemetry Metric Instruments

    /// <summary>Counter tracking per-grain memory allocations.</summary>
    private readonly Counter<long> _grainAllocations;

    /// <summary>Counter tracking per-grain memory deallocations.</summary>
    private readonly Counter<long> _grainDeallocations;

    /// <summary>Histogram tracking per-grain allocation sizes.</summary>
    private readonly Histogram<long> _grainAllocationSize;

    /// <summary>Observable gauge reporting current per-grain memory usage.</summary>
    private readonly ObservableGauge<long> _perGrainMemory;

    /// <summary>Observable gauge reporting active grain count.</summary>
    private readonly ObservableGauge<int> _activeGrainCount;

    /// <summary>Observable gauge reporting memory pool utilization.</summary>
    private readonly ObservableGauge<double> _poolUtilization;

    /// <summary>Observable gauge reporting memory pool fragmentation.</summary>
    private readonly ObservableGauge<int> _poolFragmentation;

    #endregion

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuMemoryTelemetryProvider"/> class.
    /// </summary>
    /// <param name="logger">Logger for recording telemetry operations.</param>
    /// <param name="meterFactory">Factory for creating OpenTelemetry meters.</param>
    public GpuMemoryTelemetryProvider(
        ILogger<GpuMemoryTelemetryProvider> logger,
        IMeterFactory meterFactory)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _meter = meterFactory.Create("Orleans.GpuBridge.GrainMemory", "1.0.0");

        // Initialize event streaming channel with bounded capacity
        _eventChannel = Channel.CreateBounded<GpuMemoryEvent>(
            new BoundedChannelOptions(10_000)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = false,
                SingleWriter = false
            });

        // Initialize counter instruments
        _grainAllocations = _meter.CreateCounter<long>(
            "gpu.grain.allocations",
            unit: "allocations",
            description: "Number of GPU memory allocations per grain");

        _grainDeallocations = _meter.CreateCounter<long>(
            "gpu.grain.deallocations",
            unit: "deallocations",
            description: "Number of GPU memory deallocations per grain");

        // Initialize histogram instruments
        _grainAllocationSize = _meter.CreateHistogram<long>(
            "gpu.grain.allocation.size",
            unit: "bytes",
            description: "GPU memory allocation size per grain");

        // Initialize observable gauge instruments
        _perGrainMemory = _meter.CreateObservableGauge<long>(
            "gpu.grain.memory.allocated",
            GetPerGrainMemoryMeasurements,
            unit: "bytes",
            description: "GPU memory currently allocated per grain type");

        _activeGrainCount = _meter.CreateObservableGauge<int>(
            "gpu.grain.active.count",
            () => _grainTrackers.Count,
            unit: "grains",
            description: "Number of active grains with GPU memory allocations");

        _poolUtilization = _meter.CreateObservableGauge<double>(
            "gpu.memory.pool.utilization",
            GetPoolUtilizationMeasurements,
            unit: "percent",
            description: "GPU memory pool utilization percentage per device");

        _poolFragmentation = _meter.CreateObservableGauge<int>(
            "gpu.memory.pool.fragmentation",
            GetPoolFragmentationMeasurements,
            unit: "percent",
            description: "GPU memory pool fragmentation percentage per device");

        _logger.LogInformation(
            "GPU memory telemetry provider initialized with per-grain tracking and OpenTelemetry metrics");
    }

    /// <inheritdoc />
    public void RecordGrainMemoryAllocation(
        string grainType,
        string grainId,
        int deviceIndex,
        long bytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(grainType);
        ArgumentException.ThrowIfNullOrWhiteSpace(grainId);

        var key = new GrainKey(grainType, grainId);
        var now = DateTimeOffset.UtcNow;

        var tracker = _grainTrackers.AddOrUpdate(
            key,
            _ => new GrainMemoryTracker(grainType, grainId, deviceIndex, bytes, now),
            (_, existing) =>
            {
                existing.AddAllocation(bytes, now);
                return existing;
            });

        // Record OpenTelemetry metrics
        var tags = new TagList
        {
            { "grain_type", grainType },
            { "device", deviceIndex }
        };

        _grainAllocations.Add(1, tags);
        _grainAllocationSize.Record(bytes, tags);

        // Emit event
        TryEmitEvent(new GpuMemoryEvent(
            GpuMemoryEventType.Allocated,
            grainType,
            grainId,
            deviceIndex,
            bytes,
            now));

        _logger.LogDebug(
            "Recorded GPU memory allocation for grain {GrainType}/{GrainId} on device {Device}: {Bytes} bytes",
            grainType, grainId, deviceIndex, bytes);
    }

    /// <inheritdoc />
    public void RecordGrainMemoryRelease(
        string grainType,
        string grainId,
        int deviceIndex,
        long bytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(grainType);
        ArgumentException.ThrowIfNullOrWhiteSpace(grainId);

        var key = new GrainKey(grainType, grainId);
        var now = DateTimeOffset.UtcNow;

        if (_grainTrackers.TryGetValue(key, out var tracker))
        {
            tracker.RecordRelease(bytes, now);

            // Remove tracker if fully deallocated
            if (tracker.AllocatedBytes <= 0)
            {
                _grainTrackers.TryRemove(key, out _);
            }
        }

        // Record OpenTelemetry metrics
        var tags = new TagList
        {
            { "grain_type", grainType },
            { "device", deviceIndex }
        };

        _grainDeallocations.Add(1, tags);

        // Emit event
        TryEmitEvent(new GpuMemoryEvent(
            GpuMemoryEventType.Released,
            grainType,
            grainId,
            deviceIndex,
            bytes,
            now));

        _logger.LogDebug(
            "Recorded GPU memory release for grain {GrainType}/{GrainId} on device {Device}: {Bytes} bytes",
            grainType, grainId, deviceIndex, bytes);
    }

    /// <inheritdoc />
    public GrainMemorySnapshot? GetGrainMemorySnapshot(string grainType, string grainId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(grainType);
        ArgumentException.ThrowIfNullOrWhiteSpace(grainId);

        var key = new GrainKey(grainType, grainId);

        if (_grainTrackers.TryGetValue(key, out var tracker))
        {
            return tracker.ToSnapshot();
        }

        return null;
    }

    /// <inheritdoc />
    public GrainTypeMemoryStats? GetMemoryStatsByGrainType(string grainType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(grainType);

        var grains = _grainTrackers.Values
            .Where(t => t.GrainType == grainType)
            .ToList();

        if (grains.Count == 0)
        {
            return null;
        }

        var totalBytes = grains.Sum(g => g.AllocatedBytes);
        var avgBytes = totalBytes / grains.Count;
        var peakBytes = grains.Max(g => g.PeakBytes);
        var minBytes = grains.Min(g => g.AllocatedBytes);

        return new GrainTypeMemoryStats(
            grainType,
            grains.Count,
            totalBytes,
            avgBytes,
            peakBytes,
            minBytes);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, GrainTypeMemoryStats> GetAllGrainTypeMemoryStats()
    {
        var results = new Dictionary<string, GrainTypeMemoryStats>();

        var groupedByType = _grainTrackers.Values
            .GroupBy(t => t.GrainType);

        foreach (var group in groupedByType)
        {
            var grains = group.ToList();
            var totalBytes = grains.Sum(g => g.AllocatedBytes);
            var avgBytes = grains.Count > 0 ? totalBytes / grains.Count : 0;
            var peakBytes = grains.Count > 0 ? grains.Max(g => g.PeakBytes) : 0;
            var minBytes = grains.Count > 0 ? grains.Min(g => g.AllocatedBytes) : 0;

            results[group.Key] = new GrainTypeMemoryStats(
                group.Key,
                grains.Count,
                totalBytes,
                avgBytes,
                peakBytes,
                minBytes);
        }

        return results;
    }

    /// <inheritdoc />
    public long GetTotalAllocatedMemory()
    {
        return _grainTrackers.Values.Sum(t => t.AllocatedBytes);
    }

    /// <inheritdoc />
    public int GetActiveGrainCount()
    {
        return _grainTrackers.Count;
    }

    /// <inheritdoc />
    public void RecordMemoryPoolStats(
        int deviceIndex,
        long poolUsedBytes,
        long poolTotalBytes,
        int fragmentationPercent)
    {
        var now = DateTimeOffset.UtcNow;
        var available = poolTotalBytes - poolUsedBytes;
        var utilization = poolTotalBytes > 0
            ? (double)poolUsedBytes / poolTotalBytes * 100.0
            : 0.0;

        var existingStats = _poolStats.GetValueOrDefault(deviceIndex);
        var allocationCount = existingStats?.AllocationCount ?? 0;

        var stats = new MemoryPoolStats(
            deviceIndex,
            poolUsedBytes,
            poolTotalBytes,
            available,
            utilization,
            fragmentationPercent,
            allocationCount,
            now);

        _poolStats[deviceIndex] = stats;

        // Check for threshold warnings
        if (utilization > 90)
        {
            TryEmitEvent(new GpuMemoryEvent(
                GpuMemoryEventType.PoolThresholdExceeded,
                "MemoryPool",
                deviceIndex.ToString(),
                deviceIndex,
                poolUsedBytes,
                now));

            _logger.LogWarning(
                "GPU memory pool on device {Device} exceeded 90% utilization: {Utilization:F1}%",
                deviceIndex, utilization);
        }

        // Emit pool stats update event
        TryEmitEvent(new GpuMemoryEvent(
            GpuMemoryEventType.PoolStatsUpdated,
            "MemoryPool",
            deviceIndex.ToString(),
            deviceIndex,
            poolUsedBytes,
            now));
    }

    /// <inheritdoc />
    public MemoryPoolStats? GetMemoryPoolStats(int deviceIndex)
    {
        return _poolStats.GetValueOrDefault(deviceIndex);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<GpuMemoryEvent> StreamEventsAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var evt in _eventChannel.Reader.ReadAllAsync(cancellationToken))
        {
            yield return evt;
        }
    }

    #region Private Helper Methods

    private void TryEmitEvent(GpuMemoryEvent evt)
    {
        if (!_eventChannel.Writer.TryWrite(evt))
        {
            _logger.LogDebug("Event channel full, dropping event: {EventType}", evt.EventType);
        }
    }

    private IEnumerable<Measurement<long>> GetPerGrainMemoryMeasurements()
    {
        var groupedByType = _grainTrackers.Values
            .GroupBy(t => t.GrainType);

        foreach (var group in groupedByType)
        {
            var totalBytes = group.Sum(g => g.AllocatedBytes);
            yield return new Measurement<long>(
                totalBytes,
                new TagList { { "grain_type", group.Key } });
        }
    }

    private IEnumerable<Measurement<double>> GetPoolUtilizationMeasurements()
    {
        foreach (var (device, stats) in _poolStats)
        {
            yield return new Measurement<double>(
                stats.UtilizationPercent,
                new TagList { { "device", device } });
        }
    }

    private IEnumerable<Measurement<int>> GetPoolFragmentationMeasurements()
    {
        foreach (var (device, stats) in _poolStats)
        {
            yield return new Measurement<int>(
                stats.FragmentationPercent,
                new TagList { { "device", device } });
        }
    }

    #endregion

    /// <summary>
    /// Releases all resources used by the GPU memory telemetry provider.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _eventChannel.Writer.Complete();
        _meter.Dispose();
    }

    #region Internal Types

    /// <summary>
    /// Composite key for grain tracking.
    /// </summary>
    private readonly record struct GrainKey(string GrainType, string GrainId);

    /// <summary>
    /// Internal tracker for per-grain memory usage.
    /// </summary>
    private sealed class GrainMemoryTracker
    {
        private long _allocatedBytes;
        private long _peakBytes;
        private DateTimeOffset _lastAccessTime;
        private readonly object _lock = new();

        public string GrainType { get; }
        public string GrainId { get; }
        public int DeviceIndex { get; }
        public DateTimeOffset AllocationTime { get; }

        public long AllocatedBytes
        {
            get { lock (_lock) return _allocatedBytes; }
        }

        public long PeakBytes
        {
            get { lock (_lock) return _peakBytes; }
        }

        public GrainMemoryTracker(
            string grainType,
            string grainId,
            int deviceIndex,
            long initialBytes,
            DateTimeOffset allocationTime)
        {
            GrainType = grainType;
            GrainId = grainId;
            DeviceIndex = deviceIndex;
            AllocationTime = allocationTime;
            _allocatedBytes = initialBytes;
            _peakBytes = initialBytes;
            _lastAccessTime = allocationTime;
        }

        public void AddAllocation(long bytes, DateTimeOffset timestamp)
        {
            lock (_lock)
            {
                _allocatedBytes += bytes;
                if (_allocatedBytes > _peakBytes)
                {
                    _peakBytes = _allocatedBytes;
                }
                _lastAccessTime = timestamp;
            }
        }

        public void RecordRelease(long bytes, DateTimeOffset timestamp)
        {
            lock (_lock)
            {
                _allocatedBytes = Math.Max(0, _allocatedBytes - bytes);
                _lastAccessTime = timestamp;
            }
        }

        public GrainMemorySnapshot ToSnapshot()
        {
            lock (_lock)
            {
                return new GrainMemorySnapshot(
                    GrainType,
                    GrainId,
                    DeviceIndex,
                    _allocatedBytes,
                    _peakBytes,
                    AllocationTime,
                    _lastAccessTime);
            }
        }
    }

    #endregion
}
