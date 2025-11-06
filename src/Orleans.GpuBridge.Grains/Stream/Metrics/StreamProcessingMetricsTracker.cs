using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using Orleans.GpuBridge.Grains.Stream.Configuration;

namespace Orleans.GpuBridge.Grains.Stream.Metrics;

/// <summary>
/// Thread-safe metrics tracker for stream processing
/// </summary>
internal sealed class StreamProcessingMetricsTracker
{
    private readonly StreamProcessingConfiguration _config;

    // Basic counters
    private long _totalItemsProcessed;
    private long _totalItemsFailed;
    private long _totalBatchesProcessed;
    private long _totalItemsReceived;
    private long _totalItemsDropped;
    private DateTime _startTime;
    private DateTime? _lastProcessedTime;

    // GPU metrics
    private long _totalKernelExecutionMs;
    private long _totalMemoryTransferMs;
    private long _totalGpuMemoryAllocated;

    // Latency tracking (circular buffer for percentile calculations)
    private readonly ConcurrentQueue<double> _latencyHistory;
    private const int MaxLatencySamples = 1000;

    // Throughput tracking (sliding window - last 10 seconds)
    private readonly ConcurrentQueue<(DateTime timestamp, int count)> _throughputWindow;
    private double _peakThroughput;

    // Backpressure tracking
    private long _totalPauseCount;
    private long _totalPauseDurationMs;
    private DateTime? _pauseStartTime;
    private long _bufferCurrentSize;

    // Device info
    private string _deviceType = "Unknown";
    private string _deviceName = "Unknown";

    public StreamProcessingMetricsTracker(StreamProcessingConfiguration config)
    {
        _config = config;
        _latencyHistory = new ConcurrentQueue<double>();
        _throughputWindow = new ConcurrentQueue<(DateTime, int)>();
    }

    public void Start()
    {
        _startTime = DateTime.UtcNow;
    }

    public void SetDeviceInfo(string deviceType, string deviceName)
    {
        _deviceType = deviceType;
        _deviceName = deviceName;
    }

    #region Item Tracking

    public void RecordItemReceived()
    {
        Interlocked.Increment(ref _totalItemsReceived);
    }

    public void RecordItemDropped()
    {
        Interlocked.Increment(ref _totalItemsDropped);
    }

    #endregion

    #region Batch Processing

    public void RecordBatchSuccess(int itemCount, TimeSpan elapsed, DateTime batchStartTime)
    {
        Interlocked.Add(ref _totalItemsProcessed, itemCount);
        Interlocked.Increment(ref _totalBatchesProcessed);
        _lastProcessedTime = DateTime.UtcNow;

        // Calculate and record latency per item
        var latencyPerItem = elapsed.TotalMilliseconds / itemCount;
        RecordLatency(latencyPerItem);

        // Record throughput
        RecordThroughput(itemCount);
    }

    public void RecordBatchFailure(int itemCount, TimeSpan elapsed)
    {
        Interlocked.Add(ref _totalItemsFailed, itemCount);
        _lastProcessedTime = DateTime.UtcNow;
    }

    private void RecordLatency(double latencyMs)
    {
        _latencyHistory.Enqueue(latencyMs);

        // Keep only last N samples
        while (_latencyHistory.Count > MaxLatencySamples)
        {
            _latencyHistory.TryDequeue(out _);
        }
    }

    private void RecordThroughput(int itemCount)
    {
        _throughputWindow.Enqueue((DateTime.UtcNow, itemCount));

        // Remove entries older than 10 seconds
        while (_throughputWindow.TryPeek(out var entry) &&
               (DateTime.UtcNow - entry.timestamp).TotalSeconds > 10)
        {
            _throughputWindow.TryDequeue(out _);
        }

        // Update peak throughput
        var currentThroughput = CalculateCurrentThroughput();
        if (currentThroughput > _peakThroughput)
        {
            _peakThroughput = currentThroughput;
        }
    }

    #endregion

    #region GPU Metrics

    public void RecordKernelExecution(TimeSpan elapsed)
    {
        var ms = (long)elapsed.TotalMilliseconds;
        Interlocked.Add(ref _totalKernelExecutionMs, ms);
    }

    public void RecordMemoryTransfer(TimeSpan elapsed)
    {
        var ms = (long)elapsed.TotalMilliseconds;
        Interlocked.Add(ref _totalMemoryTransferMs, ms);
    }

    public void RecordMemoryAllocation(long bytes)
    {
        Interlocked.Add(ref _totalGpuMemoryAllocated, bytes);
    }

    #endregion

    #region Backpressure Tracking

    public void RecordPause()
    {
        Interlocked.Increment(ref _totalPauseCount);
        _pauseStartTime = DateTime.UtcNow;
    }

    public void RecordResume()
    {
        if (_pauseStartTime.HasValue)
        {
            var pauseDuration = (DateTime.UtcNow - _pauseStartTime.Value).TotalMilliseconds;
            Interlocked.Add(ref _totalPauseDurationMs, (long)pauseDuration);
            _pauseStartTime = null;
        }
    }

    public void UpdateBufferSize(long currentSize)
    {
        Interlocked.Exchange(ref _bufferCurrentSize, currentSize);
    }

    #endregion

    #region Metrics Calculation

    public StreamProcessingMetrics GetMetrics()
    {
        var totalProcessed = _totalItemsProcessed;
        var totalBatches = _totalBatchesProcessed;

        // Calculate latency metrics
        var (avgLatency, p50, p99) = CalculateLatencyMetrics();

        // Calculate batch metrics
        var avgBatchSize = totalBatches > 0 ? (double)totalProcessed / totalBatches : 0;
        var batchEfficiency = _config.BatchConfig.MaxBatchSize > 0
            ? avgBatchSize / _config.BatchConfig.MaxBatchSize
            : 0;

        // Calculate GPU metrics
        var totalKernelTime = TimeSpan.FromMilliseconds(_totalKernelExecutionMs);
        var totalTransferTime = TimeSpan.FromMilliseconds(_totalMemoryTransferMs);
        var totalGpuTime = totalKernelTime + totalTransferTime;

        var kernelEfficiency = totalGpuTime.TotalMilliseconds > 0
            ? (totalKernelTime.TotalMilliseconds / totalGpuTime.TotalMilliseconds) * 100
            : 0;

        var memoryBandwidth = totalTransferTime.TotalSeconds > 0
            ? (_totalGpuMemoryAllocated / (1024.0 * 1024.0)) / totalTransferTime.TotalSeconds
            : 0;

        // Calculate throughput
        var currentThroughput = CalculateCurrentThroughput();

        // Calculate backpressure metrics
        var bufferUtilization = _config.BackpressureConfig.BufferCapacity > 0
            ? (double)_bufferCurrentSize / _config.BackpressureConfig.BufferCapacity
            : 0;

        var totalPauseDuration = TimeSpan.FromMilliseconds(_totalPauseDurationMs);

        // Add current pause duration if paused
        if (_pauseStartTime.HasValue)
        {
            totalPauseDuration += DateTime.UtcNow - _pauseStartTime.Value;
        }

        return new StreamProcessingMetrics(
            TotalItemsProcessed: totalProcessed,
            TotalItemsFailed: _totalItemsFailed,
            TotalProcessingTime: _lastProcessedTime.HasValue
                ? _lastProcessedTime.Value - _startTime
                : TimeSpan.Zero,
            TotalBatchesProcessed: totalBatches,
            AverageBatchSize: avgBatchSize,
            BatchEfficiency: batchEfficiency,
            AverageLatencyMs: avgLatency,
            P50LatencyMs: p50,
            P99LatencyMs: p99,
            TotalKernelExecutionTime: totalKernelTime,
            TotalMemoryTransferTime: totalTransferTime,
            KernelEfficiency: kernelEfficiency,
            MemoryBandwidthMBps: memoryBandwidth,
            TotalGpuMemoryAllocated: _totalGpuMemoryAllocated,
            CurrentThroughput: currentThroughput,
            PeakThroughput: _peakThroughput,
            BufferCurrentSize: _bufferCurrentSize,
            BufferCapacity: _config.BackpressureConfig.BufferCapacity,
            BufferUtilization: bufferUtilization,
            TotalPauseCount: _totalPauseCount,
            TotalPauseDuration: totalPauseDuration,
            DeviceType: _deviceType,
            DeviceName: _deviceName,
            StartTime: _startTime,
            LastProcessedTime: _lastProcessedTime);
    }

    private (double avg, double p50, double p99) CalculateLatencyMetrics()
    {
        var samples = _latencyHistory.ToArray();

        if (samples.Length == 0)
        {
            return (0, 0, 0);
        }

        var sorted = samples.OrderBy(x => x).ToArray();
        var avg = sorted.Average();
        var p50 = sorted[(int)(sorted.Length * 0.5)];
        var p99 = sorted[Math.Min((int)(sorted.Length * 0.99), sorted.Length - 1)];

        return (avg, p50, p99);
    }

    private double CalculateCurrentThroughput()
    {
        var recent = _throughputWindow
            .Where(e => (DateTime.UtcNow - e.timestamp).TotalSeconds <= 10)
            .ToArray();

        if (recent.Length == 0)
        {
            return 0;
        }

        var totalItems = recent.Sum(e => e.count);
        var duration = recent.Length > 1
            ? (recent[^1].timestamp - recent[0].timestamp).TotalSeconds
            : 10; // Assume 10 seconds if only one sample

        return duration > 0 ? totalItems / duration : 0;
    }

    #endregion
}
