using System;
using System.Threading;

namespace Orleans.GpuBridge.Grains.Stream.Internal;

/// <summary>
/// Helper class for tracking statistics
/// </summary>
internal sealed class StreamProcessingStatsTracker
{
    private long _itemsProcessed;
    private long _itemsFailed;
    private double _totalMs;
    private DateTime _startTime;
    private DateTime? _lastProcessedTime;

    public long ItemsProcessed => _itemsProcessed;

    public void Start()
    {
        _startTime = DateTime.UtcNow;
    }

    public void RecordSuccess(int count, TimeSpan elapsed)
    {
        Interlocked.Add(ref _itemsProcessed, count);
        _lastProcessedTime = DateTime.UtcNow;

        var ms = elapsed.TotalMilliseconds;
        var currentTotal = _totalMs;
        while (true)
        {
            var newTotal = currentTotal + ms;
            var original = Interlocked.CompareExchange(
                ref _totalMs, newTotal, currentTotal);
            if (original == currentTotal) break;
            currentTotal = original;
        }
    }

    public void RecordFailure(int count)
    {
        Interlocked.Add(ref _itemsFailed, count);
    }

    public StreamProcessingStats GetStats()
    {
        var processed = _itemsProcessed;
        var avgLatency = processed > 0 ? _totalMs / processed : 0;

        return new StreamProcessingStats(
            _itemsProcessed,
            _itemsFailed,
            DateTime.UtcNow - _startTime,
            avgLatency,
            _startTime,
            _lastProcessedTime);
    }
}