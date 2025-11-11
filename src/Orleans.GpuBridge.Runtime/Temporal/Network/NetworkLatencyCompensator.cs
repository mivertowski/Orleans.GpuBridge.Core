using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Runtime.Temporal.Network;

/// <summary>
/// Measures and compensates for network latency in distributed timestamps.
/// Uses RTT (Round-Trip Time) measurement to adjust remote timestamps for one-way network delay.
/// </summary>
/// <remarks>
/// Algorithm:
/// 1. Measure RTT to remote endpoint via TCP connection or ICMP ping
/// 2. Assume symmetric path: one-way latency ≈ RTT / 2
/// 3. Compensate remote timestamp: adjusted = original + (RTT / 2)
///
/// Limitations:
/// - Assumes symmetric network paths (may be inaccurate for asymmetric routes)
/// - RTT varies over time (use median of multiple samples)
/// - Does not account for clock drift (use PTP/NTP for clock sync)
///
/// Performance:
/// - RTT measurement: ~1ms per endpoint
/// - Compensation calculation: ~10ns overhead
/// - Measurement frequency: Every 1 minute per endpoint
/// </remarks>
public sealed class NetworkLatencyCompensator
{
    private readonly ILogger<NetworkLatencyCompensator> _logger;
    private readonly ConcurrentDictionary<IPEndPoint, LatencyStatistics> _latencyCache = new();
    private readonly TimeSpan _measurementInterval;
    private readonly CancellationTokenSource _backgroundCts = new();

    /// <summary>
    /// Initializes a new network latency compensator.
    /// </summary>
    /// <param name="logger">Logger for diagnostic messages.</param>
    /// <param name="measurementInterval">Interval between background measurements (default: 1 minute).</param>
    public NetworkLatencyCompensator(
        ILogger<NetworkLatencyCompensator> logger,
        TimeSpan? measurementInterval = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _measurementInterval = measurementInterval ?? TimeSpan.FromMinutes(1);
    }

    /// <summary>
    /// Measures round-trip time (RTT) to remote endpoint.
    /// Takes median of 10 TCP connection roundtrips for robustness.
    /// </summary>
    /// <param name="remote">Remote endpoint to measure.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Median RTT as TimeSpan.</returns>
    public async Task<TimeSpan> MeasureLatencyAsync(
        IPEndPoint remote,
        CancellationToken ct = default)
    {
        const int sampleCount = 10;
        var samples = new long[sampleCount];

        _logger.LogDebug("Measuring RTT to {Remote} ({Count} samples)", remote, sampleCount);

        for (int i = 0; i < sampleCount; i++)
        {
            ct.ThrowIfCancellationRequested();

            var stopwatch = Stopwatch.StartNew();

            try
            {
                // TCP connection roundtrip (more reliable than ICMP ping)
                using var client = new TcpClient();
                await client.ConnectAsync(remote.Address, remote.Port, ct);

                stopwatch.Stop();
                samples[i] = stopwatch.ElapsedTicks;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to measure RTT to {Remote} (sample {Index})", remote, i);
                samples[i] = long.MaxValue; // Mark as outlier
            }

            // Small delay between measurements to avoid overwhelming target
            if (i < sampleCount - 1)
            {
                await Task.Delay(10, ct);
            }
        }

        // Calculate median RTT (more robust than mean against outliers)
        Array.Sort(samples);

        // Remove outliers (top 20%)
        int validCount = (int)(sampleCount * 0.8);
        long medianTicks = samples[validCount / 2];

        if (medianTicks == long.MaxValue)
        {
            _logger.LogError("All RTT measurements failed for {Remote}", remote);
            throw new InvalidOperationException($"Cannot measure RTT to {remote}");
        }

        var rtt = TimeSpan.FromTicks(medianTicks);

        // Cache result
        var stats = _latencyCache.GetOrAdd(remote, _ => new LatencyStatistics(remote, _logger));
        stats.AddSample(rtt);

        _logger.LogInformation(
            "Measured RTT to {Remote}: {RTT:F3}ms (median of {Count} samples, min={Min:F3}ms, p99={P99:F3}ms)",
            remote,
            rtt.TotalMilliseconds,
            sampleCount,
            stats.MinRtt.TotalMilliseconds,
            stats.P99Rtt.TotalMilliseconds);

        return rtt;
    }

    /// <summary>
    /// Compensates remote timestamp for network latency.
    /// Adjusts timestamp assuming symmetric network path (RTT/2).
    /// </summary>
    /// <param name="remoteTimestampNanos">Timestamp from remote node (nanoseconds).</param>
    /// <param name="sourceEndpoint">Remote endpoint that sent the timestamp.</param>
    /// <returns>Compensated timestamp in nanoseconds.</returns>
    public long CompensateTimestamp(long remoteTimestampNanos, IPEndPoint sourceEndpoint)
    {
        if (!_latencyCache.TryGetValue(sourceEndpoint, out var stats))
        {
            // No latency data available - return uncompensated
            _logger.LogWarning(
                "No latency data for {Source} - returning uncompensated timestamp. " +
                "Call MeasureLatencyAsync() first.",
                sourceEndpoint);
            return remoteTimestampNanos;
        }

        // Assume symmetric path: one-way latency ≈ RTT / 2
        // Convert ticks to nanoseconds (1 tick = 100ns)
        long oneWayLatencyNanos = stats.MedianRtt.Ticks * 100 / 2;

        // Compensate by subtracting one-way latency (message was sent in the past)
        long compensatedTimestamp = remoteTimestampNanos - oneWayLatencyNanos;

        _logger.LogTrace(
            "Compensated timestamp from {Source}: {Original}ns → {Compensated}ns (Δ=-{Delta}μs)",
            sourceEndpoint,
            remoteTimestampNanos,
            compensatedTimestamp,
            oneWayLatencyNanos / 1_000);

        return compensatedTimestamp;
    }

    /// <summary>
    /// Gets cached latency statistics for an endpoint.
    /// </summary>
    /// <param name="endpoint">Remote endpoint.</param>
    /// <returns>Latency statistics, or null if no measurements available.</returns>
    public LatencyStatistics? GetStatistics(IPEndPoint endpoint)
    {
        _latencyCache.TryGetValue(endpoint, out var stats);
        return stats;
    }

    /// <summary>
    /// Starts background latency measurement for known endpoints.
    /// Measurements repeat every measurementInterval.
    /// </summary>
    /// <param name="endpoints">Endpoints to monitor.</param>
    public void StartPeriodicMeasurements(IEnumerable<IPEndPoint> endpoints)
    {
        var endpointList = endpoints.ToList();
        _logger.LogInformation(
            "Starting periodic RTT measurements for {Count} endpoints (interval={Interval})",
            endpointList.Count,
            _measurementInterval);

        foreach (var endpoint in endpointList)
        {
            _ = Task.Run(async () =>
            {
                while (!_backgroundCts.Token.IsCancellationRequested)
                {
                    try
                    {
                        await MeasureLatencyAsync(endpoint, _backgroundCts.Token);
                        await Task.Delay(_measurementInterval, _backgroundCts.Token);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Failed to measure latency to {Endpoint}", endpoint);

                        // Back off on errors
                        await Task.Delay(TimeSpan.FromSeconds(10), _backgroundCts.Token);
                    }
                }
            }, _backgroundCts.Token);
        }
    }

    /// <summary>
    /// Stops all background measurements.
    /// </summary>
    public void StopPeriodicMeasurements()
    {
        _logger.LogInformation("Stopping periodic RTT measurements");
        _backgroundCts.Cancel();
    }
}

/// <summary>
/// Statistical analysis of network latency measurements.
/// Tracks median, min, max, and p99 RTT values over sliding window.
/// </summary>
public sealed class LatencyStatistics
{
    private const int MaxSamples = 100;
    private readonly IPEndPoint _endpoint;
    private readonly ILogger _logger;
    private readonly List<TimeSpan> _samples = new();
    private readonly object _lock = new();

    /// <summary>Gets the median RTT (50th percentile).</summary>
    public TimeSpan MedianRtt { get; private set; }

    /// <summary>Gets the minimum RTT observed.</summary>
    public TimeSpan MinRtt { get; private set; } = TimeSpan.MaxValue;

    /// <summary>Gets the maximum RTT observed.</summary>
    public TimeSpan MaxRtt { get; private set; }

    /// <summary>Gets the 99th percentile RTT.</summary>
    public TimeSpan P99Rtt { get; private set; }

    /// <summary>Gets the number of samples collected.</summary>
    public int SampleCount { get; private set; }

    /// <summary>Gets the remote endpoint.</summary>
    public IPEndPoint Endpoint => _endpoint;

    /// <summary>
    /// Initializes latency statistics for an endpoint.
    /// </summary>
    public LatencyStatistics(IPEndPoint endpoint, ILogger logger)
    {
        _endpoint = endpoint ?? throw new ArgumentNullException(nameof(endpoint));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Adds a new RTT sample and updates statistics.
    /// </summary>
    /// <param name="rtt">Round-trip time measurement.</param>
    public void AddSample(TimeSpan rtt)
    {
        lock (_lock)
        {
            _samples.Add(rtt);
            SampleCount++;

            // Sliding window - keep last 100 samples
            if (_samples.Count > MaxSamples)
            {
                _samples.RemoveAt(0);
            }

            // Update statistics
            var sorted = _samples.OrderBy(s => s).ToArray();

            MedianRtt = sorted[sorted.Length / 2];
            MinRtt = sorted[0];
            MaxRtt = sorted[^1];
            P99Rtt = sorted[(int)(sorted.Length * 0.99)];

            _logger.LogTrace(
                "RTT statistics for {Endpoint}: median={Median:F3}ms, min={Min:F3}ms, max={Max:F3}ms, p99={P99:F3}ms",
                _endpoint,
                MedianRtt.TotalMilliseconds,
                MinRtt.TotalMilliseconds,
                MaxRtt.TotalMilliseconds,
                P99Rtt.TotalMilliseconds);
        }
    }

    /// <summary>
    /// Checks if RTT measurements show high variance (potential network instability).
    /// </summary>
    /// <returns>True if variance is high; false otherwise.</returns>
    public bool HasHighVariance()
    {
        lock (_lock)
        {
            if (_samples.Count < 10)
                return false;

            // High variance if p99 > 2× median
            return P99Rtt.TotalMilliseconds > MedianRtt.TotalMilliseconds * 2;
        }
    }
}
