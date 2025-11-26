// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// CPU-based timing provider fallback when GPU timing is unavailable.
/// Uses high-resolution <see cref="Stopwatch"/> for timestamp generation.
/// </summary>
/// <remarks>
/// <para>
/// This provider is used as a fallback when:
/// <list type="bullet">
/// <item><description>No GPU hardware is available</description></item>
/// <item><description>GPU drivers are not installed</description></item>
/// <item><description>GPU timing initialization fails</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Performance Characteristics:</strong>
/// <list type="bullet">
/// <item><description>Resolution: ~100ns (platform-dependent)</description></item>
/// <item><description>Timestamp query overhead: &lt;50ns</description></item>
/// <item><description>Thread-safe: Yes</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class CpuTimingProvider : IGpuTimingProvider
{
    private readonly ILogger<CpuTimingProvider> _logger;
    private readonly long _timerResolutionNanos;
    private readonly long _clockFrequencyHz;
    private readonly long _baseTimestamp;
    private bool _timestampInjectionEnabled;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuTimingProvider"/> class.
    /// </summary>
    /// <param name="logger">Logger for diagnostic output.</param>
    public CpuTimingProvider(ILogger<CpuTimingProvider> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _clockFrequencyHz = Stopwatch.Frequency;

        // Calculate resolution: 1 / frequency * 1e9 nanoseconds
        // For most systems, Stopwatch.Frequency is ~10,000,000 (100ns resolution)
        _timerResolutionNanos = Math.Max(1L, (long)(1_000_000_000.0 / _clockFrequencyHz));

        // Record base timestamp for consistent offset calculations
        _baseTimestamp = Stopwatch.GetTimestamp();

        _logger.LogInformation(
            "CpuTimingProvider initialized: Resolution={ResolutionNs}ns, Frequency={FrequencyHz}Hz",
            _timerResolutionNanos, _clockFrequencyHz);
    }

    /// <inheritdoc/>
    public Task<long> GetGpuTimestampAsync(CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        return Task.FromResult(GetCurrentTimestampNanos());
    }

    /// <inheritdoc/>
    public Task<long[]> GetGpuTimestampsBatchAsync(int count, CancellationToken ct = default)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(count, 0, nameof(count));
        ct.ThrowIfCancellationRequested();

        var timestamps = new long[count];
        for (int i = 0; i < count; i++)
        {
            ct.ThrowIfCancellationRequested();
            timestamps[i] = GetCurrentTimestampNanos();
        }

        return Task.FromResult(timestamps);
    }

    /// <inheritdoc/>
    public async Task<ClockCalibration> CalibrateAsync(int sampleCount = 100, CancellationToken ct = default)
    {
        if (sampleCount < 10)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleCount),
                "Sample count must be at least 10 for reliable calibration.");
        }

        _logger.LogInformation("Starting CPU clock calibration with {SampleCount} samples...", sampleCount);

        long calibrationStartTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // For CPU provider, GPU time == CPU time (no real offset)
        // We still collect samples to measure the provider's jitter/error
        var offsets = new long[sampleCount];
        var cpuTimes = new long[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            ct.ThrowIfCancellationRequested();

            // Get both timestamps as close together as possible
            long cpuTime1 = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
            long providerTime = await GetGpuTimestampAsync(ct);
            long cpuTime2 = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

            // Use midpoint for best estimate
            cpuTimes[i] = (cpuTime1 + cpuTime2) / 2;
            offsets[i] = providerTime - cpuTimes[i];

            // Small delay between samples
            if (i < sampleCount - 1)
            {
                await Task.Delay(1, ct);
            }
        }

        // Calculate median offset (robust to outliers)
        Array.Sort(offsets);
        long medianOffset = sampleCount % 2 == 0
            ? (offsets[sampleCount / 2 - 1] + offsets[sampleCount / 2]) / 2
            : offsets[sampleCount / 2];

        // Calculate drift using linear regression
        double driftPPM = CalculateDrift(cpuTimes, offsets, calibrationStartTime);

        // Calculate error bound (3-sigma)
        double variance = offsets.Select(o => Math.Pow(o - medianOffset, 2)).Average();
        long errorBound = (long)(3.0 * Math.Sqrt(variance));

        var calibration = new ClockCalibration(
            offsetNanos: medianOffset,
            driftPPM: driftPPM,
            errorBoundNanos: errorBound,
            sampleCount: sampleCount,
            calibrationTimestampNanos: calibrationStartTime);

        _logger.LogInformation(
            "CPU clock calibration complete: Offset={OffsetNs}ns, Drift={DriftPPM:F3}ppm, Error=±{ErrorNs}ns",
            calibration.OffsetNanos, calibration.DriftPPM, calibration.ErrorBoundNanos);

        return calibration;
    }

    /// <inheritdoc/>
    public void EnableTimestampInjection(bool enable = true)
    {
        _timestampInjectionEnabled = enable;
        _logger.LogDebug("Timestamp injection {Action} (CPU fallback - no actual injection)",
            enable ? "enabled" : "disabled");
    }

    /// <inheritdoc/>
    public bool IsTimestampInjectionEnabled => _timestampInjectionEnabled;

    /// <inheritdoc/>
    public long GetGpuClockFrequency() => _clockFrequencyHz;

    /// <inheritdoc/>
    public long GetTimerResolutionNanos() => _timerResolutionNanos;

    /// <inheritdoc/>
    public bool IsGpuBacked => false;

    /// <inheritdoc/>
    public string ProviderTypeName => "CPU Fallback (Stopwatch)";

    /// <summary>
    /// Gets the current timestamp in nanoseconds using Stopwatch.
    /// </summary>
    private long GetCurrentTimestampNanos()
    {
        long ticks = Stopwatch.GetTimestamp();
        // Convert to nanoseconds: ticks / frequency * 1e9
        return (long)((ticks / (double)_clockFrequencyHz) * 1_000_000_000);
    }

    /// <summary>
    /// Calculates clock drift using linear regression.
    /// </summary>
    private static double CalculateDrift(long[] cpuTimes, long[] offsets, long startTime)
    {
        if (cpuTimes.Length < 2)
            return 0.0;

        double n = cpuTimes.Length;
        double sumTime = 0, sumOffset = 0, sumTimeOffset = 0, sumTimeSquared = 0;

        for (int i = 0; i < cpuTimes.Length; i++)
        {
            double time = cpuTimes[i] - startTime;
            sumTime += time;
            sumOffset += offsets[i];
            sumTimeOffset += time * offsets[i];
            sumTimeSquared += time * time;
        }

        double meanTime = sumTime / n;
        double meanOffset = sumOffset / n;

        double covariance = (sumTimeOffset / n) - (meanTime * meanOffset);
        double variance = (sumTimeSquared / n) - (meanTime * meanTime);

        if (Math.Abs(variance) < double.Epsilon)
            return 0.0;

        // Drift in parts per million
        return (covariance / variance) * 1_000_000.0;
    }
}
