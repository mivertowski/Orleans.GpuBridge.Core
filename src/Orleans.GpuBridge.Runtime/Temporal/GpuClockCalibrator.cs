using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Calibrates GPU clock against CPU time for temporal correctness.
/// Performs statistical analysis of clock offset and drift to enable accurate time conversion.
/// </summary>
/// <remarks>
/// <para>
/// This calibrator uses an <see cref="IGpuTimingProvider"/> to query GPU timestamps
/// and performs linear regression to determine:
/// <list type="bullet">
/// <item><description><strong>Offset</strong>: Constant difference between GPU and CPU time</description></item>
/// <item><description><strong>Drift</strong>: Rate at which clocks diverge (PPM)</description></item>
/// <item><description><strong>Error Bound</strong>: Uncertainty in offset measurement</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Automatic Recalibration:</strong> Calibration results are cached and automatically
/// refreshed when stale (default: 5 minutes). Call <see cref="GetCalibrationAsync"/> to get
/// the current calibration, which will refresh if needed.
/// </para>
/// </remarks>
public sealed class GpuClockCalibrator : IDisposable
{
    private readonly ILogger<GpuClockCalibrator> _logger;
    private readonly IGpuTimingProvider? _timingProvider;
    private ClockCalibration? _currentCalibration;
    private readonly SemaphoreSlim _calibrationLock = new(1, 1);
    private bool _disposed;

    /// <summary>
    /// Default calibration interval (5 minutes).
    /// </summary>
    public static readonly TimeSpan DefaultCalibrationInterval = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuClockCalibrator"/> class
    /// with a timing provider for real GPU timestamps.
    /// </summary>
    /// <param name="timingProvider">GPU timing provider for timestamp queries.</param>
    /// <param name="logger">Logger for diagnostic output.</param>
    public GpuClockCalibrator(IGpuTimingProvider timingProvider, ILogger<GpuClockCalibrator> logger)
    {
        _timingProvider = timingProvider ?? throw new ArgumentNullException(nameof(timingProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _logger.LogInformation(
            "GpuClockCalibrator initialized with {ProviderType} (GPU-backed: {IsGpuBacked})",
            _timingProvider.ProviderTypeName,
            _timingProvider.IsGpuBacked);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuClockCalibrator"/> class
    /// without a timing provider (uses simulated timestamps for testing).
    /// </summary>
    /// <param name="logger">Logger for diagnostic output.</param>
    /// <remarks>
    /// This constructor is provided for backward compatibility and testing scenarios.
    /// Production code should use the constructor that accepts <see cref="IGpuTimingProvider"/>.
    /// </remarks>
    public GpuClockCalibrator(ILogger<GpuClockCalibrator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _timingProvider = null; // Will use simulated timestamps

        _logger.LogWarning(
            "GpuClockCalibrator initialized without timing provider - using simulated timestamps. " +
            "For production use, provide IGpuTimingProvider via dependency injection.");
    }

    /// <summary>
    /// Gets whether this calibrator is using real GPU timestamps.
    /// </summary>
    public bool HasGpuTimingProvider => _timingProvider != null;

    /// <summary>
    /// Gets the timing provider type name, or "Simulated" if no provider.
    /// </summary>
    public string TimingProviderType => _timingProvider?.ProviderTypeName ?? "Simulated";

    /// <summary>
    /// Gets the current clock calibration (performs calibration if not cached or stale).
    /// </summary>
    public async Task<ClockCalibration> GetCalibrationAsync(CancellationToken ct = default)
    {
        // Check if calibration exists and is fresh
        if (_currentCalibration != null)
        {
            long currentTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
            if (!_currentCalibration.Value.IsStale(currentTime))
            {
                return _currentCalibration.Value;
            }
        }

        // Perform new calibration
        await _calibrationLock.WaitAsync(ct);
        try
        {
            // Double-check after acquiring lock
            if (_currentCalibration != null)
            {
                long currentTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
                if (!_currentCalibration.Value.IsStale(currentTime))
                {
                    return _currentCalibration.Value;
                }
            }

            _currentCalibration = await CalibrateAsync(sampleCount: 1000, ct);
            return _currentCalibration.Value;
        }
        finally
        {
            _calibrationLock.Release();
        }
    }

    /// <summary>
    /// Performs GPU-CPU clock calibration with specified sample count.
    /// Uses statistical analysis to determine offset, drift, and error bounds.
    /// </summary>
    /// <param name="sampleCount">Number of round-trip samples to collect (more = more accurate).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Clock calibration result.</returns>
    public async Task<ClockCalibration> CalibrateAsync(
        int sampleCount = 100,
        CancellationToken ct = default)
    {
        if (sampleCount < 10)
            throw new ArgumentException("Sample count must be at least 10.", nameof(sampleCount));

        _logger.LogInformation("Starting GPU clock calibration with {SampleCount} samples...", sampleCount);

        long calibrationStartTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Collect timestamp samples
        // In a real implementation, this would call DotCompute timing API
        // For now, we simulate GPU timestamps with CPU time + synthetic offset
        var samples = await CollectTimestampSamplesAsync(sampleCount, ct);

        // Calculate offset (median to reject outliers)
        long medianOffset = CalculateMedian(samples.Select(s => s.offset).ToArray());

        // Calculate drift (linear regression of offset over time)
        double driftPPM = CalculateDrift(samples, calibrationStartTime);

        // Calculate error bound (standard deviation * 3 for 99.7% confidence)
        long errorBound = CalculateErrorBound(samples, medianOffset);

        var calibration = new ClockCalibration(
            offsetNanos: medianOffset,
            driftPPM: driftPPM,
            errorBoundNanos: errorBound,
            sampleCount: sampleCount,
            calibrationTimestampNanos: calibrationStartTime);

        _logger.LogInformation(
            "GPU clock calibration complete: Offset={OffsetNs}ns, Drift={DriftPPM:F3}ppm, Error=±{ErrorBoundNs}ns",
            calibration.OffsetNanos,
            calibration.DriftPPM,
            calibration.ErrorBoundNanos);

        return calibration;
    }

    /// <summary>
    /// Converts GPU timestamp to CPU time using current calibration.
    /// </summary>
    public long GpuToCpuTime(long gpuTimeNanos)
    {
        if (_currentCalibration == null)
        {
            throw new InvalidOperationException("Clock not calibrated. Call GetCalibrationAsync() first.");
        }

        return _currentCalibration.Value.GpuToCpuTime(gpuTimeNanos);
    }

    /// <summary>
    /// Converts CPU timestamp to GPU time using current calibration.
    /// </summary>
    public long CpuToGpuTime(long cpuTimeNanos)
    {
        if (_currentCalibration == null)
        {
            throw new InvalidOperationException("Clock not calibrated. Call GetCalibrationAsync() first.");
        }

        return _currentCalibration.Value.CpuToGpuTime(cpuTimeNanos);
    }

    /// <summary>
    /// Collects timestamp samples by querying both GPU and CPU clocks.
    /// </summary>
    private async Task<(long cpuTime, long gpuTime, long offset)[]> CollectTimestampSamplesAsync(
        int sampleCount,
        CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var samples = new (long cpuTime, long gpuTime, long offset)[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            ct.ThrowIfCancellationRequested();

            // Get CPU time
            long cpuTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

            // Get GPU time - use real provider if available, otherwise simulate
            long gpuTime = _timingProvider != null
                ? await _timingProvider.GetGpuTimestampAsync(ct)
                : SimulateGpuTimestamp(cpuTime);

            // Calculate offset
            long offset = gpuTime - cpuTime;

            samples[i] = (cpuTime, gpuTime, offset);

            // Small delay between samples to get temporal distribution
            if (i < sampleCount - 1)
            {
                await Task.Delay(1, ct);
            }
        }

        return samples;
    }

    /// <summary>
    /// Simulates GPU timestamp query for testing and fallback scenarios.
    /// </summary>
    /// <param name="cpuTime">Current CPU time in nanoseconds.</param>
    /// <returns>Simulated GPU timestamp with synthetic offset and drift.</returns>
    private static long SimulateGpuTimestamp(long cpuTime)
    {
        // Simulate GPU clock with synthetic offset and drift
        const long syntheticOffset = 1_000_000_000L; // 1 second offset
        const double syntheticDriftPPM = 10.0; // 10 PPM drift

        long drift = (long)(cpuTime * (syntheticDriftPPM / 1_000_000.0));

        return cpuTime + syntheticOffset + drift;
    }

    /// <summary>
    /// Calculates median value from samples (robust to outliers).
    /// </summary>
    private static long CalculateMedian(long[] values)
    {
        if (values.Length == 0)
            throw new ArgumentException("Cannot calculate median of empty array.");

        Array.Sort(values);
        int mid = values.Length / 2;

        if (values.Length % 2 == 0)
        {
            return (values[mid - 1] + values[mid]) / 2;
        }
        else
        {
            return values[mid];
        }
    }

    /// <summary>
    /// Calculates clock drift using linear regression.
    /// </summary>
    private static double CalculateDrift(
        (long cpuTime, long gpuTime, long offset)[] samples,
        long startTime)
    {
        if (samples.Length < 2)
            return 0.0;

        // Linear regression: offset = drift * time + constant
        // drift = covariance(time, offset) / variance(time)

        double n = samples.Length;
        double sumTime = 0;
        double sumOffset = 0;
        double sumTimeOffset = 0;
        double sumTimeSquared = 0;

        foreach (var (cpuTime, _, offset) in samples)
        {
            double time = cpuTime - startTime;
            sumTime += time;
            sumOffset += offset;
            sumTimeOffset += time * offset;
            sumTimeSquared += time * time;
        }

        double meanTime = sumTime / n;
        double meanOffset = sumOffset / n;

        double covariance = (sumTimeOffset / n) - (meanTime * meanOffset);
        double variance = (sumTimeSquared / n) - (meanTime * meanTime);

        if (variance == 0)
            return 0.0;

        // Drift in parts per million (PPM)
        double driftFraction = covariance / variance;
        return driftFraction * 1_000_000.0;
    }

    /// <summary>
    /// Calculates error bound using standard deviation (3-sigma for 99.7% confidence).
    /// </summary>
    private static long CalculateErrorBound(
        (long cpuTime, long gpuTime, long offset)[] samples,
        long medianOffset)
    {
        if (samples.Length < 2)
            return 0;

        // Calculate standard deviation of offsets
        double sumSquaredDiff = 0;
        foreach (var (_, _, offset) in samples)
        {
            double diff = offset - medianOffset;
            sumSquaredDiff += diff * diff;
        }

        double variance = sumSquaredDiff / samples.Length;
        double stdDev = Math.Sqrt(variance);

        // 3-sigma for 99.7% confidence interval
        return (long)(3.0 * stdDev);
    }

    public void Dispose()
    {
        _calibrationLock.Dispose();
    }
}
