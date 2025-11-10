using System;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Timing;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Adapter for DotCompute's ITimingProvider - enables GPU-native nanosecond timing.
/// Provides GPU timestamp queries, clock calibration, and automatic timestamp injection.
/// </summary>
public sealed class DotComputeTimingProvider : IDisposable
{
    private readonly ITimingProvider _dotComputeTiming;
    private readonly ILogger<DotComputeTimingProvider> _logger;
    private ClockCalibration? _cachedCalibration;
    private DateTime _calibrationTime;
    private readonly TimeSpan _recalibrationInterval = TimeSpan.FromMinutes(5);
    private bool _disposed;

    public DotComputeTimingProvider(
        ITimingProvider dotComputeTiming,
        ILogger<DotComputeTimingProvider> logger)
    {
        _dotComputeTiming = dotComputeTiming ?? throw new ArgumentNullException(nameof(dotComputeTiming));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _logger.LogInformation(
            "DotComputeTimingProvider initialized - GPU clock frequency: {Frequency}Hz, Resolution: {Resolution}ns",
            _dotComputeTiming.GetGpuClockFrequency(),
            _dotComputeTiming.GetTimerResolutionNanos());
    }

    /// <summary>
    /// Gets GPU timestamp in nanoseconds.
    /// CUDA: Uses %%globaltimer register (1ns resolution)
    /// OpenCL: Uses clock() built-in (microsecond resolution)
    /// CPU: Uses Stopwatch (100ns resolution)
    /// </summary>
    public async Task<long> GetGpuTimestampAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            return await _dotComputeTiming.GetGpuTimestampAsync(ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get GPU timestamp");
            throw;
        }
    }

    /// <summary>
    /// Gets multiple GPU timestamps in batch (more efficient than individual queries).
    /// Amortized cost: ~1μs for 1000 timestamps on CUDA.
    /// </summary>
    public async Task<long[]> GetGpuTimestampsBatchAsync(int count, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);

        try
        {
            return await _dotComputeTiming.GetGpuTimestampsBatchAsync(count, ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get batch GPU timestamps (count: {Count})", count);
            throw;
        }
    }

    /// <summary>
    /// Calibrates GPU clock against CPU clock.
    /// Performs multiple round-trip measurements to determine offset, drift, and error bounds.
    /// Should be called periodically (every 5 minutes) to account for clock drift.
    /// </summary>
    public async Task<ClockCalibration> CalibrateAsync(
        int sampleCount = 100,
        bool forceRecalibration = false,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(sampleCount, 10);

        // Return cached calibration if recent enough
        if (!forceRecalibration &&
            _cachedCalibration.HasValue &&
            DateTime.UtcNow - _calibrationTime < _recalibrationInterval)
        {
            _logger.LogDebug(
                "Returning cached calibration (age: {Age}s)",
                (DateTime.UtcNow - _calibrationTime).TotalSeconds);
            return _cachedCalibration.Value;
        }

        try
        {
            _logger.LogInformation("Calibrating GPU clock with {SampleCount} samples...", sampleCount);

            var calibration = await _dotComputeTiming.CalibrateAsync(sampleCount, ct).ConfigureAwait(false);

            _cachedCalibration = calibration;
            _calibrationTime = DateTime.UtcNow;

            _logger.LogInformation(
                "Clock calibration complete - Offset: {Offset}ns, Drift: {Drift}ppm, Error: ±{Error}ns",
                calibration.OffsetNanos,
                calibration.DriftPPM,
                calibration.ErrorBoundNanos);

            return calibration;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Clock calibration failed");
            throw;
        }
    }

    /// <summary>
    /// Enables automatic timestamp injection at kernel entry points.
    /// When enabled, kernels automatically record entry timestamp in parameter slot 0.
    /// Application must allocate device memory for timestamps.
    /// Overhead: ~10-20ns per kernel launch.
    /// </summary>
    public void EnableTimestampInjection(bool enable = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            _dotComputeTiming.EnableTimestampInjection(enable);

            _logger.LogInformation(
                "Timestamp injection {Status}",
                enable ? "enabled" : "disabled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to configure timestamp injection");
            throw;
        }
    }

    /// <summary>
    /// Gets GPU clock frequency in Hz.
    /// Typical values: 1-2 GHz for modern GPUs.
    /// </summary>
    public long GetGpuClockFrequency()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _dotComputeTiming.GetGpuClockFrequency();
    }

    /// <summary>
    /// Gets timer resolution in nanoseconds.
    /// CUDA: 1ns (%%globaltimer)
    /// OpenCL: 1000ns (microsecond resolution)
    /// CPU: 100ns (Stopwatch)
    /// </summary>
    public long GetTimerResolutionNanos()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _dotComputeTiming.GetTimerResolutionNanos();
    }

    /// <summary>
    /// Converts GPU timestamp to CPU time using calibration data.
    /// Requires recent calibration (call CalibrateAsync first).
    /// </summary>
    public long GpuToCpuTime(long gpuTimeNanos)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_cachedCalibration.HasValue)
        {
            throw new InvalidOperationException(
                "No calibration data available. Call CalibrateAsync first.");
        }

        return _cachedCalibration.Value.GpuToCpuTime(gpuTimeNanos);
    }

    /// <summary>
    /// Gets uncertainty range for timestamp conversion (min, max).
    /// </summary>
    public (long min, long max) GetUncertaintyRange(long gpuTimeNanos)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_cachedCalibration.HasValue)
        {
            throw new InvalidOperationException(
                "No calibration data available. Call CalibrateAsync first.");
        }

        return _cachedCalibration.Value.GetUncertaintyRange(gpuTimeNanos);
    }

    /// <summary>
    /// Gets current calibration status.
    /// </summary>
    public (bool IsCalibrated, TimeSpan Age, ClockCalibration? Calibration) GetCalibrationStatus()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_cachedCalibration.HasValue)
        {
            return (false, TimeSpan.Zero, null);
        }

        var age = DateTime.UtcNow - _calibrationTime;
        return (true, age, _cachedCalibration.Value);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            // Disable timestamp injection on cleanup
            if (_dotComputeTiming != null)
            {
                _dotComputeTiming.EnableTimestampInjection(false);
            }

            _disposed = true;
            _logger.LogDebug("DotComputeTimingProvider disposed");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error during DotComputeTimingProvider disposal");
        }
    }
}
