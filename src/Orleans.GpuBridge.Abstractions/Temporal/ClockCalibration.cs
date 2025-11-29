using System;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Results of GPU-CPU clock calibration.
/// Used to convert between GPU timestamps and CPU time.
/// </summary>
/// <remarks>
/// Clock calibration accounts for:
/// 1. Offset: Constant difference between GPU and CPU time bases
/// 2. Drift: Rate at which GPU and CPU clocks diverge (PPM)
/// 3. Error bound: Uncertainty in offset measurement (±ns)
/// </remarks>
public readonly struct ClockCalibration
{
    /// <summary>
    /// Clock offset in nanoseconds (GPU_time - CPU_time).
    /// </summary>
    public long OffsetNanos { get; init; }

    /// <summary>
    /// Clock drift rate in parts per million (PPM).
    /// Positive values indicate GPU clock runs faster than CPU clock.
    /// </summary>
    public double DriftPPM { get; init; }

    /// <summary>
    /// Error bound in nanoseconds (±).
    /// Actual offset is within [OffsetNanos - ErrorBoundNanos, OffsetNanos + ErrorBoundNanos].
    /// </summary>
    public long ErrorBoundNanos { get; init; }

    /// <summary>
    /// Number of samples used for calibration.
    /// More samples = lower error bound but slower calibration.
    /// </summary>
    public int SampleCount { get; init; }

    /// <summary>
    /// Calibration timestamp (CPU time in nanoseconds since Unix epoch).
    /// Used for drift correction over time.
    /// </summary>
    public long CalibrationTimestampNanos { get; init; }

    /// <summary>
    /// Creates a new clock calibration result.
    /// </summary>
    public ClockCalibration(
        long offsetNanos,
        double driftPPM,
        long errorBoundNanos,
        int sampleCount,
        long calibrationTimestampNanos)
    {
        OffsetNanos = offsetNanos;
        DriftPPM = driftPPM;
        ErrorBoundNanos = errorBoundNanos;
        SampleCount = sampleCount;
        CalibrationTimestampNanos = calibrationTimestampNanos;
    }

    /// <summary>
    /// Converts GPU timestamp to CPU time using calibration data.
    /// Accounts for both offset and drift since calibration.
    /// </summary>
    /// <param name="gpuTimeNanos">GPU timestamp in nanoseconds.</param>
    /// <returns>Equivalent CPU time in nanoseconds.</returns>
    public long GpuToCpuTime(long gpuTimeNanos)
    {
        // Compensate for drift since calibration
        long elapsedSinceCalibration = gpuTimeNanos - CalibrationTimestampNanos;
        long driftCorrection = (long)(elapsedSinceCalibration * (DriftPPM / 1_000_000.0));

        // Apply offset and drift correction
        return gpuTimeNanos - OffsetNanos - driftCorrection;
    }

    /// <summary>
    /// Converts CPU timestamp to GPU time using calibration data.
    /// </summary>
    /// <param name="cpuTimeNanos">CPU timestamp in nanoseconds.</param>
    /// <returns>Equivalent GPU time in nanoseconds.</returns>
    public long CpuToGpuTime(long cpuTimeNanos)
    {
        // Reverse conversion: GPU = CPU + offset + drift
        long elapsedSinceCalibration = cpuTimeNanos - CalibrationTimestampNanos;
        long driftCorrection = (long)(elapsedSinceCalibration * (DriftPPM / 1_000_000.0));

        return cpuTimeNanos + OffsetNanos + driftCorrection;
    }

    /// <summary>
    /// Gets uncertainty range for timestamp conversion.
    /// </summary>
    /// <param name="gpuTimeNanos">GPU timestamp to convert.</param>
    /// <returns>Min and max possible CPU times accounting for error bound.</returns>
    public (long min, long max) GetUncertaintyRange(long gpuTimeNanos)
    {
        long cpuTime = GpuToCpuTime(gpuTimeNanos);
        return (cpuTime - ErrorBoundNanos, cpuTime + ErrorBoundNanos);
    }

    /// <summary>
    /// Checks if calibration is stale and needs refresh.
    /// </summary>
    /// <param name="currentTimeNanos">Current CPU time in nanoseconds.</param>
    /// <param name="maxAgeNanos">Maximum calibration age before refresh (default: 5 minutes).</param>
    /// <returns>True if calibration needs refresh.</returns>
    public bool IsStale(long currentTimeNanos, long maxAgeNanos = 300_000_000_000L)
    {
        long age = currentTimeNanos - CalibrationTimestampNanos;
        return age > maxAgeNanos || Math.Abs(DriftPPM) > 1000.0;
    }

    /// <summary>
    /// Returns a string representation of the calibration showing offset, drift, error bounds, and sample count.
    /// </summary>
    /// <returns>A formatted string describing the calibration.</returns>
    public override string ToString() =>
        $"Calibration(Offset={OffsetNanos}ns, Drift={DriftPPM:F3}ppm, Error=±{ErrorBoundNanos}ns, Samples={SampleCount})";
}
