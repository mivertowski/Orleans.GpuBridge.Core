// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Provides GPU-native timing capabilities for high-precision temporal measurements.
/// This interface abstracts over different timing backends (CUDA, OpenCL, CPU fallback).
/// </summary>
/// <remarks>
/// <para>
/// The timing provider enables nanosecond-precision timestamp generation directly on GPU hardware,
/// eliminating CPU-GPU round-trip latency. This is critical for applications requiring precise
/// temporal ordering such as physics simulations, real-time systems, and distributed GPU computing.
/// </para>
/// <para>
/// <strong>Platform Support:</strong>
/// <list type="bullet">
/// <item><description>CUDA (CC 6.0+): 1ns resolution via %%globaltimer register</description></item>
/// <item><description>CUDA (CC &lt; 6.0): 1μs resolution via CUDA events</description></item>
/// <item><description>OpenCL: 1μs resolution via clock() built-in</description></item>
/// <item><description>CPU: ~100ns resolution via Stopwatch</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IGpuTimingProvider
{
    /// <summary>
    /// Gets the current GPU timestamp in nanoseconds since device initialization.
    /// </summary>
    /// <param name="ct">Cancellation token to cancel the async operation.</param>
    /// <returns>
    /// A task representing the async operation, containing the GPU timestamp in nanoseconds.
    /// The timestamp is monotonically increasing and has device-specific resolution.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method launches a minimal kernel to read the GPU hardware timer. The overhead
    /// is typically &lt;10ns on CUDA (CC 6.0+) and &lt;100ns on other platforms.
    /// </para>
    /// <para>
    /// <strong>Performance Targets:</strong>
    /// <list type="bullet">
    /// <item><description>CUDA (CC 6.0+): &lt;10ns per query</description></item>
    /// <item><description>CUDA Events: &lt;100ns per query</description></item>
    /// <item><description>OpenCL/CPU: &lt;1μs per query</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <exception cref="OperationCanceledException">
    /// Thrown when the cancellation token is triggered.
    /// </exception>
    Task<long> GetGpuTimestampAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets multiple GPU timestamps in a single batch operation for improved efficiency.
    /// </summary>
    /// <param name="count">Number of timestamps to retrieve (must be positive).</param>
    /// <param name="ct">Cancellation token to cancel the async operation.</param>
    /// <returns>
    /// A task representing the async operation, containing an array of GPU timestamps in nanoseconds.
    /// All timestamps are captured within a single kernel launch for minimal skew.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Batch queries amortize kernel launch overhead across multiple timestamps, achieving
    /// &lt;1μs per timestamp when <paramref name="count"/> ≥ 1000.
    /// </para>
    /// </remarks>
    /// <exception cref="System.ArgumentOutOfRangeException">
    /// Thrown when <paramref name="count"/> is less than or equal to zero.
    /// </exception>
    Task<long[]> GetGpuTimestampsBatchAsync(int count, CancellationToken ct = default);

    /// <summary>
    /// Calibrates the GPU clock against the CPU clock to enable accurate time conversions.
    /// </summary>
    /// <param name="sampleCount">
    /// Number of CPU-GPU timestamp pairs to collect for calibration (default: 100).
    /// Higher values improve accuracy but increase calibration time.
    /// </param>
    /// <param name="ct">Cancellation token to cancel the async operation.</param>
    /// <returns>
    /// A task representing the async operation, containing calibration data including
    /// offset, drift rate, and error bounds for converting between CPU and GPU time domains.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Clock calibration performs linear regression on <paramref name="sampleCount"/> paired
    /// CPU-GPU timestamps to compute:
    /// <list type="bullet">
    /// <item><description><strong>Offset</strong>: GPU_time = CPU_time + offset</description></item>
    /// <item><description><strong>Drift</strong>: Clock frequency difference (parts per million)</description></item>
    /// <item><description><strong>Error Bounds</strong>: ±uncertainty range from regression residuals</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <strong>Performance:</strong>
    /// <list type="bullet">
    /// <item><description>100 samples: ~10ms calibration time</description></item>
    /// <item><description>Typical drift: 50-200 PPM (180-720μs/hour)</description></item>
    /// <item><description>Recommended recalibration interval: 5-10 minutes</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <exception cref="System.ArgumentOutOfRangeException">
    /// Thrown when <paramref name="sampleCount"/> is less than 10 (insufficient for calibration).
    /// </exception>
    Task<ClockCalibration> CalibrateAsync(int sampleCount = 100, CancellationToken ct = default);

    /// <summary>
    /// Enables automatic timestamp injection at kernel entry points.
    /// </summary>
    /// <param name="enable">True to enable injection, false to disable.</param>
    /// <remarks>
    /// <para>
    /// When enabled, kernels automatically record a timestamp in parameter slot 0 before
    /// executing user code. This eliminates manual timestamp management in kernel code.
    /// </para>
    /// <para>
    /// <strong>Overhead:</strong> &lt;20ns per kernel launch (timestamp write by thread 0).
    /// </para>
    /// <para>
    /// <strong>Note:</strong> Timestamp injection requires kernel recompilation. Existing
    /// compiled kernels will not be affected until next compilation.
    /// </para>
    /// </remarks>
    void EnableTimestampInjection(bool enable = true);

    /// <summary>
    /// Gets whether timestamp injection is currently enabled.
    /// </summary>
    bool IsTimestampInjectionEnabled { get; }

    /// <summary>
    /// Gets the GPU clock frequency in Hertz (cycles per second).
    /// </summary>
    /// <returns>
    /// The GPU clock frequency in Hz. Typical values:
    /// <list type="bullet">
    /// <item><description>CUDA: 1,000,000,000 Hz (1 GHz) for nanosecond timers</description></item>
    /// <item><description>CUDA Events: 1,000,000 Hz (1 MHz) for microsecond precision</description></item>
    /// <item><description>OpenCL: Platform-dependent</description></item>
    /// </list>
    /// </returns>
    long GetGpuClockFrequency();

    /// <summary>
    /// Gets the timer resolution in nanoseconds (minimum measurable time interval).
    /// </summary>
    /// <returns>
    /// The timer resolution in nanoseconds. Typical values:
    /// <list type="bullet">
    /// <item><description>CUDA (CC 6.0+): 1 ns (%%globaltimer)</description></item>
    /// <item><description>CUDA (CC &lt; 6.0): 1,000 ns (CUDA events)</description></item>
    /// <item><description>OpenCL: 1,000 ns (clock() built-in)</description></item>
    /// <item><description>CPU: ~100 ns (Stopwatch)</description></item>
    /// </list>
    /// </returns>
    long GetTimerResolutionNanos();

    /// <summary>
    /// Gets whether this provider is backed by GPU hardware.
    /// </summary>
    /// <remarks>
    /// Returns false for CPU fallback providers, allowing applications to detect
    /// when real GPU timing is unavailable.
    /// </remarks>
    bool IsGpuBacked { get; }

    /// <summary>
    /// Gets the provider type name for diagnostics.
    /// </summary>
    /// <remarks>
    /// Examples: "CUDA (CC 8.6)", "OpenCL", "CPU Fallback"
    /// </remarks>
    string ProviderTypeName { get; }
}
