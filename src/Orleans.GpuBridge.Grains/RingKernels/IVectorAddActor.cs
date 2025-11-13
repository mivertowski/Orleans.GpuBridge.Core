// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using Orleans.GpuBridge.Runtime.RingKernels;

namespace Orleans.GpuBridge.Grains.RingKernels;

/// <summary>
/// GPU-native actor for vector addition operations using persistent ring kernels.
/// </summary>
/// <remarks>
/// <para>
/// This is the first proof-of-concept GPU-native actor demonstrating:
/// - Ring kernel lifecycle management
/// - GPU-to-GPU messaging with sub-microsecond latency
/// - Zero kernel launch overhead after initial activation
/// </para>
/// <para>
/// Performance targets:
/// - First call: ~10-50ms (kernel compilation + launch)
/// - Subsequent calls: 100-500ns (GPU queue operations)
/// - Throughput: 2M operations/second
/// </para>
/// </remarks>
public interface IVectorAddActor : IGpuNativeGrain
{
    /// <summary>
    /// Adds two vectors element-wise on GPU using persistent ring kernel.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>Result vector where result[i] = a[i] + b[i].</returns>
    /// <remarks>
    /// <para>
    /// This method demonstrates GPU-native message passing:
    /// 1. CPU serializes vectors to GPU message
    /// 2. Message enqueued to GPU ring buffer (~100-500ns)
    /// 3. GPU kernel dequeues and processes
    /// 4. GPU kernel enqueues result
    /// 5. CPU dequeues result (~100-500ns)
    /// </para>
    /// <para>
    /// For large vectors (&gt; 200 elements), vector data is stored in GPU memory
    /// and only pointers are passed in the 228-byte message payload.
    /// </para>
    /// </remarks>
    Task<float[]> AddVectorsAsync(float[] a, float[] b);

    /// <summary>
    /// Adds two vectors and returns their sum as a scalar.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>Sum of all elements in result vector.</returns>
    /// <remarks>
    /// This variant demonstrates scalar reduction on GPU.
    /// Useful for benchmarking pure messaging latency.
    /// </remarks>
    Task<float> AddVectorsScalarAsync(float[] a, float[] b);

    /// <summary>
    /// Gets current performance metrics from the GPU ring kernel.
    /// </summary>
    /// <returns>Ring kernel performance metrics.</returns>
    /// <remarks>
    /// Useful for monitoring:
    /// - Messages processed per second
    /// - Average processing time
    /// - Queue utilization
    /// - GPU memory usage
    /// </remarks>
    Task<VectorAddMetrics> GetMetricsAsync();
}

/// <summary>
/// Performance metrics for VectorAddActor ring kernel.
/// </summary>
public record VectorAddMetrics
{
    /// <summary>
    /// Gets the total number of vector additions processed.
    /// </summary>
    public long TotalOperations { get; init; }

    /// <summary>
    /// Gets the average processing time in nanoseconds.
    /// </summary>
    public double AvgProcessingTimeNs { get; init; }

    /// <summary>
    /// Gets the messages processed per second.
    /// </summary>
    public double ThroughputMsgsPerSec { get; init; }

    /// <summary>
    /// Gets the input queue utilization (0.0-1.0).
    /// </summary>
    public double InputQueueUtilization { get; init; }

    /// <summary>
    /// Gets the output queue utilization (0.0-1.0).
    /// </summary>
    public double OutputQueueUtilization { get; init; }

    /// <summary>
    /// Gets the current GPU memory usage in bytes.
    /// </summary>
    public long GpuMemoryBytes { get; init; }

    /// <summary>
    /// Gets the ring kernel uptime.
    /// </summary>
    public TimeSpan Uptime { get; init; }
}
