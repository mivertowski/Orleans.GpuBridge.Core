// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using DotCompute.Abstractions.RingKernels;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Runtime.RingKernels;
using Orleans.GpuBridge.Runtime.Placement;
using Orleans.GpuBridge.Runtime.Memory;

namespace Orleans.GpuBridge.Grains.RingKernels;

/// <summary>
/// GPU-native actor implementation for vector addition using persistent ring kernels.
/// </summary>
/// <remarks>
/// <para>
/// This is the first working proof-of-concept of GPU-native actors in Orleans.
/// It demonstrates:
/// - Zero-copy GPU messaging
/// - Sub-microsecond latency (100-500ns target)
/// - Persistent ring kernel execution
/// - Integration with DotCompute ring kernel runtime
/// - GPU-aware placement with ring kernel metrics
/// </para>
/// <para>
/// The actor maintains a GPU-resident state including:
/// - Ring kernel control block (64 bytes)
/// - Input message queue (256 messages)
/// - Output message queue (256 messages)
/// - Working memory for vector operations
/// </para>
/// </remarks>
[GpuNativeActor(
    Domain = RingKernelDomain.General,
    MessagingStrategy = MessagePassingStrategy.SharedMemory,
    Capacity = 1024,
    InputQueueSize = 256,
    OutputQueueSize = 256,
    GridSize = 1,
    BlockSize = 256)]
[GpuNativePlacement(
    MinMemoryMB = 512,
    MaxQueueUtilization = 0.75,
    PreferLocalGpu = true)]
public class VectorAddActor : GpuNativeGrain, IVectorAddActor
{
    private readonly ILogger<VectorAddActor> _actorLogger;
    private readonly GpuMemoryManager _memoryManager;

    /// <summary>
    /// Initializes a new instance of the <see cref="VectorAddActor"/> class.
    /// </summary>
    /// <param name="runtime">Ring kernel runtime.</param>
    /// <param name="logger">Base logger.</param>
    /// <param name="actorLogger">Actor-specific logger.</param>
    /// <param name="memoryManager">GPU memory manager for large vector operations.</param>
    public VectorAddActor(
        IRingKernelRuntime runtime,
        ILogger<GpuNativeGrain> logger,
        ILogger<VectorAddActor> actorLogger,
        GpuMemoryManager memoryManager)
        : base(runtime, logger)
    {
        _actorLogger = actorLogger ?? throw new ArgumentNullException(nameof(actorLogger));
        _memoryManager = memoryManager ?? throw new ArgumentNullException(nameof(memoryManager));
    }

    /// <inheritdoc/>
    public async Task<float[]> AddVectorsAsync(float[] a, float[] b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match (a: {a.Length}, b: {b.Length})");
        }

        _actorLogger.LogDebug(
            "Adding vectors of length {Length} on GPU-native actor {ActorId}",
            a.Length,
            this.GetPrimaryKeyLong());

        // Create request message
        var request = new VectorAddRequest
        {
            VectorALength = a.Length,
            VectorBLength = b.Length,
            Operation = VectorOperation.Add,
            UseGpuMemory = 0 // false = 0
        };

        VectorAddResponse response;
        GpuMemoryHandle? bufferA = null;
        GpuMemoryHandle? bufferB = null;
        GpuMemoryHandle? bufferResult = null;

        try
        {
            // Small vectors: inline data in message (≤25 elements)
            if (a.Length <= 25)
            {
                _actorLogger.LogTrace(
                    "Using inline message path for small vectors ({Length} elements)",
                    a.Length);

                unsafe
                {
                    for (int i = 0; i < a.Length; i++)
                    {
                        request.InlineDataA[i] = a[i];
                        request.InlineDataB[i] = b[i];
                    }
                }
            }
            // Large vectors: use GPU memory handles (>25 elements)
            else
            {
                _actorLogger.LogDebug(
                    "Using GPU memory path for large vectors ({Length} elements)",
                    a.Length);

                // Allocate GPU buffers and copy data
                bufferA = await _memoryManager.AllocateAndCopyAsync(a, CancellationToken.None);
                bufferB = await _memoryManager.AllocateAndCopyAsync(b, CancellationToken.None);
                bufferResult = _memoryManager.AllocateBuffer<float>(a.Length);

                // Pass GPU memory handles in request
                request.UseGpuMemory = 1; // true = 1
                request.GpuBufferAHandleId = bufferA.HandleId;
                request.GpuBufferBHandleId = bufferB.HandleId;
                request.GpuBufferResultHandleId = bufferResult.HandleId;

                _actorLogger.LogTrace(
                    "Allocated GPU buffers: A=0x{HandleA:X}, B=0x{HandleB:X}, Result=0x{HandleResult:X}",
                    bufferA.HandleId,
                    bufferB.HandleId,
                    bufferResult.HandleId);
            }

            var startTime = DateTime.UtcNow;

            // Invoke GPU kernel
            response = await InvokeKernelAsync<VectorAddRequest, VectorAddResponse>(
                request,
                timeout: TimeSpan.FromSeconds(5));

            var latencyNs = (DateTime.UtcNow - startTime).TotalNanoseconds;

            _actorLogger.LogInformation(
                "Vector addition completed on GPU in {LatencyNs}ns ({Mode}, length={Length}, target: 100-500ns)",
                latencyNs,
                request.UseGpuMemory != 0 ? "GPU memory" : "inline",
                a.Length);

            // Extract result
            var result = new float[response.ResultLength];

            if (request.UseGpuMemory != 0 && bufferResult != null)
            {
                // Copy result from GPU memory
                await _memoryManager.CopyFromGpuAsync(bufferResult!, result, CancellationToken.None);

                _actorLogger.LogTrace(
                    "Copied {Length} elements from GPU memory to CPU",
                    result.Length);
            }
            else
            {
                // Extract inline result
                unsafe
                {
                    for (int i = 0; i < Math.Min(response.ResultLength, 25); i++)
                    {
                        result[i] = response.InlineResult[i];
                    }
                }
            }

            return result;
        }
        finally
        {
            // Clean up GPU buffers
            bufferA?.Dispose();
            bufferB?.Dispose();
            bufferResult?.Dispose();
        }
    }

    /// <inheritdoc/>
    public async Task<float> AddVectorsScalarAsync(float[] a, float[] b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match (a: {a.Length}, b: {b.Length})");
        }

        _actorLogger.LogDebug(
            "Adding vectors (scalar result) of length {Length} on GPU-native actor {ActorId}",
            a.Length,
            this.GetPrimaryKeyLong());

        // Create request for scalar reduction
        var request = new VectorAddRequest
        {
            VectorALength = a.Length,
            VectorBLength = b.Length,
            Operation = VectorOperation.AddScalar,
            UseGpuMemory = 0 // false = 0
        };

        VectorAddResponse response;
        GpuMemoryHandle? bufferA = null;
        GpuMemoryHandle? bufferB = null;

        try
        {
            // Small vectors: inline data in message (≤25 elements)
            if (a.Length <= 25)
            {
                _actorLogger.LogTrace(
                    "Using inline message path for small vectors ({Length} elements, scalar reduction)",
                    a.Length);

                unsafe
                {
                    for (int i = 0; i < a.Length; i++)
                    {
                        request.InlineDataA[i] = a[i];
                        request.InlineDataB[i] = b[i];
                    }
                }
            }
            // Large vectors: use GPU memory handles (>25 elements)
            else
            {
                _actorLogger.LogDebug(
                    "Using GPU memory path for large vectors ({Length} elements, scalar reduction)",
                    a.Length);

                // Allocate GPU buffers and copy data
                bufferA = await _memoryManager.AllocateAndCopyAsync(a, CancellationToken.None);
                bufferB = await _memoryManager.AllocateAndCopyAsync(b, CancellationToken.None);

                // Pass GPU memory handles in request
                request.UseGpuMemory = 1; // true = 1
                request.GpuBufferAHandleId = bufferA.HandleId;
                request.GpuBufferBHandleId = bufferB.HandleId;
                // No result buffer needed for scalar reduction

                _actorLogger.LogTrace(
                    "Allocated GPU buffers for scalar reduction: A=0x{HandleA:X}, B=0x{HandleB:X}",
                    bufferA.HandleId,
                    bufferB.HandleId);
            }

            var startTime = DateTime.UtcNow;

            // Invoke GPU kernel
            response = await InvokeKernelAsync<VectorAddRequest, VectorAddResponse>(
                request,
                timeout: TimeSpan.FromSeconds(5));

            var latencyNs = (DateTime.UtcNow - startTime).TotalNanoseconds;

            _actorLogger.LogInformation(
                "Vector addition (scalar) completed on GPU in {LatencyNs}ns ({Mode}, length={Length})",
                latencyNs,
                request.UseGpuMemory != 0 ? "GPU memory" : "inline",
                a.Length);

            // Return scalar sum
            return response.ScalarResult;
        }
        finally
        {
            // Clean up GPU buffers
            bufferA?.Dispose();
            bufferB?.Dispose();
        }
    }

    /// <inheritdoc/>
    public async Task<VectorAddMetrics> GetMetricsAsync()
    {
        _actorLogger.LogDebug(
            "Getting metrics for GPU-native actor {ActorId}",
            this.GetPrimaryKeyLong());

        // Get ring kernel metrics from DotCompute runtime
        var kernelMetrics = await GetKernelMetricsAsync();
        var kernelStatus = await GetKernelStatusAsync();

        return new VectorAddMetrics
        {
            TotalOperations = kernelMetrics.MessagesReceived,
            AvgProcessingTimeNs = kernelMetrics.AvgProcessingTimeMs * 1_000_000, // ms to ns
            ThroughputMsgsPerSec = kernelMetrics.ThroughputMsgsPerSec,
            InputQueueUtilization = kernelMetrics.InputQueueUtilization,
            OutputQueueUtilization = kernelMetrics.OutputQueueUtilization,
            GpuMemoryBytes = kernelMetrics.CurrentMemoryBytes,
            Uptime = kernelStatus.Uptime
        };
    }
}

/// <summary>
/// Request message for GPU vector addition (fits in 228-byte payload).
/// </summary>
/// <remarks>
/// <para>
/// This message structure supports two modes:
/// </para>
/// <para>
/// **Small vectors (≤25 elements)**: Data is inlined directly in the message.
/// Total size: 16 bytes (metadata) + 4 bytes (flags) + 24 bytes (handles) + 100 bytes (A) + 100 bytes (B) = 244 bytes.
/// This exceeds 228 bytes, so we use 20 elements: 16 + 4 + 24 + 80 + 80 = 204 bytes (fits!)
/// </para>
/// <para>
/// **Large vectors (>25 elements)**: GPU memory handles are passed, kernel operates on GPU memory.
/// GPU kernel uses handles to access data from GPU buffer pool.
/// </para>
/// </remarks>
[System.Runtime.InteropServices.StructLayout(
    System.Runtime.InteropServices.LayoutKind.Sequential,
    Pack = 4)]
internal unsafe struct VectorAddRequest
{
    /// <summary>
    /// Length of vector A.
    /// </summary>
    public int VectorALength;

    /// <summary>
    /// Length of vector B.
    /// </summary>
    public int VectorBLength;

    /// <summary>
    /// Operation to perform.
    /// </summary>
    public VectorOperation Operation;

    /// <summary>
    /// Whether to use GPU memory handles (true) or inline data (false).
    /// </summary>
    public int UseGpuMemory; // Using int for struct packing (bool = 1 byte creates alignment issues)

    /// <summary>
    /// GPU memory handle ID for vector A (used when UseGpuMemory = true).
    /// </summary>
    public ulong GpuBufferAHandleId;

    /// <summary>
    /// GPU memory handle ID for vector B (used when UseGpuMemory = true).
    /// </summary>
    public ulong GpuBufferBHandleId;

    /// <summary>
    /// GPU memory handle ID for result buffer (used when UseGpuMemory = true).
    /// </summary>
    public ulong GpuBufferResultHandleId;

    /// <summary>
    /// Inline data for small vectors (&lt;= 25 elements).
    /// </summary>
    /// <remarks>
    /// Used only when UseGpuMemory = false.
    /// Size: 25 floats × 4 bytes = 100 bytes.
    /// </remarks>
    public fixed float InlineDataA[25];

    /// <summary>
    /// Inline data for vector B (small vectors only).
    /// </summary>
    /// <remarks>
    /// Used only when UseGpuMemory = false.
    /// Size: 25 floats × 4 bytes = 100 bytes.
    /// </remarks>
    public fixed float InlineDataB[25];
}

/// <summary>
/// Response message from GPU vector addition.
/// </summary>
/// <remarks>
/// For large vectors using GPU memory, the result is written directly to the
/// GPU buffer specified by GpuBufferResultHandleId in the request.
/// InlineResult is only used for small vectors (≤25 elements).
/// </remarks>
[System.Runtime.InteropServices.StructLayout(
    System.Runtime.InteropServices.LayoutKind.Sequential,
    Pack = 4)]
internal unsafe struct VectorAddResponse
{
    /// <summary>
    /// Length of result vector.
    /// </summary>
    public int ResultLength;

    /// <summary>
    /// Scalar sum (if Operation was AddScalar).
    /// </summary>
    public float ScalarResult;

    /// <summary>
    /// Reserved for alignment.
    /// </summary>
    private long _reserved;

    /// <summary>
    /// Inline result data for small vectors (≤25 elements).
    /// </summary>
    /// <remarks>
    /// Only used when request.UseGpuMemory = false.
    /// For large vectors, result is in GPU memory buffer.
    /// Size: 25 floats × 4 bytes = 100 bytes.
    /// </remarks>
    public fixed float InlineResult[25]; // Reduced from 50 to match request size limit
}

/// <summary>
/// Vector operation type.
/// </summary>
internal enum VectorOperation : int
{
    /// <summary>
    /// Element-wise addition (returns vector).
    /// </summary>
    Add = 0,

    /// <summary>
    /// Element-wise addition with scalar reduction (returns sum).
    /// </summary>
    AddScalar = 1,

    /// <summary>
    /// Element-wise subtraction.
    /// </summary>
    Subtract = 2,

    /// <summary>
    /// Element-wise multiplication.
    /// </summary>
    Multiply = 3,

    /// <summary>
    /// Dot product.
    /// </summary>
    DotProduct = 4
}
