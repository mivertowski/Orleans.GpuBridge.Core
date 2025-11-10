using System;
using System.Runtime.InteropServices;
using DotCompute.Memory;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels;

/// <summary>
/// GPU-resident lock-free message queue for actor-to-actor communication.
/// Resides entirely in GPU memory for sub-microsecond message latency (100-500ns).
/// Supports lock-free concurrent enqueue/dequeue using atomic operations.
/// </summary>
public sealed class GpuResidentMessageQueue : IDisposable
{
    private readonly IUnifiedMemoryManager _memoryManager;
    private readonly DotComputeMemoryOrderingProvider _memoryOrdering;
    private readonly ILogger<GpuResidentMessageQueue> _logger;
    private IDeviceMemory? _queueMemory;
    private IDeviceMemory? _metadataMemory;
    private readonly int _capacity;
    private readonly int _messageSize;
    private bool _disposed;

    public GpuResidentMessageQueue(
        IUnifiedMemoryManager memoryManager,
        DotComputeMemoryOrderingProvider memoryOrdering,
        ILogger<GpuResidentMessageQueue> logger,
        int capacity,
        int messageSize)
    {
        _memoryManager = memoryManager ?? throw new ArgumentNullException(nameof(memoryManager));
        _memoryOrdering = memoryOrdering ?? throw new ArgumentNullException(nameof(memoryOrdering));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(messageSize);

        _capacity = capacity;
        _messageSize = messageSize;

        _logger.LogInformation(
            "Creating GPU-resident message queue - Capacity: {Capacity}, Message size: {MessageSize} bytes, " +
            "Total memory: {TotalMemory} bytes",
            capacity,
            messageSize,
            capacity * messageSize + Marshal.SizeOf<QueueMetadata>());
    }

    /// <summary>
    /// Initializes queue in GPU memory.
    /// Must be called before use.
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_queueMemory != null)
        {
            throw new InvalidOperationException("Queue already initialized");
        }

        try
        {
            _logger.LogInformation("Allocating GPU memory for message queue...");

            // Allocate memory for queue data (circular buffer)
            var queueBytes = _capacity * _messageSize;
            _queueMemory = await _memoryManager.AllocateAsync<byte>(queueBytes, ct).ConfigureAwait(false);

            // Allocate memory for queue metadata (head, tail, count)
            var metadataBytes = Marshal.SizeOf<QueueMetadata>();
            _metadataMemory = await _memoryManager.AllocateAsync<byte>(metadataBytes, ct).ConfigureAwait(false);

            // Initialize metadata
            var metadata = new QueueMetadata
            {
                Head = 0,
                Tail = 0,
                Count = 0,
                Capacity = _capacity
            };

            await _metadataMemory.WriteAsync(
                MemoryMarshal.Cast<QueueMetadata, byte>(MemoryMarshal.CreateSpan(ref metadata, 1)),
                ct).ConfigureAwait(false);

            // Configure memory ordering for causal message passing
            _memoryOrdering.ConfigureActorMessageOrdering();

            _logger.LogInformation(
                "GPU-resident message queue initialized - " +
                "Queue memory: {QueueMemory:X}, Metadata memory: {MetadataMemory:X}",
                _queueMemory.DevicePointer,
                _metadataMemory.DevicePointer);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize GPU-resident message queue");

            // Cleanup on failure
            _queueMemory?.Dispose();
            _queueMemory = null;
            _metadataMemory?.Dispose();
            _metadataMemory = null;

            throw;
        }
    }

    /// <summary>
    /// Gets device pointer to queue memory (for kernel access).
    /// </summary>
    public nint GetQueueDevicePointer()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_queueMemory == null)
        {
            throw new InvalidOperationException("Queue not initialized. Call InitializeAsync first.");
        }

        return _queueMemory.DevicePointer;
    }

    /// <summary>
    /// Gets device pointer to metadata memory (for kernel access).
    /// </summary>
    public nint GetMetadataDevicePointer()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_metadataMemory == null)
        {
            throw new InvalidOperationException("Queue not initialized. Call InitializeAsync first.");
        }

        return _metadataMemory.DevicePointer;
    }

    /// <summary>
    /// Gets queue configuration for kernel launch.
    /// </summary>
    public QueueConfiguration GetConfiguration()
    {
        return new QueueConfiguration
        {
            QueuePointer = GetQueueDevicePointer(),
            MetadataPointer = GetMetadataDevicePointer(),
            Capacity = _capacity,
            MessageSize = _messageSize
        };
    }

    /// <summary>
    /// Gets current queue statistics from GPU.
    /// </summary>
    public async Task<QueueStatistics> GetStatisticsAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_metadataMemory == null)
        {
            throw new InvalidOperationException("Queue not initialized");
        }

        try
        {
            // Read metadata from GPU
            var metadataBytes = new byte[Marshal.SizeOf<QueueMetadata>()];
            await _metadataMemory.ReadAsync(metadataBytes, ct).ConfigureAwait(false);

            var metadata = MemoryMarshal.Read<QueueMetadata>(metadataBytes);

            return new QueueStatistics
            {
                Capacity = metadata.Capacity,
                CurrentCount = metadata.Count,
                UtilizationPercent = (double)metadata.Count / metadata.Capacity * 100.0,
                IsEmpty = metadata.Count == 0,
                IsFull = metadata.Count >= metadata.Capacity,
                HeadPosition = metadata.Head,
                TailPosition = metadata.Tail
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get queue statistics");
            throw;
        }
    }

    /// <summary>
    /// Generates CUDA kernel code for lock-free enqueue operation.
    /// This code should be embedded in actor kernels.
    /// </summary>
    public static string GetEnqueueKernelCode()
    {
        return @"
// Lock-free enqueue operation (CUDA)
__device__ bool gpu_queue_enqueue(
    QueueMetadata* metadata,
    char* queue_data,
    int message_size,
    const char* message)
{
    // Atomically increment tail and get old value
    int tail = atomicAdd(&metadata->Tail, 1);
    int index = tail % metadata->Capacity;

    // Check if queue is full
    int count = atomicAdd(&metadata->Count, 0); // Atomic read
    if (count >= metadata->Capacity)
    {
        // Queue full - decrement tail and return false
        atomicSub(&metadata->Tail, 1);
        return false;
    }

    // Copy message to queue (with release fence)
    char* slot = queue_data + (index * message_size);
    for (int i = 0; i < message_size; i++)
    {
        slot[i] = message[i];
    }

    // RELEASE fence: ensure message write completes before count increment
    __threadfence_system();

    // Atomically increment count
    atomicAdd(&metadata->Count, 1);

    return true;
}
";
    }

    /// <summary>
    /// Generates CUDA kernel code for lock-free dequeue operation.
    /// This code should be embedded in actor kernels.
    /// </summary>
    public static string GetDequeueKernelCode()
    {
        return @"
// Lock-free dequeue operation (CUDA)
__device__ bool gpu_queue_dequeue(
    QueueMetadata* metadata,
    char* queue_data,
    int message_size,
    char* message_out)
{
    // Check if queue is empty
    int count = atomicAdd(&metadata->Count, 0); // Atomic read
    if (count <= 0)
    {
        return false; // Queue empty
    }

    // Atomically increment head and get old value
    int head = atomicAdd(&metadata->Head, 1);
    int index = head % metadata->Capacity;

    // ACQUIRE fence: ensure all prior operations complete before read
    __threadfence_system();

    // Copy message from queue
    char* slot = queue_data + (index * message_size);
    for (int i = 0; i < message_size; i++)
    {
        message_out[i] = slot[i];
    }

    // ACQUIRE fence: ensure message read completes
    __threadfence_system();

    // Atomically decrement count
    atomicSub(&metadata->Count, 1);

    return true;
}
";
    }

    /// <summary>
    /// Gets complete CUDA header for GPU queue operations.
    /// </summary>
    public static string GetCudaHeader()
    {
        return @"
#ifndef GPU_QUEUE_H
#define GPU_QUEUE_H

// Queue metadata structure (must match C# struct layout)
struct QueueMetadata
{
    int Head;       // Next dequeue position
    int Tail;       // Next enqueue position
    int Count;      // Current message count
    int Capacity;   // Maximum capacity
};

// Lock-free enqueue (returns true on success)
__device__ bool gpu_queue_enqueue(
    QueueMetadata* metadata,
    char* queue_data,
    int message_size,
    const char* message);

// Lock-free dequeue (returns true on success)
__device__ bool gpu_queue_dequeue(
    QueueMetadata* metadata,
    char* queue_data,
    int message_size,
    char* message_out);

#endif // GPU_QUEUE_H
";
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogDebug("Disposing GPU-resident message queue");

            _queueMemory?.Dispose();
            _queueMemory = null;

            _metadataMemory?.Dispose();
            _metadataMemory = null;

            _disposed = true;

            _logger.LogInformation("GPU-resident message queue disposed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during GPU-resident message queue disposal");
        }
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    private struct QueueMetadata
    {
        public int Head;       // Next dequeue position
        public int Tail;       // Next enqueue position
        public int Count;      // Current message count
        public int Capacity;   // Maximum capacity
    }
}

/// <summary>
/// Configuration for GPU queue access in kernels.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 8)]
public readonly struct QueueConfiguration
{
    public readonly nint QueuePointer;      // Device pointer to queue data
    public readonly nint MetadataPointer;   // Device pointer to metadata
    public readonly int Capacity;           // Maximum queue capacity
    public readonly int MessageSize;        // Size of each message in bytes
}

/// <summary>
/// Statistics about queue state.
/// </summary>
public sealed class QueueStatistics
{
    public required int Capacity { get; init; }
    public required int CurrentCount { get; init; }
    public required double UtilizationPercent { get; init; }
    public required bool IsEmpty { get; init; }
    public required bool IsFull { get; init; }
    public required int HeadPosition { get; init; }
    public required int TailPosition { get; init; }

    public override string ToString()
    {
        return $"Queue: {CurrentCount}/{Capacity} ({UtilizationPercent:F1}%), " +
               $"Empty: {IsEmpty}, Full: {IsFull}";
    }
}
