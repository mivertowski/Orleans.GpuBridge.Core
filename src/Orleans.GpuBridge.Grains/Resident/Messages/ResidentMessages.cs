using System;
using System.Collections.Generic;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Models;

namespace Orleans.GpuBridge.Grains.Resident.Messages;

/// <summary>
/// Base message type for Ring Kernel communication.
/// Ring Kernels process these messages in a persistent GPU-resident loop.
/// </summary>
public abstract record ResidentMessage
{
    /// <summary>
    /// Unique identifier for this message request.
    /// Used to match responses back to pending grain operations.
    /// </summary>
    public Guid RequestId { get; init; } = Guid.NewGuid();

    /// <summary>
    /// Timestamp when the message was created (for latency tracking).
    /// </summary>
    public long TimestampTicks { get; init; } = DateTime.UtcNow.Ticks;
}

/// <summary>
/// Message to allocate GPU memory through the Ring Kernel.
/// The Ring Kernel will allocate from its memory pool or create new allocation.
/// </summary>
public sealed record AllocateMessage : ResidentMessage
{
    /// <summary>
    /// Size of memory to allocate in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Type of memory allocation (default, pinned, shared).
    /// </summary>
    public GpuMemoryType MemoryType { get; init; }

    /// <summary>
    /// Preferred device index for allocation (-1 for any available).
    /// </summary>
    public int DeviceIndex { get; init; } = -1;

    public AllocateMessage(long sizeBytes, GpuMemoryType memoryType, int deviceIndex = -1)
    {
        SizeBytes = sizeBytes;
        MemoryType = memoryType;
        DeviceIndex = deviceIndex;
    }
}

/// <summary>
/// Response message for successful allocation.
/// </summary>
public sealed record AllocateResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Handle to the allocated memory.
    /// </summary>
    public GpuMemoryHandle Handle { get; init; }

    /// <summary>
    /// Whether allocation came from memory pool (true) or was newly allocated (false).
    /// </summary>
    public bool IsPoolHit { get; init; }

    public AllocateResponse(Guid originalRequestId, GpuMemoryHandle handle, bool isPoolHit)
    {
        OriginalRequestId = originalRequestId;
        Handle = handle;
        IsPoolHit = isPoolHit;
    }
}

/// <summary>
/// Message to write data to GPU memory.
/// Data transfer happens via staged memory (not embedded in message due to size limits).
/// </summary>
public sealed record WriteMessage : ResidentMessage
{
    /// <summary>
    /// ID of the memory allocation to write to.
    /// </summary>
    public string AllocationId { get; init; }

    /// <summary>
    /// Offset in bytes within the allocation.
    /// </summary>
    public long OffsetBytes { get; init; }

    /// <summary>
    /// Size of data to write in bytes.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Pointer to staged host memory (pinned) for DMA transfer.
    /// Data is staged separately to avoid message size limits.
    /// </summary>
    public IntPtr StagedDataPointer { get; init; }

    public WriteMessage(string allocationId, long offsetBytes, long sizeBytes, IntPtr stagedDataPointer)
    {
        AllocationId = allocationId;
        OffsetBytes = offsetBytes;
        SizeBytes = sizeBytes;
        StagedDataPointer = stagedDataPointer;
    }
}

/// <summary>
/// Response message for successful write operation.
/// </summary>
public sealed record WriteResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Bytes actually written.
    /// </summary>
    public long BytesWritten { get; init; }

    /// <summary>
    /// Transfer time in microseconds.
    /// </summary>
    public double TransferTimeMicroseconds { get; init; }

    public WriteResponse(Guid originalRequestId, long bytesWritten, double transferTimeMicroseconds)
    {
        OriginalRequestId = originalRequestId;
        BytesWritten = bytesWritten;
        TransferTimeMicroseconds = transferTimeMicroseconds;
    }
}

/// <summary>
/// Message to read data from GPU memory.
/// </summary>
public sealed record ReadMessage : ResidentMessage
{
    /// <summary>
    /// ID of the memory allocation to read from.
    /// </summary>
    public string AllocationId { get; init; }

    /// <summary>
    /// Offset in bytes within the allocation.
    /// </summary>
    public long OffsetBytes { get; init; }

    /// <summary>
    /// Number of bytes to read.
    /// </summary>
    public long SizeBytes { get; init; }

    /// <summary>
    /// Pointer to staged host memory (pinned) for DMA transfer.
    /// </summary>
    public IntPtr StagedDataPointer { get; init; }

    public ReadMessage(string allocationId, long offsetBytes, long sizeBytes, IntPtr stagedDataPointer)
    {
        AllocationId = allocationId;
        OffsetBytes = offsetBytes;
        SizeBytes = sizeBytes;
        StagedDataPointer = stagedDataPointer;
    }
}

/// <summary>
/// Response message for successful read operation.
/// </summary>
public sealed record ReadResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Bytes actually read.
    /// </summary>
    public long BytesRead { get; init; }

    /// <summary>
    /// Transfer time in microseconds.
    /// </summary>
    public double TransferTimeMicroseconds { get; init; }

    public ReadResponse(Guid originalRequestId, long bytesRead, double transferTimeMicroseconds)
    {
        OriginalRequestId = originalRequestId;
        BytesRead = bytesRead;
        TransferTimeMicroseconds = transferTimeMicroseconds;
    }
}

/// <summary>
/// Message to execute a kernel on GPU with resident memory.
/// </summary>
public sealed record ComputeMessage : ResidentMessage
{
    /// <summary>
    /// Kernel identifier (from KernelId).
    /// </summary>
    public string KernelId { get; init; }

    /// <summary>
    /// Input memory allocation ID.
    /// </summary>
    public string InputAllocationId { get; init; }

    /// <summary>
    /// Output memory allocation ID.
    /// </summary>
    public string OutputAllocationId { get; init; }

    /// <summary>
    /// Optional kernel execution parameters (work group size, constants, etc.).
    /// </summary>
    public Dictionary<string, object>? Parameters { get; init; }

    public ComputeMessage(string kernelId, string inputAllocationId, string outputAllocationId, Dictionary<string, object>? parameters = null)
    {
        KernelId = kernelId;
        InputAllocationId = inputAllocationId;
        OutputAllocationId = outputAllocationId;
        Parameters = parameters;
    }
}

/// <summary>
/// Response message for successful kernel execution.
/// </summary>
public sealed record ComputeResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Whether execution was successful.
    /// </summary>
    public bool Success { get; init; }

    /// <summary>
    /// Kernel execution time in microseconds.
    /// </summary>
    public double KernelTimeMicroseconds { get; init; }

    /// <summary>
    /// Total execution time (including memory transfers) in microseconds.
    /// </summary>
    public double TotalTimeMicroseconds { get; init; }

    /// <summary>
    /// Error message if execution failed.
    /// </summary>
    public string? Error { get; init; }

    /// <summary>
    /// Whether kernel was cached (true) or newly compiled (false).
    /// </summary>
    public bool IsCacheHit { get; init; }

    public ComputeResponse(Guid originalRequestId, bool success, double kernelTimeMicroseconds, double totalTimeMicroseconds, string? error = null, bool isCacheHit = false)
    {
        OriginalRequestId = originalRequestId;
        Success = success;
        KernelTimeMicroseconds = kernelTimeMicroseconds;
        TotalTimeMicroseconds = totalTimeMicroseconds;
        Error = error;
        IsCacheHit = isCacheHit;
    }
}

/// <summary>
/// Message to release GPU memory allocation.
/// </summary>
public sealed record ReleaseMessage : ResidentMessage
{
    /// <summary>
    /// ID of the memory allocation to release.
    /// </summary>
    public string AllocationId { get; init; }

    /// <summary>
    /// Whether to return memory to pool (true) or dispose immediately (false).
    /// </summary>
    public bool ReturnToPool { get; init; }

    public ReleaseMessage(string allocationId, bool returnToPool = true)
    {
        AllocationId = allocationId;
        ReturnToPool = returnToPool;
    }
}

/// <summary>
/// Response message for successful release operation.
/// </summary>
public sealed record ReleaseResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Bytes freed.
    /// </summary>
    public long BytesFreed { get; init; }

    /// <summary>
    /// Whether memory was returned to pool (true) or disposed (false).
    /// </summary>
    public bool ReturnedToPool { get; init; }

    public ReleaseResponse(Guid originalRequestId, long bytesFreed, bool returnedToPool)
    {
        OriginalRequestId = originalRequestId;
        BytesFreed = bytesFreed;
        ReturnedToPool = returnedToPool;
    }
}

/// <summary>
/// Message to get Ring Kernel metrics.
/// </summary>
public sealed record GetMetricsMessage : ResidentMessage
{
    /// <summary>
    /// Whether to include detailed per-allocation metrics.
    /// </summary>
    public bool IncludeDetails { get; init; }

    public GetMetricsMessage(bool includeDetails = false)
    {
        IncludeDetails = includeDetails;
    }
}

/// <summary>
/// Response message containing Ring Kernel metrics.
/// </summary>
public sealed record MetricsResponse : ResidentMessage
{
    /// <summary>
    /// Request ID this response corresponds to.
    /// </summary>
    public Guid OriginalRequestId { get; init; }

    /// <summary>
    /// Total messages processed by Ring Kernel.
    /// </summary>
    public long TotalMessagesProcessed { get; init; }

    /// <summary>
    /// Current messages per second throughput.
    /// </summary>
    public double MessagesPerSecond { get; init; }

    /// <summary>
    /// Average message processing latency in nanoseconds.
    /// </summary>
    public double AverageLatencyNanoseconds { get; init; }

    /// <summary>
    /// Memory pool hit count.
    /// </summary>
    public long PoolHitCount { get; init; }

    /// <summary>
    /// Memory pool miss count.
    /// </summary>
    public long PoolMissCount { get; init; }

    /// <summary>
    /// Kernel cache hit count.
    /// </summary>
    public long KernelCacheHitCount { get; init; }

    /// <summary>
    /// Kernel cache miss count.
    /// </summary>
    public long KernelCacheMissCount { get; init; }

    /// <summary>
    /// Total GPU memory allocated in bytes.
    /// </summary>
    public long TotalAllocatedBytes { get; init; }

    /// <summary>
    /// Active allocation count.
    /// </summary>
    public int ActiveAllocationCount { get; init; }

    public MetricsResponse(
        Guid originalRequestId,
        long totalMessagesProcessed,
        double messagesPerSecond,
        double averageLatencyNanoseconds,
        long poolHitCount,
        long poolMissCount,
        long kernelCacheHitCount,
        long kernelCacheMissCount,
        long totalAllocatedBytes,
        int activeAllocationCount)
    {
        OriginalRequestId = originalRequestId;
        TotalMessagesProcessed = totalMessagesProcessed;
        MessagesPerSecond = messagesPerSecond;
        AverageLatencyNanoseconds = averageLatencyNanoseconds;
        PoolHitCount = poolHitCount;
        PoolMissCount = poolMissCount;
        KernelCacheHitCount = kernelCacheHitCount;
        KernelCacheMissCount = kernelCacheMissCount;
        TotalAllocatedBytes = totalAllocatedBytes;
        ActiveAllocationCount = activeAllocationCount;
    }
}

/// <summary>
/// Message to initialize the Ring Kernel.
/// Sent once at launch.
/// </summary>
public sealed record InitializeMessage : ResidentMessage
{
    /// <summary>
    /// Maximum memory pool size in bytes.
    /// </summary>
    public long MaxPoolSizeBytes { get; init; }

    /// <summary>
    /// Maximum kernel cache size.
    /// </summary>
    public int MaxKernelCacheSize { get; init; }

    /// <summary>
    /// Device index to use.
    /// </summary>
    public int DeviceIndex { get; init; }

    public InitializeMessage(long maxPoolSizeBytes, int maxKernelCacheSize, int deviceIndex)
    {
        MaxPoolSizeBytes = maxPoolSizeBytes;
        MaxKernelCacheSize = maxKernelCacheSize;
        DeviceIndex = deviceIndex;
    }
}

/// <summary>
/// Message to shutdown the Ring Kernel gracefully.
/// </summary>
public sealed record ShutdownMessage : ResidentMessage
{
    /// <summary>
    /// Whether to drain pending messages before shutdown.
    /// </summary>
    public bool DrainPendingMessages { get; init; }

    public ShutdownMessage(bool drainPendingMessages = true)
    {
        DrainPendingMessages = drainPendingMessages;
    }
}
