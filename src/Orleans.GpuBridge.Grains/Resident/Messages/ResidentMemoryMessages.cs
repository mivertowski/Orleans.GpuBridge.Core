using System;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Grains.Enums;
using Orleans.GpuBridge.Grains.Models;

namespace Orleans.GpuBridge.Grains.Resident.Messages;

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
