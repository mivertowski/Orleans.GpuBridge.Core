// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using DotCompute.Abstractions;

namespace Orleans.GpuBridge.Runtime.Memory;

/// <summary>
/// Handle to GPU-allocated memory with automatic reference counting and cleanup.
/// </summary>
/// <remarks>
/// <para>
/// GpuMemoryHandle provides a safe wrapper around raw GPU memory pointers.
/// It tracks allocation size, usage, and automatically releases memory when
/// the last reference is disposed.
/// </para>
/// <para>
/// This class is thread-safe and uses atomic reference counting for
/// multi-threaded access scenarios.
/// </para>
/// </remarks>
public sealed class GpuMemoryHandle : IDisposable, IEquatable<GpuMemoryHandle>
{
    private readonly IntPtr _devicePointer;
    private readonly long _sizeBytes;
    private readonly GpuBufferPool? _pool;
    private readonly IUnifiedMemoryBuffer? _dotComputeBuffer;
    private int _referenceCount = 1;
    private bool _disposed;
    private readonly object _disposeLock = new();

    /// <summary>
    /// Initializes a new GPU memory handle.
    /// </summary>
    /// <param name="devicePointer">GPU device pointer.</param>
    /// <param name="sizeBytes">Allocation size in bytes.</param>
    /// <param name="pool">Optional buffer pool for recycling.</param>
    /// <param name="dotComputeBuffer">Optional DotCompute unified memory buffer.</param>
    internal GpuMemoryHandle(IntPtr devicePointer, long sizeBytes, GpuBufferPool? pool = null, IUnifiedMemoryBuffer? dotComputeBuffer = null)
    {
        _devicePointer = devicePointer;
        _sizeBytes = sizeBytes;
        _pool = pool;
        _dotComputeBuffer = dotComputeBuffer;
    }

    /// <summary>
    /// Gets the GPU device pointer.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Handle has been disposed.</exception>
    public IntPtr DevicePointer
    {
        get
        {
            ThrowIfDisposed();
            return _devicePointer;
        }
    }

    /// <summary>
    /// Gets the size of the allocation in bytes.
    /// </summary>
    public long SizeBytes => _sizeBytes;

    /// <summary>
    /// Gets the unique handle ID (derived from device pointer).
    /// </summary>
    public ulong HandleId => (ulong)_devicePointer.ToInt64();

    /// <summary>
    /// Gets whether this handle is from a buffer pool (recyclable).
    /// </summary>
    public bool IsPooled => _pool != null;

    /// <summary>
    /// Gets the underlying DotCompute buffer (if allocated via DotCompute).
    /// </summary>
    internal IUnifiedMemoryBuffer? DotComputeBuffer => _dotComputeBuffer;

    /// <summary>
    /// Gets current reference count (for diagnostics).
    /// </summary>
    public int ReferenceCount => Interlocked.CompareExchange(ref _referenceCount, 0, 0);

    /// <summary>
    /// Adds a reference to this handle.
    /// </summary>
    /// <returns>This handle for fluent API.</returns>
    /// <exception cref="ObjectDisposedException">Handle has been disposed.</exception>
    public GpuMemoryHandle AddReference()
    {
        lock (_disposeLock)
        {
            ThrowIfDisposed();
            Interlocked.Increment(ref _referenceCount);
            return this;
        }
    }

    /// <summary>
    /// Removes a reference. When reference count reaches zero, memory is released.
    /// </summary>
    public void RemoveReference()
    {
        int newCount = Interlocked.Decrement(ref _referenceCount);
        if (newCount == 0)
        {
            Dispose();
        }
        else if (newCount < 0)
        {
            throw new InvalidOperationException("Reference count cannot be negative.");
        }
    }

    /// <summary>
    /// Disposes the GPU memory handle and releases memory.
    /// </summary>
    public void Dispose()
    {
        lock (_disposeLock)
        {
            if (_disposed)
                return;

            if (_pool != null)
            {
                // Return to pool for recycling
                _pool.ReturnBuffer(this);
            }
            else
            {
                // Free GPU memory directly via DotCompute
                if (_dotComputeBuffer != null)
                {
                    _dotComputeBuffer.Dispose();
                }
                // Fallback for legacy non-DotCompute allocations
                else if (_devicePointer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(_devicePointer);
                }
            }

            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// Checks equality based on device pointer.
    /// </summary>
    public bool Equals(GpuMemoryHandle? other)
    {
        if (other is null) return false;
        return _devicePointer == other._devicePointer;
    }

    /// <summary>
    /// Checks equality based on device pointer.
    /// </summary>
    public override bool Equals(object? obj) => Equals(obj as GpuMemoryHandle);

    /// <summary>
    /// Gets hash code based on device pointer.
    /// </summary>
    public override int GetHashCode() => _devicePointer.GetHashCode();

    /// <summary>
    /// Returns string representation of handle.
    /// </summary>
    public override string ToString() =>
        $"GpuMemoryHandle(Ptr=0x{_devicePointer.ToInt64():X}, Size={_sizeBytes} bytes, " +
        $"Refs={ReferenceCount}, Pooled={IsPooled})";

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GpuMemoryHandle),
                "Cannot access disposed GPU memory handle.");
        }
    }

    /// <summary>
    /// Finalizer to ensure GPU memory is released even if Dispose() is not called.
    /// </summary>
    ~GpuMemoryHandle()
    {
        Dispose();
    }
}

/// <summary>
/// GPU memory allocation statistics.
/// </summary>
public readonly struct GpuMemoryStats
{
    /// <summary>
    /// Total allocated GPU memory in bytes.
    /// </summary>
    public long TotalAllocatedBytes { get; init; }

    /// <summary>
    /// Currently in-use GPU memory in bytes.
    /// </summary>
    public long InUseBytes { get; init; }

    /// <summary>
    /// Available GPU memory in bytes.
    /// </summary>
    public long AvailableBytes { get; init; }

    /// <summary>
    /// Number of active allocations.
    /// </summary>
    public int ActiveAllocations { get; init; }

    /// <summary>
    /// Number of pooled buffers ready for reuse.
    /// </summary>
    public int PooledBuffers { get; init; }

    /// <summary>
    /// GPU memory utilization percentage (0-1).
    /// </summary>
    public double UtilizationRatio => TotalAllocatedBytes > 0
        ? (double)InUseBytes / TotalAllocatedBytes
        : 0.0;

    public override string ToString() =>
        $"GpuMemoryStats(Total={TotalAllocatedBytes / (1024 * 1024)}MB, " +
        $"InUse={InUseBytes / (1024 * 1024)}MB, " +
        $"Available={AvailableBytes / (1024 * 1024)}MB, " +
        $"Allocations={ActiveAllocations}, Pooled={PooledBuffers}, " +
        $"Utilization={UtilizationRatio:P1})";
}
