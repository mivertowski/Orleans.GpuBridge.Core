using System;
using System.Collections.Concurrent;

namespace Orleans.GpuBridge.Backends.DotCompute.Serialization;

/// <summary>
/// Buffer pool for serialization operations
/// </summary>
public sealed class SerializationBufferPool
{
    private readonly ConcurrentBag<byte[]> _smallBuffers = new();
    private readonly ConcurrentBag<byte[]> _mediumBuffers = new();
    private readonly ConcurrentBag<byte[]> _largeBuffers = new();

    private const int SmallSize = 4 * 1024;      // 4KB
    private const int MediumSize = 64 * 1024;    // 64KB
    private const int LargeSize = 1024 * 1024;   // 1MB

    /// <summary>
    /// Rents a buffer of at least the specified size from the pool
    /// </summary>
    /// <param name="minSize">Minimum required buffer size in bytes</param>
    /// <returns>A byte array that is at least minSize bytes</returns>
    public byte[] Rent(int minSize)
    {
        if (minSize <= SmallSize)
        {
            if (_smallBuffers.TryTake(out var small))
                return small;
            return new byte[SmallSize];
        }

        if (minSize <= MediumSize)
        {
            if (_mediumBuffers.TryTake(out var medium))
                return medium;
            return new byte[MediumSize];
        }

        if (minSize <= LargeSize)
        {
            if (_largeBuffers.TryTake(out var large))
                return large;
            return new byte[LargeSize];
        }

        // For very large buffers, don't pool
        return new byte[minSize];
    }

    /// <summary>
    /// Returns a rented buffer back to the pool after clearing sensitive data
    /// </summary>
    /// <param name="buffer">The buffer to return to the pool</param>
    public void Return(byte[] buffer)
    {
        if (buffer == null) return;

        // Clear sensitive data
        Array.Clear(buffer);

        switch (buffer.Length)
        {
            case SmallSize:
                _smallBuffers.Add(buffer);
                break;
            case MediumSize:
                _mediumBuffers.Add(buffer);
                break;
            case LargeSize:
                _largeBuffers.Add(buffer);
                break;
            // Don't pool non-standard sizes
        }
    }

    /// <summary>
    /// Clears all buffers from the pool, releasing memory
    /// </summary>
    public void Clear()
    {
        _smallBuffers.Clear();
        _mediumBuffers.Clear();
        _largeBuffers.Clear();
    }
}
