// Copyright (c) 2025 Michael Ivertowski. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Runtime.Persistent;

/// <summary>
/// Manages ring buffers for persistent kernel I/O
/// </summary>
public sealed class RingBufferManager : IDisposable
{
    private readonly ILogger<RingBufferManager> _logger;
    private readonly ConcurrentDictionary<string, RingBuffer> _buffers;
    private readonly int _defaultBufferSize;
    private bool _disposed;

    public RingBufferManager(
        ILogger<RingBufferManager> logger,
        int defaultBufferSize = 1024 * 1024 * 16) // 16MB default
    {
        _logger = logger;
        _defaultBufferSize = defaultBufferSize;
        _buffers = new ConcurrentDictionary<string, RingBuffer>();
    }

    /// <summary>
    /// Creates a new ring buffer for kernel I/O
    /// </summary>
    public RingBuffer CreateBuffer(string kernelId, int? bufferSize = null)
    {
        var size = bufferSize ?? _defaultBufferSize;
        var buffer = new RingBuffer(kernelId, size, _logger);

        if (!_buffers.TryAdd(kernelId, buffer))
        {
            buffer.Dispose();
            throw new InvalidOperationException($"Buffer already exists for kernel {kernelId}");
        }

        _logger.LogInformation(
            "Created ring buffer for kernel {KernelId} with size {Size:N0} bytes",
            kernelId, size);

        return buffer;
    }

    /// <summary>
    /// Gets an existing ring buffer
    /// </summary>
    public RingBuffer? GetBuffer(string kernelId)
    {
        return _buffers.TryGetValue(kernelId, out var buffer) ? buffer : null;
    }

    /// <summary>
    /// Removes and disposes a ring buffer
    /// </summary>
    public void RemoveBuffer(string kernelId)
    {
        if (_buffers.TryRemove(kernelId, out var buffer))
        {
            buffer.Dispose();
            _logger.LogInformation("Removed ring buffer for kernel {KernelId}", kernelId);
        }
    }

    /// <summary>
    /// Gets statistics for all buffers
    /// </summary>
    public Dictionary<string, RingBufferStats> GetStatistics()
    {
        var stats = new Dictionary<string, RingBufferStats>();

        foreach (var (kernelId, buffer) in _buffers)
        {
            stats[kernelId] = buffer.GetStats();
        }

        return stats;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var buffer in _buffers.Values)
        {
            buffer.Dispose();
        }

        _buffers.Clear();
    }
}

/// <summary>
/// Lock-free ring buffer for kernel I/O
/// </summary>
public sealed class RingBuffer : IDisposable
{
    private readonly string _kernelId;
    private readonly ILogger _logger;
    private readonly byte[] _buffer;
    private readonly GCHandle _handle;
    private readonly int _size;
    private readonly Channel<DataBatch> _inputChannel;
    private readonly Channel<DataBatch> _outputChannel;
    private int _readPosition;
    private int _writePosition;
    private long _totalBytesWritten;
    private long _totalBytesRead;
    private long _totalWrites;
    private long _totalReads;
    private bool _disposed;

    public IntPtr BufferPointer { get; }
    public int Size => _size;
    public bool IsPinned => _handle.IsAllocated;

    public RingBuffer(string kernelId, int size, ILogger logger)
    {
        _kernelId = kernelId;
        _size = size;
        _logger = logger;
        _buffer = new byte[size];

        // Pin the buffer for direct GPU access
        _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
        BufferPointer = _handle.AddrOfPinnedObject();

        // Create channels for async data flow
        var channelOptions = new UnboundedChannelOptions
        {
            SingleReader = false,
            SingleWriter = false
        };

        _inputChannel = Channel.CreateUnbounded<DataBatch>(channelOptions);
        _outputChannel = Channel.CreateUnbounded<DataBatch>(channelOptions);
    }

    /// <summary>
    /// Writes data to the ring buffer
    /// </summary>
    public async Task<bool> WriteAsync(ReadOnlyMemory<byte> data, CancellationToken ct = default)
    {
        if (data.Length > _size)
        {
            _logger.LogWarning(
                "Data size {DataSize} exceeds buffer size {BufferSize}",
                data.Length, _size);
            return false;
        }

        // Wait for space to be available
        while (!TryWrite(data))
        {
            await Task.Delay(1, ct);
            if (ct.IsCancellationRequested) return false;
        }

        return true;
    }

    /// <summary>
    /// Tries to write data without waiting
    /// </summary>
    public bool TryWrite(ReadOnlyMemory<byte> data)
    {
        var dataSize = data.Length;
        var currentWrite = Volatile.Read(ref _writePosition);
        var currentRead = Volatile.Read(ref _readPosition);

        // Calculate available space
        var available = currentRead <= currentWrite
            ? _size - (currentWrite - currentRead)
            : currentRead - currentWrite;

        if (available < dataSize + sizeof(int)) // Need space for size header
            return false;

        // Write size header
        var newWritePos = currentWrite;
        WriteInt32(_buffer, newWritePos, dataSize);
        newWritePos = (newWritePos + sizeof(int)) % _size;

        // Write data (handle wrap-around)
        if (newWritePos + dataSize <= _size)
        {
            data.Span.CopyTo(_buffer.AsSpan(newWritePos, dataSize));
        }
        else
        {
            var firstChunk = _size - newWritePos;
            data.Span[..firstChunk].CopyTo(_buffer.AsSpan(newWritePos));
            data.Span[firstChunk..].CopyTo(_buffer.AsSpan(0));
        }

        newWritePos = (newWritePos + dataSize) % _size;

        // Update position atomically
        Volatile.Write(ref _writePosition, newWritePos);

        // Update statistics
        Interlocked.Add(ref _totalBytesWritten, dataSize);
        Interlocked.Increment(ref _totalWrites);

        return true;
    }

    /// <summary>
    /// Reads data from the ring buffer
    /// </summary>
    public async Task<Memory<byte>?> ReadAsync(CancellationToken ct = default)
    {
        while (true)
        {
            var result = TryRead();
            if (result.HasValue)
                return result;

            await Task.Delay(1, ct);
            if (ct.IsCancellationRequested)
                return null;
        }
    }

    /// <summary>
    /// Tries to read data without waiting
    /// </summary>
    public Memory<byte>? TryRead()
    {
        var currentRead = Volatile.Read(ref _readPosition);
        var currentWrite = Volatile.Read(ref _writePosition);

        if (currentRead == currentWrite)
            return null; // Buffer is empty

        // Read size header
        var dataSize = ReadInt32(_buffer, currentRead);
        var newReadPos = (currentRead + sizeof(int)) % _size;

        // Allocate result buffer
        var result = new byte[dataSize];

        // Read data (handle wrap-around)
        if (newReadPos + dataSize <= _size)
        {
            _buffer.AsSpan(newReadPos, dataSize).CopyTo(result);
        }
        else
        {
            var firstChunk = _size - newReadPos;
            _buffer.AsSpan(newReadPos, firstChunk).CopyTo(result.AsSpan(0, firstChunk));
            _buffer.AsSpan(0, dataSize - firstChunk).CopyTo(result.AsSpan(firstChunk));
        }

        newReadPos = (newReadPos + dataSize) % _size;

        // Update position atomically
        Volatile.Write(ref _readPosition, newReadPos);

        // Update statistics
        Interlocked.Add(ref _totalBytesRead, dataSize);
        Interlocked.Increment(ref _totalReads);

        return result;
    }

    /// <summary>
    /// Submits a batch for processing
    /// </summary>
    public async Task SubmitBatchAsync(DataBatch batch)
    {
        await _inputChannel.Writer.WriteAsync(batch);
    }

    /// <summary>
    /// Retrieves processed results
    /// </summary>
    public async IAsyncEnumerable<DataBatch> GetResultsAsync(
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var batch in _outputChannel.Reader.ReadAllAsync(ct))
        {
            yield return batch;
        }
    }

    /// <summary>
    /// Gets buffer statistics
    /// </summary>
    public RingBufferStats GetStats()
    {
        var currentRead = Volatile.Read(ref _readPosition);
        var currentWrite = Volatile.Read(ref _writePosition);

        var used = currentWrite >= currentRead
            ? currentWrite - currentRead
            : _size - (currentRead - currentWrite);

        return new RingBufferStats
        {
            KernelId = _kernelId,
            BufferSize = _size,
            UsedBytes = used,
            AvailableBytes = _size - used,
            TotalBytesWritten = Interlocked.Read(ref _totalBytesWritten),
            TotalBytesRead = Interlocked.Read(ref _totalBytesRead),
            TotalWrites = Interlocked.Read(ref _totalWrites),
            TotalReads = Interlocked.Read(ref _totalReads),
            UtilizationPercent = (used / (double)_size) * 100
        };
    }

    private static void WriteInt32(byte[] buffer, int position, int value)
    {
        BitConverter.GetBytes(value).CopyTo(buffer, position);
    }

    private static int ReadInt32(byte[] buffer, int position)
    {
        return BitConverter.ToInt32(buffer, position);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _inputChannel.Writer.TryComplete();
        _outputChannel.Writer.TryComplete();

        if (_handle.IsAllocated)
        {
            _handle.Free();
        }
    }
}

/// <summary>
/// Represents a batch of data for kernel processing
/// </summary>
public sealed class DataBatch
{
    public string Id { get; init; } = Guid.NewGuid().ToString();
    public Memory<byte> Data { get; init; }
    public Dictionary<string, object>? Metadata { get; init; }
    public DateTime SubmittedAt { get; init; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }
}

/// <summary>
/// Ring buffer statistics
/// </summary>
public sealed class RingBufferStats
{
    public string KernelId { get; init; } = string.Empty;
    public int BufferSize { get; init; }
    public int UsedBytes { get; init; }
    public int AvailableBytes { get; init; }
    public long TotalBytesWritten { get; init; }
    public long TotalBytesRead { get; init; }
    public long TotalWrites { get; init; }
    public long TotalReads { get; init; }
    public double UtilizationPercent { get; init; }
    public double WriteThroughputMBps => TotalWrites > 0
        ? (TotalBytesWritten / 1024.0 / 1024.0) / (TotalWrites / 1000.0)
        : 0;
    public double ReadThroughputMBps => TotalReads > 0
        ? (TotalBytesRead / 1024.0 / 1024.0) / (TotalReads / 1000.0)
        : 0;
}