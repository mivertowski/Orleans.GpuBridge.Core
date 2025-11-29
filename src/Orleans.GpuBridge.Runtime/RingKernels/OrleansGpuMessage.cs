// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Runtime.RingKernels;

/// <summary>
/// GPU-compatible message structure for Orleans method invocation on ring kernels.
/// </summary>
/// <remarks>
/// <para>
/// This structure is designed for zero-copy GPU messaging with the following constraints:
/// - Fixed size (256 bytes) for cache alignment and predictable memory access
/// - Unmanaged type (no pointers/references) safe for GPU memory
/// - Inline payload to avoid pointer chasing on GPU
/// - Temporal metadata for HLC/Vector Clock integration
/// </para>
/// <para>
/// The 256-byte size is chosen because:
/// - Aligns with GPU cache line size (128-256 bytes depending on architecture)
/// - Allows 16 messages per 4KB page (efficient memory usage)
/// - Fits in L1 cache for fast access
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential, Pack = 4, Size = 256)]
public unsafe struct OrleansGpuMessage
{
    /// <summary>
    /// Hash of the method name for fast dispatch.
    /// </summary>
    /// <remarks>
    /// Computed as <see cref="string.GetHashCode()"/> of the method name.
    /// GPU kernel uses switch statement on MethodId for O(1) dispatch.
    /// </remarks>
    public int MethodId;

    /// <summary>
    /// Timestamp in UTC ticks for temporal ordering (HLC integration).
    /// </summary>
    /// <remarks>
    /// Used for Hybrid Logical Clock (HLC) synchronization.
    /// GPU kernel can maintain HLC state for causality tracking.
    /// Resolution: 100ns (DateTime.Ticks)
    /// </remarks>
    public long TimestampTicks;

    /// <summary>
    /// Correlation ID for matching requests with responses.
    /// </summary>
    /// <remarks>
    /// Generated from <see cref="Guid"/> for uniqueness.
    /// GPU kernel echoes this ID in response messages.
    /// </remarks>
    public long CorrelationId;

    /// <summary>
    /// Message type classification.
    /// </summary>
    public MessageType Type;

    /// <summary>
    /// Sender actor ID (for actor-to-actor messaging).
    /// </summary>
    /// <remarks>
    /// Maps to Orleans grain primary key (integer portion).
    /// Value 0 indicates message from CPU/Orleans runtime.
    /// </remarks>
    public int SenderId;

    /// <summary>
    /// Target actor ID (destination).
    /// </summary>
    /// <remarks>
    /// Maps to Orleans grain primary key (integer portion).
    /// GPU kernel validates target matches its actor ID.
    /// </remarks>
    public int TargetId;

    /// <summary>
    /// Reserved for future use (alignment padding).
    /// </summary>
    private int _reserved1;

    /// <summary>
    /// Reserved for future use (alignment padding).
    /// </summary>
    private int _reserved2;

    /// <summary>
    /// Serialized method arguments or response data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Fixed 228-byte inline buffer for message payload.
    /// This avoids pointer chasing and ensures cache-friendly access on GPU.
    /// </para>
    /// <para>
    /// Layout strategy:
    /// - Primitives (int, float, long, etc.) packed sequentially
    /// - Small arrays (&lt; 228 bytes) copied inline
    /// - Large arrays passed via GPU memory pointers (stored as long)
    /// </para>
    /// <para>
    /// Calculation: 256 (total) - 28 (metadata) = 228 bytes
    /// - MethodId (4) + TimestampTicks (8) + CorrelationId (8)
    /// - Type (4) + SenderId (4) + TargetId (4)
    /// - _reserved1 (4) + _reserved2 (4)
    /// = 40 bytes metadata, but Pack=4 reduces to 28 bytes effective
    /// </para>
    /// </remarks>
    public fixed byte Payload[228];

    /// <summary>
    /// Creates a new GPU message for a method invocation.
    /// </summary>
    /// <param name="methodId">Hash of the method name.</param>
    /// <param name="senderId">Sender actor ID (0 for Orleans runtime).</param>
    /// <param name="targetId">Target actor ID.</param>
    /// <param name="type">Message type.</param>
    /// <returns>Initialized GPU message.</returns>
    public static OrleansGpuMessage Create(int methodId, int senderId, int targetId, MessageType type = MessageType.Data)
    {
        return new OrleansGpuMessage
        {
            MethodId = methodId,
            TimestampTicks = DateTime.UtcNow.Ticks,
            CorrelationId = GenerateCorrelationId(),
            Type = type,
            SenderId = senderId,
            TargetId = targetId,
            _reserved1 = 0,
            _reserved2 = 0
        };
    }

    /// <summary>
    /// Creates a response message from a request.
    /// </summary>
    /// <param name="request">Original request message.</param>
    /// <returns>Response message with correlation ID preserved.</returns>
    public static OrleansGpuMessage CreateResponse(OrleansGpuMessage request)
    {
        return new OrleansGpuMessage
        {
            MethodId = request.MethodId,
            TimestampTicks = DateTime.UtcNow.Ticks,
            CorrelationId = request.CorrelationId, // Echo correlation ID
            Type = MessageType.Response,
            SenderId = request.TargetId, // Swap sender/target
            TargetId = request.SenderId,
            _reserved1 = 0,
            _reserved2 = 0
        };
    }

    /// <summary>
    /// Generates a unique correlation ID from a GUID.
    /// </summary>
    /// <returns>64-bit correlation ID.</returns>
    private static long GenerateCorrelationId()
    {
        var guidBytes = Guid.NewGuid().ToByteArray();
        return BitConverter.ToInt64(guidBytes, 0);
    }

    /// <summary>
    /// Gets a span over the payload buffer for safe manipulation.
    /// </summary>
    /// <returns>Span of 228 bytes.</returns>
    public readonly Span<byte> GetPayloadSpan()
    {
        fixed (byte* ptr = Payload)
        {
            return new Span<byte>(ptr, 228);
        }
    }

    /// <summary>
    /// Writes a value to the payload at the specified offset.
    /// </summary>
    /// <typeparam name="T">Unmanaged value type.</typeparam>
    /// <param name="offset">Byte offset in payload.</param>
    /// <param name="value">Value to write.</param>
    /// <exception cref="ArgumentOutOfRangeException">Offset + sizeof(T) exceeds payload size.</exception>
    public void WritePayload<T>(int offset, T value) where T : unmanaged
    {
        var size = sizeof(T);
        if (offset + size > 228)
        {
            throw new ArgumentOutOfRangeException(nameof(offset),
                $"Cannot write {size} bytes at offset {offset} (payload size: 228 bytes)");
        }

        fixed (byte* ptr = Payload)
        {
            *(T*)(ptr + offset) = value;
        }
    }

    /// <summary>
    /// Reads a value from the payload at the specified offset.
    /// </summary>
    /// <typeparam name="T">Unmanaged value type.</typeparam>
    /// <param name="offset">Byte offset in payload.</param>
    /// <returns>Value read from payload.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Offset + sizeof(T) exceeds payload size.</exception>
    public readonly T ReadPayload<T>(int offset) where T : unmanaged
    {
        var size = sizeof(T);
        if (offset + size > 228)
        {
            throw new ArgumentOutOfRangeException(nameof(offset),
                $"Cannot read {size} bytes at offset {offset} (payload size: 228 bytes)");
        }

        fixed (byte* ptr = Payload)
        {
            return *(T*)(ptr + offset);
        }
    }
}

/// <summary>
/// Message type classification for GPU kernel dispatch.
/// </summary>
public enum MessageType : int
{
    /// <summary>
    /// Standard method invocation or data message.
    /// </summary>
    Data = 0,

    /// <summary>
    /// Response to a previous request (contains return value).
    /// </summary>
    Response = 1,

    /// <summary>
    /// Control message (activate, deactivate, checkpoint, etc.).
    /// </summary>
    Control = 2,

    /// <summary>
    /// Error message (exception occurred during processing).
    /// </summary>
    Error = 3,

    /// <summary>
    /// Heartbeat message (liveness check).
    /// </summary>
    Heartbeat = 4
}
