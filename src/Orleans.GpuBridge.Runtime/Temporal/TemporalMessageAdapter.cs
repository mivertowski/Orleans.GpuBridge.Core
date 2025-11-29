// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using DotCompute.Abstractions.RingKernels;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Adapts between DotCompute KernelMessage and temporal ActorMessage structures.
/// </summary>
/// <remarks>
/// <para>
/// This adapter enables HLC timestamp injection into GPU-bound messages without
/// modifying the DotCompute ring kernel infrastructure.
/// </para>
/// <para>
/// Message Flow:
/// 1. GpuNativeGrain creates typed request with InvokeKernelAsync
/// 2. HLC timestamp is generated
/// 3. TemporalMessageAdapter wraps request + timestamp into ActorMessage
/// 4. GPU temporal kernel processes ActorMessage and updates HLC
/// 5. Response is unwrapped and returned to grain
/// </para>
/// </remarks>
public static class TemporalMessageAdapter
{
    /// <summary>
    /// Thread-safe buffer pool for large payloads that exceed inline capacity.
    /// Uses negative handle values to distinguish from inline payloads.
    /// </summary>
    private static readonly ConcurrentDictionary<long, byte[]> LargePayloadBuffer = new();

    /// <summary>
    /// Counter for generating unique payload handles.
    /// Starts at 1 to reserve 0 for empty/null payloads.
    /// </summary>
    private static long _nextPayloadHandle = 1;

    /// <summary>
    /// Marker bit indicating the payload field contains a handle, not inline data.
    /// Uses the sign bit (MSB) - negative values indicate handles.
    /// </summary>
    private const long HandleMarkerBit = unchecked((long)0x8000_0000_0000_0000);

    /// <summary>
    /// Maximum size for inline payload embedding (8 bytes = sizeof(long)).
    /// </summary>
    private const int MaxInlinePayloadSize = sizeof(long);

    /// <summary>
    /// Wraps a typed request into an ActorMessage with HLC timestamp.
    /// </summary>
    /// <typeparam name="TRequest">Request payload type.</typeparam>
    /// <param name="senderId">Sender actor ID.</param>
    /// <param name="receiverId">Receiver actor ID.</param>
    /// <param name="request">Request payload.</param>
    /// <param name="timestamp">HLC timestamp for causal ordering.</param>
    /// <param name="sequenceNumber">Optional sequence number.</param>
    /// <returns>ActorMessage with embedded timestamp and payload reference.</returns>
    /// <remarks>
    /// <para>
    /// For small payloads (&lt;= 8 bytes), the data is embedded directly in Payload field.
    /// For larger payloads, Payload contains a reference/handle to GPU memory.
    /// </para>
    /// <para>
    /// This method is used on the CPU side before sending to GPU.
    /// The GPU temporal kernel (ProcessActorMessageWithTimestamp) will extract
    /// the timestamp and update actor HLC state.
    /// </para>
    /// </remarks>
    public static ActorMessage WrapWithTimestamp<TRequest>(
        ulong senderId,
        ulong receiverId,
        TRequest request,
        HybridTimestamp timestamp,
        ulong sequenceNumber = 0)
        where TRequest : unmanaged
    {
        // For small payloads (8 bytes or less), embed directly
        // For larger payloads, store in buffer pool and use handle
        long payloadValue;
        unsafe
        {
            int payloadSize = sizeof(TRequest);
            if (payloadSize <= MaxInlinePayloadSize)
            {
                // Direct embedding for small types - use stackalloc for local buffer
                byte* buffer = stackalloc byte[sizeof(long)];
                *(TRequest*)buffer = request;
                payloadValue = *(long*)buffer;
            }
            else
            {
                // Large payload: serialize to buffer and store handle
                payloadValue = StoreLargePayload(ref request, payloadSize);
            }
        }

        // Create ActorMessage envelope with timestamp
        var actorMessage = new ActorMessage
        {
            MessageId = Guid.NewGuid(),
            SourceActorId = senderId,
            TargetActorId = receiverId,
            Timestamp = timestamp, // HLC timestamp embedded here
            Type = Abstractions.Temporal.MessageType.Command, // Fully qualified to avoid ambiguity
            Payload = payloadValue,
            SequenceNumber = sequenceNumber,
            Priority = 0 // Normal priority
        };

        return actorMessage;
    }

    /// <summary>
    /// Unwraps an ActorMessage response into a typed response payload.
    /// </summary>
    /// <typeparam name="TResponse">Response payload type.</typeparam>
    /// <param name="message">ActorMessage from GPU kernel.</param>
    /// <returns>Typed response payload.</returns>
    /// <remarks>
    /// Extracts the response data from ActorMessage.Payload field.
    /// Must match the message type sent from GPU temporal kernel.
    /// </remarks>
    public static TResponse UnwrapResponse<TResponse>(ActorMessage message)
        where TResponse : unmanaged
    {
        unsafe
        {
            int responseSize = sizeof(TResponse);
            long payload = message.Payload;

            // Check if this is a handle (negative value = handle marker set)
            if ((payload & HandleMarkerBit) != 0)
            {
                // Large payload: retrieve from buffer and deserialize
                return RetrieveLargePayload<TResponse>(payload);
            }
            else if (responseSize <= MaxInlinePayloadSize)
            {
                // Direct extraction for small types
                return *(TResponse*)&payload;
            }
            else
            {
                // Large type requested but payload doesn't have handle marker
                // This could happen if the GPU kernel wrote inline data for a type
                // that's larger than 8 bytes - treat as inline attempt
                throw new InvalidOperationException(
                    $"Response type {typeof(TResponse).Name} ({responseSize} bytes) exceeds inline capacity " +
                    $"but payload {payload} does not contain a valid handle marker.");
            }
        }
    }

    /// <summary>
    /// Creates a KernelMessage from an ActorMessage for DotCompute compatibility.
    /// </summary>
    /// <remarks>
    /// This allows us to use existing DotCompute ring kernel infrastructure
    /// while embedding temporal information.
    /// </remarks>
    public static KernelMessage<ActorMessage> ToKernelMessage(ActorMessage actorMessage)
    {
        return KernelMessage<ActorMessage>.Create(
            senderId: (int)actorMessage.SourceActorId,
            receiverId: (int)actorMessage.TargetActorId,
            type: DotCompute.Abstractions.RingKernels.MessageType.Data,
            payload: actorMessage);
    }

    /// <summary>
    /// Extracts ActorMessage from a DotCompute KernelMessage.
    /// </summary>
    public static ActorMessage FromKernelMessage(KernelMessage<ActorMessage> kernelMessage)
    {
        return kernelMessage.Payload;
    }

    /// <summary>
    /// Updates HLC timestamp in an existing ActorMessage.
    /// </summary>
    /// <remarks>
    /// Used when forwarding messages to update causality chain.
    /// </remarks>
    public static ActorMessage UpdateTimestamp(ActorMessage message, HybridTimestamp newTimestamp)
    {
        return message with { Timestamp = newTimestamp };
    }

    /// <summary>
    /// Stores a large payload in the buffer pool and returns a handle.
    /// </summary>
    /// <typeparam name="T">Payload type (unmanaged).</typeparam>
    /// <param name="payload">Reference to the payload data.</param>
    /// <param name="payloadSize">Size of the payload in bytes.</param>
    /// <returns>Handle value with marker bit set (negative value).</returns>
    private static unsafe long StoreLargePayload<T>(ref T payload, int payloadSize)
        where T : unmanaged
    {
        // Generate unique handle
        long handle = Interlocked.Increment(ref _nextPayloadHandle);

        // Serialize payload to byte array
        byte[] buffer = ArrayPool<byte>.Shared.Rent(payloadSize);
        try
        {
            fixed (T* sourcePtr = &payload)
            fixed (byte* destPtr = buffer)
            {
                Buffer.MemoryCopy(sourcePtr, destPtr, payloadSize, payloadSize);
            }

            // Create exact-size copy for storage (rent may give larger array)
            byte[] exactBuffer = new byte[payloadSize];
            Array.Copy(buffer, exactBuffer, payloadSize);

            // Store in buffer pool
            if (!LargePayloadBuffer.TryAdd(handle, exactBuffer))
            {
                throw new InvalidOperationException(
                    $"Failed to store large payload with handle {handle}. Handle collision detected.");
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        // Return handle with marker bit set (makes it negative)
        return handle | HandleMarkerBit;
    }

    /// <summary>
    /// Retrieves and deserializes a large payload from the buffer pool.
    /// </summary>
    /// <typeparam name="T">Expected payload type (unmanaged).</typeparam>
    /// <param name="handleWithMarker">Handle value with marker bit set.</param>
    /// <returns>Deserialized payload.</returns>
    /// <remarks>
    /// This method removes the payload from the buffer pool after retrieval
    /// to prevent memory leaks. Each payload can only be retrieved once.
    /// </remarks>
    private static unsafe T RetrieveLargePayload<T>(long handleWithMarker)
        where T : unmanaged
    {
        // Extract actual handle by clearing marker bit
        long handle = handleWithMarker & ~HandleMarkerBit;

        // Remove and retrieve from buffer pool (one-time retrieval)
        if (!LargePayloadBuffer.TryRemove(handle, out byte[]? buffer))
        {
            throw new InvalidOperationException(
                $"Large payload with handle {handle} not found. " +
                "It may have already been retrieved or never stored.");
        }

        // Verify size matches expected type
        int expectedSize = sizeof(T);
        if (buffer.Length != expectedSize)
        {
            throw new InvalidOperationException(
                $"Payload size mismatch: stored {buffer.Length} bytes, " +
                $"but type {typeof(T).Name} requires {expectedSize} bytes.");
        }

        // Deserialize from byte array
        T result;
        fixed (byte* sourcePtr = buffer)
        {
            result = *(T*)sourcePtr;
        }

        return result;
    }

    /// <summary>
    /// Gets the number of large payloads currently stored in the buffer pool.
    /// </summary>
    /// <remarks>
    /// Useful for diagnostics and detecting potential memory leaks from
    /// unretrieved payloads.
    /// </remarks>
    public static int PendingLargePayloadCount => LargePayloadBuffer.Count;

    /// <summary>
    /// Clears all pending large payloads from the buffer pool.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use with caution - this will invalidate any handles that haven't
    /// been retrieved yet. Primarily useful for testing and cleanup
    /// during shutdown.
    /// </para>
    /// </remarks>
    /// <returns>Number of payloads that were cleared.</returns>
    public static int ClearPendingPayloads()
    {
        int count = LargePayloadBuffer.Count;
        LargePayloadBuffer.Clear();
        return count;
    }

    /// <summary>
    /// Checks if a payload value represents a large payload handle.
    /// </summary>
    /// <param name="payloadValue">The payload field value from ActorMessage.</param>
    /// <returns>True if this is a handle to a large payload; false if inline data.</returns>
    public static bool IsLargePayloadHandle(long payloadValue)
    {
        return (payloadValue & HandleMarkerBit) != 0;
    }
}
