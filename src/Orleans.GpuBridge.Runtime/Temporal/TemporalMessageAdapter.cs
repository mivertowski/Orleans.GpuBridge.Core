// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
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
        // For larger payloads, this would store a GPU memory handle
        long payloadValue;
        unsafe
        {
            int payloadSize = sizeof(TRequest);
            if (payloadSize <= sizeof(long))
            {
                // Direct embedding for small types - use stackalloc for local buffer
                byte* buffer = stackalloc byte[sizeof(long)];
                *(TRequest*)buffer = request;
                payloadValue = *(long*)buffer;
            }
            else
            {
                // For larger payloads, GPU memory handle would go here
                // This requires GPU memory management integration (Phase 4)
                throw new NotImplementedException(
                    $"Payload type {typeof(TRequest).Name} ({payloadSize} bytes) exceeds inline capacity. " +
                    "GPU memory management for large payloads is pending Phase 4 implementation.");
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
            if (responseSize <= sizeof(long))
            {
                // Direct extraction for small types
                long payload = message.Payload;
                return *(TResponse*)&payload;
            }
            else
            {
                // For larger responses, GPU memory handle extraction would go here
                throw new NotImplementedException(
                    $"Response type {typeof(TResponse).Name} ({responseSize} bytes) exceeds inline capacity.");
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
}
