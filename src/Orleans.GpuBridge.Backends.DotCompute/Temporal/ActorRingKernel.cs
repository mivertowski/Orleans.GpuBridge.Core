// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using DotCompute.Abstractions.Attributes;
using DotCompute.Abstractions.RingKernels;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// GPU-resident ring kernel for processing Orleans.GpuBridge actor messages.
/// </summary>
/// <remarks>
/// <para>
/// This kernel replaces the three pre-v1.0 raw-Span kernels
/// (<c>ActorMessageProcessorRing</c>, <c>BatchedActorMessageProcessorRing</c>,
/// <c>CoordinatedActorRing</c>) with a single handler that uses the DotCompute v1.0
/// <see cref="RingKernelContext"/> model. The runtime handles the persistent dispatch
/// loop, message dequeue, and HLC propagation — the handler is called once per message.
/// </para>
/// <para>
/// Batching and coordination, which previously required separate kernels, are now
/// orthogonal configuration knobs on the <see cref="RingKernelAttribute"/>:
/// <see cref="RingKernelAttribute.MaxMessagesPerIteration"/> controls batch size and
/// <see cref="RingKernelAttribute.UseBarriers"/> enables named device barriers that
/// the handler can trigger via <see cref="RingKernelContext.NamedBarrier(string)"/>.
/// </para>
/// <para>
/// The actor state itself is managed by <c>Orleans.GpuBridge.Runtime.Temporal.RingKernelManager</c>,
/// which allocates GPU buffers for <c>ActorState</c>, HLC arrays, etc. and passes them to the
/// backend's <see cref="Orleans.GpuBridge.Runtime.Temporal.IRingKernelExecutor"/> implementation.
/// This kernel operates on the message-queue front end of that pipeline: it receives
/// <see cref="ActorRingRequest"/> items and emits <see cref="ActorRingResponse"/> items
/// carrying the post-processing HLC and state snapshot.
/// </para>
/// <para>
/// Performance target: &lt;500 ns per message on an RTX 2000 Ada (CC 8.9).
/// </para>
/// </remarks>
public static class ActorRingKernel
{
    /// <summary>Kernel identifier, matched by <c>RingKernelManager.KernelName</c>.</summary>
    public const string KernelId = "ActorMessageProcessorRing";

    /// <summary>
    /// Processes a single actor message: merges HLC, applies the payload to the actor's
    /// data field based on <see cref="ActorRingRequest.MessageTypeCode"/>, and emits a
    /// response carrying the updated state.
    /// </summary>
    /// <param name="ctx">Ring kernel context providing intrinsics and message dispatch.</param>
    /// <param name="request">Inbound actor message with HLC and payload.</param>
    /// <remarks>
    /// <para>
    /// Ring kernel message handlers are executed by a SINGLE thread per message (thread 0,
    /// block 0), so <c>SyncThreads()</c> must not be called here — it would deadlock since
    /// the full block hasn't arrived at this point. <see cref="RingKernelContext.ThreadFence"/>
    /// is used for memory ordering instead.
    /// </para>
    /// </remarks>
    [RingKernel(
        KernelId = KernelId,
        Capacity = 4096,
        InputQueueSize = 1024,
        OutputQueueSize = 1024,
        MaxInputMessageSizeBytes = 1024,
        MaxOutputMessageSizeBytes = 1024,
        ProcessingMode = RingProcessingMode.Continuous,
        MaxMessagesPerIteration = 4,
        EnableTimestamps = true,
        EnableCausalOrdering = true,
        Domain = RingKernelDomain.ActorModel,
        MessagingStrategy = MessagePassingStrategy.SharedMemory,
        Backends = KernelBackends.CUDA,
        OutputMessageType = typeof(ActorRingResponse))]
    public static void ProcessActorMessage(RingKernelContext ctx, ActorRingRequest request)
    {
        // --- HLC merge ---------------------------------------------------------
        // Classic HLC update: max(local, sender, wallClock) for the physical part,
        // logical counter increments when the dominant physical time is a tie.
        var localNow = ctx.Now();  // Device-local HLC sample.
        long localPhysical = localNow.PhysicalTimeNanos;
        long senderPhysical = request.SenderHlcPhysical;
        long senderLogical = request.SenderHlcLogical;

        // We don't have persistent local HLC per-actor in this handler — the
        // RingKernelManager owns that buffer. The merged value computed here is
        // emitted back to the host via the response so the manager can store it
        // under the correct actor slot.
        long mergedPhysical = localPhysical;
        if (senderPhysical > mergedPhysical)
        {
            mergedPhysical = senderPhysical;
        }

        long mergedLogical;
        if (mergedPhysical == senderPhysical && mergedPhysical == localPhysical)
        {
            mergedLogical = (senderLogical > 0 ? senderLogical : 0) + 1;
        }
        else if (mergedPhysical == senderPhysical)
        {
            mergedLogical = senderLogical + 1;
        }
        else
        {
            mergedLogical = 1;
        }

        // --- Apply payload to derived "data" field ----------------------------
        // State update semantics are intentionally narrow because the real
        // actor state lives in the manager-owned buffer. Here we just compute
        // the post-transition Data value so the manager can persist it atomically.
        long data;
        byte status = ActorStatusBits.Active | ActorStatusBits.Processing;
        byte errorCode = 0;

        byte typeCode = request.MessageTypeCode;
        if (typeCode == ActorMessageTypeCode.StateUpdate)
        {
            // Replace data with payload.
            data = request.Payload;
            status |= ActorStatusBits.Active;
        }
        else if (typeCode == ActorMessageTypeCode.Command)
        {
            // Commands accumulate into data (as in the legacy kernel).
            data = request.Payload;
        }
        else if (typeCode == ActorMessageTypeCode.Query)
        {
            // Queries do not mutate state.
            data = 0;
        }
        else if (typeCode == ActorMessageTypeCode.Event)
        {
            data = 0;
            status |= ActorStatusBits.Processing;
        }
        else if (typeCode == ActorMessageTypeCode.Response)
        {
            data = 0;
            // Clear blocked flag: responding unblocks the actor in the legacy model.
            status &= unchecked((byte)(~ActorStatusBits.Blocked));
        }
        else
        {
            // Unknown message type - flag error but still produce a response.
            data = 0;
            errorCode = 1;
            status |= ActorStatusBits.Error;
        }

        // Ensure merged HLC and data writes are visible to the host before enqueue.
        ctx.ThreadFence();

        var response = new ActorRingResponse
        {
            MessageId = request.MessageId,
            Priority = request.Priority,
            CorrelationId = request.CorrelationId,
            ActorId = request.TargetActorId,
            UpdatedHlcPhysical = mergedPhysical,
            UpdatedHlcLogical = mergedLogical,
            MessageCount = request.SequenceNumber,
            Data = data,
            Status = status,
            ErrorCode = errorCode
        };

        ctx.EnqueueOutput(response);
    }
}
