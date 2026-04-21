// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System;
using System.Runtime.InteropServices;
using DotCompute.Abstractions.Messaging;
using MemoryPack;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Request message for the actor ring kernel - CUDA/MemoryPack compatible.
/// </summary>
/// <remarks>
/// <para>
/// Represents a single actor message to be processed by the GPU-resident actor ring kernel.
/// The kernel updates the target actor's HLC and applies the message payload to the
/// actor state according to <see cref="ActorMessageTypeCode"/>.
/// </para>
/// <para>
/// Uses ONLY primitives supported by the CUDA MemoryPack serializer generator:
/// ints, longs, floats, bools, bytes, and Guid (emitted as two ulongs). Nested structs,
/// arrays, and enums are intentionally avoided — enums are expressed as byte/int fields.
/// </para>
/// </remarks>
[MemoryPackable]
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public partial struct ActorRingRequest : IRingKernelMessage
{
    // Message metadata ---------------------------------------------------------
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }

    [MemoryPackIgnore]
    public readonly string MessageType => nameof(ActorRingRequest);

    [MemoryPackIgnore]
    public readonly int PayloadSize => 16 + 1 + 17 + 8 + 8 + 8 + 8 + 8 + 8 + 1 + 1;

    // Actor routing ------------------------------------------------------------

    /// <summary>Target actor identifier (also indexes the actor state buffer).</summary>
    public ulong TargetActorId { get; set; }

    /// <summary>Source actor identifier for causality tracking.</summary>
    public ulong SourceActorId { get; set; }

    /// <summary>Monotonic sequence number for ordering messages from the same source.</summary>
    public ulong SequenceNumber { get; set; }

    // HLC carried with the message --------------------------------------------

    /// <summary>Physical component of the sender's HLC timestamp (nanoseconds).</summary>
    public long SenderHlcPhysical { get; set; }

    /// <summary>Logical component of the sender's HLC timestamp.</summary>
    public long SenderHlcLogical { get; set; }

    // Payload -----------------------------------------------------------------

    /// <summary>
    /// Message type as byte (maps to <c>Orleans.GpuBridge.Abstractions.Temporal.MessageType</c>):
    /// 0=StateUpdate, 1=Query, 2=Command, 3=Event, 4=Response.
    /// </summary>
    public byte MessageTypeCode { get; set; }

    /// <summary>
    /// Inline payload value (small-payload fast path). Larger payloads should be
    /// addressed by handle/offset in a separate GPU buffer managed by the host.
    /// </summary>
    public long Payload { get; set; }

    public readonly ReadOnlySpan<byte> Serialize() => MemoryPackSerializer.Serialize(this);

    public void Deserialize(ReadOnlySpan<byte> data) =>
        this = MemoryPackSerializer.Deserialize<ActorRingRequest>(data);
}

/// <summary>
/// Response message from the actor ring kernel.
/// </summary>
[MemoryPackable]
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public partial struct ActorRingResponse : IRingKernelMessage
{
    // Message metadata ---------------------------------------------------------
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }

    [MemoryPackIgnore]
    public readonly string MessageType => nameof(ActorRingResponse);

    [MemoryPackIgnore]
    public readonly int PayloadSize => 16 + 1 + 17 + 8 + 8 + 8 + 8 + 8 + 1 + 1;

    // Post-processing state ---------------------------------------------------

    /// <summary>Actor identifier that processed the message.</summary>
    public ulong ActorId { get; set; }

    /// <summary>Updated actor HLC physical time after merging the request's HLC.</summary>
    public long UpdatedHlcPhysical { get; set; }

    /// <summary>Updated actor HLC logical counter after merging the request's HLC.</summary>
    public long UpdatedHlcLogical { get; set; }

    /// <summary>Number of messages the actor has processed since activation.</summary>
    public ulong MessageCount { get; set; }

    /// <summary>Actor data field after applying the message payload.</summary>
    public long Data { get; set; }

    /// <summary>
    /// Current actor status flags (bitwise OR of
    /// <c>Orleans.GpuBridge.Abstractions.Temporal.ActorStatusFlags</c>).
    /// </summary>
    public byte Status { get; set; }

    /// <summary>
    /// Processing result: 0 = success, 1 = unknown message type code.
    /// </summary>
    public byte ErrorCode { get; set; }

    public readonly ReadOnlySpan<byte> Serialize() => MemoryPackSerializer.Serialize(this);

    public void Deserialize(ReadOnlySpan<byte> data) =>
        this = MemoryPackSerializer.Deserialize<ActorRingResponse>(data);
}

/// <summary>
/// Numeric constants mirroring <c>Orleans.GpuBridge.Abstractions.Temporal.MessageType</c>.
/// Duplicated here as byte constants so GPU handlers can switch on them without
/// pulling a managed enum through the CUDA serializer.
/// </summary>
public static class ActorMessageTypeCode
{
    /// <summary>State update message.</summary>
    public const byte StateUpdate = 0;

    /// <summary>Query message requesting information.</summary>
    public const byte Query = 1;

    /// <summary>Command message requesting an action.</summary>
    public const byte Command = 2;

    /// <summary>Event notification message.</summary>
    public const byte Event = 3;

    /// <summary>Response message.</summary>
    public const byte Response = 4;
}

/// <summary>
/// Status flag bits mirroring <c>Orleans.GpuBridge.Abstractions.Temporal.ActorStatusFlags</c>.
/// </summary>
public static class ActorStatusBits
{
    /// <summary>Actor is active.</summary>
    public const byte Active = 0x01;

    /// <summary>Actor is currently processing a message.</summary>
    public const byte Processing = 0x02;

    /// <summary>Actor is blocked.</summary>
    public const byte Blocked = 0x04;

    /// <summary>Actor encountered an error.</summary>
    public const byte Error = 0x08;
}
