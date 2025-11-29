// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System;
using DotCompute.Abstractions.Messaging;
using MemoryPack;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Vector operation types supported by vector add kernel.
/// </summary>
/// <remarks>
/// Note: Enum is kept for C# code but NOT used in CUDA serialization.
/// CUDA messages use int OperationType instead.
/// </remarks>
public enum VectorOperation
{
    /// <summary>Element-wise addition: result[i] = a[i] + b[i]</summary>
    Add = 0,
    /// <summary>Element-wise subtraction: result[i] = a[i] - b[i]</summary>
    Subtract = 1,
    /// <summary>Element-wise multiplication: result[i] = a[i] * b[i]</summary>
    Multiply = 2,
    /// <summary>Element-wise division: result[i] = a[i] / b[i]</summary>
    Divide = 3
}

/// <summary>
/// Request message for vector addition ring kernel - CUDA-compatible version.
/// </summary>
/// <remarks>
/// <para>
/// Uses ONLY primitives supported by CudaMemoryPackSerializerGenerator:
/// - Primitives: int, float, bool, byte, long, ulong, double, short, ushort, uint, sbyte
/// - Guid (serialized as two ulongs)
/// - Nullable&lt;T&gt; of supported types
/// </para>
/// <para>
/// IMPORTANT: Arrays, strings, and enums are NOT supported for CUDA serialization.
/// For validation, we use 4 fixed float fields instead of arrays.
/// </para>
/// </remarks>
[MemoryPackable]
public partial struct VectorAddProcessorRingRequest : IRingKernelMessage
{
    // Message metadata
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }

    // IRingKernelMessage implementation
    [MemoryPackIgnore]
    public readonly string MessageType => nameof(VectorAddProcessorRingRequest);

    [MemoryPackIgnore]
    public readonly int PayloadSize => 16 + 1 + 17 + 4 + 4 + (4 * 8); // Guid + byte + Guid? + ints + floats

    // Application data - primitives only for CUDA compatibility
    /// <summary>Number of elements to process (max 4 for validation)</summary>
    public int VectorLength { get; set; }

    /// <summary>Operation type as int: 0=Add, 1=Subtract, 2=Multiply, 3=Divide</summary>
    public int OperationType { get; set; }

    // Fixed inline data (4 elements for validation - no arrays!)
    public float A0 { get; set; }
    public float A1 { get; set; }
    public float A2 { get; set; }
    public float A3 { get; set; }

    public float B0 { get; set; }
    public float B1 { get; set; }
    public float B2 { get; set; }
    public float B3 { get; set; }

    // Serialization handled by MemoryPack on host, CUDA serializer on GPU
    public readonly ReadOnlySpan<byte> Serialize() => MemoryPackSerializer.Serialize(this);

    public void Deserialize(ReadOnlySpan<byte> data) => this = MemoryPackSerializer.Deserialize<VectorAddProcessorRingRequest>(data);
}

/// <summary>
/// Response message from vector addition ring kernel - CUDA-compatible version.
/// </summary>
/// <remarks>
/// Uses ONLY primitives for CUDA serialization compatibility.
/// </remarks>
[MemoryPackable]
public partial struct VectorAddProcessorRingResponse : IRingKernelMessage
{
    // Message metadata
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }

    // IRingKernelMessage implementation
    [MemoryPackIgnore]
    public readonly string MessageType => nameof(VectorAddProcessorRingResponse);

    [MemoryPackIgnore]
    public readonly int PayloadSize => 16 + 1 + 17 + 1 + 4 + 4 + 8 + (4 * 4); // Guid + byte + Guid? + bool + ints + long + floats

    // Application data - primitives only
    public bool Success { get; set; }
    public int ErrorCode { get; set; }  // 0 = success, non-zero = error type
    public int ProcessedElements { get; set; }
    public long ProcessingTimeNs { get; set; }

    // Fixed result data (4 elements for validation - no arrays!)
    public float R0 { get; set; }
    public float R1 { get; set; }
    public float R2 { get; set; }
    public float R3 { get; set; }

    // Serialization handled by MemoryPack on host, CUDA serializer on GPU
    public readonly ReadOnlySpan<byte> Serialize() => MemoryPackSerializer.Serialize(this);

    public void Deserialize(ReadOnlySpan<byte> data) => this = MemoryPackSerializer.Deserialize<VectorAddProcessorRingResponse>(data);
}
