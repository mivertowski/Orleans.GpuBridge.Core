// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System;
using DotCompute.Abstractions.Messaging;
using MemoryPack;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Vector operation types supported by vector add kernel.
/// </summary>
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
/// Request message for vector addition ring kernel.
/// </summary>
/// <remarks>
/// MemoryPack source generator auto-generates high-performance serialization
/// for this message type (2-5x faster than MessagePack, AOT-compatible).
/// </remarks>
[MemoryPackable]
public partial class VectorAddRequestMessage : IRingKernelMessage
{
    // IRingKernelMessage core properties
    public Guid MessageId { get; set; } = Guid.NewGuid();
    public byte Priority { get; set; } = 128;
    public Guid? CorrelationId { get; set; }

    // Application data
    public int VectorALength { get; set; }
    public VectorOperation Operation { get; set; } = VectorOperation.Add;
    public bool UseGpuMemory { get; set; }
    public ulong GpuBufferAHandleId { get; set; }
    public ulong GpuBufferBHandleId { get; set; }
    public ulong GpuBufferResultHandleId { get; set; }
    public float[] InlineDataA { get; set; } = Array.Empty<float>();
    public float[] InlineDataB { get; set; } = Array.Empty<float>();
}

/// <summary>
/// Response message from vector addition ring kernel.
/// </summary>
[MemoryPackable]
public partial class VectorAddResponseMessage : IRingKernelMessage
{
    // IRingKernelMessage core properties
    public Guid MessageId { get; set; } = Guid.NewGuid();
    public byte Priority { get; set; } = 128;
    public Guid? CorrelationId { get; set; }

    // Application data
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public int ProcessedElements { get; set; }
    public ulong GpuResultBufferHandleId { get; set; }
    public float[] InlineResult { get; set; } = Array.Empty<float>();
    public long ProcessingTimeNs { get; set; }
}
