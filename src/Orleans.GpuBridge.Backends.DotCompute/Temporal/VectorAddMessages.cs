// Copyright (c) 2025 Michael Ivertowski
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Vector addition operation type.
/// </summary>
public enum VectorOperation
{
    /// <summary>
    /// Element-wise vector addition.
    /// </summary>
    Add = 0,

    /// <summary>
    /// Scalar reduction (sum all elements).
    /// </summary>
    AddScalar = 1
}

/// <summary>
/// Request message for vector addition ring kernel.
/// </summary>
/// <remarks>
/// <para>
/// <strong>NOTE: TEMPORARY PLACEHOLDER</strong>
/// </para>
/// <para>
/// This type is a placeholder until DotCompute.Generators source generator can be used
/// (requires .NET SDK 9.0.300+ with Roslyn 4.14.0). Once the SDK is upgraded, this will
/// be replaced by auto-generated code from the [RingKernel] attribute.
/// </para>
/// <para>
/// Message layout: 228 bytes total
/// - Inline data path: 2 × 25 floats (200 bytes) + metadata (28 bytes)
/// - GPU memory path: 3 × 8-byte handles (24 bytes) + metadata (28 bytes)
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VectorAddRequest
{
    /// <summary>
    /// Length of vector A (and B).
    /// </summary>
    public int VectorALength;

    /// <summary>
    /// Operation type (Add or AddScalar).
    /// </summary>
    public VectorOperation Operation;

    /// <summary>
    /// 1 = use GPU memory handles, 0 = use inline data.
    /// </summary>
    public int UseGpuMemory;

    /// <summary>
    /// GPU buffer handle for vector A (when UseGpuMemory = 1).
    /// </summary>
    public ulong GpuBufferAHandleId;

    /// <summary>
    /// GPU buffer handle for vector B (when UseGpuMemory = 1).
    /// </summary>
    public ulong GpuBufferBHandleId;

    /// <summary>
    /// GPU buffer handle for result vector (when UseGpuMemory = 1).
    /// </summary>
    public ulong GpuBufferResultHandleId;

    /// <summary>
    /// Inline data for vector A (max 25 elements).
    /// </summary>
    public fixed float InlineDataA[25];

    /// <summary>
    /// Inline data for vector B (max 25 elements).
    /// </summary>
    public fixed float InlineDataB[25];
}

/// <summary>
/// Response message from vector addition ring kernel.
/// </summary>
/// <remarks>
/// <para>
/// <strong>NOTE: TEMPORARY PLACEHOLDER</strong>
/// </para>
/// <para>
/// This type is a placeholder until DotCompute.Generators source generator can be used.
/// Will be replaced by auto-generated code from the [RingKernel] attribute.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct VectorAddResponse
{
    /// <summary>
    /// Scalar result (when Operation = AddScalar).
    /// </summary>
    public float ScalarResult;

    /// <summary>
    /// Length of result vector (0 for scalar results).
    /// </summary>
    public int ResultLength;

    /// <summary>
    /// Inline result data (max 25 elements).
    /// </summary>
    public fixed float InlineResult[25];
}
