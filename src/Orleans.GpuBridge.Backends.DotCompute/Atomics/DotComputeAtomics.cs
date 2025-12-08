// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Runtime.CompilerServices;
using DotCompute.Abstractions.Atomics;
using Orleans.GpuBridge.Abstractions.Atomics;

namespace Orleans.GpuBridge.Backends.DotCompute.Atomics;

/// <summary>
/// GPU atomic operations implementation using DotCompute.Atomics.
/// </summary>
/// <remarks>
/// <para>
/// This implementation wraps DotCompute's atomic operations, which automatically
/// translate to native GPU atomics on each backend:
/// </para>
/// <list type="bullet">
/// <item><description>CUDA: atomicAdd, atomicSub, atomicExch, atomicCAS, atomicMin, atomicMax, etc.</description></item>
/// <item><description>OpenCL: atomic_add, atomic_sub, atomic_xchg, atomic_cmpxchg, etc.</description></item>
/// <item><description>CPU: System.Threading.Interlocked operations with appropriate barriers.</description></item>
/// </list>
/// <para>
/// <b>Performance Characteristics:</b>
/// </para>
/// <list type="bullet">
/// <item><description>GPU atomic latency: 100-500 cycles (vs 4-200 for regular memory access)</description></item>
/// <item><description>Contention on hot atomics can cause serialization (warp divergence)</description></item>
/// <item><description>Memory fences add synchronization overhead but ensure visibility</description></item>
/// </list>
/// </remarks>
public sealed class DotComputeAtomics : IGpuAtomics
{
    /// <summary>
    /// Singleton instance for stateless atomic operations.
    /// </summary>
    public static readonly DotComputeAtomics Instance = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="DotComputeAtomics"/> class.
    /// </summary>
    private DotComputeAtomics()
    {
    }

    // ============================================================================
    // Atomic Add Operations
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicAdd(ref int target, int value)
        => AtomicOps.AtomicAdd(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicAdd(ref long target, long value)
        => AtomicOps.AtomicAdd(ref target, value);

    // ============================================================================
    // Atomic Subtract Operations
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicSub(ref int target, int value)
        => AtomicOps.AtomicSub(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicSub(ref long target, long value)
        => AtomicOps.AtomicSub(ref target, value);

    // ============================================================================
    // Atomic Exchange Operations
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicExchange(ref int target, int value)
        => AtomicOps.AtomicExchange(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicExchange(ref long target, long value)
        => AtomicOps.AtomicExchange(ref target, value);

    // ============================================================================
    // Atomic Compare-Exchange Operations
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicCompareExchange(ref int target, int expected, int desired)
        => AtomicOps.AtomicCompareExchange(ref target, expected, desired);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicCompareExchange(ref long target, long expected, long desired)
        => AtomicOps.AtomicCompareExchange(ref target, expected, desired);

    // ============================================================================
    // Atomic Min/Max Operations
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicMin(ref int target, int value)
        => AtomicOps.AtomicMin(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicMax(ref int target, int value)
        => AtomicOps.AtomicMax(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicMin(ref long target, long value)
        => AtomicOps.AtomicMin(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicMax(ref long target, long value)
        => AtomicOps.AtomicMax(ref target, value);

    // ============================================================================
    // Atomic Bitwise Operations
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicAnd(ref int target, int value)
        => AtomicOps.AtomicAnd(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicOr(ref int target, int value)
        => AtomicOps.AtomicOr(ref target, value);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicXor(ref int target, int value)
        => AtomicOps.AtomicXor(ref target, value);

    // ============================================================================
    // Atomic Load/Store with Memory Ordering
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AtomicLoad(ref int target, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent)
        => AtomicOps.AtomicLoad(ref target, ConvertMemoryOrder(order));

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AtomicLoad(ref long target, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent)
        => AtomicOps.AtomicLoad(ref target, ConvertMemoryOrder(order));

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AtomicStore(ref int target, int value, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent)
        => AtomicOps.AtomicStore(ref target, value, ConvertMemoryOrder(order));

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AtomicStore(ref long target, long value, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent)
        => AtomicOps.AtomicStore(ref target, value, ConvertMemoryOrder(order));

    // ============================================================================
    // Memory Fences
    // ============================================================================

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ThreadFence(GpuMemoryScope scope = GpuMemoryScope.Device)
        => AtomicOps.ThreadFence(ConvertMemoryScope(scope));

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MemoryBarrier()
        => AtomicOps.MemoryBarrier();

    // ============================================================================
    // Enum Conversion Helpers
    // ============================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static MemoryOrder ConvertMemoryOrder(GpuMemoryOrder order) => order switch
    {
        GpuMemoryOrder.Relaxed => MemoryOrder.Relaxed,
        GpuMemoryOrder.Acquire => MemoryOrder.Acquire,
        GpuMemoryOrder.Release => MemoryOrder.Release,
        GpuMemoryOrder.AcquireRelease => MemoryOrder.AcquireRelease,
        GpuMemoryOrder.SequentiallyConsistent => MemoryOrder.SequentiallyConsistent,
        _ => MemoryOrder.SequentiallyConsistent
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static MemoryScope ConvertMemoryScope(GpuMemoryScope scope) => scope switch
    {
        GpuMemoryScope.Workgroup => MemoryScope.Workgroup,
        GpuMemoryScope.Device => MemoryScope.Device,
        GpuMemoryScope.System => MemoryScope.System,
        _ => MemoryScope.Device
    };
}
