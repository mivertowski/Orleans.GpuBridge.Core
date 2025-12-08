// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Atomics;

/// <summary>
/// Memory ordering semantics for atomic operations.
/// </summary>
/// <remarks>
/// Maps to backend-specific memory orderings:
/// <list type="bullet">
/// <item><description>CUDA: memory_order_relaxed, memory_order_acquire, etc.</description></item>
/// <item><description>OpenCL: memory_order_relaxed, memory_order_acquire, etc.</description></item>
/// <item><description>CPU: Volatile operations with appropriate barriers.</description></item>
/// </list>
/// </remarks>
public enum GpuMemoryOrder
{
    /// <summary>No ordering guarantees beyond atomicity.</summary>
    Relaxed = 0,

    /// <summary>Acquire semantics - subsequent reads see all prior writes.</summary>
    Acquire = 1,

    /// <summary>Release semantics - prior writes are visible to subsequent reads.</summary>
    Release = 2,

    /// <summary>Both acquire and release semantics.</summary>
    AcquireRelease = 3,

    /// <summary>Full sequential consistency - total ordering of all atomic operations.</summary>
    SequentiallyConsistent = 4
}

/// <summary>
/// Memory scope for fence operations.
/// </summary>
/// <remarks>
/// Maps to backend-specific scopes:
/// <list type="bullet">
/// <item><description>CUDA: __threadfence_block(), __threadfence(), __threadfence_system()</description></item>
/// <item><description>OpenCL: CLK_LOCAL_MEM_FENCE, CLK_GLOBAL_MEM_FENCE</description></item>
/// </list>
/// </remarks>
public enum GpuMemoryScope
{
    /// <summary>Within workgroup/block only.</summary>
    Workgroup = 0,

    /// <summary>Within device (all GPU threads).</summary>
    Device = 1,

    /// <summary>System-wide (including CPU, for unified memory).</summary>
    System = 2
}

/// <summary>
/// Provides GPU-compatible atomic operations for lock-free data structures.
/// </summary>
/// <remarks>
/// <para>
/// This interface abstracts GPU atomic operations, enabling lock-free queue management,
/// counter updates, and synchronization primitives that work across CPU and GPU backends.
/// </para>
/// <para>
/// On GPU backends (CUDA, OpenCL), methods are translated to native atomics.
/// On CPU, they use <see cref="System.Threading.Interlocked"/> operations.
/// </para>
/// </remarks>
public interface IGpuAtomics
{
    // ============================================================================
    // Atomic Add Operations
    // ============================================================================

    /// <summary>
    /// Atomically adds a value to an integer and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to add.</param>
    /// <returns>The original value before the addition.</returns>
    int AtomicAdd(ref int target, int value);

    /// <summary>
    /// Atomically adds a value to a long integer and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target long integer.</param>
    /// <param name="value">The value to add.</param>
    /// <returns>The original value before the addition.</returns>
    long AtomicAdd(ref long target, long value);

    // ============================================================================
    // Atomic Subtract Operations
    // ============================================================================

    /// <summary>
    /// Atomically subtracts a value from an integer and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to subtract.</param>
    /// <returns>The original value before the subtraction.</returns>
    int AtomicSub(ref int target, int value);

    /// <summary>
    /// Atomically subtracts a value from a long integer and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target long integer.</param>
    /// <param name="value">The value to subtract.</param>
    /// <returns>The original value before the subtraction.</returns>
    long AtomicSub(ref long target, long value);

    // ============================================================================
    // Atomic Exchange Operations
    // ============================================================================

    /// <summary>
    /// Atomically exchanges an integer value and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to store.</param>
    /// <returns>The original value before the exchange.</returns>
    int AtomicExchange(ref int target, int value);

    /// <summary>
    /// Atomically exchanges a long integer value and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target long integer.</param>
    /// <param name="value">The value to store.</param>
    /// <returns>The original value before the exchange.</returns>
    long AtomicExchange(ref long target, long value);

    // ============================================================================
    // Atomic Compare-Exchange Operations
    // ============================================================================

    /// <summary>
    /// Atomically compares and exchanges an integer value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="expected">The value to compare against.</param>
    /// <param name="desired">The value to store if comparison succeeds.</param>
    /// <returns>The original value (compare with expected to check success).</returns>
    int AtomicCompareExchange(ref int target, int expected, int desired);

    /// <summary>
    /// Atomically compares and exchanges a long integer value.
    /// </summary>
    /// <param name="target">Reference to the target long integer.</param>
    /// <param name="expected">The value to compare against.</param>
    /// <param name="desired">The value to store if comparison succeeds.</param>
    /// <returns>The original value (compare with expected to check success).</returns>
    long AtomicCompareExchange(ref long target, long expected, long desired);

    // ============================================================================
    // Atomic Min/Max Operations
    // ============================================================================

    /// <summary>
    /// Atomically computes the minimum and stores it.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to compare.</param>
    /// <returns>The original value before the operation.</returns>
    int AtomicMin(ref int target, int value);

    /// <summary>
    /// Atomically computes the maximum and stores it.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to compare.</param>
    /// <returns>The original value before the operation.</returns>
    int AtomicMax(ref int target, int value);

    /// <summary>
    /// Atomically computes the minimum and stores it.
    /// </summary>
    /// <param name="target">Reference to the target long integer.</param>
    /// <param name="value">The value to compare.</param>
    /// <returns>The original value before the operation.</returns>
    long AtomicMin(ref long target, long value);

    /// <summary>
    /// Atomically computes the maximum and stores it.
    /// </summary>
    /// <param name="target">Reference to the target long integer.</param>
    /// <param name="value">The value to compare.</param>
    /// <returns>The original value before the operation.</returns>
    long AtomicMax(ref long target, long value);

    // ============================================================================
    // Atomic Bitwise Operations
    // ============================================================================

    /// <summary>
    /// Atomically performs a bitwise AND operation and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to AND with.</param>
    /// <returns>The original value before the operation.</returns>
    int AtomicAnd(ref int target, int value);

    /// <summary>
    /// Atomically performs a bitwise OR operation and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to OR with.</param>
    /// <returns>The original value before the operation.</returns>
    int AtomicOr(ref int target, int value);

    /// <summary>
    /// Atomically performs a bitwise XOR operation and returns the original value.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to XOR with.</param>
    /// <returns>The original value before the operation.</returns>
    int AtomicXor(ref int target, int value);

    // ============================================================================
    // Atomic Load/Store with Memory Ordering
    // ============================================================================

    /// <summary>
    /// Atomically loads an integer value with specified memory ordering.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="order">Memory ordering semantics.</param>
    /// <returns>The current value.</returns>
    int AtomicLoad(ref int target, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent);

    /// <summary>
    /// Atomically loads a long value with specified memory ordering.
    /// </summary>
    /// <param name="target">Reference to the target long.</param>
    /// <param name="order">Memory ordering semantics.</param>
    /// <returns>The current value.</returns>
    long AtomicLoad(ref long target, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent);

    /// <summary>
    /// Atomically stores an integer value with specified memory ordering.
    /// </summary>
    /// <param name="target">Reference to the target integer.</param>
    /// <param name="value">The value to store.</param>
    /// <param name="order">Memory ordering semantics.</param>
    void AtomicStore(ref int target, int value, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent);

    /// <summary>
    /// Atomically stores a long value with specified memory ordering.
    /// </summary>
    /// <param name="target">Reference to the target long.</param>
    /// <param name="value">The value to store.</param>
    /// <param name="order">Memory ordering semantics.</param>
    void AtomicStore(ref long target, long value, GpuMemoryOrder order = GpuMemoryOrder.SequentiallyConsistent);

    // ============================================================================
    // Memory Fences
    // ============================================================================

    /// <summary>
    /// Issues a memory fence with the specified scope.
    /// </summary>
    /// <param name="scope">The memory scope for the fence.</param>
    void ThreadFence(GpuMemoryScope scope = GpuMemoryScope.Device);

    /// <summary>
    /// Issues a full memory barrier ensuring all prior memory operations
    /// are visible before any subsequent operations.
    /// </summary>
    void MemoryBarrier();
}
