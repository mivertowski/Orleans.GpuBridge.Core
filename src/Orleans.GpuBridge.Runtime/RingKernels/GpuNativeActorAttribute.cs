// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using DotCompute.Abstractions.RingKernels;

namespace Orleans.GpuBridge.Runtime.RingKernels;

/// <summary>
/// Marks an Orleans grain as a GPU-native actor using persistent ring kernels.
/// </summary>
/// <remarks>
/// <para>
/// This attribute configures how the grain's ring kernel is compiled and executed on GPU.
/// It maps to DotCompute's RingKernelAttribute for kernel compilation.
/// </para>
/// <para>
/// Example usage:
/// <code>
/// [GpuNativeActor(
///     Domain = RingKernelDomain.GraphAnalytics,
///     MessagingStrategy = MessagePassingStrategy.SharedMemory,
///     Capacity = 1024)]
/// public class GraphVertexActor : GpuNativeGrain, IGraphVertexActor
/// {
///     // Actor implementation
/// }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class GpuNativeActorAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the ring buffer capacity for message queue.
    /// </summary>
    /// <remarks>
    /// Must be a power of 2. Default: 1024 messages.
    /// Larger capacity = higher memory usage but lower queue contention.
    /// </remarks>
    public int Capacity { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the input queue size.
    /// </summary>
    /// <remarks>
    /// Default: 256 messages. Increase for high-throughput actors.
    /// </remarks>
    public int InputQueueSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the output queue size.
    /// </summary>
    /// <remarks>
    /// Default: 256 messages. Increase if actor produces many responses.
    /// </remarks>
    public int OutputQueueSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the ring kernel execution mode.
    /// </summary>
    /// <remarks>
    /// - Persistent: Infinite loop, always running (best for latency)
    /// - EventDriven: Process burst and exit (best for power efficiency)
    /// Default: Persistent
    /// </remarks>
    public RingKernelMode Mode { get; set; } = RingKernelMode.Persistent;

    /// <summary>
    /// Gets or sets the message passing strategy.
    /// </summary>
    /// <remarks>
    /// - SharedMemory: Lock-free queues in GPU shared memory (lowest latency)
    /// - AtomicQueue: Atomic operations on global memory (moderate latency)
    /// - P2P: Peer-to-peer GPU-to-GPU (multi-GPU scenarios)
    /// - NCCL: NVIDIA Collective Communications Library (distributed)
    /// Default: SharedMemory
    /// </remarks>
    public MessagePassingStrategy MessagingStrategy { get; set; } = MessagePassingStrategy.SharedMemory;

    /// <summary>
    /// Gets or sets the ring kernel domain for optimization hints.
    /// </summary>
    /// <remarks>
    /// - General: Standard actor model
    /// - GraphAnalytics: Hypergraph/graph processing (grid sync enabled)
    /// - SpatialSimulation: Physics/spatial queries
    /// - ActorModel: Pure actor messaging (minimal overhead)
    /// Default: General
    /// </remarks>
    public RingKernelDomain Domain { get; set; } = RingKernelDomain.General;

    /// <summary>
    /// Gets or sets the grid size for kernel launch.
    /// </summary>
    /// <remarks>
    /// Number of GPU blocks. Default: 1 (suitable for single-actor workloads).
    /// Increase for multi-tenant scenarios or data-parallel actors.
    /// </remarks>
    public int GridSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the block size for kernel launch.
    /// </summary>
    /// <remarks>
    /// Number of threads per block. Default: 256.
    /// Must be multiple of warp size (32). Max: 1024 on most GPUs.
    /// </remarks>
    public int BlockSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets a value indicating whether to enable temporal alignment on GPU.
    /// </summary>
    /// <remarks>
    /// When true, integrates with Orleans.GpuBridge.Runtime Temporal subsystem:
    /// - Maintains HLC clock in GPU memory (20ns vs 50ns CPU)
    /// - Enables causal ordering for actor messages
    /// - Supports temporal pattern detection
    /// Default: false
    /// </remarks>
    public bool EnableTemporalAlignment { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to enable state persistence to GPU memory.
    /// </summary>
    /// <remarks>
    /// When true, actor state is persisted to GPU memory for durability.
    /// Requires GPUDirect Storage or periodic GPU→CPU sync.
    /// Default: false (volatile GPU state)
    /// </remarks>
    public bool EnableStatePersistence { get; set; } = false;

    /// <summary>
    /// Gets or sets the preferred GPU device index.
    /// </summary>
    /// <remarks>
    /// -1 = Auto-select (placement strategy decides)
    /// 0+ = Pin to specific GPU device
    /// Default: -1
    /// </remarks>
    public int PreferredGpuDevice { get; set; } = -1;
}
