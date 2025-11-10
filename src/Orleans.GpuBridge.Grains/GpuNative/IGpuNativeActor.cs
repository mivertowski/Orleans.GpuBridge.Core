using System;
using System.Threading.Tasks;
using Orleans;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

namespace Orleans.GpuBridge.Grains.GpuNative;

/// <summary>
/// Interface for GPU-native actors that live permanently on the GPU.
/// These actors process messages at sub-microsecond latency (100-500ns) using ring kernels.
/// </summary>
/// <remarks>
/// GPU-native actors differ from traditional grains:
/// - State resides permanently in GPU memory
/// - Ring kernels run indefinitely processing messages
/// - Zero kernel launch overhead
/// - Sub-microsecond message passing (2M msgs/s/actor)
/// - Temporal ordering with GPU-native HLC (20ns updates)
/// </remarks>
public interface IGpuNativeActor : IGrainWithGuidKey
{
    /// <summary>
    /// Initializes the GPU-native actor and launches its ring kernel.
    /// The ring kernel runs indefinitely until the actor is deactivated.
    /// </summary>
    /// <param name="configuration">Actor configuration (queue sizes, threads, etc.)</param>
    Task InitializeAsync(GpuNativeActorConfiguration configuration);

    /// <summary>
    /// Sends a message to this GPU-native actor.
    /// Messages are queued in GPU memory and processed by the ring kernel.
    /// Performance: 100-500ns message latency.
    /// </summary>
    /// <param name="message">Message payload (must be unmanaged for GPU transfer)</param>
    /// <returns>HLC timestamp when message was enqueued</returns>
    Task<HLCTimestamp> SendMessageAsync(ActorMessage message);

    /// <summary>
    /// Gets current actor status (running, message count, HLC time, etc.).
    /// </summary>
    Task<GpuActorStatus> GetStatusAsync();

    /// <summary>
    /// Gets actor statistics (throughput, latency, queue depth, etc.).
    /// </summary>
    Task<GpuActorStatistics> GetStatisticsAsync();

    /// <summary>
    /// Stops the ring kernel and deactivates the actor.
    /// Gracefully drains message queues before shutdown.
    /// </summary>
    Task ShutdownAsync();
}

/// <summary>
/// Configuration for GPU-native actors.
/// </summary>
public sealed class GpuNativeActorConfiguration
{
    /// <summary>
    /// Message queue capacity (messages).
    /// </summary>
    public int MessageQueueCapacity { get; init; } = 10000;

    /// <summary>
    /// Size of each message in bytes.
    /// </summary>
    public int MessageSize { get; init; } = 256;

    /// <summary>
    /// Number of threads per actor (typically 1).
    /// </summary>
    public int ThreadsPerActor { get; init; } = 1;

    /// <summary>
    /// Enable temporal ordering with HLC.
    /// Performance impact: ~15% overhead.
    /// </summary>
    public bool EnableTemporalOrdering { get; init; } = true;

    /// <summary>
    /// Enable automatic timestamp injection.
    /// </summary>
    public bool EnableTimestamps { get; init; } = true;

    /// <summary>
    /// Ring kernel source code (CUDA).
    /// </summary>
    public required string RingKernelSource { get; init; }

    /// <summary>
    /// Kernel entry point function name.
    /// </summary>
    public string KernelEntryPoint { get; init; } = "actor_ring_kernel";

    /// <summary>
    /// Additional kernel arguments (actor state, custom buffers, etc.).
    /// </summary>
    public object[]? AdditionalArguments { get; init; }
}

/// <summary>
/// Message payload for GPU-native actors.
/// Must be unmanaged for GPU transfer.
/// </summary>
[System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack = 8)]
public struct ActorMessage
{
    /// <summary>
    /// Message type identifier.
    /// </summary>
    public int MessageType;

    /// <summary>
    /// Source actor ID.
    /// </summary>
    public Guid SourceActorId;

    /// <summary>
    /// Target actor ID.
    /// </summary>
    public Guid TargetActorId;

    /// <summary>
    /// HLC timestamp when message was sent.
    /// </summary>
    public long TimestampPhysical;
    public int TimestampLogical;

    /// <summary>
    /// Message payload data (up to 224 bytes for 256-byte messages).
    /// </summary>
    public unsafe fixed byte PayloadData[224];

    public ActorMessage(int messageType, Guid sourceId, Guid targetId, HLCTimestamp timestamp)
    {
        MessageType = messageType;
        SourceActorId = sourceId;
        TargetActorId = targetId;
        TimestampPhysical = timestamp.PhysicalTime;
        TimestampLogical = timestamp.LogicalCounter;
    }
}

/// <summary>
/// Status information for a GPU-native actor.
/// </summary>
public sealed class GpuActorStatus
{
    public required Guid ActorId { get; init; }
    public required bool IsRunning { get; init; }
    public required int PendingMessages { get; init; }
    public required HLCTimestamp CurrentTimestamp { get; init; }
    public required TimeSpan Uptime { get; init; }
    public required DateTimeOffset ActivationTime { get; init; }
}

/// <summary>
/// Performance statistics for a GPU-native actor.
/// </summary>
public sealed class GpuActorStatistics
{
    public required long TotalMessagesProcessed { get; init; }
    public required long TotalMessagesSent { get; init; }
    public required double AverageLatencyNanos { get; init; }
    public required double ThroughputMessagesPerSecond { get; init; }
    public required int CurrentQueueDepth { get; init; }
    public required int MaxQueueDepth { get; init; }
    public required double QueueUtilization { get; init; }
}
