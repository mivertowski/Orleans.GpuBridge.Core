// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime.K2K;

/// <summary>
/// Interface for kernel-to-kernel message dispatching.
/// K2K dispatch enables GPU-to-GPU communication without CPU involvement.
/// </summary>
public interface IK2KDispatcher
{
    /// <summary>
    /// Dispatches a message from one kernel to another using GPU-native messaging.
    /// </summary>
    /// <typeparam name="TMessage">The blittable message type.</typeparam>
    /// <param name="sourceActorId">The source actor identifier.</param>
    /// <param name="targetActorType">The target actor type name.</param>
    /// <param name="targetMethod">The target method name.</param>
    /// <param name="targetActorId">The target actor identifier.</param>
    /// <param name="message">The message to dispatch.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the dispatch operation.</returns>
    ValueTask DispatchAsync<TMessage>(
        long sourceActorId,
        string targetActorType,
        string targetMethod,
        long targetActorId,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged;

    /// <summary>
    /// Dispatches a message and awaits a response (request-response pattern).
    /// </summary>
    /// <typeparam name="TRequest">The blittable request type.</typeparam>
    /// <typeparam name="TResponse">The blittable response type.</typeparam>
    /// <param name="sourceActorId">The source actor identifier.</param>
    /// <param name="targetActorType">The target actor type name.</param>
    /// <param name="targetMethod">The target method name.</param>
    /// <param name="targetActorId">The target actor identifier.</param>
    /// <param name="request">The request message.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The response from the target kernel.</returns>
    ValueTask<TResponse> DispatchWithResponseAsync<TRequest, TResponse>(
        long sourceActorId,
        string targetActorType,
        string targetMethod,
        long targetActorId,
        TRequest request,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TResponse : unmanaged;

    /// <summary>
    /// Broadcasts a message to multiple target actors.
    /// </summary>
    /// <typeparam name="TMessage">The blittable message type.</typeparam>
    /// <param name="sourceActorId">The source actor identifier.</param>
    /// <param name="targetActorType">The target actor type name.</param>
    /// <param name="targetMethod">The target method name.</param>
    /// <param name="targetActorIds">The target actor identifiers.</param>
    /// <param name="message">The message to broadcast.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the broadcast operation.</returns>
    ValueTask BroadcastAsync<TMessage>(
        long sourceActorId,
        string targetActorType,
        string targetMethod,
        ReadOnlySpan<long> targetActorIds,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged;

    /// <summary>
    /// Gets the queue pointer for a target actor for direct GPU-level dispatch.
    /// </summary>
    /// <param name="targetActorType">The target actor type name.</param>
    /// <param name="targetActorId">The target actor identifier.</param>
    /// <returns>The GPU memory pointer to the target's message queue, or IntPtr.Zero if not found.</returns>
    IntPtr GetTargetQueuePointer(string targetActorType, long targetActorId);

    /// <summary>
    /// Registers a local actor's queue for K2K messaging.
    /// </summary>
    /// <param name="actorType">The actor type name.</param>
    /// <param name="actorId">The actor identifier.</param>
    /// <param name="queuePointer">The GPU memory pointer to the actor's message queue.</param>
    void RegisterQueue(string actorType, long actorId, IntPtr queuePointer);

    /// <summary>
    /// Unregisters an actor's queue when the actor is deactivated.
    /// </summary>
    /// <param name="actorType">The actor type name.</param>
    /// <param name="actorId">The actor identifier.</param>
    void UnregisterQueue(string actorType, long actorId);

    /// <summary>
    /// Registers an actor's device placement for P2P routing decisions.
    /// </summary>
    /// <param name="actorType">The actor type name.</param>
    /// <param name="actorId">The actor identifier.</param>
    /// <param name="deviceId">The GPU device ID where the actor resides.</param>
    void RegisterActorDevice(string actorType, long actorId, int deviceId);

    /// <summary>
    /// Gets the device ID where a target actor resides.
    /// </summary>
    /// <param name="actorType">The target actor type name.</param>
    /// <param name="actorId">The target actor identifier.</param>
    /// <returns>The device ID, or -1 if not found.</returns>
    int GetActorDevice(string actorType, long actorId);

    /// <summary>
    /// Gets K2K routing statistics for monitoring and diagnostics.
    /// </summary>
    /// <returns>Current routing statistics.</returns>
    K2KRoutingStats GetRoutingStats();
}

/// <summary>
/// Statistics for K2K routing operations.
/// </summary>
public sealed record K2KRoutingStats
{
    /// <summary>Total messages dispatched.</summary>
    public long TotalDispatches { get; init; }

    /// <summary>Messages sent via P2P path.</summary>
    public long P2PDispatches { get; init; }

    /// <summary>Messages sent via CPU-routed path.</summary>
    public long CpuRoutedDispatches { get; init; }

    /// <summary>Messages that failed to dispatch.</summary>
    public long FailedDispatches { get; init; }

    /// <summary>Broadcast operations performed.</summary>
    public long BroadcastOperations { get; init; }

    /// <summary>Request-response operations performed.</summary>
    public long RequestResponseOperations { get; init; }

    /// <summary>Average dispatch latency in nanoseconds.</summary>
    public double AverageLatencyNs { get; init; }

    /// <summary>P99 dispatch latency in nanoseconds.</summary>
    public double P99LatencyNs { get; init; }

    /// <summary>Number of registered actor queues.</summary>
    public int RegisteredQueues { get; init; }

    /// <summary>Number of P2P-enabled device pairs.</summary>
    public int P2PEnabledPairs { get; init; }
}
