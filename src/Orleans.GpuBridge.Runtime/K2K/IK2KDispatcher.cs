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
}
