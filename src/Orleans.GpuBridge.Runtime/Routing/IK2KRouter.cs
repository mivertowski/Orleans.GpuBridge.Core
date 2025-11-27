// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Interface for K2K routing strategies.
/// </summary>
public interface IK2KRouter
{
    /// <summary>
    /// Gets the routing strategy type.
    /// </summary>
    K2KRoutingStrategy Strategy { get; }

    /// <summary>
    /// Routes a message to target actor(s) based on the strategy.
    /// </summary>
    /// <typeparam name="TMessage">The blittable message type.</typeparam>
    /// <param name="context">The routing context with source and target information.</param>
    /// <param name="message">The message to route.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A task representing the routing operation.</returns>
    ValueTask RouteAsync<TMessage>(
        K2KRoutingContext context,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged;

    /// <summary>
    /// Resolves target actor ID(s) based on the routing strategy.
    /// </summary>
    /// <param name="context">The routing context.</param>
    /// <returns>The resolved target actor ID(s).</returns>
    ReadOnlySpan<long> ResolveTargets(K2KRoutingContext context);
}

/// <summary>
/// Context information for K2K routing decisions.
/// </summary>
public readonly record struct K2KRoutingContext
{
    /// <summary>
    /// Gets the source actor ID.
    /// </summary>
    public required long SourceActorId { get; init; }

    /// <summary>
    /// Gets the source actor type.
    /// </summary>
    public required string SourceActorType { get; init; }

    /// <summary>
    /// Gets the target actor type.
    /// </summary>
    public required string TargetActorType { get; init; }

    /// <summary>
    /// Gets the target method name.
    /// </summary>
    public required string TargetMethod { get; init; }

    /// <summary>
    /// Gets the explicit target actor ID (for Direct routing).
    /// </summary>
    public long? ExplicitTargetId { get; init; }

    /// <summary>
    /// Gets the message hash (for HashRouted routing).
    /// </summary>
    public int MessageHash { get; init; }

    /// <summary>
    /// Gets the current hop count (for Ring routing).
    /// </summary>
    public int CurrentHopCount { get; init; }

    /// <summary>
    /// Gets the maximum hop count (for Ring routing).
    /// </summary>
    public int MaxHopCount { get; init; }

    /// <summary>
    /// Gets whether CPU fallback is allowed.
    /// </summary>
    public bool AllowCpuFallback { get; init; }
}
