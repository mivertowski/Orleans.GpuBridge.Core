// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.K2K;

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Ring routing strategy for circular topology K2K messaging.
/// Messages are forwarded in a ring pattern until max hops reached.
/// Useful for consensus protocols and aggregation patterns.
/// </summary>
public sealed class RingRouter : IK2KRouter
{
    private readonly K2KDispatcher _dispatcher;

    /// <summary>
    /// Initializes a new instance of the ring router.
    /// </summary>
    public RingRouter(K2KDispatcher dispatcher)
    {
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
    }

    /// <inheritdoc />
    public K2KRoutingStrategy Strategy => K2KRoutingStrategy.Ring;

    /// <inheritdoc />
    public ValueTask RouteAsync<TMessage>(
        K2KRoutingContext context,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged
    {
        // Check hop count to prevent infinite loops
        if (context.CurrentHopCount >= context.MaxHopCount)
        {
            return ValueTask.CompletedTask;
        }

        var nextActorId = _dispatcher.GetNextInRing(
            context.TargetActorType,
            context.SourceActorId);

        if (nextActorId < 0)
        {
            return ValueTask.CompletedTask;
        }

        return _dispatcher.DispatchAsync(
            context.SourceActorId,
            context.TargetActorType,
            context.TargetMethod,
            nextActorId,
            message,
            cancellationToken);
    }

    /// <inheritdoc />
    public ReadOnlySpan<long> ResolveTargets(K2KRoutingContext context)
    {
        if (context.CurrentHopCount >= context.MaxHopCount)
        {
            return ReadOnlySpan<long>.Empty;
        }

        var nextActorId = _dispatcher.GetNextInRing(
            context.TargetActorType,
            context.SourceActorId);

        if (nextActorId < 0)
        {
            return ReadOnlySpan<long>.Empty;
        }

        return new[] { nextActorId };
    }
}
