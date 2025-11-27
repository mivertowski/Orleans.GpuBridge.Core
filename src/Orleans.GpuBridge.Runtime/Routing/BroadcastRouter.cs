// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.K2K;

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Broadcast routing strategy for one-to-many K2K messaging.
/// Sends messages to all actors of the target type.
/// </summary>
public sealed class BroadcastRouter : IK2KRouter
{
    private readonly K2KDispatcher _dispatcher;

    /// <summary>
    /// Initializes a new instance of the broadcast router.
    /// </summary>
    public BroadcastRouter(K2KDispatcher dispatcher)
    {
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
    }

    /// <inheritdoc />
    public K2KRoutingStrategy Strategy => K2KRoutingStrategy.Broadcast;

    /// <inheritdoc />
    public ValueTask RouteAsync<TMessage>(
        K2KRoutingContext context,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged
    {
        var targets = _dispatcher.GetActorIdsByType(context.TargetActorType);

        if (targets.IsEmpty)
        {
            return ValueTask.CompletedTask;
        }

        return _dispatcher.BroadcastAsync(
            context.SourceActorId,
            context.TargetActorType,
            context.TargetMethod,
            targets,
            message,
            cancellationToken);
    }

    /// <inheritdoc />
    public ReadOnlySpan<long> ResolveTargets(K2KRoutingContext context)
    {
        return _dispatcher.GetActorIdsByType(context.TargetActorType);
    }
}
