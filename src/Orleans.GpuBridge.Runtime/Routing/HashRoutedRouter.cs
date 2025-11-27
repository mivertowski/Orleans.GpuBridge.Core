// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.K2K;

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Hash-based routing strategy using consistent hashing.
/// Distributes load evenly across actors based on message content hash.
/// </summary>
public sealed class HashRoutedRouter : IK2KRouter
{
    private readonly K2KDispatcher _dispatcher;

    /// <summary>
    /// Initializes a new instance of the hash-routed router.
    /// </summary>
    public HashRoutedRouter(K2KDispatcher dispatcher)
    {
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
    }

    /// <inheritdoc />
    public K2KRoutingStrategy Strategy => K2KRoutingStrategy.HashRouted;

    /// <inheritdoc />
    public ValueTask RouteAsync<TMessage>(
        K2KRoutingContext context,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged
    {
        var targetActorId = _dispatcher.GetHashRoutedTarget(
            context.TargetActorType,
            context.MessageHash);

        if (targetActorId < 0)
        {
            return ValueTask.CompletedTask;
        }

        return _dispatcher.DispatchAsync(
            context.SourceActorId,
            context.TargetActorType,
            context.TargetMethod,
            targetActorId,
            message,
            cancellationToken);
    }

    /// <inheritdoc />
    public ReadOnlySpan<long> ResolveTargets(K2KRoutingContext context)
    {
        var targetActorId = _dispatcher.GetHashRoutedTarget(
            context.TargetActorType,
            context.MessageHash);

        if (targetActorId < 0)
        {
            return ReadOnlySpan<long>.Empty;
        }

        return new[] { targetActorId };
    }
}
