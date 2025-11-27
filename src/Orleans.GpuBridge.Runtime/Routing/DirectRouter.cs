// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Runtime.K2K;

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Direct routing strategy for point-to-point K2K messaging.
/// Provides lowest latency (100-500ns) for known targets.
/// </summary>
public sealed class DirectRouter : IK2KRouter
{
    private readonly IK2KDispatcher _dispatcher;

    /// <summary>
    /// Initializes a new instance of the direct router.
    /// </summary>
    public DirectRouter(IK2KDispatcher dispatcher)
    {
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
    }

    /// <inheritdoc />
    public K2KRoutingStrategy Strategy => K2KRoutingStrategy.Direct;

    /// <inheritdoc />
    public ValueTask RouteAsync<TMessage>(
        K2KRoutingContext context,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged
    {
        if (!context.ExplicitTargetId.HasValue)
        {
            throw new InvalidOperationException("Direct routing requires an explicit target actor ID");
        }

        return _dispatcher.DispatchAsync(
            context.SourceActorId,
            context.TargetActorType,
            context.TargetMethod,
            context.ExplicitTargetId.Value,
            message,
            cancellationToken);
    }

    /// <inheritdoc />
    public ReadOnlySpan<long> ResolveTargets(K2KRoutingContext context)
    {
        if (!context.ExplicitTargetId.HasValue)
        {
            return ReadOnlySpan<long>.Empty;
        }

        // Return single-element span
        return new[] { context.ExplicitTargetId.Value };
    }
}
