// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Runtime.K2K;

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Factory for creating K2K routers based on routing strategy.
/// </summary>
public sealed class K2KRouterFactory
{
    private readonly IServiceProvider _serviceProvider;
    private readonly K2KDispatcher _dispatcher;

    /// <summary>
    /// Initializes a new instance of the K2K router factory.
    /// </summary>
    public K2KRouterFactory(IServiceProvider serviceProvider, K2KDispatcher dispatcher)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
    }

    /// <summary>
    /// Creates a router for the specified routing strategy.
    /// </summary>
    public IK2KRouter CreateRouter(K2KRoutingStrategy strategy)
    {
        return strategy switch
        {
            K2KRoutingStrategy.Direct => new DirectRouter(_dispatcher),
            K2KRoutingStrategy.Broadcast => new BroadcastRouter(_dispatcher),
            K2KRoutingStrategy.Ring => new RingRouter(_dispatcher),
            K2KRoutingStrategy.HashRouted => new HashRoutedRouter(_dispatcher),
            _ => throw new ArgumentOutOfRangeException(nameof(strategy), strategy, "Unknown routing strategy")
        };
    }

    /// <summary>
    /// Gets a router from DI or creates one if not registered.
    /// </summary>
    public IK2KRouter GetRouter(K2KRoutingStrategy strategy)
    {
        // Try to get from DI first
        var router = _serviceProvider.GetKeyedService<IK2KRouter>(strategy);

        return router ?? CreateRouter(strategy);
    }
}
