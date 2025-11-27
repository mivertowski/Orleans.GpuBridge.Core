// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Runtime.Routing;

/// <summary>
/// Defines routing strategies for kernel-to-kernel (K2K) messaging.
/// </summary>
public enum K2KRoutingStrategy
{
    /// <summary>
    /// Direct point-to-point routing to a specific target actor.
    /// Provides lowest latency (100-500ns) for known targets.
    /// </summary>
    Direct = 0,

    /// <summary>
    /// Broadcasts the message to all actors of the target type.
    /// Uses parallel GPU-native message enqueue operations.
    /// </summary>
    Broadcast = 1,

    /// <summary>
    /// Ring topology routing where messages are forwarded in a circular pattern.
    /// Each actor forwards to its successor until the message completes the ring.
    /// Useful for consensus protocols and aggregation patterns.
    /// </summary>
    Ring = 2,

    /// <summary>
    /// Hash-based routing using consistent hashing to select target actors.
    /// Distributes load evenly across actors based on message content hash.
    /// </summary>
    HashRouted = 3
}

/// <summary>
/// Extension methods for K2K routing strategies.
/// </summary>
public static class K2KRoutingStrategyExtensions
{
    /// <summary>
    /// Gets whether this routing strategy requires a target actor ID.
    /// </summary>
    public static bool RequiresTargetId(this K2KRoutingStrategy strategy) =>
        strategy == K2KRoutingStrategy.Direct;

    /// <summary>
    /// Gets whether this routing strategy targets multiple actors.
    /// </summary>
    public static bool IsMultiTarget(this K2KRoutingStrategy strategy) =>
        strategy is K2KRoutingStrategy.Broadcast or K2KRoutingStrategy.Ring;

    /// <summary>
    /// Gets the default hop count for ring routing.
    /// </summary>
    public static int GetDefaultMaxHops(this K2KRoutingStrategy strategy) =>
        strategy == K2KRoutingStrategy.Ring ? 32 : 1;
}
