// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Generation;

/// <summary>
/// Marks a GPU handler method as targeting another GPU-native actor via kernel-to-kernel (K2K) messaging.
/// K2K allows direct GPU-to-GPU communication without CPU involvement.
/// </summary>
/// <remarks>
/// <para>
/// K2K messaging provides:
/// <list type="bullet">
/// <item><description>Direct GPU memory-to-memory message passing</description></item>
/// <item><description>Sub-microsecond latency (100-500ns)</description></item>
/// <item><description>No CPU context switches or system calls</description></item>
/// <item><description>Automatic routing table generation at compile time</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Requirements:</strong>
/// <list type="bullet">
/// <item><description>Target actor must also be a [GpuNativeActor]</description></item>
/// <item><description>Target method must be a [GpuHandler]</description></item>
/// <item><description>Message payload sizes must be compatible</description></item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// [GpuNativeActor]
/// public interface IRouterActor : IGrainWithIntegerKey
/// {
///     [GpuHandler]
///     [K2KTarget(typeof(IWorkerActor), nameof(IWorkerActor.ProcessAsync))]
///     Task RouteToWorkerAsync(WorkItem item, long workerId);
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true, Inherited = false)]
public sealed class K2KTargetAttribute : Attribute
{
    /// <summary>
    /// Gets the target actor interface type.
    /// </summary>
    public Type TargetActorType { get; }

    /// <summary>
    /// Gets the name of the target method on the target actor.
    /// </summary>
    public string TargetMethodName { get; }

    /// <summary>
    /// Gets or sets whether to fall back to CPU-based routing if K2K is unavailable.
    /// Default is true.
    /// </summary>
    public bool AllowCpuFallback { get; set; } = true;

    /// <summary>
    /// Gets or sets the routing strategy for K2K messages.
    /// </summary>
    public K2KRoutingStrategy RoutingStrategy { get; set; } = K2KRoutingStrategy.Direct;

    /// <summary>
    /// Initializes a new instance of the <see cref="K2KTargetAttribute"/> class.
    /// </summary>
    /// <param name="targetActorType">The type of the target actor interface.</param>
    /// <param name="targetMethodName">The name of the target method.</param>
    public K2KTargetAttribute(Type targetActorType, string targetMethodName)
    {
        TargetActorType = targetActorType ?? throw new ArgumentNullException(nameof(targetActorType));
        TargetMethodName = targetMethodName ?? throw new ArgumentNullException(nameof(targetMethodName));
    }
}

/// <summary>
/// Specifies the routing strategy for K2K messages.
/// </summary>
public enum K2KRoutingStrategy
{
    /// <summary>
    /// Direct point-to-point routing to a specific actor instance.
    /// </summary>
    Direct = 0,

    /// <summary>
    /// Broadcast to all instances of the target actor type.
    /// </summary>
    Broadcast = 1,

    /// <summary>
    /// Round-robin load balancing across target instances.
    /// </summary>
    RoundRobin = 2,

    /// <summary>
    /// Route based on message content hash (consistent hashing).
    /// </summary>
    ContentBased = 3
}
