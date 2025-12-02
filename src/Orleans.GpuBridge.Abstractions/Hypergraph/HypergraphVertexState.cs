// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Abstractions.Hypergraph;

/// <summary>
/// Persistent state for hypergraph vertex grains.
/// </summary>
/// <remarks>
/// This state is serialized and persisted to the configured Orleans storage provider,
/// allowing vertex state to survive grain deactivation and silo restarts.
/// </remarks>
[GenerateSerializer]
public sealed class HypergraphVertexState
{
    /// <summary>
    /// Vertex type/label.
    /// </summary>
    [Id(0)]
    public string VertexType { get; set; } = string.Empty;

    /// <summary>
    /// Current version number for optimistic concurrency.
    /// </summary>
    [Id(1)]
    public long Version { get; set; }

    /// <summary>
    /// Vertex properties dictionary.
    /// </summary>
    [Id(2)]
    public Dictionary<string, object> Properties { get; set; } = new();

    /// <summary>
    /// Hyperedge memberships keyed by hyperedge ID.
    /// </summary>
    [Id(3)]
    public Dictionary<string, HyperedgeMembershipState> Hyperedges { get; set; } = new();

    /// <summary>
    /// GPU affinity group for colocation with related grains.
    /// </summary>
    [Id(4)]
    public string? AffinityGroup { get; set; }

    /// <summary>
    /// Created timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(5)]
    public long CreatedAtNanos { get; set; }

    /// <summary>
    /// Last modified timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(6)]
    public long ModifiedAtNanos { get; set; }

    /// <summary>
    /// Hybrid logical clock timestamp for causal ordering.
    /// </summary>
    [Id(7)]
    public HybridTimestamp HlcTimestamp { get; set; }

    /// <summary>
    /// Whether the vertex has been initialized.
    /// </summary>
    [Id(8)]
    public bool IsInitialized { get; set; }

    /// <summary>
    /// Message handlers registered for this vertex (by message type).
    /// </summary>
    [Id(9)]
    public HashSet<string> RegisteredMessageTypes { get; set; } = new();

    /// <summary>
    /// Metrics tracking state.
    /// </summary>
    [Id(10)]
    public VertexMetricsState Metrics { get; set; } = new();
}

/// <summary>
/// Persistent state for hyperedge membership information.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeMembershipState
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    [Id(0)]
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Role within the hyperedge.
    /// </summary>
    [Id(1)]
    public string? Role { get; set; }

    /// <summary>
    /// When membership started (nanoseconds since Unix epoch).
    /// </summary>
    [Id(2)]
    public long JoinedAtNanos { get; set; }

    /// <summary>
    /// Cached count of other vertices in this hyperedge.
    /// </summary>
    [Id(3)]
    public int CachedPeerCount { get; set; }

    /// <summary>
    /// Last updated timestamp for cached peer count.
    /// </summary>
    [Id(4)]
    public long CachedPeerCountUpdatedNanos { get; set; }
}

/// <summary>
/// Persistent state for vertex performance metrics.
/// </summary>
[GenerateSerializer]
public sealed class VertexMetricsState
{
    /// <summary>
    /// Total messages received.
    /// </summary>
    [Id(0)]
    public long MessagesReceived { get; set; }

    /// <summary>
    /// Total messages sent.
    /// </summary>
    [Id(1)]
    public long MessagesSent { get; set; }

    /// <summary>
    /// Total processing time in nanoseconds.
    /// </summary>
    [Id(2)]
    public long TotalProcessingTimeNanos { get; set; }

    /// <summary>
    /// Total pattern matches performed.
    /// </summary>
    [Id(3)]
    public long PatternMatchCount { get; set; }

    /// <summary>
    /// Peak GPU memory usage (bytes).
    /// </summary>
    [Id(4)]
    public long PeakGpuMemoryBytes { get; set; }

    /// <summary>
    /// Number of broadcasts initiated.
    /// </summary>
    [Id(5)]
    public long BroadcastsInitiated { get; set; }

    /// <summary>
    /// Number of neighbor queries performed.
    /// </summary>
    [Id(6)]
    public long NeighborQueries { get; set; }

    /// <summary>
    /// Number of aggregations performed.
    /// </summary>
    [Id(7)]
    public long AggregationsPerformed { get; set; }
}
