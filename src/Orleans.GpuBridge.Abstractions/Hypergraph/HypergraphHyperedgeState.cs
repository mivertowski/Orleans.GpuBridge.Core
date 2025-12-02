// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Abstractions.Hypergraph;

/// <summary>
/// Persistent state for hypergraph hyperedge grains.
/// </summary>
/// <remarks>
/// This state is serialized and persisted to the configured Orleans storage provider,
/// allowing hyperedge state to survive grain deactivation and silo restarts.
/// </remarks>
[GenerateSerializer]
public sealed class HypergraphHyperedgeState
{
    /// <summary>
    /// Hyperedge type/label.
    /// </summary>
    [Id(0)]
    public string HyperedgeType { get; set; } = string.Empty;

    /// <summary>
    /// Current version number for optimistic concurrency.
    /// </summary>
    [Id(1)]
    public long Version { get; set; }

    /// <summary>
    /// Hyperedge properties dictionary.
    /// </summary>
    [Id(2)]
    public Dictionary<string, object> Properties { get; set; } = new();

    /// <summary>
    /// Member vertices keyed by vertex ID.
    /// </summary>
    [Id(3)]
    public Dictionary<string, HyperedgeMemberState> Members { get; set; } = new();

    /// <summary>
    /// Minimum cardinality constraint.
    /// </summary>
    [Id(4)]
    public int MinCardinality { get; set; } = 2;

    /// <summary>
    /// Maximum cardinality constraint (0 = unlimited).
    /// </summary>
    [Id(5)]
    public int MaxCardinality { get; set; }

    /// <summary>
    /// Whether hyperedge is directed (has source/target roles).
    /// </summary>
    [Id(6)]
    public bool IsDirected { get; set; }

    /// <summary>
    /// GPU affinity group for colocation with members.
    /// </summary>
    [Id(7)]
    public string? AffinityGroup { get; set; }

    /// <summary>
    /// Created timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(8)]
    public long CreatedAtNanos { get; set; }

    /// <summary>
    /// Last modified timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(9)]
    public long ModifiedAtNanos { get; set; }

    /// <summary>
    /// Hybrid logical clock timestamp for causal ordering.
    /// </summary>
    [Id(10)]
    public HybridTimestamp HlcTimestamp { get; set; }

    /// <summary>
    /// Whether the hyperedge has been initialized.
    /// </summary>
    [Id(11)]
    public bool IsInitialized { get; set; }

    /// <summary>
    /// History of hyperedge events (bounded).
    /// </summary>
    [Id(12)]
    public List<HyperedgeHistoryEvent> History { get; set; } = new();

    /// <summary>
    /// Maximum number of history events to retain.
    /// </summary>
    [Id(13)]
    public int MaxHistorySize { get; set; } = 1000;

    /// <summary>
    /// Metrics tracking state.
    /// </summary>
    [Id(14)]
    public HyperedgeMetricsState Metrics { get; set; } = new();
}

/// <summary>
/// Persistent state for hyperedge member information.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeMemberState
{
    /// <summary>
    /// Vertex ID.
    /// </summary>
    [Id(0)]
    public required string VertexId { get; init; }

    /// <summary>
    /// Role in hyperedge.
    /// </summary>
    [Id(1)]
    public string? Role { get; set; }

    /// <summary>
    /// Position in ordered hyperedge.
    /// </summary>
    [Id(2)]
    public int? Position { get; set; }

    /// <summary>
    /// When vertex joined (nanoseconds since Unix epoch).
    /// </summary>
    [Id(3)]
    public long JoinedAtNanos { get; set; }

    /// <summary>
    /// Custom properties for this membership.
    /// </summary>
    [Id(4)]
    public Dictionary<string, object>? MembershipProperties { get; set; }

    /// <summary>
    /// Last message send attempt timestamp.
    /// </summary>
    [Id(5)]
    public long LastMessageAtNanos { get; set; }

    /// <summary>
    /// Whether the member is currently reachable.
    /// </summary>
    [Id(6)]
    public bool IsReachable { get; set; } = true;
}

/// <summary>
/// Persistent state for hyperedge performance metrics.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeMetricsState
{
    /// <summary>
    /// Total messages broadcast.
    /// </summary>
    [Id(0)]
    public long MessagesBroadcast { get; set; }

    /// <summary>
    /// Total membership changes (adds + removes).
    /// </summary>
    [Id(1)]
    public long MembershipChanges { get; set; }

    /// <summary>
    /// Total broadcast latency in nanoseconds.
    /// </summary>
    [Id(2)]
    public long TotalBroadcastLatencyNanos { get; set; }

    /// <summary>
    /// Peak GPU memory usage (bytes).
    /// </summary>
    [Id(3)]
    public long PeakGpuMemoryBytes { get; set; }

    /// <summary>
    /// Total aggregations performed.
    /// </summary>
    [Id(4)]
    public long AggregationsPerformed { get; set; }

    /// <summary>
    /// Number of successful merge operations.
    /// </summary>
    [Id(5)]
    public long MergeOperations { get; set; }

    /// <summary>
    /// Number of successful split operations.
    /// </summary>
    [Id(6)]
    public long SplitOperations { get; set; }

    /// <summary>
    /// Number of constraint checks performed.
    /// </summary>
    [Id(7)]
    public long ConstraintChecks { get; set; }
}
