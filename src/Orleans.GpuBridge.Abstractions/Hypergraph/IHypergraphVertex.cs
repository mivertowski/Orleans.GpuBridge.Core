// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans;

namespace Orleans.GpuBridge.Abstractions.Hypergraph;

/// <summary>
/// GPU-native hypergraph vertex actor interface for multi-way relationship graphs.
/// </summary>
/// <remarks>
/// <para>
/// Hypergraph vertices represent entities that can participate in multiple hyperedges.
/// Unlike standard graphs where edges connect exactly two vertices, hyperedges can
/// connect any number of vertices, enabling modeling of complex multi-way relationships.
/// </para>
/// <para>
/// <strong>GPU-Native Features:</strong>
/// <list type="bullet">
/// <item><description>CSR (Compressed Sparse Row) format for efficient GPU memory layout</description></item>
/// <item><description>Sub-microsecond message passing between connected vertices</description></item>
/// <item><description>GPU-accelerated pattern matching across hyperedges</description></item>
/// <item><description>Temporal ordering with HLC timestamps for causal correctness</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Use Cases:</strong>
/// <list type="bullet">
/// <item><description>Knowledge graphs with complex entity relationships</description></item>
/// <item><description>Social networks with group interactions</description></item>
/// <item><description>Chemical compound modeling (atoms in molecules)</description></item>
/// <item><description>Database query optimization (join hypergraphs)</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IHypergraphVertex : IGrainWithStringKey
{
    /// <summary>
    /// Initializes the vertex with properties.
    /// </summary>
    /// <param name="request">Initialization request with vertex properties.</param>
    /// <returns>Result of initialization.</returns>
    Task<VertexInitResult> InitializeAsync(VertexInitRequest request);

    /// <summary>
    /// Gets vertex state including all properties and hyperedge memberships.
    /// </summary>
    /// <returns>Current vertex state.</returns>
    Task<VertexState> GetStateAsync();

    /// <summary>
    /// Updates vertex properties.
    /// </summary>
    /// <param name="properties">Properties to update.</param>
    /// <returns>Update result with new version.</returns>
    Task<VertexUpdateResult> UpdatePropertiesAsync(IReadOnlyDictionary<string, object> properties);

    /// <summary>
    /// Joins a hyperedge (adds vertex to hyperedge membership).
    /// </summary>
    /// <param name="hyperedgeId">Hyperedge identifier to join.</param>
    /// <param name="role">Role within the hyperedge (optional).</param>
    /// <returns>Result of joining.</returns>
    Task<HyperedgeMembershipResult> JoinHyperedgeAsync(string hyperedgeId, string? role = null);

    /// <summary>
    /// Leaves a hyperedge (removes vertex from hyperedge membership).
    /// </summary>
    /// <param name="hyperedgeId">Hyperedge identifier to leave.</param>
    /// <returns>Result of leaving.</returns>
    Task<HyperedgeMembershipResult> LeaveHyperedgeAsync(string hyperedgeId);

    /// <summary>
    /// Gets all hyperedges this vertex participates in.
    /// </summary>
    /// <returns>List of hyperedge memberships.</returns>
    Task<IReadOnlyList<HyperedgeMembership>> GetHyperedgesAsync();

    /// <summary>
    /// Sends a message to all vertices in a shared hyperedge.
    /// </summary>
    /// <param name="hyperedgeId">Hyperedge to broadcast within.</param>
    /// <param name="message">Message to broadcast.</param>
    /// <returns>Broadcast result with delivery status.</returns>
    Task<BroadcastResult> BroadcastToHyperedgeAsync(string hyperedgeId, VertexMessage message);

    /// <summary>
    /// Receives a message from another vertex.
    /// </summary>
    /// <param name="message">Incoming message.</param>
    /// <returns>Message processing result.</returns>
    Task<MessageResult> ReceiveMessageAsync(VertexMessage message);

    /// <summary>
    /// Queries neighbors reachable through hyperedges within specified hops.
    /// </summary>
    /// <param name="maxHops">Maximum number of hyperedge traversals.</param>
    /// <param name="filter">Optional filter for neighbors.</param>
    /// <returns>Neighbor query result.</returns>
    Task<NeighborQueryResult> QueryNeighborsAsync(int maxHops = 1, NeighborFilter? filter = null);

    /// <summary>
    /// Executes a pattern match starting from this vertex.
    /// </summary>
    /// <param name="pattern">Pattern to match.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Pattern match results.</returns>
    Task<PatternMatchResult> MatchPatternAsync(HypergraphPattern pattern, CancellationToken ct = default);

    /// <summary>
    /// Computes aggregate values across connected vertices via hyperedges.
    /// </summary>
    /// <param name="aggregation">Aggregation specification.</param>
    /// <returns>Aggregation result.</returns>
    Task<AggregationResult> AggregateAsync(VertexAggregation aggregation);

    /// <summary>
    /// Gets performance metrics for this vertex.
    /// </summary>
    /// <returns>Vertex performance metrics.</returns>
    Task<VertexMetrics> GetMetricsAsync();
}

/// <summary>
/// Request to initialize a hypergraph vertex.
/// </summary>
public sealed class VertexInitRequest
{
    /// <summary>
    /// Vertex type/label.
    /// </summary>
    public required string VertexType { get; init; }

    /// <summary>
    /// Initial properties.
    /// </summary>
    public required IReadOnlyDictionary<string, object> Properties { get; init; }

    /// <summary>
    /// Initial hyperedge memberships.
    /// </summary>
    public IReadOnlyList<string>? InitialHyperedges { get; init; }

    /// <summary>
    /// GPU affinity group for colocation.
    /// </summary>
    public string? AffinityGroup { get; init; }
}

/// <summary>
/// Result of vertex initialization.
/// </summary>
public readonly record struct VertexInitResult
{
    /// <summary>
    /// Whether initialization succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Vertex ID assigned.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// Initial version number.
    /// </summary>
    public required long Version { get; init; }

    /// <summary>
    /// Timestamp of creation (nanoseconds).
    /// </summary>
    public required long CreatedAtNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Current state of a vertex.
/// </summary>
public sealed class VertexState
{
    /// <summary>
    /// Vertex ID.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// Vertex type/label.
    /// </summary>
    public required string VertexType { get; init; }

    /// <summary>
    /// Current version number.
    /// </summary>
    public required long Version { get; init; }

    /// <summary>
    /// Current properties.
    /// </summary>
    public required IReadOnlyDictionary<string, object> Properties { get; init; }

    /// <summary>
    /// Hyperedge memberships.
    /// </summary>
    public required IReadOnlyList<HyperedgeMembership> Hyperedges { get; init; }

    /// <summary>
    /// Created timestamp (nanoseconds).
    /// </summary>
    public required long CreatedAtNanos { get; init; }

    /// <summary>
    /// Last modified timestamp (nanoseconds).
    /// </summary>
    public required long ModifiedAtNanos { get; init; }

    /// <summary>
    /// HLC timestamp for causal ordering.
    /// </summary>
    public required long HlcTimestamp { get; init; }
}

/// <summary>
/// Result of vertex property update.
/// </summary>
public readonly record struct VertexUpdateResult
{
    /// <summary>
    /// Whether update succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// New version number.
    /// </summary>
    public required long NewVersion { get; init; }

    /// <summary>
    /// Timestamp of update (nanoseconds).
    /// </summary>
    public required long UpdatedAtNanos { get; init; }

    /// <summary>
    /// Properties that were changed.
    /// </summary>
    public required IReadOnlyList<string> ChangedProperties { get; init; }
}

/// <summary>
/// Membership information for a hyperedge.
/// </summary>
public sealed class HyperedgeMembership
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Role within hyperedge.
    /// </summary>
    public string? Role { get; init; }

    /// <summary>
    /// When membership started (nanoseconds).
    /// </summary>
    public required long JoinedAtNanos { get; init; }

    /// <summary>
    /// Number of other vertices in this hyperedge.
    /// </summary>
    public required int PeerCount { get; init; }
}

/// <summary>
/// Result of joining/leaving a hyperedge.
/// </summary>
public readonly record struct HyperedgeMembershipResult
{
    /// <summary>
    /// Whether operation succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Operation type (Join/Leave).
    /// </summary>
    public required string Operation { get; init; }

    /// <summary>
    /// Timestamp of operation (nanoseconds).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Number of vertices now in hyperedge.
    /// </summary>
    public required int CurrentMemberCount { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Message between vertices.
/// </summary>
public sealed class VertexMessage
{
    /// <summary>
    /// Unique message ID.
    /// </summary>
    public required string MessageId { get; init; }

    /// <summary>
    /// Source vertex ID.
    /// </summary>
    public required string SourceVertexId { get; init; }

    /// <summary>
    /// Target vertex ID (null for broadcast).
    /// </summary>
    public string? TargetVertexId { get; init; }

    /// <summary>
    /// Hyperedge through which message is sent.
    /// </summary>
    public string? ViaHyperedgeId { get; init; }

    /// <summary>
    /// Message type for routing.
    /// </summary>
    public required string MessageType { get; init; }

    /// <summary>
    /// Message payload.
    /// </summary>
    public required IReadOnlyDictionary<string, object> Payload { get; init; }

    /// <summary>
    /// HLC timestamp for causal ordering.
    /// </summary>
    public required long HlcTimestamp { get; init; }

    /// <summary>
    /// Time-to-live in hops.
    /// </summary>
    public int TtlHops { get; init; } = 3;
}

/// <summary>
/// Result of broadcasting to hyperedge.
/// </summary>
public readonly record struct BroadcastResult
{
    /// <summary>
    /// Whether broadcast was sent.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Number of vertices targeted.
    /// </summary>
    public required int TargetCount { get; init; }

    /// <summary>
    /// Number of successful deliveries.
    /// </summary>
    public required int DeliveredCount { get; init; }

    /// <summary>
    /// Broadcast latency (nanoseconds).
    /// </summary>
    public required long LatencyNanos { get; init; }
}

/// <summary>
/// Result of processing an incoming message.
/// </summary>
public readonly record struct MessageResult
{
    /// <summary>
    /// Whether message was processed successfully.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Processing latency (nanoseconds).
    /// </summary>
    public required long ProcessingTimeNanos { get; init; }

    /// <summary>
    /// Response payload (if any).
    /// </summary>
    public IReadOnlyDictionary<string, object>? Response { get; init; }
}

/// <summary>
/// Filter for neighbor queries.
/// </summary>
public sealed class NeighborFilter
{
    /// <summary>
    /// Filter by vertex types.
    /// </summary>
    public IReadOnlyList<string>? VertexTypes { get; init; }

    /// <summary>
    /// Filter by hyperedge types.
    /// </summary>
    public IReadOnlyList<string>? HyperedgeTypes { get; init; }

    /// <summary>
    /// Property filter predicates.
    /// </summary>
    public IReadOnlyDictionary<string, object>? PropertyFilters { get; init; }

    /// <summary>
    /// Maximum results to return.
    /// </summary>
    public int MaxResults { get; init; } = 100;
}

/// <summary>
/// Result of neighbor query.
/// </summary>
public sealed class NeighborQueryResult
{
    /// <summary>
    /// Source vertex ID.
    /// </summary>
    public required string SourceVertexId { get; init; }

    /// <summary>
    /// Found neighbors.
    /// </summary>
    public required IReadOnlyList<NeighborInfo> Neighbors { get; init; }

    /// <summary>
    /// Query execution time (nanoseconds).
    /// </summary>
    public required long QueryTimeNanos { get; init; }

    /// <summary>
    /// Whether results were truncated.
    /// </summary>
    public required bool IsTruncated { get; init; }
}

/// <summary>
/// Information about a neighbor vertex.
/// </summary>
public sealed class NeighborInfo
{
    /// <summary>
    /// Neighbor vertex ID.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// Neighbor vertex type.
    /// </summary>
    public required string VertexType { get; init; }

    /// <summary>
    /// Distance in hops.
    /// </summary>
    public required int Distance { get; init; }

    /// <summary>
    /// Hyperedges connecting to this neighbor.
    /// </summary>
    public required IReadOnlyList<string> ViaHyperedges { get; init; }

    /// <summary>
    /// Selected properties.
    /// </summary>
    public IReadOnlyDictionary<string, object>? Properties { get; init; }
}

/// <summary>
/// Pattern for hypergraph pattern matching.
/// </summary>
public sealed class HypergraphPattern
{
    /// <summary>
    /// Pattern identifier.
    /// </summary>
    public required string PatternId { get; init; }

    /// <summary>
    /// Vertex constraints in the pattern.
    /// </summary>
    public required IReadOnlyList<PatternVertexConstraint> VertexConstraints { get; init; }

    /// <summary>
    /// Hyperedge constraints in the pattern.
    /// </summary>
    public required IReadOnlyList<PatternHyperedgeConstraint> HyperedgeConstraints { get; init; }

    /// <summary>
    /// Maximum matches to return.
    /// </summary>
    public int MaxMatches { get; init; } = 100;

    /// <summary>
    /// Timeout for pattern matching.
    /// </summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);
}

/// <summary>
/// Constraint on a vertex in a pattern.
/// </summary>
public sealed class PatternVertexConstraint
{
    /// <summary>
    /// Variable name for this vertex in pattern.
    /// </summary>
    public required string VariableName { get; init; }

    /// <summary>
    /// Required vertex types (any of).
    /// </summary>
    public IReadOnlyList<string>? VertexTypes { get; init; }

    /// <summary>
    /// Property constraints.
    /// </summary>
    public IReadOnlyDictionary<string, object>? PropertyConstraints { get; init; }
}

/// <summary>
/// Constraint on a hyperedge in a pattern.
/// </summary>
public sealed class PatternHyperedgeConstraint
{
    /// <summary>
    /// Variable name for this hyperedge in pattern.
    /// </summary>
    public required string VariableName { get; init; }

    /// <summary>
    /// Required hyperedge types (any of).
    /// </summary>
    public IReadOnlyList<string>? HyperedgeTypes { get; init; }

    /// <summary>
    /// Vertex variables that must be in this hyperedge.
    /// </summary>
    public required IReadOnlyList<string> ContainedVertices { get; init; }

    /// <summary>
    /// Minimum cardinality.
    /// </summary>
    public int MinCardinality { get; init; } = 2;

    /// <summary>
    /// Maximum cardinality (0 = unbounded).
    /// </summary>
    public int MaxCardinality { get; init; } = 0;
}

/// <summary>
/// Result of pattern matching.
/// </summary>
public sealed class PatternMatchResult
{
    /// <summary>
    /// Pattern that was matched.
    /// </summary>
    public required string PatternId { get; init; }

    /// <summary>
    /// Matches found.
    /// </summary>
    public required IReadOnlyList<PatternMatch> Matches { get; init; }

    /// <summary>
    /// Matching execution time (nanoseconds).
    /// </summary>
    public required long ExecutionTimeNanos { get; init; }

    /// <summary>
    /// Whether matches were truncated.
    /// </summary>
    public required bool IsTruncated { get; init; }

    /// <summary>
    /// Total matches found (before truncation).
    /// </summary>
    public required int TotalMatchCount { get; init; }
}

/// <summary>
/// Single pattern match.
/// </summary>
public sealed class PatternMatch
{
    /// <summary>
    /// Vertex bindings (variable name → vertex ID).
    /// </summary>
    public required IReadOnlyDictionary<string, string> VertexBindings { get; init; }

    /// <summary>
    /// Hyperedge bindings (variable name → hyperedge ID).
    /// </summary>
    public required IReadOnlyDictionary<string, string> HyperedgeBindings { get; init; }

    /// <summary>
    /// Match score (for ranked results).
    /// </summary>
    public double Score { get; init; } = 1.0;
}

/// <summary>
/// Aggregation specification for vertex computations.
/// </summary>
public sealed class VertexAggregation
{
    /// <summary>
    /// Aggregation type.
    /// </summary>
    public required AggregationType Type { get; init; }

    /// <summary>
    /// Property to aggregate.
    /// </summary>
    public required string PropertyName { get; init; }

    /// <summary>
    /// Scope of aggregation.
    /// </summary>
    public AggregationScope Scope { get; init; } = AggregationScope.DirectNeighbors;

    /// <summary>
    /// Maximum hops for multi-hop aggregation.
    /// </summary>
    public int MaxHops { get; init; } = 1;

    /// <summary>
    /// Filter for vertices to include.
    /// </summary>
    public NeighborFilter? Filter { get; init; }
}

/// <summary>
/// Result of aggregation.
/// </summary>
public readonly record struct AggregationResult
{
    /// <summary>
    /// Aggregation type performed.
    /// </summary>
    public required AggregationType Type { get; init; }

    /// <summary>
    /// Result value.
    /// </summary>
    public required double Value { get; init; }

    /// <summary>
    /// Number of vertices included.
    /// </summary>
    public required int VertexCount { get; init; }

    /// <summary>
    /// Execution time (nanoseconds).
    /// </summary>
    public required long ExecutionTimeNanos { get; init; }
}

/// <summary>
/// Types of aggregation.
/// </summary>
public enum AggregationType
{
    /// <summary>
    /// Sum of values.
    /// </summary>
    Sum = 0,

    /// <summary>
    /// Average of values.
    /// </summary>
    Average = 1,

    /// <summary>
    /// Minimum value.
    /// </summary>
    Min = 2,

    /// <summary>
    /// Maximum value.
    /// </summary>
    Max = 3,

    /// <summary>
    /// Count of matching vertices.
    /// </summary>
    Count = 4,

    /// <summary>
    /// Standard deviation.
    /// </summary>
    StdDev = 5
}

/// <summary>
/// Scope of aggregation.
/// </summary>
public enum AggregationScope
{
    /// <summary>
    /// Only direct neighbors (1 hop).
    /// </summary>
    DirectNeighbors = 0,

    /// <summary>
    /// Same hyperedge members.
    /// </summary>
    SameHyperedge = 1,

    /// <summary>
    /// Multi-hop traversal.
    /// </summary>
    MultiHop = 2
}

/// <summary>
/// Performance metrics for a vertex.
/// </summary>
public readonly record struct VertexMetrics
{
    /// <summary>
    /// Vertex ID.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// Messages received count.
    /// </summary>
    public required long MessagesReceived { get; init; }

    /// <summary>
    /// Messages sent count.
    /// </summary>
    public required long MessagesSent { get; init; }

    /// <summary>
    /// Average message processing time (nanoseconds).
    /// </summary>
    public required double AvgProcessingTimeNanos { get; init; }

    /// <summary>
    /// Number of hyperedges.
    /// </summary>
    public required int HyperedgeCount { get; init; }

    /// <summary>
    /// Pattern matches performed.
    /// </summary>
    public required long PatternMatchCount { get; init; }

    /// <summary>
    /// GPU memory used (bytes).
    /// </summary>
    public required long GpuMemoryBytes { get; init; }
}
