// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans;

namespace Orleans.GpuBridge.Abstractions.Hypergraph;

/// <summary>
/// GPU-native hyperedge actor interface for multi-way relationship management.
/// </summary>
/// <remarks>
/// <para>
/// Hyperedges connect multiple vertices in a single relationship, enabling
/// modeling of complex multi-way interactions that cannot be represented
/// with standard binary edges.
/// </para>
/// <para>
/// <strong>GPU-Native Features:</strong>
/// <list type="bullet">
/// <item><description>Efficient incidence matrix storage in GPU memory</description></item>
/// <item><description>Parallel broadcast to all member vertices</description></item>
/// <item><description>GPU-accelerated cardinality operations</description></item>
/// <item><description>Temporal ordering for event sequences</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Examples of Hyperedges:</strong>
/// <list type="bullet">
/// <item><description>A meeting (connects all attendees)</description></item>
/// <item><description>A transaction (buyer, seller, item, payment)</description></item>
/// <item><description>A chemical reaction (all reactants and products)</description></item>
/// <item><description>A database join (all tables involved)</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IHypergraphHyperedge : IGrainWithStringKey
{
    /// <summary>
    /// Initializes the hyperedge with properties and initial members.
    /// </summary>
    /// <param name="request">Initialization request.</param>
    /// <returns>Result of initialization.</returns>
    Task<HyperedgeInitResult> InitializeAsync(HyperedgeInitRequest request);

    /// <summary>
    /// Gets current hyperedge state including all members.
    /// </summary>
    /// <returns>Current hyperedge state.</returns>
    Task<HyperedgeState> GetStateAsync();

    /// <summary>
    /// Adds a vertex to this hyperedge.
    /// </summary>
    /// <param name="vertexId">Vertex ID to add.</param>
    /// <param name="role">Role of vertex in hyperedge (optional).</param>
    /// <returns>Result of adding vertex.</returns>
    Task<HyperedgeMutationResult> AddVertexAsync(string vertexId, string? role = null);

    /// <summary>
    /// Removes a vertex from this hyperedge.
    /// </summary>
    /// <param name="vertexId">Vertex ID to remove.</param>
    /// <returns>Result of removing vertex.</returns>
    Task<HyperedgeMutationResult> RemoveVertexAsync(string vertexId);

    /// <summary>
    /// Gets all vertices in this hyperedge.
    /// </summary>
    /// <returns>List of member vertices with their roles.</returns>
    Task<IReadOnlyList<HyperedgeMember>> GetMembersAsync();

    /// <summary>
    /// Updates hyperedge properties.
    /// </summary>
    /// <param name="properties">Properties to update.</param>
    /// <returns>Update result.</returns>
    Task<HyperedgeUpdateResult> UpdatePropertiesAsync(IReadOnlyDictionary<string, object> properties);

    /// <summary>
    /// Broadcasts a message to all member vertices.
    /// </summary>
    /// <param name="message">Message to broadcast.</param>
    /// <returns>Broadcast result with delivery statistics.</returns>
    Task<HyperedgeBroadcastResult> BroadcastAsync(HyperedgeMessage message);

    /// <summary>
    /// Executes an aggregation across all member vertices.
    /// </summary>
    /// <param name="aggregation">Aggregation specification.</param>
    /// <returns>Aggregation result.</returns>
    Task<HyperedgeAggregationResult> AggregateAsync(HyperedgeAggregation aggregation);

    /// <summary>
    /// Checks if this hyperedge satisfies a constraint.
    /// </summary>
    /// <param name="constraint">Constraint to check.</param>
    /// <returns>Whether constraint is satisfied.</returns>
    Task<bool> SatisfiesConstraintAsync(HyperedgeConstraint constraint);

    /// <summary>
    /// Merges this hyperedge with another (combines members).
    /// </summary>
    /// <param name="otherHyperedgeId">ID of hyperedge to merge with.</param>
    /// <returns>Merge result.</returns>
    Task<HyperedgeMergeResult> MergeWithAsync(string otherHyperedgeId);

    /// <summary>
    /// Splits this hyperedge into two based on predicate.
    /// </summary>
    /// <param name="splitPredicate">Predicate to determine split.</param>
    /// <returns>Split result with new hyperedge ID.</returns>
    Task<HyperedgeSplitResult> SplitAsync(HyperedgeSplitPredicate splitPredicate);

    /// <summary>
    /// Gets temporal history of hyperedge changes.
    /// </summary>
    /// <param name="since">Start timestamp (nanoseconds).</param>
    /// <param name="maxEvents">Maximum events to return.</param>
    /// <returns>History of changes.</returns>
    Task<HyperedgeHistory> GetHistoryAsync(long? since = null, int maxEvents = 100);

    /// <summary>
    /// Gets performance metrics for this hyperedge.
    /// </summary>
    /// <returns>Hyperedge performance metrics.</returns>
    Task<HyperedgeMetrics> GetMetricsAsync();
}

/// <summary>
/// Request to initialize a hyperedge.
/// </summary>
public sealed class HyperedgeInitRequest
{
    /// <summary>
    /// Hyperedge type/label.
    /// </summary>
    public required string HyperedgeType { get; init; }

    /// <summary>
    /// Initial properties.
    /// </summary>
    public required IReadOnlyDictionary<string, object> Properties { get; init; }

    /// <summary>
    /// Initial member vertices.
    /// </summary>
    public required IReadOnlyList<HyperedgeMemberInit> InitialMembers { get; init; }

    /// <summary>
    /// Minimum cardinality (vertices required).
    /// </summary>
    public int MinCardinality { get; init; } = 2;

    /// <summary>
    /// Maximum cardinality (0 = unlimited).
    /// </summary>
    public int MaxCardinality { get; init; } = 0;

    /// <summary>
    /// Whether hyperedge is directed (has source/target roles).
    /// </summary>
    public bool IsDirected { get; init; } = false;

    /// <summary>
    /// GPU affinity group for colocation with members.
    /// </summary>
    public string? AffinityGroup { get; init; }
}

/// <summary>
/// Initial member specification.
/// </summary>
public sealed class HyperedgeMemberInit
{
    /// <summary>
    /// Vertex ID.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// Role in hyperedge.
    /// </summary>
    public string? Role { get; init; }

    /// <summary>
    /// Position in ordered hyperedge (optional).
    /// </summary>
    public int? Position { get; init; }
}

/// <summary>
/// Result of hyperedge initialization.
/// </summary>
public readonly record struct HyperedgeInitResult
{
    /// <summary>
    /// Whether initialization succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Initial version number.
    /// </summary>
    public required long Version { get; init; }

    /// <summary>
    /// Number of initial members.
    /// </summary>
    public required int MemberCount { get; init; }

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
/// Current state of a hyperedge.
/// </summary>
public sealed class HyperedgeState
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Hyperedge type/label.
    /// </summary>
    public required string HyperedgeType { get; init; }

    /// <summary>
    /// Current version number.
    /// </summary>
    public required long Version { get; init; }

    /// <summary>
    /// Current properties.
    /// </summary>
    public required IReadOnlyDictionary<string, object> Properties { get; init; }

    /// <summary>
    /// Current members.
    /// </summary>
    public required IReadOnlyList<HyperedgeMember> Members { get; init; }

    /// <summary>
    /// Cardinality (number of members).
    /// </summary>
    public int Cardinality => Members.Count;

    /// <summary>
    /// Minimum cardinality constraint.
    /// </summary>
    public required int MinCardinality { get; init; }

    /// <summary>
    /// Maximum cardinality constraint (0 = unlimited).
    /// </summary>
    public required int MaxCardinality { get; init; }

    /// <summary>
    /// Whether hyperedge is directed.
    /// </summary>
    public required bool IsDirected { get; init; }

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
/// Member of a hyperedge.
/// </summary>
public sealed class HyperedgeMember
{
    /// <summary>
    /// Vertex ID.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// Role in hyperedge.
    /// </summary>
    public string? Role { get; init; }

    /// <summary>
    /// Position in ordered hyperedge.
    /// </summary>
    public int? Position { get; init; }

    /// <summary>
    /// When vertex joined (nanoseconds).
    /// </summary>
    public required long JoinedAtNanos { get; init; }

    /// <summary>
    /// Custom properties for this membership.
    /// </summary>
    public IReadOnlyDictionary<string, object>? MembershipProperties { get; init; }
}

/// <summary>
/// Result of adding/removing a vertex.
/// </summary>
public readonly record struct HyperedgeMutationResult
{
    /// <summary>
    /// Whether operation succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Operation performed.
    /// </summary>
    public required string Operation { get; init; }

    /// <summary>
    /// Vertex ID affected.
    /// </summary>
    public required string VertexId { get; init; }

    /// <summary>
    /// New version number.
    /// </summary>
    public required long NewVersion { get; init; }

    /// <summary>
    /// Current cardinality.
    /// </summary>
    public required int CurrentCardinality { get; init; }

    /// <summary>
    /// Timestamp of operation (nanoseconds).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Result of property update.
/// </summary>
public readonly record struct HyperedgeUpdateResult
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
    /// Properties changed.
    /// </summary>
    public required IReadOnlyList<string> ChangedProperties { get; init; }

    /// <summary>
    /// Timestamp of update (nanoseconds).
    /// </summary>
    public required long TimestampNanos { get; init; }
}

/// <summary>
/// Message for broadcast to hyperedge members.
/// </summary>
public sealed class HyperedgeMessage
{
    /// <summary>
    /// Unique message ID.
    /// </summary>
    public required string MessageId { get; init; }

    /// <summary>
    /// Message type for routing.
    /// </summary>
    public required string MessageType { get; init; }

    /// <summary>
    /// Source vertex (null if from hyperedge itself).
    /// </summary>
    public string? SourceVertexId { get; init; }

    /// <summary>
    /// Message payload.
    /// </summary>
    public required IReadOnlyDictionary<string, object> Payload { get; init; }

    /// <summary>
    /// HLC timestamp.
    /// </summary>
    public required long HlcTimestamp { get; init; }

    /// <summary>
    /// Roles to include in broadcast (null = all).
    /// </summary>
    public IReadOnlyList<string>? TargetRoles { get; init; }

    /// <summary>
    /// Vertices to exclude from broadcast.
    /// </summary>
    public IReadOnlyList<string>? ExcludeVertices { get; init; }
}

/// <summary>
/// Result of broadcast operation.
/// </summary>
public readonly record struct HyperedgeBroadcastResult
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
    /// Number of failed deliveries.
    /// </summary>
    public required int FailedCount { get; init; }

    /// <summary>
    /// Total broadcast latency (nanoseconds).
    /// </summary>
    public required long TotalLatencyNanos { get; init; }

    /// <summary>
    /// Average per-vertex latency (nanoseconds).
    /// </summary>
    public double AvgLatencyNanos => TargetCount > 0 ? (double)TotalLatencyNanos / TargetCount : 0;
}

/// <summary>
/// Aggregation specification for hyperedge.
/// </summary>
public sealed class HyperedgeAggregation
{
    /// <summary>
    /// Aggregation type.
    /// </summary>
    public required AggregationType Type { get; init; }

    /// <summary>
    /// Property to aggregate from member vertices.
    /// </summary>
    public required string PropertyName { get; init; }

    /// <summary>
    /// Roles to include (null = all).
    /// </summary>
    public IReadOnlyList<string>? IncludeRoles { get; init; }

    /// <summary>
    /// Whether to include properties from hyperedge itself.
    /// </summary>
    public bool IncludeHyperedgeProperties { get; init; } = false;
}

/// <summary>
/// Result of hyperedge aggregation.
/// </summary>
public readonly record struct HyperedgeAggregationResult
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
/// Constraint for hyperedge validation.
/// </summary>
public sealed class HyperedgeConstraint
{
    /// <summary>
    /// Constraint type.
    /// </summary>
    public required HyperedgeConstraintType Type { get; init; }

    /// <summary>
    /// Required minimum cardinality.
    /// </summary>
    public int? MinCardinality { get; init; }

    /// <summary>
    /// Required maximum cardinality.
    /// </summary>
    public int? MaxCardinality { get; init; }

    /// <summary>
    /// Required roles (all must be present).
    /// </summary>
    public IReadOnlyList<string>? RequiredRoles { get; init; }

    /// <summary>
    /// Property constraints.
    /// </summary>
    public IReadOnlyDictionary<string, object>? PropertyConstraints { get; init; }
}

/// <summary>
/// Types of hyperedge constraints.
/// </summary>
public enum HyperedgeConstraintType
{
    /// <summary>
    /// Cardinality constraint.
    /// </summary>
    Cardinality = 0,

    /// <summary>
    /// Role presence constraint.
    /// </summary>
    RolePresence = 1,

    /// <summary>
    /// Property value constraint.
    /// </summary>
    PropertyValue = 2,

    /// <summary>
    /// Unique vertex constraint.
    /// </summary>
    UniqueVertices = 3
}

/// <summary>
/// Result of merging two hyperedges.
/// </summary>
public readonly record struct HyperedgeMergeResult
{
    /// <summary>
    /// Whether merge succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// New cardinality after merge.
    /// </summary>
    public required int NewCardinality { get; init; }

    /// <summary>
    /// Number of new members added.
    /// </summary>
    public required int MembersAdded { get; init; }

    /// <summary>
    /// New version number.
    /// </summary>
    public required long NewVersion { get; init; }

    /// <summary>
    /// Timestamp of merge (nanoseconds).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Predicate for splitting a hyperedge.
/// </summary>
public sealed class HyperedgeSplitPredicate
{
    /// <summary>
    /// Split by role.
    /// </summary>
    public IReadOnlyList<string>? SplitRoles { get; init; }

    /// <summary>
    /// Split by property value.
    /// </summary>
    public string? SplitPropertyName { get; init; }

    /// <summary>
    /// Split threshold for numeric properties.
    /// </summary>
    public double? SplitThreshold { get; init; }
}

/// <summary>
/// Result of splitting a hyperedge.
/// </summary>
public readonly record struct HyperedgeSplitResult
{
    /// <summary>
    /// Whether split succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// ID of new hyperedge created.
    /// </summary>
    public required string NewHyperedgeId { get; init; }

    /// <summary>
    /// Members moved to new hyperedge.
    /// </summary>
    public required int MembersMoved { get; init; }

    /// <summary>
    /// Members remaining in original.
    /// </summary>
    public required int MembersRemaining { get; init; }

    /// <summary>
    /// Timestamp of split (nanoseconds).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// History of hyperedge changes.
/// </summary>
public sealed class HyperedgeHistory
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// History events.
    /// </summary>
    public required IReadOnlyList<HyperedgeHistoryEvent> Events { get; init; }

    /// <summary>
    /// Whether history is complete or truncated.
    /// </summary>
    public required bool IsComplete { get; init; }
}

/// <summary>
/// Single event in hyperedge history.
/// </summary>
public sealed class HyperedgeHistoryEvent
{
    /// <summary>
    /// Event type.
    /// </summary>
    public required HyperedgeEventType EventType { get; init; }

    /// <summary>
    /// Timestamp of event (nanoseconds).
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// HLC timestamp.
    /// </summary>
    public required long HlcTimestamp { get; init; }

    /// <summary>
    /// Version after event.
    /// </summary>
    public required long Version { get; init; }

    /// <summary>
    /// Vertex ID involved (for membership events).
    /// </summary>
    public string? VertexId { get; init; }

    /// <summary>
    /// Event details.
    /// </summary>
    public IReadOnlyDictionary<string, object>? Details { get; init; }
}

/// <summary>
/// Types of hyperedge events.
/// </summary>
public enum HyperedgeEventType
{
    /// <summary>
    /// Hyperedge created.
    /// </summary>
    Created = 0,

    /// <summary>
    /// Vertex added.
    /// </summary>
    VertexAdded = 1,

    /// <summary>
    /// Vertex removed.
    /// </summary>
    VertexRemoved = 2,

    /// <summary>
    /// Properties updated.
    /// </summary>
    PropertiesUpdated = 3,

    /// <summary>
    /// Message broadcast.
    /// </summary>
    MessageBroadcast = 4,

    /// <summary>
    /// Merged with another hyperedge.
    /// </summary>
    Merged = 5,

    /// <summary>
    /// Split into two hyperedges.
    /// </summary>
    Split = 6
}

/// <summary>
/// Performance metrics for a hyperedge.
/// </summary>
public readonly record struct HyperedgeMetrics
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Current cardinality.
    /// </summary>
    public required int Cardinality { get; init; }

    /// <summary>
    /// Total messages broadcast.
    /// </summary>
    public required long MessagesBroadcast { get; init; }

    /// <summary>
    /// Total membership changes.
    /// </summary>
    public required long MembershipChanges { get; init; }

    /// <summary>
    /// Average broadcast latency (nanoseconds).
    /// </summary>
    public required double AvgBroadcastLatencyNanos { get; init; }

    /// <summary>
    /// GPU memory used (bytes).
    /// </summary>
    public required long GpuMemoryBytes { get; init; }

    /// <summary>
    /// Total aggregations performed.
    /// </summary>
    public required long AggregationsPerformed { get; init; }
}
