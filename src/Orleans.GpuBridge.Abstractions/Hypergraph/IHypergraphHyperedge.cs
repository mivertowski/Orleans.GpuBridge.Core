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
[GenerateSerializer]
public sealed class HyperedgeInitRequest
{
    /// <summary>
    /// Hyperedge type/label.
    /// </summary>
    [Id(0)]
    public required string HyperedgeType { get; init; }

    /// <summary>
    /// Initial properties.
    /// </summary>
    [Id(1)]
    public required IReadOnlyDictionary<string, object> Properties { get; init; }

    /// <summary>
    /// Initial member vertices.
    /// </summary>
    [Id(2)]
    public required IReadOnlyList<HyperedgeMemberInit> InitialMembers { get; init; }

    /// <summary>
    /// Minimum cardinality (vertices required).
    /// </summary>
    [Id(3)]
    public int MinCardinality { get; init; } = 2;

    /// <summary>
    /// Maximum cardinality (0 = unlimited).
    /// </summary>
    [Id(4)]
    public int MaxCardinality { get; init; } = 0;

    /// <summary>
    /// Whether hyperedge is directed (has source/target roles).
    /// </summary>
    [Id(5)]
    public bool IsDirected { get; init; } = false;

    /// <summary>
    /// GPU affinity group for colocation with members.
    /// </summary>
    [Id(6)]
    public string? AffinityGroup { get; init; }
}

/// <summary>
/// Initial member specification.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeMemberInit
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
    public string? Role { get; init; }

    /// <summary>
    /// Position in ordered hyperedge (optional).
    /// </summary>
    [Id(2)]
    public int? Position { get; init; }
}

/// <summary>
/// Result of hyperedge initialization.
/// </summary>
[GenerateSerializer]
public readonly record struct HyperedgeInitResult
{
    /// <summary>
    /// Whether initialization succeeded.
    /// </summary>
    [Id(0)]
    public required bool Success { get; init; }

    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    [Id(1)]
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Initial version number.
    /// </summary>
    [Id(2)]
    public required long Version { get; init; }

    /// <summary>
    /// Number of initial members.
    /// </summary>
    [Id(3)]
    public required int MemberCount { get; init; }

    /// <summary>
    /// Timestamp of creation (nanoseconds).
    /// </summary>
    [Id(4)]
    public required long CreatedAtNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    [Id(5)]
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Current state of a hyperedge.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeState
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    [Id(0)]
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Hyperedge type/label.
    /// </summary>
    [Id(1)]
    public required string HyperedgeType { get; init; }

    /// <summary>
    /// Current version number.
    /// </summary>
    [Id(2)]
    public required long Version { get; init; }

    /// <summary>
    /// Current properties.
    /// </summary>
    [Id(3)]
    public required IReadOnlyDictionary<string, object> Properties { get; init; }

    /// <summary>
    /// Current members.
    /// </summary>
    [Id(4)]
    public required IReadOnlyList<HyperedgeMember> Members { get; init; }

    /// <summary>
    /// Cardinality (number of members).
    /// </summary>
    public int Cardinality => Members.Count;

    /// <summary>
    /// Minimum cardinality constraint.
    /// </summary>
    [Id(5)]
    public required int MinCardinality { get; init; }

    /// <summary>
    /// Maximum cardinality constraint (0 = unlimited).
    /// </summary>
    [Id(6)]
    public required int MaxCardinality { get; init; }

    /// <summary>
    /// Whether hyperedge is directed.
    /// </summary>
    [Id(7)]
    public required bool IsDirected { get; init; }

    /// <summary>
    /// Created timestamp (nanoseconds).
    /// </summary>
    [Id(8)]
    public required long CreatedAtNanos { get; init; }

    /// <summary>
    /// Last modified timestamp (nanoseconds).
    /// </summary>
    [Id(9)]
    public required long ModifiedAtNanos { get; init; }

    /// <summary>
    /// HLC timestamp for causal ordering.
    /// </summary>
    [Id(10)]
    public required long HlcTimestamp { get; init; }
}

/// <summary>
/// Member of a hyperedge.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeMember
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
    public string? Role { get; init; }

    /// <summary>
    /// Position in ordered hyperedge.
    /// </summary>
    [Id(2)]
    public int? Position { get; init; }

    /// <summary>
    /// When vertex joined (nanoseconds).
    /// </summary>
    [Id(3)]
    public required long JoinedAtNanos { get; init; }

    /// <summary>
    /// Custom properties for this membership.
    /// </summary>
    [Id(4)]
    public IReadOnlyDictionary<string, object>? MembershipProperties { get; init; }
}

/// <summary>
/// Result of adding/removing a vertex.
/// </summary>
[GenerateSerializer]
public readonly record struct HyperedgeMutationResult
{
    /// <summary>
    /// Whether operation succeeded.
    /// </summary>
    [Id(0)]
    public required bool Success { get; init; }

    /// <summary>
    /// Operation performed.
    /// </summary>
    [Id(1)]
    public required string Operation { get; init; }

    /// <summary>
    /// Vertex ID affected.
    /// </summary>
    [Id(2)]
    public required string VertexId { get; init; }

    /// <summary>
    /// New version number.
    /// </summary>
    [Id(3)]
    public required long NewVersion { get; init; }

    /// <summary>
    /// Current cardinality.
    /// </summary>
    [Id(4)]
    public required int CurrentCardinality { get; init; }

    /// <summary>
    /// Timestamp of operation (nanoseconds).
    /// </summary>
    [Id(5)]
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    [Id(6)]
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Result of property update.
/// </summary>
[GenerateSerializer]
public readonly record struct HyperedgeUpdateResult
{
    /// <summary>
    /// Whether update succeeded.
    /// </summary>
    [Id(0)]
    public required bool Success { get; init; }

    /// <summary>
    /// New version number.
    /// </summary>
    [Id(1)]
    public required long NewVersion { get; init; }

    /// <summary>
    /// Properties changed.
    /// </summary>
    [Id(2)]
    public required IReadOnlyList<string> ChangedProperties { get; init; }

    /// <summary>
    /// Timestamp of update (nanoseconds).
    /// </summary>
    [Id(3)]
    public required long TimestampNanos { get; init; }
}

/// <summary>
/// Message for broadcast to hyperedge members.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeMessage
{
    /// <summary>
    /// Unique message ID.
    /// </summary>
    [Id(0)]
    public required string MessageId { get; init; }

    /// <summary>
    /// Message type for routing.
    /// </summary>
    [Id(1)]
    public required string MessageType { get; init; }

    /// <summary>
    /// Source vertex (null if from hyperedge itself).
    /// </summary>
    [Id(2)]
    public string? SourceVertexId { get; init; }

    /// <summary>
    /// Message payload.
    /// </summary>
    [Id(3)]
    public required IReadOnlyDictionary<string, object> Payload { get; init; }

    /// <summary>
    /// HLC timestamp.
    /// </summary>
    [Id(4)]
    public required long HlcTimestamp { get; init; }

    /// <summary>
    /// Roles to include in broadcast (null = all).
    /// </summary>
    [Id(5)]
    public IReadOnlyList<string>? TargetRoles { get; init; }

    /// <summary>
    /// Vertices to exclude from broadcast.
    /// </summary>
    [Id(6)]
    public IReadOnlyList<string>? ExcludeVertices { get; init; }
}

/// <summary>
/// Result of broadcast operation.
/// </summary>
[GenerateSerializer]
public readonly record struct HyperedgeBroadcastResult
{
    /// <summary>
    /// Whether broadcast was sent.
    /// </summary>
    [Id(0)]
    public required bool Success { get; init; }

    /// <summary>
    /// Number of vertices targeted.
    /// </summary>
    [Id(1)]
    public required int TargetCount { get; init; }

    /// <summary>
    /// Number of successful deliveries.
    /// </summary>
    [Id(2)]
    public required int DeliveredCount { get; init; }

    /// <summary>
    /// Number of failed deliveries.
    /// </summary>
    [Id(3)]
    public required int FailedCount { get; init; }

    /// <summary>
    /// Total broadcast latency (nanoseconds).
    /// </summary>
    [Id(4)]
    public required long TotalLatencyNanos { get; init; }

    /// <summary>
    /// Average per-vertex latency (nanoseconds).
    /// </summary>
    public double AvgLatencyNanos => TargetCount > 0 ? (double)TotalLatencyNanos / TargetCount : 0;
}

/// <summary>
/// Aggregation specification for hyperedge.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeAggregation
{
    /// <summary>
    /// Aggregation type.
    /// </summary>
    [Id(0)]
    public required AggregationType Type { get; init; }

    /// <summary>
    /// Property to aggregate from member vertices.
    /// </summary>
    [Id(1)]
    public required string PropertyName { get; init; }

    /// <summary>
    /// Roles to include (null = all).
    /// </summary>
    [Id(2)]
    public IReadOnlyList<string>? IncludeRoles { get; init; }

    /// <summary>
    /// Whether to include properties from hyperedge itself.
    /// </summary>
    [Id(3)]
    public bool IncludeHyperedgeProperties { get; init; } = false;
}

/// <summary>
/// Result of hyperedge aggregation.
/// </summary>
[GenerateSerializer]
public readonly record struct HyperedgeAggregationResult
{
    /// <summary>
    /// Aggregation type performed.
    /// </summary>
    [Id(0)]
    public required AggregationType Type { get; init; }

    /// <summary>
    /// Result value.
    /// </summary>
    [Id(1)]
    public required double Value { get; init; }

    /// <summary>
    /// Number of vertices included.
    /// </summary>
    [Id(2)]
    public required int VertexCount { get; init; }

    /// <summary>
    /// Execution time (nanoseconds).
    /// </summary>
    [Id(3)]
    public required long ExecutionTimeNanos { get; init; }
}

/// <summary>
/// Constraint for hyperedge validation.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeConstraint
{
    /// <summary>
    /// Constraint type.
    /// </summary>
    [Id(0)]
    public required HyperedgeConstraintType Type { get; init; }

    /// <summary>
    /// Required minimum cardinality.
    /// </summary>
    [Id(1)]
    public int? MinCardinality { get; init; }

    /// <summary>
    /// Required maximum cardinality.
    /// </summary>
    [Id(2)]
    public int? MaxCardinality { get; init; }

    /// <summary>
    /// Required roles (all must be present).
    /// </summary>
    [Id(3)]
    public IReadOnlyList<string>? RequiredRoles { get; init; }

    /// <summary>
    /// Property constraints.
    /// </summary>
    [Id(4)]
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
[GenerateSerializer]
public readonly record struct HyperedgeMergeResult
{
    /// <summary>
    /// Whether merge succeeded.
    /// </summary>
    [Id(0)]
    public required bool Success { get; init; }

    /// <summary>
    /// New cardinality after merge.
    /// </summary>
    [Id(1)]
    public required int NewCardinality { get; init; }

    /// <summary>
    /// Number of new members added.
    /// </summary>
    [Id(2)]
    public required int MembersAdded { get; init; }

    /// <summary>
    /// New version number.
    /// </summary>
    [Id(3)]
    public required long NewVersion { get; init; }

    /// <summary>
    /// Timestamp of merge (nanoseconds).
    /// </summary>
    [Id(4)]
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    [Id(5)]
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Predicate for splitting a hyperedge.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeSplitPredicate
{
    /// <summary>
    /// Split by role.
    /// </summary>
    [Id(0)]
    public IReadOnlyList<string>? SplitRoles { get; init; }

    /// <summary>
    /// Split by property value.
    /// </summary>
    [Id(1)]
    public string? SplitPropertyName { get; init; }

    /// <summary>
    /// Split threshold for numeric properties.
    /// </summary>
    [Id(2)]
    public double? SplitThreshold { get; init; }
}

/// <summary>
/// Result of splitting a hyperedge.
/// </summary>
[GenerateSerializer]
public readonly record struct HyperedgeSplitResult
{
    /// <summary>
    /// Whether split succeeded.
    /// </summary>
    [Id(0)]
    public required bool Success { get; init; }

    /// <summary>
    /// ID of new hyperedge created.
    /// </summary>
    [Id(1)]
    public required string NewHyperedgeId { get; init; }

    /// <summary>
    /// Members moved to new hyperedge.
    /// </summary>
    [Id(2)]
    public required int MembersMoved { get; init; }

    /// <summary>
    /// Members remaining in original.
    /// </summary>
    [Id(3)]
    public required int MembersRemaining { get; init; }

    /// <summary>
    /// Timestamp of split (nanoseconds).
    /// </summary>
    [Id(4)]
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    [Id(5)]
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// History of hyperedge changes.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeHistory
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    [Id(0)]
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// History events.
    /// </summary>
    [Id(1)]
    public required IReadOnlyList<HyperedgeHistoryEvent> Events { get; init; }

    /// <summary>
    /// Whether history is complete or truncated.
    /// </summary>
    [Id(2)]
    public required bool IsComplete { get; init; }
}

/// <summary>
/// Single event in hyperedge history.
/// </summary>
[GenerateSerializer]
public sealed class HyperedgeHistoryEvent
{
    /// <summary>
    /// Event type.
    /// </summary>
    [Id(0)]
    public required HyperedgeEventType EventType { get; init; }

    /// <summary>
    /// Timestamp of event (nanoseconds).
    /// </summary>
    [Id(1)]
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// HLC timestamp.
    /// </summary>
    [Id(2)]
    public required long HlcTimestamp { get; init; }

    /// <summary>
    /// Version after event.
    /// </summary>
    [Id(3)]
    public required long Version { get; init; }

    /// <summary>
    /// Vertex ID involved (for membership events).
    /// </summary>
    [Id(4)]
    public string? VertexId { get; init; }

    /// <summary>
    /// Event details.
    /// </summary>
    [Id(5)]
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
[GenerateSerializer]
public readonly record struct HyperedgeMetrics
{
    /// <summary>
    /// Hyperedge ID.
    /// </summary>
    [Id(0)]
    public required string HyperedgeId { get; init; }

    /// <summary>
    /// Current cardinality.
    /// </summary>
    [Id(1)]
    public required int Cardinality { get; init; }

    /// <summary>
    /// Total messages broadcast.
    /// </summary>
    [Id(2)]
    public required long MessagesBroadcast { get; init; }

    /// <summary>
    /// Total membership changes.
    /// </summary>
    [Id(3)]
    public required long MembershipChanges { get; init; }

    /// <summary>
    /// Average broadcast latency (nanoseconds).
    /// </summary>
    [Id(4)]
    public required double AvgBroadcastLatencyNanos { get; init; }

    /// <summary>
    /// GPU memory used (bytes).
    /// </summary>
    [Id(5)]
    public required long GpuMemoryBytes { get; init; }

    /// <summary>
    /// Total aggregations performed.
    /// </summary>
    [Id(6)]
    public required long AggregationsPerformed { get; init; }
}
