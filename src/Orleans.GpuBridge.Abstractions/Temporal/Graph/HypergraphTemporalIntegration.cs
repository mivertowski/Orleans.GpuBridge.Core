// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans;

namespace Orleans.GpuBridge.Abstractions.Temporal.Graph;

/// <summary>
/// Represents a temporal hyperedge - a many-to-many relationship with time validity.
/// </summary>
/// <remarks>
/// <para>
/// Unlike regular temporal edges (binary relationships), temporal hyperedges connect
/// multiple vertices simultaneously, enabling:
/// <list type="bullet">
/// <item><description>Group transactions (multiple parties involved)</description></item>
/// <item><description>Meeting attendance (many participants at once)</description></item>
/// <item><description>Multi-party contracts (effective during time window)</description></item>
/// </list>
/// </para>
/// </remarks>
[GenerateSerializer]
[Immutable]
public sealed record TemporalHyperedgeData
{
    /// <summary>
    /// Unique hyperedge identifier.
    /// </summary>
    [Id(0)]
    public string HyperedgeId { get; init; } = string.Empty;

    /// <summary>
    /// Vertex identifiers in this hyperedge.
    /// </summary>
    [Id(1)]
    public IReadOnlyList<ulong> VertexIds { get; init; } = Array.Empty<ulong>();

    /// <summary>
    /// Start of the time range when this hyperedge was valid (inclusive, nanoseconds).
    /// </summary>
    [Id(2)]
    public long ValidFrom { get; init; }

    /// <summary>
    /// End of the time range when this hyperedge was valid (inclusive, nanoseconds).
    /// </summary>
    [Id(3)]
    public long ValidTo { get; init; }

    /// <summary>
    /// Hybrid Logical Clock timestamp when the hyperedge was created.
    /// </summary>
    [Id(4)]
    public HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Hyperedge type identifier.
    /// </summary>
    [Id(5)]
    public string HyperedgeType { get; init; } = "default";

    /// <summary>
    /// Aggregate weight of the hyperedge.
    /// </summary>
    [Id(6)]
    public double Weight { get; init; } = 1.0;

    /// <summary>
    /// Additional hyperedge metadata.
    /// </summary>
    [Id(7)]
    public IReadOnlyDictionary<string, object>? Properties { get; init; }

    /// <summary>
    /// Role assignments for vertices (vertex ID → role).
    /// </summary>
    [Id(8)]
    public IReadOnlyDictionary<ulong, string>? VertexRoles { get; init; }

    /// <summary>
    /// Number of vertices in the hyperedge.
    /// </summary>
    public int Cardinality => VertexIds.Count;

    /// <summary>
    /// Duration of the hyperedge's validity in nanoseconds.
    /// </summary>
    public long DurationNanos => ValidTo - ValidFrom;

    /// <summary>
    /// Checks if this hyperedge was valid at the specified time.
    /// </summary>
    public bool IsValidAt(long timeNanos)
    {
        return timeNanos >= ValidFrom && timeNanos <= ValidTo;
    }

    /// <summary>
    /// Checks if this hyperedge overlaps with the specified time range.
    /// </summary>
    public bool OverlapsWith(long startTime, long endTime)
    {
        return ValidFrom <= endTime && ValidTo >= startTime;
    }

    /// <summary>
    /// Checks if this hyperedge contains a specific vertex.
    /// </summary>
    public bool ContainsVertex(ulong vertexId)
    {
        return VertexIds.Contains(vertexId);
    }
}

/// <summary>
/// Represents a temporal vertex state snapshot.
/// </summary>
[GenerateSerializer]
public sealed record TemporalVertexSnapshot
{
    /// <summary>
    /// Vertex identifier.
    /// </summary>
    [Id(0)]
    public ulong VertexId { get; init; }

    /// <summary>
    /// Vertex string identifier (for hypergraph grains).
    /// </summary>
    [Id(1)]
    public string? VertexStringId { get; init; }

    /// <summary>
    /// Snapshot timestamp (nanoseconds).
    /// </summary>
    [Id(2)]
    public long TimestampNanos { get; init; }

    /// <summary>
    /// HLC timestamp at snapshot time.
    /// </summary>
    [Id(3)]
    public HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Vertex type.
    /// </summary>
    [Id(4)]
    public string VertexType { get; init; } = string.Empty;

    /// <summary>
    /// Vertex properties at snapshot time.
    /// </summary>
    [Id(5)]
    public IReadOnlyDictionary<string, object> Properties { get; init; } = new Dictionary<string, object>();

    /// <summary>
    /// Hyperedge memberships at snapshot time.
    /// </summary>
    [Id(6)]
    public IReadOnlyList<string> HyperedgeIds { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Outgoing edges at snapshot time.
    /// </summary>
    [Id(7)]
    public IReadOnlyList<TemporalEdgeData> OutgoingEdges { get; init; } = Array.Empty<TemporalEdgeData>();

    /// <summary>
    /// Incoming edges at snapshot time.
    /// </summary>
    [Id(8)]
    public IReadOnlyList<TemporalEdgeData> IncomingEdges { get; init; } = Array.Empty<TemporalEdgeData>();
}

/// <summary>
/// Represents a temporal event in the hypergraph.
/// </summary>
[GenerateSerializer]
public sealed record TemporalHypergraphEvent
{
    /// <summary>
    /// Event type.
    /// </summary>
    [Id(0)]
    public HypergraphEventType EventType { get; init; }

    /// <summary>
    /// Event timestamp (nanoseconds).
    /// </summary>
    [Id(1)]
    public long TimestampNanos { get; init; }

    /// <summary>
    /// HLC timestamp for causal ordering.
    /// </summary>
    [Id(2)]
    public HybridTimestamp HLC { get; init; }

    /// <summary>
    /// Affected vertex ID (if applicable).
    /// </summary>
    [Id(3)]
    public ulong? VertexId { get; init; }

    /// <summary>
    /// Affected hyperedge ID (if applicable).
    /// </summary>
    [Id(4)]
    public string? HyperedgeId { get; init; }

    /// <summary>
    /// Edge data (for edge events).
    /// </summary>
    [Id(5)]
    public TemporalEdgeData? EdgeData { get; init; }

    /// <summary>
    /// Hyperedge data (for hyperedge events).
    /// </summary>
    [Id(6)]
    public TemporalHyperedgeData? HyperedgeData { get; init; }

    /// <summary>
    /// Event metadata.
    /// </summary>
    [Id(7)]
    public IReadOnlyDictionary<string, object>? Metadata { get; init; }
}

/// <summary>
/// Types of hypergraph events.
/// </summary>
public enum HypergraphEventType
{
    /// <summary>
    /// Vertex was created.
    /// </summary>
    VertexCreated,

    /// <summary>
    /// Vertex was deleted.
    /// </summary>
    VertexDeleted,

    /// <summary>
    /// Vertex properties were updated.
    /// </summary>
    VertexUpdated,

    /// <summary>
    /// Edge was created.
    /// </summary>
    EdgeCreated,

    /// <summary>
    /// Edge was deleted/expired.
    /// </summary>
    EdgeDeleted,

    /// <summary>
    /// Hyperedge was created.
    /// </summary>
    HyperedgeCreated,

    /// <summary>
    /// Hyperedge was deleted/expired.
    /// </summary>
    HyperedgeDeleted,

    /// <summary>
    /// Vertex joined a hyperedge.
    /// </summary>
    VertexJoinedHyperedge,

    /// <summary>
    /// Vertex left a hyperedge.
    /// </summary>
    VertexLeftHyperedge,

    /// <summary>
    /// Message was sent between vertices.
    /// </summary>
    MessageSent,

    /// <summary>
    /// Broadcast was sent through a hyperedge.
    /// </summary>
    BroadcastSent
}

/// <summary>
/// Query request for temporal hypergraph analysis.
/// </summary>
[GenerateSerializer]
public sealed record TemporalHypergraphQuery
{
    /// <summary>
    /// Query type.
    /// </summary>
    [Id(0)]
    public TemporalQueryType QueryType { get; init; }

    /// <summary>
    /// Start time for the query window (nanoseconds).
    /// </summary>
    [Id(1)]
    public long StartTimeNanos { get; init; }

    /// <summary>
    /// End time for the query window (nanoseconds).
    /// </summary>
    [Id(2)]
    public long EndTimeNanos { get; init; }

    /// <summary>
    /// Target vertex ID (for vertex-centric queries).
    /// </summary>
    [Id(3)]
    public ulong? TargetVertexId { get; init; }

    /// <summary>
    /// Target hyperedge ID (for hyperedge-centric queries).
    /// </summary>
    [Id(4)]
    public string? TargetHyperedgeId { get; init; }

    /// <summary>
    /// Maximum results to return.
    /// </summary>
    [Id(5)]
    public int MaxResults { get; init; } = 1000;

    /// <summary>
    /// Filter by event types.
    /// </summary>
    [Id(6)]
    public IReadOnlyList<HypergraphEventType>? EventTypeFilter { get; init; }

    /// <summary>
    /// Filter by vertex types.
    /// </summary>
    [Id(7)]
    public IReadOnlyList<string>? VertexTypeFilter { get; init; }

    /// <summary>
    /// Filter by hyperedge types.
    /// </summary>
    [Id(8)]
    public IReadOnlyList<string>? HyperedgeTypeFilter { get; init; }
}

/// <summary>
/// Types of temporal hypergraph queries.
/// </summary>
public enum TemporalQueryType
{
    /// <summary>
    /// Get vertex history over time.
    /// </summary>
    VertexHistory,

    /// <summary>
    /// Get hyperedge evolution over time.
    /// </summary>
    HyperedgeEvolution,

    /// <summary>
    /// Get graph snapshot at specific time.
    /// </summary>
    GraphSnapshot,

    /// <summary>
    /// Find causally related events.
    /// </summary>
    CausalChain,

    /// <summary>
    /// Find concurrent events (happened at same time).
    /// </summary>
    ConcurrentEvents,

    /// <summary>
    /// Get event stream in time range.
    /// </summary>
    EventStream,

    /// <summary>
    /// Detect temporal patterns.
    /// </summary>
    PatternDetection
}

/// <summary>
/// Result of a temporal hypergraph query.
/// </summary>
[GenerateSerializer]
public sealed record TemporalHypergraphQueryResult
{
    /// <summary>
    /// Query type that was executed.
    /// </summary>
    [Id(0)]
    public TemporalQueryType QueryType { get; init; }

    /// <summary>
    /// Events matching the query.
    /// </summary>
    [Id(1)]
    public IReadOnlyList<TemporalHypergraphEvent> Events { get; init; } = Array.Empty<TemporalHypergraphEvent>();

    /// <summary>
    /// Vertex snapshots (for snapshot queries).
    /// </summary>
    [Id(2)]
    public IReadOnlyList<TemporalVertexSnapshot> VertexSnapshots { get; init; } = Array.Empty<TemporalVertexSnapshot>();

    /// <summary>
    /// Whether results were truncated due to limits.
    /// </summary>
    [Id(3)]
    public bool IsTruncated { get; init; }

    /// <summary>
    /// Total matching items (may exceed returned count).
    /// </summary>
    [Id(4)]
    public int TotalCount { get; init; }

    /// <summary>
    /// Query execution time (nanoseconds).
    /// </summary>
    [Id(5)]
    public long QueryTimeNanos { get; init; }
}

/// <summary>
/// Interface for temporal hypergraph event storage.
/// </summary>
public interface ITemporalEventStore
{
    /// <summary>
    /// Records a temporal event.
    /// </summary>
    Task RecordEventAsync(TemporalHypergraphEvent evt, CancellationToken cancellationToken = default);

    /// <summary>
    /// Queries events in a time range.
    /// </summary>
    Task<IReadOnlyList<TemporalHypergraphEvent>> QueryEventsAsync(
        long startTimeNanos,
        long endTimeNanos,
        IReadOnlyList<HypergraphEventType>? eventTypeFilter = null,
        int maxResults = 1000,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets events for a specific vertex.
    /// </summary>
    Task<IReadOnlyList<TemporalHypergraphEvent>> GetVertexEventsAsync(
        ulong vertexId,
        long startTimeNanos,
        long endTimeNanos,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets events for a specific hyperedge.
    /// </summary>
    Task<IReadOnlyList<TemporalHypergraphEvent>> GetHyperedgeEventsAsync(
        string hyperedgeId,
        long startTimeNanos,
        long endTimeNanos,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Finds causally related events (using HLC ordering).
    /// </summary>
    Task<IReadOnlyList<TemporalHypergraphEvent>> FindCausalChainAsync(
        HybridTimestamp startHlc,
        int maxEvents = 100,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Orleans grain interface for temporal event storage.
/// </summary>
public interface ITemporalEventStoreGrain : IGrainWithStringKey
{
    /// <summary>
    /// Records an event.
    /// </summary>
    Task<bool> RecordEventAsync(TemporalHypergraphEvent evt);

    /// <summary>
    /// Queries events matching criteria.
    /// </summary>
    Task<TemporalHypergraphQueryResult> QueryAsync(TemporalHypergraphQuery query);

    /// <summary>
    /// Gets event count.
    /// </summary>
    Task<long> GetEventCountAsync();

    /// <summary>
    /// Clears all events.
    /// </summary>
    Task ClearAsync();
}
