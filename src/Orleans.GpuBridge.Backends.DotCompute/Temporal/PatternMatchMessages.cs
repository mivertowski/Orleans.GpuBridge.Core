// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Runtime.InteropServices;
using DotCompute.Abstractions.Messaging;
using MemoryPack;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Pattern matching operations supported by the pattern match kernel.
/// </summary>
public enum PatternMatchOperation
{
    /// <summary>Match vertices by property value</summary>
    MatchByProperty = 0,
    /// <summary>Match vertices by degree (neighbor count)</summary>
    MatchByDegree = 1,
    /// <summary>Match vertices connected to a specific vertex</summary>
    MatchNeighbors = 2,
    /// <summary>Match path pattern (source -> target via edge)</summary>
    MatchPath = 3,
    /// <summary>Match triangle pattern (3-clique)</summary>
    MatchTriangle = 4,
    /// <summary>Match star pattern (hub with N spokes)</summary>
    MatchStar = 5
}

/// <summary>
/// Request message for GPU pattern matching in hypergraphs.
/// </summary>
/// <remarks>
/// <para>
/// Uses fixed-size fields for CUDA compatibility:
/// - Vertices represented by integer IDs (0-based index)
/// - Adjacency encoded as fixed-size neighbor arrays
/// - Property values as float for comparison
/// </para>
/// <para>
/// For larger graphs, use GPU memory handles for adjacency data
/// and send only the handle reference in messages.
/// </para>
/// </remarks>
[MemoryPackable]
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public partial struct PatternMatchRingRequest : IRingKernelMessage
{
    // Message metadata
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }

    [MemoryPackIgnore]
    public readonly string MessageType => nameof(PatternMatchRingRequest);

    [MemoryPackIgnore]
    public readonly int PayloadSize => 256; // Approximate size

    // Pattern specification
    /// <summary>Operation type (0-5)</summary>
    public int OperationType { get; set; }

    /// <summary>Number of vertices in local subgraph</summary>
    public int VertexCount { get; set; }

    /// <summary>Number of edges in local subgraph</summary>
    public int EdgeCount { get; set; }

    /// <summary>Property key hash to match (for MatchByProperty)</summary>
    public int PropertyKeyHash { get; set; }

    /// <summary>Property value to match (float comparison)</summary>
    public float PropertyValue { get; set; }

    /// <summary>Property comparison operator: 0=equal, 1=less, 2=greater, 3=not equal</summary>
    public int ComparisonOp { get; set; }

    /// <summary>Target degree for MatchByDegree</summary>
    public int TargetDegree { get; set; }

    /// <summary>Source vertex ID for path/neighbor matching</summary>
    public int SourceVertexId { get; set; }

    /// <summary>Target vertex ID for path matching</summary>
    public int TargetVertexId { get; set; }

    /// <summary>Maximum search depth (for path patterns)</summary>
    public int MaxDepth { get; set; }

    // Fixed-size adjacency list for small graphs (8 vertices, 8 edges each)
    // For larger graphs, use GPU memory handles
    /// <summary>Vertex 0 neighbor count</summary>
    public int V0NeighborCount { get; set; }
    /// <summary>Vertex 0 neighbors</summary>
    public int V0N0 { get; set; }
    public int V0N1 { get; set; }
    public int V0N2 { get; set; }
    public int V0N3 { get; set; }
    public int V0N4 { get; set; }
    public int V0N5 { get; set; }
    public int V0N6 { get; set; }
    public int V0N7 { get; set; }

    /// <summary>Vertex 1 neighbor count</summary>
    public int V1NeighborCount { get; set; }
    public int V1N0 { get; set; }
    public int V1N1 { get; set; }
    public int V1N2 { get; set; }
    public int V1N3 { get; set; }
    public int V1N4 { get; set; }
    public int V1N5 { get; set; }
    public int V1N6 { get; set; }
    public int V1N7 { get; set; }

    /// <summary>Vertex 2 neighbor count</summary>
    public int V2NeighborCount { get; set; }
    public int V2N0 { get; set; }
    public int V2N1 { get; set; }
    public int V2N2 { get; set; }
    public int V2N3 { get; set; }
    public int V2N4 { get; set; }
    public int V2N5 { get; set; }
    public int V2N6 { get; set; }
    public int V2N7 { get; set; }

    /// <summary>Vertex 3 neighbor count</summary>
    public int V3NeighborCount { get; set; }
    public int V3N0 { get; set; }
    public int V3N1 { get; set; }
    public int V3N2 { get; set; }
    public int V3N3 { get; set; }
    public int V3N4 { get; set; }
    public int V3N5 { get; set; }
    public int V3N6 { get; set; }
    public int V3N7 { get; set; }

    // Vertex property values for matching
    /// <summary>Vertex property values (one per vertex)</summary>
    public float V0Property { get; set; }
    public float V1Property { get; set; }
    public float V2Property { get; set; }
    public float V3Property { get; set; }
    public float V4Property { get; set; }
    public float V5Property { get; set; }
    public float V6Property { get; set; }
    public float V7Property { get; set; }

    public readonly ReadOnlySpan<byte> Serialize() => MemoryPackSerializer.Serialize(this);
    public void Deserialize(ReadOnlySpan<byte> data) => this = MemoryPackSerializer.Deserialize<PatternMatchRingRequest>(data);
}

/// <summary>
/// Response message from GPU pattern matching.
/// </summary>
/// <remarks>
/// Returns matched vertex IDs and match statistics.
/// </remarks>
[MemoryPackable]
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public partial struct PatternMatchRingResponse : IRingKernelMessage
{
    // Message metadata
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }

    [MemoryPackIgnore]
    public readonly string MessageType => nameof(PatternMatchRingResponse);

    [MemoryPackIgnore]
    public readonly int PayloadSize => 128; // Approximate size

    // Result status
    public bool Success { get; set; }
    public int ErrorCode { get; set; }
    public long ProcessingTimeNs { get; set; }

    // Match results
    /// <summary>Number of vertices matched</summary>
    public int MatchCount { get; set; }

    /// <summary>Total vertices examined</summary>
    public int VerticesExamined { get; set; }

    /// <summary>Total edges traversed</summary>
    public int EdgesTraversed { get; set; }

    /// <summary>Pattern match confidence (0.0-1.0)</summary>
    public float MatchConfidence { get; set; }

    // Matched vertex IDs (up to 8 results)
    public int Match0 { get; set; }
    public int Match1 { get; set; }
    public int Match2 { get; set; }
    public int Match3 { get; set; }
    public int Match4 { get; set; }
    public int Match5 { get; set; }
    public int Match6 { get; set; }
    public int Match7 { get; set; }

    // Match scores for each matched vertex (relevance ranking)
    public float Score0 { get; set; }
    public float Score1 { get; set; }
    public float Score2 { get; set; }
    public float Score3 { get; set; }
    public float Score4 { get; set; }
    public float Score5 { get; set; }
    public float Score6 { get; set; }
    public float Score7 { get; set; }

    // Triangle detection results (for MatchTriangle)
    /// <summary>Number of triangles found</summary>
    public int TriangleCount { get; set; }

    /// <summary>Triangle vertices (up to 2 triangles, 3 vertices each)</summary>
    public int Tri0V0 { get; set; }
    public int Tri0V1 { get; set; }
    public int Tri0V2 { get; set; }
    public int Tri1V0 { get; set; }
    public int Tri1V1 { get; set; }
    public int Tri1V2 { get; set; }

    public readonly ReadOnlySpan<byte> Serialize() => MemoryPackSerializer.Serialize(this);
    public void Deserialize(ReadOnlySpan<byte> data) => this = MemoryPackSerializer.Deserialize<PatternMatchRingResponse>(data);
}
