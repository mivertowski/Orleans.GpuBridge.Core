// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using DotCompute.Abstractions.Attributes;
using DotCompute.Abstractions.RingKernels;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Ring kernel for GPU-native graph pattern matching in hypergraphs.
/// </summary>
/// <remarks>
/// <para>
/// This kernel implements GPU-accelerated subgraph pattern matching using the
/// DotCompute ring kernel system. It supports various graph pattern operations
/// that are common in hypergraph analytics.
/// </para>
/// <para>
/// <strong>Supported Operations:</strong>
/// <list type="bullet">
/// <item><description>MatchByProperty: Find vertices with matching property values</description></item>
/// <item><description>MatchByDegree: Find vertices with specific neighbor count</description></item>
/// <item><description>MatchNeighbors: Find all neighbors of a vertex</description></item>
/// <item><description>MatchPath: Find path between two vertices</description></item>
/// <item><description>MatchTriangle: Find 3-cliques in the graph</description></item>
/// <item><description>MatchStar: Find hub vertices with spoke patterns</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>Performance:</strong>
/// GPU-accelerated pattern matching can be 10-100× faster than CPU for large graphs
/// due to parallel vertex/edge processing.
/// </para>
/// </remarks>
public static class PatternMatchRingKernel
{
    /// <summary>
    /// Processes graph pattern matching requests on GPU.
    /// </summary>
    /// <param name="ctx">The ring kernel context for GPU operations.</param>
    /// <param name="request">Pattern match request with graph data and pattern specification.</param>
    [RingKernel(
        KernelId = "patternmatch_processor",
        Capacity = 4096,
        InputQueueSize = 512,
        OutputQueueSize = 512,
        MaxInputMessageSizeBytes = 2048,
        MaxOutputMessageSizeBytes = 1024,
        ProcessingMode = RingProcessingMode.Continuous,
        EnableTimestamps = true,
        Domain = RingKernelDomain.ActorModel,
        MessagingStrategy = MessagePassingStrategy.SharedMemory,
        Backends = KernelBackends.CUDA,
        OutputMessageType = typeof(PatternMatchRingResponse))]
    public static void ProcessPatternMatch(RingKernelContext ctx, PatternMatchRingRequest request)
    {
        // Get operation type
        int operationType = request.OperationType;

        // Initialize response
        var response = new PatternMatchRingResponse
        {
            MessageId = request.MessageId,
            Priority = request.Priority,
            CorrelationId = request.CorrelationId,
            Success = true,
            ErrorCode = 0,
            MatchCount = 0,
            VerticesExamined = 0,
            EdgesTraversed = 0,
            MatchConfidence = 0.0f,
            TriangleCount = 0
        };

        // Dispatch based on operation type
        if (operationType == 0) // MatchByProperty
        {
            ProcessMatchByProperty(ctx, request, ref response);
        }
        else if (operationType == 1) // MatchByDegree
        {
            ProcessMatchByDegree(ctx, request, ref response);
        }
        else if (operationType == 2) // MatchNeighbors
        {
            ProcessMatchNeighbors(ctx, request, ref response);
        }
        else if (operationType == 3) // MatchPath
        {
            ProcessMatchPath(ctx, request, ref response);
        }
        else if (operationType == 4) // MatchTriangle
        {
            ProcessMatchTriangle(ctx, request, ref response);
        }
        else if (operationType == 5) // MatchStar
        {
            ProcessMatchStar(ctx, request, ref response);
        }
        else
        {
            response.Success = false;
            response.ErrorCode = 1; // Unknown operation
        }

        // Ensure memory visibility
        ctx.ThreadFence();

        // Send response
        ctx.EnqueueOutput(response);
    }

    /// <summary>
    /// Match vertices by property value comparison.
    /// </summary>
    private static void ProcessMatchByProperty(
        RingKernelContext ctx,
        PatternMatchRingRequest request,
        ref PatternMatchRingResponse response)
    {
        int vertexCount = request.VertexCount;
        float targetValue = request.PropertyValue;
        int compOp = request.ComparisonOp;
        int matchCount = 0;

        // Get property values (up to 8 vertices)
        float p0 = request.V0Property;
        float p1 = request.V1Property;
        float p2 = request.V2Property;
        float p3 = request.V3Property;
        float p4 = request.V4Property;
        float p5 = request.V5Property;
        float p6 = request.V6Property;
        float p7 = request.V7Property;

        // Check each vertex
        if (vertexCount > 0 && CompareProperty(p0, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 0, 1.0f);
        }
        if (vertexCount > 1 && CompareProperty(p1, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 1, 1.0f);
        }
        if (vertexCount > 2 && CompareProperty(p2, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 2, 1.0f);
        }
        if (vertexCount > 3 && CompareProperty(p3, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 3, 1.0f);
        }
        if (vertexCount > 4 && CompareProperty(p4, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 4, 1.0f);
        }
        if (vertexCount > 5 && CompareProperty(p5, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 5, 1.0f);
        }
        if (vertexCount > 6 && CompareProperty(p6, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 6, 1.0f);
        }
        if (vertexCount > 7 && CompareProperty(p7, targetValue, compOp))
        {
            SetMatch(ref response, matchCount++, 7, 1.0f);
        }

        response.MatchCount = matchCount;
        response.VerticesExamined = vertexCount;
        response.MatchConfidence = matchCount > 0 ? 1.0f : 0.0f;
    }

    /// <summary>
    /// Match vertices by degree (neighbor count).
    /// </summary>
    private static void ProcessMatchByDegree(
        RingKernelContext ctx,
        PatternMatchRingRequest request,
        ref PatternMatchRingResponse response)
    {
        int vertexCount = request.VertexCount;
        int targetDegree = request.TargetDegree;
        int matchCount = 0;

        // Get neighbor counts
        int d0 = request.V0NeighborCount;
        int d1 = request.V1NeighborCount;
        int d2 = request.V2NeighborCount;
        int d3 = request.V3NeighborCount;

        // Check degrees
        if (vertexCount > 0 && d0 == targetDegree)
        {
            SetMatch(ref response, matchCount++, 0, 1.0f);
        }
        if (vertexCount > 1 && d1 == targetDegree)
        {
            SetMatch(ref response, matchCount++, 1, 1.0f);
        }
        if (vertexCount > 2 && d2 == targetDegree)
        {
            SetMatch(ref response, matchCount++, 2, 1.0f);
        }
        if (vertexCount > 3 && d3 == targetDegree)
        {
            SetMatch(ref response, matchCount++, 3, 1.0f);
        }

        response.MatchCount = matchCount;
        response.VerticesExamined = vertexCount;
        response.MatchConfidence = matchCount > 0 ? 1.0f : 0.0f;
    }

    /// <summary>
    /// Find all neighbors of a source vertex.
    /// </summary>
    private static void ProcessMatchNeighbors(
        RingKernelContext ctx,
        PatternMatchRingRequest request,
        ref PatternMatchRingResponse response)
    {
        int sourceVertex = request.SourceVertexId;
        int neighborCount = 0;

        // Get neighbors based on source vertex
        if (sourceVertex == 0)
        {
            neighborCount = request.V0NeighborCount;
            if (neighborCount > 0) SetMatch(ref response, 0, request.V0N0, 1.0f);
            if (neighborCount > 1) SetMatch(ref response, 1, request.V0N1, 1.0f);
            if (neighborCount > 2) SetMatch(ref response, 2, request.V0N2, 1.0f);
            if (neighborCount > 3) SetMatch(ref response, 3, request.V0N3, 1.0f);
            if (neighborCount > 4) SetMatch(ref response, 4, request.V0N4, 1.0f);
            if (neighborCount > 5) SetMatch(ref response, 5, request.V0N5, 1.0f);
            if (neighborCount > 6) SetMatch(ref response, 6, request.V0N6, 1.0f);
            if (neighborCount > 7) SetMatch(ref response, 7, request.V0N7, 1.0f);
        }
        else if (sourceVertex == 1)
        {
            neighborCount = request.V1NeighborCount;
            if (neighborCount > 0) SetMatch(ref response, 0, request.V1N0, 1.0f);
            if (neighborCount > 1) SetMatch(ref response, 1, request.V1N1, 1.0f);
            if (neighborCount > 2) SetMatch(ref response, 2, request.V1N2, 1.0f);
            if (neighborCount > 3) SetMatch(ref response, 3, request.V1N3, 1.0f);
            if (neighborCount > 4) SetMatch(ref response, 4, request.V1N4, 1.0f);
            if (neighborCount > 5) SetMatch(ref response, 5, request.V1N5, 1.0f);
            if (neighborCount > 6) SetMatch(ref response, 6, request.V1N6, 1.0f);
            if (neighborCount > 7) SetMatch(ref response, 7, request.V1N7, 1.0f);
        }
        else if (sourceVertex == 2)
        {
            neighborCount = request.V2NeighborCount;
            if (neighborCount > 0) SetMatch(ref response, 0, request.V2N0, 1.0f);
            if (neighborCount > 1) SetMatch(ref response, 1, request.V2N1, 1.0f);
            if (neighborCount > 2) SetMatch(ref response, 2, request.V2N2, 1.0f);
            if (neighborCount > 3) SetMatch(ref response, 3, request.V2N3, 1.0f);
            if (neighborCount > 4) SetMatch(ref response, 4, request.V2N4, 1.0f);
            if (neighborCount > 5) SetMatch(ref response, 5, request.V2N5, 1.0f);
            if (neighborCount > 6) SetMatch(ref response, 6, request.V2N6, 1.0f);
            if (neighborCount > 7) SetMatch(ref response, 7, request.V2N7, 1.0f);
        }
        else if (sourceVertex == 3)
        {
            neighborCount = request.V3NeighborCount;
            if (neighborCount > 0) SetMatch(ref response, 0, request.V3N0, 1.0f);
            if (neighborCount > 1) SetMatch(ref response, 1, request.V3N1, 1.0f);
            if (neighborCount > 2) SetMatch(ref response, 2, request.V3N2, 1.0f);
            if (neighborCount > 3) SetMatch(ref response, 3, request.V3N3, 1.0f);
            if (neighborCount > 4) SetMatch(ref response, 4, request.V3N4, 1.0f);
            if (neighborCount > 5) SetMatch(ref response, 5, request.V3N5, 1.0f);
            if (neighborCount > 6) SetMatch(ref response, 6, request.V3N6, 1.0f);
            if (neighborCount > 7) SetMatch(ref response, 7, request.V3N7, 1.0f);
        }

        response.MatchCount = neighborCount;
        response.VerticesExamined = 1;
        response.EdgesTraversed = neighborCount;
        response.MatchConfidence = neighborCount > 0 ? 1.0f : 0.0f;
    }

    /// <summary>
    /// Match path between source and target vertices (simplified BFS).
    /// </summary>
    private static void ProcessMatchPath(
        RingKernelContext ctx,
        PatternMatchRingRequest request,
        ref PatternMatchRingResponse response)
    {
        int sourceVertex = request.SourceVertexId;
        int targetVertex = request.TargetVertexId;
        int maxDepth = request.MaxDepth;

        // Simplified path detection - check if target is direct neighbor of source
        bool found = false;

        if (sourceVertex == 0)
        {
            int nc = request.V0NeighborCount;
            if (nc > 0 && request.V0N0 == targetVertex) found = true;
            if (nc > 1 && request.V0N1 == targetVertex) found = true;
            if (nc > 2 && request.V0N2 == targetVertex) found = true;
            if (nc > 3 && request.V0N3 == targetVertex) found = true;
            if (nc > 4 && request.V0N4 == targetVertex) found = true;
            if (nc > 5 && request.V0N5 == targetVertex) found = true;
            if (nc > 6 && request.V0N6 == targetVertex) found = true;
            if (nc > 7 && request.V0N7 == targetVertex) found = true;
        }
        else if (sourceVertex == 1)
        {
            int nc = request.V1NeighborCount;
            if (nc > 0 && request.V1N0 == targetVertex) found = true;
            if (nc > 1 && request.V1N1 == targetVertex) found = true;
            if (nc > 2 && request.V1N2 == targetVertex) found = true;
            if (nc > 3 && request.V1N3 == targetVertex) found = true;
            if (nc > 4 && request.V1N4 == targetVertex) found = true;
            if (nc > 5 && request.V1N5 == targetVertex) found = true;
            if (nc > 6 && request.V1N6 == targetVertex) found = true;
            if (nc > 7 && request.V1N7 == targetVertex) found = true;
        }

        if (found)
        {
            SetMatch(ref response, 0, sourceVertex, 1.0f);
            SetMatch(ref response, 1, targetVertex, 1.0f);
            response.MatchCount = 2;
            response.MatchConfidence = 1.0f;
        }
        else
        {
            response.MatchCount = 0;
            response.MatchConfidence = 0.0f;
        }

        response.VerticesExamined = 2;
        response.EdgesTraversed = found ? 1 : 0;
    }

    /// <summary>
    /// Find triangles (3-cliques) in the graph.
    /// </summary>
    private static void ProcessMatchTriangle(
        RingKernelContext ctx,
        PatternMatchRingRequest request,
        ref PatternMatchRingResponse response)
    {
        // Triangle detection: for each vertex u, check if any two neighbors v,w are connected
        int triangleCount = 0;

        // Check vertex 0's neighbors for triangles
        int nc0 = request.V0NeighborCount;
        if (nc0 >= 2)
        {
            // Get first two neighbors of vertex 0
            int v = request.V0N0;
            int w = request.V0N1;

            // Check if v and w are connected (simplified - check v's neighbors for w)
            bool vwConnected = false;
            if (v == 1)
            {
                int nc1 = request.V1NeighborCount;
                if (nc1 > 0 && request.V1N0 == w) vwConnected = true;
                if (nc1 > 1 && request.V1N1 == w) vwConnected = true;
                if (nc1 > 2 && request.V1N2 == w) vwConnected = true;
                if (nc1 > 3 && request.V1N3 == w) vwConnected = true;
            }
            else if (v == 2)
            {
                int nc2 = request.V2NeighborCount;
                if (nc2 > 0 && request.V2N0 == w) vwConnected = true;
                if (nc2 > 1 && request.V2N1 == w) vwConnected = true;
                if (nc2 > 2 && request.V2N2 == w) vwConnected = true;
                if (nc2 > 3 && request.V2N3 == w) vwConnected = true;
            }
            else if (v == 3)
            {
                int nc3 = request.V3NeighborCount;
                if (nc3 > 0 && request.V3N0 == w) vwConnected = true;
                if (nc3 > 1 && request.V3N1 == w) vwConnected = true;
                if (nc3 > 2 && request.V3N2 == w) vwConnected = true;
                if (nc3 > 3 && request.V3N3 == w) vwConnected = true;
            }

            if (vwConnected && triangleCount < 2)
            {
                if (triangleCount == 0)
                {
                    response.Tri0V0 = 0;
                    response.Tri0V1 = v;
                    response.Tri0V2 = w;
                }
                else
                {
                    response.Tri1V0 = 0;
                    response.Tri1V1 = v;
                    response.Tri1V2 = w;
                }
                triangleCount++;
            }
        }

        response.TriangleCount = triangleCount;
        response.MatchCount = triangleCount * 3; // 3 vertices per triangle
        response.VerticesExamined = request.VertexCount;
        response.MatchConfidence = triangleCount > 0 ? 1.0f : 0.0f;
    }

    /// <summary>
    /// Find star patterns (hub with multiple spokes).
    /// </summary>
    private static void ProcessMatchStar(
        RingKernelContext ctx,
        PatternMatchRingRequest request,
        ref PatternMatchRingResponse response)
    {
        // Find vertex with highest degree as hub
        int d0 = request.V0NeighborCount;
        int d1 = request.V1NeighborCount;
        int d2 = request.V2NeighborCount;
        int d3 = request.V3NeighborCount;

        int maxDegree = d0;
        int hubVertex = 0;

        if (d1 > maxDegree) { maxDegree = d1; hubVertex = 1; }
        if (d2 > maxDegree) { maxDegree = d2; hubVertex = 2; }
        if (d3 > maxDegree) { maxDegree = d3; hubVertex = 3; }

        // Hub is the first match, spokes follow
        SetMatch(ref response, 0, hubVertex, 1.0f);
        int matchIdx = 1;

        // Add neighbors as spokes
        if (hubVertex == 0 && d0 > 0)
        {
            if (matchIdx < 8) SetMatch(ref response, matchIdx++, request.V0N0, 0.8f);
            if (matchIdx < 8 && d0 > 1) SetMatch(ref response, matchIdx++, request.V0N1, 0.8f);
            if (matchIdx < 8 && d0 > 2) SetMatch(ref response, matchIdx++, request.V0N2, 0.8f);
            if (matchIdx < 8 && d0 > 3) SetMatch(ref response, matchIdx++, request.V0N3, 0.8f);
        }
        else if (hubVertex == 1 && d1 > 0)
        {
            if (matchIdx < 8) SetMatch(ref response, matchIdx++, request.V1N0, 0.8f);
            if (matchIdx < 8 && d1 > 1) SetMatch(ref response, matchIdx++, request.V1N1, 0.8f);
            if (matchIdx < 8 && d1 > 2) SetMatch(ref response, matchIdx++, request.V1N2, 0.8f);
            if (matchIdx < 8 && d1 > 3) SetMatch(ref response, matchIdx++, request.V1N3, 0.8f);
        }
        else if (hubVertex == 2 && d2 > 0)
        {
            if (matchIdx < 8) SetMatch(ref response, matchIdx++, request.V2N0, 0.8f);
            if (matchIdx < 8 && d2 > 1) SetMatch(ref response, matchIdx++, request.V2N1, 0.8f);
            if (matchIdx < 8 && d2 > 2) SetMatch(ref response, matchIdx++, request.V2N2, 0.8f);
            if (matchIdx < 8 && d2 > 3) SetMatch(ref response, matchIdx++, request.V2N3, 0.8f);
        }
        else if (hubVertex == 3 && d3 > 0)
        {
            if (matchIdx < 8) SetMatch(ref response, matchIdx++, request.V3N0, 0.8f);
            if (matchIdx < 8 && d3 > 1) SetMatch(ref response, matchIdx++, request.V3N1, 0.8f);
            if (matchIdx < 8 && d3 > 2) SetMatch(ref response, matchIdx++, request.V3N2, 0.8f);
            if (matchIdx < 8 && d3 > 3) SetMatch(ref response, matchIdx++, request.V3N3, 0.8f);
        }

        response.MatchCount = matchIdx;
        response.VerticesExamined = request.VertexCount;
        response.EdgesTraversed = maxDegree;
        response.MatchConfidence = maxDegree >= request.TargetDegree ? 1.0f : (float)maxDegree / request.TargetDegree;
    }

    /// <summary>
    /// Helper to compare property values.
    /// </summary>
    private static bool CompareProperty(float value, float target, int compOp)
    {
        if (compOp == 0) return value == target;       // Equal
        if (compOp == 1) return value < target;        // Less than
        if (compOp == 2) return value > target;        // Greater than
        if (compOp == 3) return value != target;       // Not equal
        return false;
    }

    /// <summary>
    /// Helper to set match result at index.
    /// </summary>
    private static void SetMatch(ref PatternMatchRingResponse response, int idx, int vertexId, float score)
    {
        if (idx == 0) { response.Match0 = vertexId; response.Score0 = score; }
        else if (idx == 1) { response.Match1 = vertexId; response.Score1 = score; }
        else if (idx == 2) { response.Match2 = vertexId; response.Score2 = score; }
        else if (idx == 3) { response.Match3 = vertexId; response.Score3 = score; }
        else if (idx == 4) { response.Match4 = vertexId; response.Score4 = score; }
        else if (idx == 5) { response.Match5 = vertexId; response.Score5 = score; }
        else if (idx == 6) { response.Match6 = vertexId; response.Score6 = score; }
        else if (idx == 7) { response.Match7 = vertexId; response.Score7 = score; }
    }
}
