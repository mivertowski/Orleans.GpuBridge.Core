// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Backends.DotCompute.Temporal;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels.CpuFallbackHandlers;

/// <summary>
/// CPU fallback handler for pattern matching ring kernel operations.
/// </summary>
/// <remarks>
/// <para>
/// This handler mirrors the GPU ring kernel logic for graph pattern matching,
/// providing equivalent functionality when GPU execution is unavailable.
/// </para>
/// <para>
/// Supported operations:
/// <list type="bullet">
/// <item><description>0 = MatchByProperty: Match vertices by property value</description></item>
/// <item><description>1 = MatchByDegree: Match vertices by degree (neighbor count)</description></item>
/// <item><description>2 = MatchNeighbors: Match vertices connected to a specific vertex</description></item>
/// <item><description>3 = MatchPath: Match path pattern (source → target via edge)</description></item>
/// <item><description>4 = MatchTriangle: Match triangle pattern (3-clique)</description></item>
/// <item><description>5 = MatchStar: Match star pattern (hub with N spokes)</description></item>
/// </list>
/// </para>
/// </remarks>
[CpuFallbackHandler("patternmatch_processor", 0)]
public sealed class PatternMatchCpuHandler
    : IStatelessCpuFallbackHandler<PatternMatchRingRequest, PatternMatchRingResponse>
{
    /// <inheritdoc/>
    public string KernelId => "patternmatch_processor";

    /// <inheritdoc/>
    public int HandlerId => 0;

    /// <inheritdoc/>
    public string Description => "CPU fallback for graph pattern matching operations";

    /// <inheritdoc/>
    public PatternMatchRingResponse Execute(PatternMatchRingRequest request)
    {
        var startTicks = System.Diagnostics.Stopwatch.GetTimestamp();

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
            MatchConfidence = 0.0f
        };

        // Extract graph data from request
        var neighborCounts = new int[] { request.V0NeighborCount, request.V1NeighborCount, request.V2NeighborCount, request.V3NeighborCount };
        var properties = new float[] { request.V0Property, request.V1Property, request.V2Property, request.V3Property,
                                        request.V4Property, request.V5Property, request.V6Property, request.V7Property };

        // Execute pattern matching based on operation type
        switch ((PatternMatchOperation)request.OperationType)
        {
            case PatternMatchOperation.MatchByProperty:
                ExecuteMatchByProperty(ref response, request, properties);
                break;

            case PatternMatchOperation.MatchByDegree:
                ExecuteMatchByDegree(ref response, request, neighborCounts);
                break;

            case PatternMatchOperation.MatchNeighbors:
                ExecuteMatchNeighbors(ref response, request);
                break;

            case PatternMatchOperation.MatchPath:
                ExecuteMatchPath(ref response, request);
                break;

            case PatternMatchOperation.MatchTriangle:
                ExecuteMatchTriangle(ref response, request);
                break;

            case PatternMatchOperation.MatchStar:
                ExecuteMatchStar(ref response, request, neighborCounts);
                break;

            default:
                response.Success = false;
                response.ErrorCode = -1; // Unknown operation
                break;
        }

        var endTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        response.ProcessingTimeNs = (endTicks - startTicks) * 1_000_000_000L / System.Diagnostics.Stopwatch.Frequency;

        return response;
    }

    private static void ExecuteMatchByProperty(
        ref PatternMatchRingResponse response,
        PatternMatchRingRequest request,
        float[] properties)
    {
        var matchCount = 0;
        var vertexCount = Math.Min(request.VertexCount, 8);
        response.VerticesExamined = vertexCount;

        for (var i = 0; i < vertexCount; i++)
        {
            var matches = request.ComparisonOp switch
            {
                0 => Math.Abs(properties[i] - request.PropertyValue) < 0.0001f, // Equal
                1 => properties[i] < request.PropertyValue, // Less than
                2 => properties[i] > request.PropertyValue, // Greater than
                3 => Math.Abs(properties[i] - request.PropertyValue) >= 0.0001f, // Not equal
                _ => false
            };

            if (matches && matchCount < 8)
            {
                SetMatch(ref response, matchCount, i, 1.0f);
                matchCount++;
            }
        }

        response.MatchCount = matchCount;
        response.MatchConfidence = vertexCount > 0 ? (float)matchCount / vertexCount : 0.0f;
    }

    private static void ExecuteMatchByDegree(
        ref PatternMatchRingResponse response,
        PatternMatchRingRequest request,
        int[] neighborCounts)
    {
        var matchCount = 0;
        var vertexCount = Math.Min(request.VertexCount, 4);
        response.VerticesExamined = vertexCount;

        for (var i = 0; i < vertexCount; i++)
        {
            if (neighborCounts[i] == request.TargetDegree && matchCount < 8)
            {
                SetMatch(ref response, matchCount, i, 1.0f);
                matchCount++;
            }
        }

        response.MatchCount = matchCount;
        response.MatchConfidence = vertexCount > 0 ? (float)matchCount / vertexCount : 0.0f;
    }

    private static void ExecuteMatchNeighbors(
        ref PatternMatchRingResponse response,
        PatternMatchRingRequest request)
    {
        var sourceVertex = request.SourceVertexId;
        if (sourceVertex < 0 || sourceVertex >= 4)
        {
            response.Success = false;
            response.ErrorCode = -2; // Invalid source vertex
            return;
        }

        response.VerticesExamined = 1;

        // Get neighbors of source vertex
        var neighbors = GetNeighbors(request, sourceVertex);
        var matchCount = 0;

        foreach (var neighbor in neighbors)
        {
            if (neighbor >= 0 && matchCount < 8)
            {
                SetMatch(ref response, matchCount, neighbor, 1.0f);
                matchCount++;
                response.EdgesTraversed++;
            }
        }

        response.MatchCount = matchCount;
        response.MatchConfidence = matchCount > 0 ? 1.0f : 0.0f;
    }

    private static void ExecuteMatchPath(
        ref PatternMatchRingResponse response,
        PatternMatchRingRequest request)
    {
        var sourceVertex = request.SourceVertexId;
        var targetVertex = request.TargetVertexId;
        var maxDepth = Math.Min(request.MaxDepth, 3);

        if (sourceVertex < 0 || sourceVertex >= 4 || targetVertex < 0 || targetVertex >= 4)
        {
            response.Success = false;
            response.ErrorCode = -2; // Invalid vertices
            return;
        }

        response.VerticesExamined = 2;

        // BFS to find path
        var visited = new bool[4];
        var queue = new Queue<(int vertex, int depth, int[] path)>();
        queue.Enqueue((sourceVertex, 0, new[] { sourceVertex }));
        visited[sourceVertex] = true;

        while (queue.Count > 0)
        {
            var (current, depth, path) = queue.Dequeue();

            if (current == targetVertex)
            {
                // Found path - return vertices in path
                var matchCount = 0;
                foreach (var v in path)
                {
                    if (matchCount < 8)
                    {
                        SetMatch(ref response, matchCount, v, 1.0f);
                        matchCount++;
                    }
                }
                response.MatchCount = matchCount;
                response.MatchConfidence = 1.0f;
                return;
            }

            if (depth >= maxDepth)
            {
                continue;
            }

            var neighbors = GetNeighbors(request, current);
            foreach (var neighbor in neighbors)
            {
                if (neighbor >= 0 && neighbor < 4 && !visited[neighbor])
                {
                    visited[neighbor] = true;
                    response.VerticesExamined++;
                    response.EdgesTraversed++;
                    var newPath = new int[path.Length + 1];
                    Array.Copy(path, newPath, path.Length);
                    newPath[path.Length] = neighbor;
                    queue.Enqueue((neighbor, depth + 1, newPath));
                }
            }
        }

        // No path found
        response.MatchCount = 0;
        response.MatchConfidence = 0.0f;
    }

    private static void ExecuteMatchTriangle(
        ref PatternMatchRingResponse response,
        PatternMatchRingRequest request)
    {
        var vertexCount = Math.Min(request.VertexCount, 4);
        response.VerticesExamined = vertexCount;

        var triangleCount = 0;

        // Check all possible triangles (i, j, k) where i < j < k
        for (var i = 0; i < vertexCount && triangleCount < 2; i++)
        {
            var neighborsI = GetNeighbors(request, i);

            for (var j = i + 1; j < vertexCount && triangleCount < 2; j++)
            {
                if (!neighborsI.Contains(j))
                {
                    continue;
                }

                var neighborsJ = GetNeighbors(request, j);
                response.EdgesTraversed++;

                for (var k = j + 1; k < vertexCount && triangleCount < 2; k++)
                {
                    if (neighborsI.Contains(k) && neighborsJ.Contains(k))
                    {
                        // Found triangle (i, j, k)
                        response.EdgesTraversed += 2;

                        if (triangleCount == 0)
                        {
                            response.Tri0V0 = i;
                            response.Tri0V1 = j;
                            response.Tri0V2 = k;
                        }
                        else
                        {
                            response.Tri1V0 = i;
                            response.Tri1V1 = j;
                            response.Tri1V2 = k;
                        }

                        triangleCount++;
                    }
                }
            }
        }

        response.TriangleCount = triangleCount;
        response.MatchCount = triangleCount * 3; // Each triangle has 3 vertices
        response.MatchConfidence = triangleCount > 0 ? 1.0f : 0.0f;
    }

    private static void ExecuteMatchStar(
        ref PatternMatchRingResponse response,
        PatternMatchRingRequest request,
        int[] neighborCounts)
    {
        var vertexCount = Math.Min(request.VertexCount, 4);
        var targetDegree = request.TargetDegree;
        response.VerticesExamined = vertexCount;

        // Find hub vertex with required degree
        var hubIndex = -1;
        var maxDegree = 0;

        for (var i = 0; i < vertexCount; i++)
        {
            if (neighborCounts[i] >= targetDegree && neighborCounts[i] > maxDegree)
            {
                hubIndex = i;
                maxDegree = neighborCounts[i];
            }
        }

        if (hubIndex < 0)
        {
            response.MatchCount = 0;
            response.MatchConfidence = 0.0f;
            return;
        }

        // Return hub and its neighbors
        var matchCount = 0;
        SetMatch(ref response, matchCount++, hubIndex, 1.0f); // Hub vertex

        var neighbors = GetNeighbors(request, hubIndex);
        foreach (var neighbor in neighbors)
        {
            if (neighbor >= 0 && matchCount < 8)
            {
                SetMatch(ref response, matchCount++, neighbor, 0.8f); // Spoke vertices
                response.EdgesTraversed++;
            }
        }

        response.MatchCount = matchCount;
        response.MatchConfidence = matchCount >= targetDegree + 1 ? 1.0f : (float)matchCount / (targetDegree + 1);
    }

    private static int[] GetNeighbors(PatternMatchRingRequest request, int vertexIndex)
    {
        return vertexIndex switch
        {
            0 => new[] { request.V0N0, request.V0N1, request.V0N2, request.V0N3,
                          request.V0N4, request.V0N5, request.V0N6, request.V0N7 }
                      .Take(request.V0NeighborCount).ToArray(),
            1 => new[] { request.V1N0, request.V1N1, request.V1N2, request.V1N3,
                          request.V1N4, request.V1N5, request.V1N6, request.V1N7 }
                      .Take(request.V1NeighborCount).ToArray(),
            2 => new[] { request.V2N0, request.V2N1, request.V2N2, request.V2N3,
                          request.V2N4, request.V2N5, request.V2N6, request.V2N7 }
                      .Take(request.V2NeighborCount).ToArray(),
            3 => new[] { request.V3N0, request.V3N1, request.V3N2, request.V3N3,
                          request.V3N4, request.V3N5, request.V3N6, request.V3N7 }
                      .Take(request.V3NeighborCount).ToArray(),
            _ => Array.Empty<int>()
        };
    }

    private static void SetMatch(ref PatternMatchRingResponse response, int index, int vertexId, float score)
    {
        switch (index)
        {
            case 0: response.Match0 = vertexId; response.Score0 = score; break;
            case 1: response.Match1 = vertexId; response.Score1 = score; break;
            case 2: response.Match2 = vertexId; response.Score2 = score; break;
            case 3: response.Match3 = vertexId; response.Score3 = score; break;
            case 4: response.Match4 = vertexId; response.Score4 = score; break;
            case 5: response.Match5 = vertexId; response.Score5 = score; break;
            case 6: response.Match6 = vertexId; response.Score6 = score; break;
            case 7: response.Match7 = vertexId; response.Score7 = score; break;
        }
    }
}
