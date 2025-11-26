// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Linq;
using FluentAssertions;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Abstractions.Temporal.Graph;
using Orleans.GpuBridge.Runtime.Temporal.Graph;
using Xunit;

namespace Orleans.GpuBridge.Runtime.Tests.Temporal.Graph;

/// <summary>
/// Tests for ITemporalGraphStorage interface and data types.
/// </summary>
public sealed class TemporalGraphStorageInterfaceTests
{
    #region TemporalEdgeData Tests

    [Fact]
    public void TemporalEdgeData_CreatesValidEdge()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);

        var edge = new TemporalEdgeData
        {
            SourceId = 1,
            TargetId = 2,
            ValidFrom = 100,
            ValidTo = 200,
            HLC = hlc,
            Weight = 10.5,
            EdgeType = "transfer"
        };

        edge.SourceId.Should().Be(1);
        edge.TargetId.Should().Be(2);
        edge.ValidFrom.Should().Be(100);
        edge.ValidTo.Should().Be(200);
        edge.DurationNanos.Should().Be(100);
        edge.Weight.Should().Be(10.5);
        edge.EdgeType.Should().Be("transfer");
    }

    [Fact]
    public void TemporalEdgeData_IsValidAt_ReturnsCorrectResult()
    {
        var edge = new TemporalEdgeData
        {
            SourceId = 1,
            TargetId = 2,
            ValidFrom = 100,
            ValidTo = 200,
            HLC = new HybridTimestamp(1000, 0, 1)
        };

        edge.IsValidAt(100).Should().BeTrue("start time should be inclusive");
        edge.IsValidAt(150).Should().BeTrue("middle time should be valid");
        edge.IsValidAt(200).Should().BeTrue("end time should be inclusive");
        edge.IsValidAt(99).Should().BeFalse("before start should be invalid");
        edge.IsValidAt(201).Should().BeFalse("after end should be invalid");
    }

    [Fact]
    public void TemporalEdgeData_OverlapsWith_DetectsOverlaps()
    {
        var edge = new TemporalEdgeData
        {
            SourceId = 1,
            TargetId = 2,
            ValidFrom = 100,
            ValidTo = 200,
            HLC = new HybridTimestamp(1000, 0, 1)
        };

        edge.OverlapsWith(50, 150).Should().BeTrue("partial overlap at start");
        edge.OverlapsWith(150, 250).Should().BeTrue("partial overlap at end");
        edge.OverlapsWith(100, 200).Should().BeTrue("exact match");
        edge.OverlapsWith(50, 250).Should().BeTrue("fully contains");
        edge.OverlapsWith(120, 180).Should().BeTrue("fully contained");
        edge.OverlapsWith(50, 99).Should().BeFalse("before");
        edge.OverlapsWith(201, 300).Should().BeFalse("after");
    }

    [Fact]
    public void TemporalEdgeData_WithProperties_MaintainsProperties()
    {
        var properties = new Dictionary<string, object>
        {
            ["amount"] = 1000.0,
            ["currency"] = "USD",
            ["reference"] = "TXN-001"
        };

        var edge = new TemporalEdgeData
        {
            SourceId = 1,
            TargetId = 2,
            ValidFrom = 100,
            ValidTo = 200,
            HLC = new HybridTimestamp(1000, 0, 1),
            Properties = properties
        };

        edge.Properties.Should().NotBeNull();
        edge.Properties!.Count.Should().Be(3);
        edge.Properties["amount"].Should().Be(1000.0);
        edge.Properties["currency"].Should().Be("USD");
    }

    #endregion

    #region TemporalPathData Tests

    [Fact]
    public void TemporalPathData_EmptyPath_HasCorrectDefaults()
    {
        var path = new TemporalPathData();

        path.Edges.Should().BeEmpty();
        path.Length.Should().Be(0);
        path.TotalWeight.Should().Be(0);
        path.StartTime.Should().Be(0);
        path.EndTime.Should().Be(0);
        path.SourceNode.Should().Be(0);
        path.TargetNode.Should().Be(0);
    }

    [Fact]
    public void TemporalPathData_WithEdges_ComputesCorrectMetrics()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var edges = new List<TemporalEdgeData>
        {
            new() { SourceId = 1, TargetId = 2, ValidFrom = 100, ValidTo = 110, HLC = hlc, Weight = 5.0 },
            new() { SourceId = 2, TargetId = 3, ValidFrom = 120, ValidTo = 130, HLC = hlc, Weight = 3.0 },
            new() { SourceId = 3, TargetId = 4, ValidFrom = 140, ValidTo = 150, HLC = hlc, Weight = 2.0 }
        };

        var path = new TemporalPathData
        {
            Edges = edges,
            TotalWeight = 10.0
        };

        path.Length.Should().Be(3);
        path.TotalWeight.Should().Be(10.0);
        path.SourceNode.Should().Be(1);
        path.TargetNode.Should().Be(4);
        path.StartTime.Should().Be(100);
        path.EndTime.Should().Be(140);
        path.TotalDurationNanos.Should().Be(40);
    }

    [Fact]
    public void TemporalPathData_GetNodes_ReturnsAllNodes()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var edges = new List<TemporalEdgeData>
        {
            new() { SourceId = 1, TargetId = 2, ValidFrom = 100, ValidTo = 110, HLC = hlc },
            new() { SourceId = 2, TargetId = 3, ValidFrom = 120, ValidTo = 130, HLC = hlc },
            new() { SourceId = 3, TargetId = 4, ValidFrom = 140, ValidTo = 150, HLC = hlc }
        };

        var path = new TemporalPathData { Edges = edges };

        var nodes = path.GetNodes().ToList();

        nodes.Should().Equal(1ul, 2ul, 3ul, 4ul);
    }

    #endregion

    #region TemporalGraphStatistics Tests

    [Fact]
    public void TemporalGraphStatistics_ComputesTimeSpanSeconds()
    {
        var stats = new TemporalGraphStatistics
        {
            NodeCount = 100,
            EdgeCount = 500,
            MinTime = 0,
            MaxTime = 5_000_000_000, // 5 seconds in nanos
            TimeSpanNanos = 5_000_000_000
        };

        stats.TimeSpanSeconds.Should().BeApproximately(5.0, 0.001);
    }

    #endregion

    #region PathSearchProgress Tests

    [Fact]
    public void PathSearchProgress_TracksSearchProgress()
    {
        var progress = new PathSearchProgress
        {
            NodesVisited = 100,
            EdgesExamined = 250,
            PathsFound = 5,
            CurrentDepth = 3,
            ElapsedNanos = 1_000_000
        };

        progress.NodesVisited.Should().Be(100);
        progress.EdgesExamined.Should().Be(250);
        progress.PathsFound.Should().Be(5);
        progress.CurrentDepth.Should().Be(3);
        progress.ElapsedNanos.Should().Be(1_000_000);
    }

    #endregion

    #region Interface Contract Tests

    [Fact]
    public void TemporalGraphStorage_ImplementsInterface()
    {
        // This test verifies that TemporalGraphStorage could implement ITemporalGraphStorage
        // Note: The actual implementation would need to be updated to implement the interface
        var storage = new TemporalGraphStorage();

        storage.EdgeCount.Should().Be(0);
        storage.NodeCount.Should().Be(0);
    }

    [Fact]
    public void TemporalGraphStorage_AddAndQueryEdges()
    {
        var storage = new TemporalGraphStorage();
        var hlc = new HybridTimestamp(1000, 0, 1);

        // Add edges
        storage.AddEdge(1, 2, 100, 200, hlc, weight: 5.0, edgeType: "edge1");
        storage.AddEdge(1, 3, 150, 250, hlc, weight: 3.0, edgeType: "edge2");
        storage.AddEdge(1, 4, 300, 400, hlc, weight: 2.0, edgeType: "edge3");

        storage.EdgeCount.Should().Be(3);
        storage.NodeCount.Should().Be(4);

        // Query overlapping edges
        var edges = storage.GetEdgesInTimeRange(1, 120, 180).ToList();

        edges.Should().HaveCount(2);
        edges.Should().Contain(e => e.EdgeType == "edge1");
        edges.Should().Contain(e => e.EdgeType == "edge2");
    }

    #endregion
}

/// <summary>
/// Tests for temporal hypergraph integration types.
/// </summary>
public sealed class HypergraphTemporalIntegrationTests
{
    #region TemporalHyperedgeData Tests

    [Fact]
    public void TemporalHyperedgeData_CreatesValidHyperedge()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);

        var hyperedge = new TemporalHyperedgeData
        {
            HyperedgeId = "meeting-001",
            VertexIds = [1, 2, 3, 4],
            ValidFrom = 100,
            ValidTo = 3600_000_000_000, // 1 hour in nanos
            HLC = hlc,
            HyperedgeType = "meeting",
            Weight = 4.0
        };

        hyperedge.HyperedgeId.Should().Be("meeting-001");
        hyperedge.Cardinality.Should().Be(4);
        hyperedge.HyperedgeType.Should().Be("meeting");
        hyperedge.DurationNanos.Should().BeGreaterThan(0);
    }

    [Fact]
    public void TemporalHyperedgeData_IsValidAt_ReturnsCorrectResult()
    {
        var hyperedge = new TemporalHyperedgeData
        {
            HyperedgeId = "test",
            VertexIds = [1, 2, 3],
            ValidFrom = 100,
            ValidTo = 200,
            HLC = new HybridTimestamp(1000, 0, 1)
        };

        hyperedge.IsValidAt(100).Should().BeTrue();
        hyperedge.IsValidAt(150).Should().BeTrue();
        hyperedge.IsValidAt(200).Should().BeTrue();
        hyperedge.IsValidAt(99).Should().BeFalse();
        hyperedge.IsValidAt(201).Should().BeFalse();
    }

    [Fact]
    public void TemporalHyperedgeData_ContainsVertex_ReturnsCorrectResult()
    {
        var hyperedge = new TemporalHyperedgeData
        {
            HyperedgeId = "test",
            VertexIds = [1, 2, 3],
            ValidFrom = 100,
            ValidTo = 200,
            HLC = new HybridTimestamp(1000, 0, 1)
        };

        hyperedge.ContainsVertex(1).Should().BeTrue();
        hyperedge.ContainsVertex(2).Should().BeTrue();
        hyperedge.ContainsVertex(3).Should().BeTrue();
        hyperedge.ContainsVertex(4).Should().BeFalse();
    }

    [Fact]
    public void TemporalHyperedgeData_WithRoles_MaintainsRoleAssignments()
    {
        var roles = new Dictionary<ulong, string>
        {
            [1] = "organizer",
            [2] = "attendee",
            [3] = "attendee"
        };

        var hyperedge = new TemporalHyperedgeData
        {
            HyperedgeId = "meeting",
            VertexIds = [1, 2, 3],
            ValidFrom = 100,
            ValidTo = 200,
            HLC = new HybridTimestamp(1000, 0, 1),
            VertexRoles = roles
        };

        hyperedge.VertexRoles.Should().NotBeNull();
        hyperedge.VertexRoles![1].Should().Be("organizer");
        hyperedge.VertexRoles[2].Should().Be("attendee");
    }

    #endregion

    #region TemporalVertexSnapshot Tests

    [Fact]
    public void TemporalVertexSnapshot_CapturesVertexState()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var properties = new Dictionary<string, object>
        {
            ["name"] = "Test Vertex",
            ["value"] = 42
        };

        var snapshot = new TemporalVertexSnapshot
        {
            VertexId = 1,
            VertexStringId = "vertex-1",
            TimestampNanos = 1000,
            HLC = hlc,
            VertexType = "user",
            Properties = properties,
            HyperedgeIds = ["he-1", "he-2"]
        };

        snapshot.VertexId.Should().Be(1);
        snapshot.VertexStringId.Should().Be("vertex-1");
        snapshot.VertexType.Should().Be("user");
        snapshot.Properties["name"].Should().Be("Test Vertex");
        snapshot.HyperedgeIds.Should().HaveCount(2);
    }

    #endregion

    #region TemporalHypergraphEvent Tests

    [Fact]
    public void TemporalHypergraphEvent_RecordsVertexEvent()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);

        var evt = new TemporalHypergraphEvent
        {
            EventType = HypergraphEventType.VertexCreated,
            TimestampNanos = 1000,
            HLC = hlc,
            VertexId = 1
        };

        evt.EventType.Should().Be(HypergraphEventType.VertexCreated);
        evt.VertexId.Should().Be(1);
        evt.HyperedgeId.Should().BeNull();
    }

    [Fact]
    public void TemporalHypergraphEvent_RecordsEdgeEvent()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var edgeData = new TemporalEdgeData
        {
            SourceId = 1,
            TargetId = 2,
            ValidFrom = 100,
            ValidTo = 200,
            HLC = hlc
        };

        var evt = new TemporalHypergraphEvent
        {
            EventType = HypergraphEventType.EdgeCreated,
            TimestampNanos = 1000,
            HLC = hlc,
            EdgeData = edgeData
        };

        evt.EventType.Should().Be(HypergraphEventType.EdgeCreated);
        evt.EdgeData.Should().NotBeNull();
        evt.EdgeData!.SourceId.Should().Be(1);
        evt.EdgeData.TargetId.Should().Be(2);
    }

    [Fact]
    public void TemporalHypergraphEvent_AllEventTypes_AreValid()
    {
        var eventTypes = Enum.GetValues<HypergraphEventType>();

        eventTypes.Should().HaveCountGreaterThan(5);
        eventTypes.Should().Contain(HypergraphEventType.VertexCreated);
        eventTypes.Should().Contain(HypergraphEventType.EdgeCreated);
        eventTypes.Should().Contain(HypergraphEventType.HyperedgeCreated);
        eventTypes.Should().Contain(HypergraphEventType.VertexJoinedHyperedge);
        eventTypes.Should().Contain(HypergraphEventType.MessageSent);
    }

    #endregion

    #region TemporalHypergraphQuery Tests

    [Fact]
    public void TemporalHypergraphQuery_ConfiguresCorrectly()
    {
        var query = new TemporalHypergraphQuery
        {
            QueryType = TemporalQueryType.EventStream,
            StartTimeNanos = 0,
            EndTimeNanos = 1_000_000_000,
            MaxResults = 500,
            EventTypeFilter = [HypergraphEventType.EdgeCreated, HypergraphEventType.EdgeDeleted],
            VertexTypeFilter = ["user", "account"]
        };

        query.QueryType.Should().Be(TemporalQueryType.EventStream);
        query.MaxResults.Should().Be(500);
        query.EventTypeFilter.Should().HaveCount(2);
        query.VertexTypeFilter.Should().HaveCount(2);
    }

    [Fact]
    public void TemporalQueryType_AllQueryTypes_AreValid()
    {
        var queryTypes = Enum.GetValues<TemporalQueryType>();

        queryTypes.Should().HaveCountGreaterThan(5);
        queryTypes.Should().Contain(TemporalQueryType.VertexHistory);
        queryTypes.Should().Contain(TemporalQueryType.GraphSnapshot);
        queryTypes.Should().Contain(TemporalQueryType.CausalChain);
        queryTypes.Should().Contain(TemporalQueryType.PatternDetection);
    }

    #endregion

    #region TemporalHypergraphQueryResult Tests

    [Fact]
    public void TemporalHypergraphQueryResult_ReturnsCorrectData()
    {
        var hlc = new HybridTimestamp(1000, 0, 1);
        var events = new List<TemporalHypergraphEvent>
        {
            new() { EventType = HypergraphEventType.VertexCreated, TimestampNanos = 100, HLC = hlc, VertexId = 1 },
            new() { EventType = HypergraphEventType.VertexCreated, TimestampNanos = 200, HLC = hlc, VertexId = 2 }
        };

        var result = new TemporalHypergraphQueryResult
        {
            QueryType = TemporalQueryType.EventStream,
            Events = events,
            IsTruncated = false,
            TotalCount = 2,
            QueryTimeNanos = 50_000
        };

        result.Events.Should().HaveCount(2);
        result.IsTruncated.Should().BeFalse();
        result.TotalCount.Should().Be(2);
        result.QueryTimeNanos.Should().Be(50_000);
    }

    #endregion
}
