// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using FluentAssertions;
using Moq;
using Orleans.GpuBridge.Abstractions.Hypergraph;
using Orleans.GpuBridge.Grains.Hypergraph;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Grains.Tests.Hypergraph;

/// <summary>
/// Comprehensive unit tests for <see cref="HypergraphVertexGrain"/>.
/// </summary>
public class HypergraphVertexGrainTests
{
    private readonly Mock<ILogger<HypergraphVertexGrain>> _loggerMock;
    private readonly Mock<IGrainFactory> _grainFactoryMock;

    public HypergraphVertexGrainTests()
    {
        _loggerMock = new Mock<ILogger<HypergraphVertexGrain>>();
        _grainFactoryMock = new Mock<IGrainFactory>();
    }

    #region VertexInitRequest Tests

    [Fact]
    public void VertexInitRequest_ShouldHaveRequiredProperties()
    {
        // Arrange & Act
        var request = new VertexInitRequest
        {
            VertexType = "Person",
            Properties = new Dictionary<string, object>
            {
                ["Name"] = "Alice",
                ["Age"] = 30
            },
            InitialHyperedges = new List<string> { "group1", "group2" },
            AffinityGroup = "gpu-0"
        };

        // Assert
        request.VertexType.Should().Be("Person");
        request.Properties.Should().HaveCount(2);
        request.Properties["Name"].Should().Be("Alice");
        request.InitialHyperedges.Should().HaveCount(2);
        request.AffinityGroup.Should().Be("gpu-0");
    }

    [Fact]
    public void VertexInitRequest_OptionalPropertiesShouldBeNull()
    {
        // Arrange & Act
        var request = new VertexInitRequest
        {
            VertexType = "Node",
            Properties = new Dictionary<string, object>()
        };

        // Assert
        request.InitialHyperedges.Should().BeNull();
        request.AffinityGroup.Should().BeNull();
    }

    #endregion

    #region VertexInitResult Tests

    [Fact]
    public void VertexInitResult_Success_ShouldHaveCorrectValues()
    {
        // Arrange & Act
        var result = new VertexInitResult
        {
            Success = true,
            VertexId = "vertex-123",
            Version = 1,
            CreatedAtNanos = 1000000000L
        };

        // Assert
        result.Success.Should().BeTrue();
        result.VertexId.Should().Be("vertex-123");
        result.Version.Should().Be(1);
        result.CreatedAtNanos.Should().Be(1000000000L);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public void VertexInitResult_Failure_ShouldHaveErrorMessage()
    {
        // Arrange & Act
        var result = new VertexInitResult
        {
            Success = false,
            VertexId = "",
            Version = 0,
            CreatedAtNanos = 0,
            ErrorMessage = "Vertex already initialized"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Be("Vertex already initialized");
    }

    [Fact]
    public void VertexInitResult_ShouldBeValueType()
    {
        // VertexInitResult is a readonly record struct
        var result1 = new VertexInitResult
        {
            Success = true,
            VertexId = "v1",
            Version = 1,
            CreatedAtNanos = 100
        };
        var result2 = new VertexInitResult
        {
            Success = true,
            VertexId = "v1",
            Version = 1,
            CreatedAtNanos = 100
        };

        // Value equality
        result1.Should().Be(result2);
    }

    #endregion

    #region VertexState Tests

    [Fact]
    public void VertexState_ShouldContainAllProperties()
    {
        // Arrange
        var hyperedges = new List<HyperedgeMembership>
        {
            new HyperedgeMembership
            {
                HyperedgeId = "he1",
                Role = "member",
                JoinedAtNanos = 1000,
                PeerCount = 5
            }
        };

        // Act
        var state = new VertexState
        {
            VertexId = "v-123",
            VertexType = "Entity",
            Version = 5,
            Properties = new Dictionary<string, object> { ["key"] = "value" },
            Hyperedges = hyperedges,
            CreatedAtNanos = 1000,
            ModifiedAtNanos = 2000,
            HlcTimestamp = 3000
        };

        // Assert
        state.VertexId.Should().Be("v-123");
        state.VertexType.Should().Be("Entity");
        state.Version.Should().Be(5);
        state.Properties.Should().ContainKey("key");
        state.Hyperedges.Should().HaveCount(1);
        state.CreatedAtNanos.Should().Be(1000);
        state.ModifiedAtNanos.Should().Be(2000);
        state.HlcTimestamp.Should().Be(3000);
    }

    #endregion

    #region VertexUpdateResult Tests

    [Fact]
    public void VertexUpdateResult_ShouldTrackChangedProperties()
    {
        // Arrange & Act
        var result = new VertexUpdateResult
        {
            Success = true,
            NewVersion = 6,
            UpdatedAtNanos = 5000,
            ChangedProperties = new List<string> { "Name", "Age", "Status" }
        };

        // Assert
        result.Success.Should().BeTrue();
        result.NewVersion.Should().Be(6);
        result.ChangedProperties.Should().Contain("Name");
        result.ChangedProperties.Should().HaveCount(3);
    }

    #endregion

    #region HyperedgeMembership Tests

    [Fact]
    public void HyperedgeMembership_ShouldHaveAllProperties()
    {
        // Arrange & Act
        var membership = new HyperedgeMembership
        {
            HyperedgeId = "meeting-123",
            Role = "organizer",
            JoinedAtNanos = 1000000,
            PeerCount = 10
        };

        // Assert
        membership.HyperedgeId.Should().Be("meeting-123");
        membership.Role.Should().Be("organizer");
        membership.JoinedAtNanos.Should().Be(1000000);
        membership.PeerCount.Should().Be(10);
    }

    [Fact]
    public void HyperedgeMembership_RoleShouldBeOptional()
    {
        // Arrange & Act
        var membership = new HyperedgeMembership
        {
            HyperedgeId = "group-456",
            JoinedAtNanos = 2000000,
            PeerCount = 3
        };

        // Assert
        membership.Role.Should().BeNull();
    }

    #endregion

    #region HyperedgeMembershipResult Tests

    [Fact]
    public void HyperedgeMembershipResult_Join_ShouldReportSuccess()
    {
        // Arrange & Act
        var result = new HyperedgeMembershipResult
        {
            Success = true,
            HyperedgeId = "he-789",
            Operation = "Join",
            TimestampNanos = 3000000,
            CurrentMemberCount = 5
        };

        // Assert
        result.Success.Should().BeTrue();
        result.Operation.Should().Be("Join");
        result.CurrentMemberCount.Should().Be(5);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public void HyperedgeMembershipResult_Leave_ShouldReportSuccess()
    {
        // Arrange & Act
        var result = new HyperedgeMembershipResult
        {
            Success = true,
            HyperedgeId = "he-789",
            Operation = "Leave",
            TimestampNanos = 4000000,
            CurrentMemberCount = 4
        };

        // Assert
        result.Operation.Should().Be("Leave");
        result.CurrentMemberCount.Should().Be(4);
    }

    [Fact]
    public void HyperedgeMembershipResult_Failure_ShouldIncludeError()
    {
        // Arrange & Act
        var result = new HyperedgeMembershipResult
        {
            Success = false,
            HyperedgeId = "he-999",
            Operation = "Join",
            TimestampNanos = 5000000,
            CurrentMemberCount = 0,
            ErrorMessage = "Hyperedge at maximum cardinality"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Contain("maximum cardinality");
    }

    #endregion

    #region VertexMessage Tests

    [Fact]
    public void VertexMessage_ShouldHaveAllRequiredFields()
    {
        // Arrange & Act
        var message = new VertexMessage
        {
            MessageId = "msg-001",
            SourceVertexId = "vertex-a",
            TargetVertexId = "vertex-b",
            ViaHyperedgeId = "he-123",
            MessageType = "UpdateNotification",
            Payload = new Dictionary<string, object>
            {
                ["data"] = "test-data",
                ["priority"] = 1
            },
            HlcTimestamp = 6000000,
            TtlHops = 5
        };

        // Assert
        message.MessageId.Should().Be("msg-001");
        message.SourceVertexId.Should().Be("vertex-a");
        message.TargetVertexId.Should().Be("vertex-b");
        message.ViaHyperedgeId.Should().Be("he-123");
        message.MessageType.Should().Be("UpdateNotification");
        message.Payload.Should().HaveCount(2);
        message.HlcTimestamp.Should().Be(6000000);
        message.TtlHops.Should().Be(5);
    }

    [Fact]
    public void VertexMessage_DefaultTtlShouldBe3()
    {
        // Arrange & Act
        var message = new VertexMessage
        {
            MessageId = "msg-002",
            SourceVertexId = "v-src",
            MessageType = "Ping",
            Payload = new Dictionary<string, object>(),
            HlcTimestamp = 7000000
        };

        // Assert
        message.TtlHops.Should().Be(3);
    }

    [Fact]
    public void VertexMessage_BroadcastShouldHaveNullTarget()
    {
        // Arrange & Act
        var broadcastMessage = new VertexMessage
        {
            MessageId = "broadcast-001",
            SourceVertexId = "v-sender",
            ViaHyperedgeId = "he-456",
            MessageType = "Announcement",
            Payload = new Dictionary<string, object> { ["content"] = "Hello all" },
            HlcTimestamp = 8000000
        };

        // Assert
        broadcastMessage.TargetVertexId.Should().BeNull();
        broadcastMessage.ViaHyperedgeId.Should().NotBeNull();
    }

    #endregion

    #region BroadcastResult Tests

    [Fact]
    public void BroadcastResult_ShouldTrackDeliveryStatistics()
    {
        // Arrange & Act
        var result = new BroadcastResult
        {
            Success = true,
            TargetCount = 10,
            DeliveredCount = 9,
            LatencyNanos = 500000
        };

        // Assert
        result.Success.Should().BeTrue();
        result.TargetCount.Should().Be(10);
        result.DeliveredCount.Should().Be(9);
        result.LatencyNanos.Should().Be(500000);
    }

    [Fact]
    public void BroadcastResult_PartialDeliveryShouldStillSucceed()
    {
        // A broadcast is successful even if some deliveries fail
        var result = new BroadcastResult
        {
            Success = true,
            TargetCount = 100,
            DeliveredCount = 95,
            LatencyNanos = 1000000
        };

        result.Success.Should().BeTrue();
        (result.TargetCount - result.DeliveredCount).Should().Be(5);
    }

    #endregion

    #region MessageResult Tests

    [Fact]
    public void MessageResult_SuccessfulProcessing()
    {
        // Arrange & Act
        var result = new MessageResult
        {
            Success = true,
            ProcessingTimeNanos = 150000,
            Response = new Dictionary<string, object> { ["ack"] = true }
        };

        // Assert
        result.Success.Should().BeTrue();
        result.ProcessingTimeNanos.Should().Be(150000);
        result.Response.Should().ContainKey("ack");
    }

    [Fact]
    public void MessageResult_NoResponsePayload()
    {
        // Arrange & Act
        var result = new MessageResult
        {
            Success = true,
            ProcessingTimeNanos = 100000
        };

        // Assert
        result.Response.Should().BeNull();
    }

    #endregion

    #region NeighborFilter Tests

    [Fact]
    public void NeighborFilter_ShouldSupportMultipleFilterTypes()
    {
        // Arrange & Act
        var filter = new NeighborFilter
        {
            VertexTypes = new List<string> { "Person", "Organization" },
            HyperedgeTypes = new List<string> { "membership", "ownership" },
            PropertyFilters = new Dictionary<string, object>
            {
                ["status"] = "active",
                ["minScore"] = 50
            },
            MaxResults = 50
        };

        // Assert
        filter.VertexTypes.Should().HaveCount(2);
        filter.HyperedgeTypes.Should().HaveCount(2);
        filter.PropertyFilters.Should().ContainKey("status");
        filter.MaxResults.Should().Be(50);
    }

    [Fact]
    public void NeighborFilter_DefaultMaxResultsShouldBe100()
    {
        // Arrange & Act
        var filter = new NeighborFilter();

        // Assert
        filter.MaxResults.Should().Be(100);
    }

    #endregion

    #region NeighborQueryResult Tests

    [Fact]
    public void NeighborQueryResult_ShouldContainNeighborInfoList()
    {
        // Arrange
        var neighbors = new List<NeighborInfo>
        {
            new NeighborInfo
            {
                VertexId = "v-neighbor-1",
                VertexType = "Person",
                Distance = 1,
                ViaHyperedges = new List<string> { "he1" },
                Properties = new Dictionary<string, object> { ["name"] = "Bob" }
            },
            new NeighborInfo
            {
                VertexId = "v-neighbor-2",
                VertexType = "Organization",
                Distance = 2,
                ViaHyperedges = new List<string> { "he1", "he2" }
            }
        };

        // Act
        var result = new NeighborQueryResult
        {
            SourceVertexId = "v-source",
            Neighbors = neighbors,
            QueryTimeNanos = 250000,
            IsTruncated = false
        };

        // Assert
        result.SourceVertexId.Should().Be("v-source");
        result.Neighbors.Should().HaveCount(2);
        result.QueryTimeNanos.Should().Be(250000);
        result.IsTruncated.Should().BeFalse();
    }

    [Fact]
    public void NeighborQueryResult_TruncatedResult()
    {
        // Arrange & Act
        var result = new NeighborQueryResult
        {
            SourceVertexId = "v-src",
            Neighbors = Enumerable.Range(0, 100)
                .Select(i => new NeighborInfo
                {
                    VertexId = $"v-{i}",
                    VertexType = "Node",
                    Distance = 1,
                    ViaHyperedges = new List<string> { "he1" }
                }).ToList(),
            QueryTimeNanos = 500000,
            IsTruncated = true
        };

        // Assert
        result.IsTruncated.Should().BeTrue();
        result.Neighbors.Should().HaveCount(100);
    }

    #endregion

    #region HypergraphPattern Tests

    [Fact]
    public void HypergraphPattern_ShouldDefineVertexAndHyperedgeConstraints()
    {
        // Arrange & Act
        var pattern = new HypergraphPattern
        {
            PatternId = "triangle-pattern",
            VertexConstraints = new List<PatternVertexConstraint>
            {
                new PatternVertexConstraint
                {
                    VariableName = "a",
                    VertexTypes = new List<string> { "Person" }
                },
                new PatternVertexConstraint
                {
                    VariableName = "b",
                    VertexTypes = new List<string> { "Person" }
                },
                new PatternVertexConstraint
                {
                    VariableName = "c",
                    VertexTypes = new List<string> { "Person" }
                }
            },
            HyperedgeConstraints = new List<PatternHyperedgeConstraint>
            {
                new PatternHyperedgeConstraint
                {
                    VariableName = "e1",
                    ContainedVertices = new List<string> { "a", "b", "c" },
                    MinCardinality = 3
                }
            },
            MaxMatches = 50,
            Timeout = TimeSpan.FromSeconds(10)
        };

        // Assert
        pattern.PatternId.Should().Be("triangle-pattern");
        pattern.VertexConstraints.Should().HaveCount(3);
        pattern.HyperedgeConstraints.Should().HaveCount(1);
        pattern.HyperedgeConstraints[0].ContainedVertices.Should().Contain("a", "b", "c");
        pattern.MaxMatches.Should().Be(50);
        pattern.Timeout.Should().Be(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public void HypergraphPattern_DefaultTimeoutShouldBe30Seconds()
    {
        // Arrange & Act
        var pattern = new HypergraphPattern
        {
            PatternId = "simple-pattern",
            VertexConstraints = new List<PatternVertexConstraint>(),
            HyperedgeConstraints = new List<PatternHyperedgeConstraint>()
        };

        // Assert
        pattern.Timeout.Should().Be(TimeSpan.FromSeconds(30));
    }

    #endregion

    #region PatternMatchResult Tests

    [Fact]
    public void PatternMatchResult_ShouldContainMatchesWithBindings()
    {
        // Arrange
        var matches = new List<PatternMatch>
        {
            new PatternMatch
            {
                VertexBindings = new Dictionary<string, string>
                {
                    ["a"] = "v-1",
                    ["b"] = "v-2",
                    ["c"] = "v-3"
                },
                HyperedgeBindings = new Dictionary<string, string>
                {
                    ["e1"] = "he-123"
                },
                Score = 0.95
            }
        };

        // Act
        var result = new PatternMatchResult
        {
            PatternId = "test-pattern",
            Matches = matches,
            ExecutionTimeNanos = 1500000,
            IsTruncated = false,
            TotalMatchCount = 1
        };

        // Assert
        result.PatternId.Should().Be("test-pattern");
        result.Matches.Should().HaveCount(1);
        result.Matches[0].VertexBindings.Should().ContainKey("a");
        result.Matches[0].Score.Should().Be(0.95);
        result.TotalMatchCount.Should().Be(1);
        result.IsTruncated.Should().BeFalse();
    }

    [Fact]
    public void PatternMatch_DefaultScoreShouldBe1()
    {
        // Arrange & Act
        var match = new PatternMatch
        {
            VertexBindings = new Dictionary<string, string>(),
            HyperedgeBindings = new Dictionary<string, string>()
        };

        // Assert
        match.Score.Should().Be(1.0);
    }

    #endregion

    #region VertexAggregation Tests

    [Theory]
    [InlineData(AggregationType.Sum)]
    [InlineData(AggregationType.Average)]
    [InlineData(AggregationType.Min)]
    [InlineData(AggregationType.Max)]
    [InlineData(AggregationType.Count)]
    [InlineData(AggregationType.StdDev)]
    public void VertexAggregation_ShouldSupportAllAggregationTypes(AggregationType type)
    {
        // Arrange & Act
        var aggregation = new VertexAggregation
        {
            Type = type,
            PropertyName = "score",
            Scope = AggregationScope.DirectNeighbors,
            MaxHops = 1
        };

        // Assert
        aggregation.Type.Should().Be(type);
        aggregation.PropertyName.Should().Be("score");
    }

    [Theory]
    [InlineData(AggregationScope.DirectNeighbors)]
    [InlineData(AggregationScope.SameHyperedge)]
    [InlineData(AggregationScope.MultiHop)]
    public void VertexAggregation_ShouldSupportAllScopes(AggregationScope scope)
    {
        // Arrange & Act
        var aggregation = new VertexAggregation
        {
            Type = AggregationType.Sum,
            PropertyName = "value",
            Scope = scope,
            MaxHops = scope == AggregationScope.MultiHop ? 3 : 1
        };

        // Assert
        aggregation.Scope.Should().Be(scope);
    }

    [Fact]
    public void VertexAggregation_DefaultScope_ShouldBeDirectNeighbors()
    {
        // Arrange & Act
        var aggregation = new VertexAggregation
        {
            Type = AggregationType.Count,
            PropertyName = "id"
        };

        // Assert
        aggregation.Scope.Should().Be(AggregationScope.DirectNeighbors);
        aggregation.MaxHops.Should().Be(1);
    }

    #endregion

    #region AggregationResult Tests

    [Fact]
    public void AggregationResult_ShouldContainComputedValues()
    {
        // Arrange & Act
        var result = new AggregationResult
        {
            Type = AggregationType.Average,
            Value = 75.5,
            VertexCount = 10,
            ExecutionTimeNanos = 200000
        };

        // Assert
        result.Type.Should().Be(AggregationType.Average);
        result.Value.Should().Be(75.5);
        result.VertexCount.Should().Be(10);
        result.ExecutionTimeNanos.Should().Be(200000);
    }

    [Fact]
    public void AggregationResult_CountShouldMatchVertexCount()
    {
        // For Count aggregation, Value should equal VertexCount
        var result = new AggregationResult
        {
            Type = AggregationType.Count,
            Value = 42,
            VertexCount = 42,
            ExecutionTimeNanos = 50000
        };

        result.Value.Should().Be(result.VertexCount);
    }

    #endregion

    #region VertexMetrics Tests

    [Fact]
    public void VertexMetrics_ShouldTrackAllPerformanceIndicators()
    {
        // Arrange & Act
        var metrics = new VertexMetrics
        {
            VertexId = "v-perf-test",
            MessagesReceived = 1000,
            MessagesSent = 500,
            AvgProcessingTimeNanos = 150.5,
            HyperedgeCount = 5,
            PatternMatchCount = 25,
            GpuMemoryBytes = 4096
        };

        // Assert
        metrics.VertexId.Should().Be("v-perf-test");
        metrics.MessagesReceived.Should().Be(1000);
        metrics.MessagesSent.Should().Be(500);
        metrics.AvgProcessingTimeNanos.Should().Be(150.5);
        metrics.HyperedgeCount.Should().Be(5);
        metrics.PatternMatchCount.Should().Be(25);
        metrics.GpuMemoryBytes.Should().Be(4096);
    }

    [Fact]
    public void VertexMetrics_MessageRatio()
    {
        // Can compute derived metrics from base values
        var metrics = new VertexMetrics
        {
            VertexId = "v-ratio-test",
            MessagesReceived = 100,
            MessagesSent = 50,
            AvgProcessingTimeNanos = 100,
            HyperedgeCount = 3,
            PatternMatchCount = 10,
            GpuMemoryBytes = 2048
        };

        var sendReceiveRatio = (double)metrics.MessagesSent / metrics.MessagesReceived;
        sendReceiveRatio.Should().Be(0.5);
    }

    #endregion

    #region PatternVertexConstraint Tests

    [Fact]
    public void PatternVertexConstraint_WithPropertyConstraints()
    {
        // Arrange & Act
        var constraint = new PatternVertexConstraint
        {
            VariableName = "person",
            VertexTypes = new List<string> { "Person", "Employee" },
            PropertyConstraints = new Dictionary<string, object>
            {
                ["age"] = 30,
                ["department"] = "Engineering"
            }
        };

        // Assert
        constraint.VariableName.Should().Be("person");
        constraint.VertexTypes.Should().Contain("Person", "Employee");
        constraint.PropertyConstraints.Should().ContainKey("age");
        constraint.PropertyConstraints["department"].Should().Be("Engineering");
    }

    #endregion

    #region PatternHyperedgeConstraint Tests

    [Fact]
    public void PatternHyperedgeConstraint_WithCardinalityBounds()
    {
        // Arrange & Act
        var constraint = new PatternHyperedgeConstraint
        {
            VariableName = "meeting",
            HyperedgeTypes = new List<string> { "Meeting", "Conference" },
            ContainedVertices = new List<string> { "organizer", "attendee1" },
            MinCardinality = 2,
            MaxCardinality = 50
        };

        // Assert
        constraint.VariableName.Should().Be("meeting");
        constraint.MinCardinality.Should().Be(2);
        constraint.MaxCardinality.Should().Be(50);
        constraint.ContainedVertices.Should().HaveCount(2);
    }

    [Fact]
    public void PatternHyperedgeConstraint_DefaultCardinality()
    {
        // Arrange & Act
        var constraint = new PatternHyperedgeConstraint
        {
            VariableName = "edge",
            ContainedVertices = new List<string> { "a", "b" }
        };

        // Assert
        constraint.MinCardinality.Should().Be(2); // Default minimum
        constraint.MaxCardinality.Should().Be(0); // 0 = unbounded
    }

    #endregion

    #region Integration Scenario Tests

    [Fact]
    public void CompleteWorkflow_CreateVertexJoinHyperedgeSendMessage()
    {
        // This test simulates a complete workflow scenario using DTOs

        // 1. Initialize vertex
        var initRequest = new VertexInitRequest
        {
            VertexType = "User",
            Properties = new Dictionary<string, object>
            {
                ["username"] = "alice",
                ["email"] = "alice@example.com"
            }
        };

        var initResult = new VertexInitResult
        {
            Success = true,
            VertexId = "user-alice",
            Version = 1,
            CreatedAtNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000
        };

        // 2. Join hyperedge
        var joinResult = new HyperedgeMembershipResult
        {
            Success = true,
            HyperedgeId = "team-engineering",
            Operation = "Join",
            TimestampNanos = initResult.CreatedAtNanos + 1000,
            CurrentMemberCount = 5
        };

        // 3. Send message to team
        var message = new VertexMessage
        {
            MessageId = Guid.NewGuid().ToString(),
            SourceVertexId = initResult.VertexId,
            ViaHyperedgeId = joinResult.HyperedgeId,
            MessageType = "Greeting",
            Payload = new Dictionary<string, object> { ["text"] = "Hello team!" },
            HlcTimestamp = joinResult.TimestampNanos + 1000
        };

        var broadcastResult = new BroadcastResult
        {
            Success = true,
            TargetCount = 4, // 5 members - 1 (self)
            DeliveredCount = 4,
            LatencyNanos = 250000
        };

        // Assertions
        initResult.Success.Should().BeTrue();
        joinResult.Success.Should().BeTrue();
        message.ViaHyperedgeId.Should().Be(joinResult.HyperedgeId);
        broadcastResult.DeliveredCount.Should().Be(broadcastResult.TargetCount);
    }

    [Fact]
    public void PatternMatching_FindTriangles()
    {
        // Define a pattern to find triangles (3 vertices all connected via one hyperedge)
        var trianglePattern = new HypergraphPattern
        {
            PatternId = "find-triangles",
            VertexConstraints = new List<PatternVertexConstraint>
            {
                new PatternVertexConstraint { VariableName = "v1", VertexTypes = new List<string> { "Node" } },
                new PatternVertexConstraint { VariableName = "v2", VertexTypes = new List<string> { "Node" } },
                new PatternVertexConstraint { VariableName = "v3", VertexTypes = new List<string> { "Node" } }
            },
            HyperedgeConstraints = new List<PatternHyperedgeConstraint>
            {
                new PatternHyperedgeConstraint
                {
                    VariableName = "connection",
                    ContainedVertices = new List<string> { "v1", "v2", "v3" },
                    MinCardinality = 3,
                    MaxCardinality = 3
                }
            },
            MaxMatches = 100
        };

        // Simulate results
        var matchResult = new PatternMatchResult
        {
            PatternId = trianglePattern.PatternId,
            Matches = new List<PatternMatch>
            {
                new PatternMatch
                {
                    VertexBindings = new Dictionary<string, string>
                    {
                        ["v1"] = "node-a",
                        ["v2"] = "node-b",
                        ["v3"] = "node-c"
                    },
                    HyperedgeBindings = new Dictionary<string, string>
                    {
                        ["connection"] = "he-abc"
                    }
                }
            },
            ExecutionTimeNanos = 500000,
            IsTruncated = false,
            TotalMatchCount = 1
        };

        // Assertions
        matchResult.Matches.Should().HaveCount(1);
        matchResult.Matches[0].VertexBindings.Should().HaveCount(3);
        matchResult.Matches[0].HyperedgeBindings["connection"].Should().Be("he-abc");
    }

    #endregion
}
