// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using FluentAssertions;
using Moq;
using Orleans.GpuBridge.Abstractions.Hypergraph;
using Orleans.GpuBridge.Grains.Hypergraph;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Grains.Tests.Hypergraph;

/// <summary>
/// Comprehensive unit tests for <see cref="HypergraphHyperedgeGrain"/>.
/// </summary>
public class HypergraphHyperedgeGrainTests
{
    private readonly Mock<ILogger<HypergraphHyperedgeGrain>> _loggerMock;
    private readonly Mock<IGrainFactory> _grainFactoryMock;

    public HypergraphHyperedgeGrainTests()
    {
        _loggerMock = new Mock<ILogger<HypergraphHyperedgeGrain>>();
        _grainFactoryMock = new Mock<IGrainFactory>();
    }

    #region HyperedgeInitRequest Tests

    [Fact]
    public void HyperedgeInitRequest_ShouldHaveAllRequiredProperties()
    {
        // Arrange & Act
        var request = new HyperedgeInitRequest
        {
            HyperedgeType = "Meeting",
            Properties = new Dictionary<string, object>
            {
                ["title"] = "Team Standup",
                ["duration"] = 30
            },
            InitialMembers = new List<HyperedgeMemberInit>
            {
                new HyperedgeMemberInit { VertexId = "user-1", Role = "organizer", Position = 0 },
                new HyperedgeMemberInit { VertexId = "user-2", Role = "attendee", Position = 1 },
                new HyperedgeMemberInit { VertexId = "user-3", Role = "attendee", Position = 2 }
            },
            MinCardinality = 2,
            MaxCardinality = 50,
            IsDirected = false,
            AffinityGroup = "gpu-0"
        };

        // Assert
        request.HyperedgeType.Should().Be("Meeting");
        request.Properties.Should().HaveCount(2);
        request.InitialMembers.Should().HaveCount(3);
        request.MinCardinality.Should().Be(2);
        request.MaxCardinality.Should().Be(50);
        request.IsDirected.Should().BeFalse();
        request.AffinityGroup.Should().Be("gpu-0");
    }

    [Fact]
    public void HyperedgeInitRequest_DefaultValues()
    {
        // Arrange & Act
        var request = new HyperedgeInitRequest
        {
            HyperedgeType = "Edge",
            Properties = new Dictionary<string, object>(),
            InitialMembers = new List<HyperedgeMemberInit>()
        };

        // Assert
        request.MinCardinality.Should().Be(2);
        request.MaxCardinality.Should().Be(0); // 0 = unlimited
        request.IsDirected.Should().BeFalse();
        request.AffinityGroup.Should().BeNull();
    }

    #endregion

    #region HyperedgeMemberInit Tests

    [Fact]
    public void HyperedgeMemberInit_WithRole()
    {
        // Arrange & Act
        var member = new HyperedgeMemberInit
        {
            VertexId = "vertex-123",
            Role = "leader",
            Position = 0
        };

        // Assert
        member.VertexId.Should().Be("vertex-123");
        member.Role.Should().Be("leader");
        member.Position.Should().Be(0);
    }

    [Fact]
    public void HyperedgeMemberInit_OptionalFieldsShouldBeNull()
    {
        // Arrange & Act
        var member = new HyperedgeMemberInit
        {
            VertexId = "vertex-456"
        };

        // Assert
        member.Role.Should().BeNull();
        member.Position.Should().BeNull();
    }

    #endregion

    #region HyperedgeInitResult Tests

    [Fact]
    public void HyperedgeInitResult_Success()
    {
        // Arrange & Act
        var result = new HyperedgeInitResult
        {
            Success = true,
            HyperedgeId = "he-123",
            Version = 1,
            MemberCount = 3,
            CreatedAtNanos = 1000000000L
        };

        // Assert
        result.Success.Should().BeTrue();
        result.HyperedgeId.Should().Be("he-123");
        result.Version.Should().Be(1);
        result.MemberCount.Should().Be(3);
        result.CreatedAtNanos.Should().Be(1000000000L);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public void HyperedgeInitResult_Failure()
    {
        // Arrange & Act
        var result = new HyperedgeInitResult
        {
            Success = false,
            HyperedgeId = "",
            Version = 0,
            MemberCount = 0,
            CreatedAtNanos = 0,
            ErrorMessage = "Minimum cardinality not met"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Contain("cardinality");
    }

    #endregion

    #region HyperedgeState Tests

    [Fact]
    public void HyperedgeState_ShouldContainAllStateInformation()
    {
        // Arrange
        var members = new List<HyperedgeMember>
        {
            new HyperedgeMember
            {
                VertexId = "v1",
                Role = "source",
                Position = 0,
                JoinedAtNanos = 1000,
                MembershipProperties = new Dictionary<string, object> { ["weight"] = 1.0 }
            },
            new HyperedgeMember
            {
                VertexId = "v2",
                Role = "target",
                Position = 1,
                JoinedAtNanos = 2000
            }
        };

        // Act
        var state = new HyperedgeState
        {
            HyperedgeId = "he-state-test",
            HyperedgeType = "DirectedEdge",
            Version = 5,
            Properties = new Dictionary<string, object> { ["weight"] = 1.5 },
            Members = members,
            MinCardinality = 2,
            MaxCardinality = 2,
            IsDirected = true,
            CreatedAtNanos = 1000,
            ModifiedAtNanos = 3000,
            HlcTimestamp = 4000
        };

        // Assert
        state.HyperedgeId.Should().Be("he-state-test");
        state.HyperedgeType.Should().Be("DirectedEdge");
        state.Version.Should().Be(5);
        state.Cardinality.Should().Be(2);
        state.Members.Should().HaveCount(2);
        state.IsDirected.Should().BeTrue();
        state.CreatedAtNanos.Should().Be(1000);
        state.ModifiedAtNanos.Should().Be(3000);
        state.HlcTimestamp.Should().Be(4000);
    }

    [Fact]
    public void HyperedgeState_CardinalityShouldEqualMemberCount()
    {
        // Arrange
        var members = Enumerable.Range(0, 5)
            .Select(i => new HyperedgeMember
            {
                VertexId = $"v{i}",
                JoinedAtNanos = i * 1000
            })
            .ToList();

        // Act
        var state = new HyperedgeState
        {
            HyperedgeId = "he-card-test",
            HyperedgeType = "Group",
            Version = 1,
            Properties = new Dictionary<string, object>(),
            Members = members,
            MinCardinality = 2,
            MaxCardinality = 10,
            IsDirected = false,
            CreatedAtNanos = 0,
            ModifiedAtNanos = 0,
            HlcTimestamp = 0
        };

        // Assert
        state.Cardinality.Should().Be(5);
        state.Cardinality.Should().Be(state.Members.Count);
    }

    #endregion

    #region HyperedgeMember Tests

    [Fact]
    public void HyperedgeMember_WithMembershipProperties()
    {
        // Arrange & Act
        var member = new HyperedgeMember
        {
            VertexId = "v-member",
            Role = "contributor",
            Position = 3,
            JoinedAtNanos = 5000000,
            MembershipProperties = new Dictionary<string, object>
            {
                ["contributionScore"] = 85.5,
                ["isActive"] = true
            }
        };

        // Assert
        member.VertexId.Should().Be("v-member");
        member.Role.Should().Be("contributor");
        member.Position.Should().Be(3);
        member.JoinedAtNanos.Should().Be(5000000);
        member.MembershipProperties.Should().ContainKey("contributionScore");
        member.MembershipProperties!["isActive"].Should().Be(true);
    }

    #endregion

    #region HyperedgeMutationResult Tests

    [Fact]
    public void HyperedgeMutationResult_AddVertex_Success()
    {
        // Arrange & Act
        var result = new HyperedgeMutationResult
        {
            Success = true,
            Operation = "Add",
            VertexId = "v-new",
            NewVersion = 6,
            CurrentCardinality = 5,
            TimestampNanos = 6000000
        };

        // Assert
        result.Success.Should().BeTrue();
        result.Operation.Should().Be("Add");
        result.VertexId.Should().Be("v-new");
        result.NewVersion.Should().Be(6);
        result.CurrentCardinality.Should().Be(5);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public void HyperedgeMutationResult_RemoveVertex_Success()
    {
        // Arrange & Act
        var result = new HyperedgeMutationResult
        {
            Success = true,
            Operation = "Remove",
            VertexId = "v-old",
            NewVersion = 7,
            CurrentCardinality = 4,
            TimestampNanos = 7000000
        };

        // Assert
        result.Operation.Should().Be("Remove");
        result.CurrentCardinality.Should().Be(4);
    }

    [Fact]
    public void HyperedgeMutationResult_Failure_MaxCardinalityExceeded()
    {
        // Arrange & Act
        var result = new HyperedgeMutationResult
        {
            Success = false,
            Operation = "Add",
            VertexId = "v-overflow",
            NewVersion = 5,
            CurrentCardinality = 10,
            TimestampNanos = 8000000,
            ErrorMessage = "Maximum cardinality (10) exceeded"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Contain("Maximum cardinality");
    }

    [Fact]
    public void HyperedgeMutationResult_Failure_MinCardinalityViolation()
    {
        // Arrange & Act
        var result = new HyperedgeMutationResult
        {
            Success = false,
            Operation = "Remove",
            VertexId = "v-last",
            NewVersion = 3,
            CurrentCardinality = 2,
            TimestampNanos = 9000000,
            ErrorMessage = "Cannot remove: would violate minimum cardinality (2)"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Contain("minimum cardinality");
    }

    #endregion

    #region HyperedgeUpdateResult Tests

    [Fact]
    public void HyperedgeUpdateResult_ShouldTrackChangedProperties()
    {
        // Arrange & Act
        var result = new HyperedgeUpdateResult
        {
            Success = true,
            NewVersion = 8,
            ChangedProperties = new List<string> { "title", "description", "priority" },
            TimestampNanos = 10000000
        };

        // Assert
        result.Success.Should().BeTrue();
        result.NewVersion.Should().Be(8);
        result.ChangedProperties.Should().HaveCount(3);
        result.ChangedProperties.Should().Contain("title");
    }

    #endregion

    #region HyperedgeMessage Tests

    [Fact]
    public void HyperedgeMessage_BroadcastToAll()
    {
        // Arrange & Act
        var message = new HyperedgeMessage
        {
            MessageId = "msg-broadcast-001",
            MessageType = "Notification",
            SourceVertexId = "v-sender",
            Payload = new Dictionary<string, object>
            {
                ["event"] = "StatusChanged",
                ["newStatus"] = "Active"
            },
            HlcTimestamp = 11000000
        };

        // Assert
        message.MessageId.Should().Be("msg-broadcast-001");
        message.MessageType.Should().Be("Notification");
        message.TargetRoles.Should().BeNull(); // Broadcast to all
        message.ExcludeVertices.Should().BeNull();
    }

    [Fact]
    public void HyperedgeMessage_BroadcastToSpecificRoles()
    {
        // Arrange & Act
        var message = new HyperedgeMessage
        {
            MessageId = "msg-role-001",
            MessageType = "Alert",
            Payload = new Dictionary<string, object> { ["level"] = "High" },
            HlcTimestamp = 12000000,
            TargetRoles = new List<string> { "admin", "moderator" }
        };

        // Assert
        message.TargetRoles.Should().HaveCount(2);
        message.TargetRoles.Should().Contain("admin", "moderator");
    }

    [Fact]
    public void HyperedgeMessage_BroadcastExcludingVertices()
    {
        // Arrange & Act
        var message = new HyperedgeMessage
        {
            MessageId = "msg-exclude-001",
            MessageType = "Update",
            SourceVertexId = "v-origin",
            Payload = new Dictionary<string, object>(),
            HlcTimestamp = 13000000,
            ExcludeVertices = new List<string> { "v-origin", "v-busy" }
        };

        // Assert
        message.ExcludeVertices.Should().HaveCount(2);
        message.ExcludeVertices.Should().Contain("v-origin"); // Don't echo back to sender
    }

    #endregion

    #region HyperedgeBroadcastResult Tests

    [Fact]
    public void HyperedgeBroadcastResult_FullDelivery()
    {
        // Arrange & Act
        var result = new HyperedgeBroadcastResult
        {
            Success = true,
            TargetCount = 10,
            DeliveredCount = 10,
            FailedCount = 0,
            TotalLatencyNanos = 500000
        };

        // Assert
        result.Success.Should().BeTrue();
        result.TargetCount.Should().Be(10);
        result.DeliveredCount.Should().Be(10);
        result.FailedCount.Should().Be(0);
        result.AvgLatencyNanos.Should().Be(50000); // 500000 / 10
    }

    [Fact]
    public void HyperedgeBroadcastResult_PartialDelivery()
    {
        // Arrange & Act
        var result = new HyperedgeBroadcastResult
        {
            Success = true,
            TargetCount = 20,
            DeliveredCount = 18,
            FailedCount = 2,
            TotalLatencyNanos = 1000000
        };

        // Assert
        result.DeliveredCount.Should().BeLessThan(result.TargetCount);
        result.FailedCount.Should().Be(2);
        result.AvgLatencyNanos.Should().Be(50000); // 1000000 / 20
    }

    [Fact]
    public void HyperedgeBroadcastResult_AvgLatency_ZeroTargets()
    {
        // Arrange & Act
        var result = new HyperedgeBroadcastResult
        {
            Success = false,
            TargetCount = 0,
            DeliveredCount = 0,
            FailedCount = 0,
            TotalLatencyNanos = 0
        };

        // Assert - Should not throw division by zero
        result.AvgLatencyNanos.Should().Be(0);
    }

    #endregion

    #region HyperedgeAggregation Tests

    [Fact]
    public void HyperedgeAggregation_SumAllMembers()
    {
        // Arrange & Act
        var aggregation = new HyperedgeAggregation
        {
            Type = AggregationType.Sum,
            PropertyName = "score"
        };

        // Assert
        aggregation.Type.Should().Be(AggregationType.Sum);
        aggregation.PropertyName.Should().Be("score");
        aggregation.IncludeRoles.Should().BeNull(); // All roles
        aggregation.IncludeHyperedgeProperties.Should().BeFalse();
    }

    [Fact]
    public void HyperedgeAggregation_AverageForSpecificRoles()
    {
        // Arrange & Act
        var aggregation = new HyperedgeAggregation
        {
            Type = AggregationType.Average,
            PropertyName = "rating",
            IncludeRoles = new List<string> { "reviewer", "expert" }
        };

        // Assert
        aggregation.IncludeRoles.Should().HaveCount(2);
    }

    [Fact]
    public void HyperedgeAggregation_IncludeHyperedgeProperties()
    {
        // Arrange & Act
        var aggregation = new HyperedgeAggregation
        {
            Type = AggregationType.Max,
            PropertyName = "priority",
            IncludeHyperedgeProperties = true
        };

        // Assert
        aggregation.IncludeHyperedgeProperties.Should().BeTrue();
    }

    #endregion

    #region HyperedgeAggregationResult Tests

    [Fact]
    public void HyperedgeAggregationResult_ShouldContainAllFields()
    {
        // Arrange & Act
        var result = new HyperedgeAggregationResult
        {
            Type = AggregationType.Average,
            Value = 85.3,
            VertexCount = 15,
            ExecutionTimeNanos = 300000
        };

        // Assert
        result.Type.Should().Be(AggregationType.Average);
        result.Value.Should().Be(85.3);
        result.VertexCount.Should().Be(15);
        result.ExecutionTimeNanos.Should().Be(300000);
    }

    #endregion

    #region HyperedgeConstraint Tests

    [Theory]
    [InlineData(HyperedgeConstraintType.Cardinality)]
    [InlineData(HyperedgeConstraintType.RolePresence)]
    [InlineData(HyperedgeConstraintType.PropertyValue)]
    [InlineData(HyperedgeConstraintType.UniqueVertices)]
    public void HyperedgeConstraint_AllTypesSupported(HyperedgeConstraintType type)
    {
        // Arrange & Act
        var constraint = new HyperedgeConstraint
        {
            Type = type
        };

        // Assert
        constraint.Type.Should().Be(type);
    }

    [Fact]
    public void HyperedgeConstraint_CardinalityConstraint()
    {
        // Arrange & Act
        var constraint = new HyperedgeConstraint
        {
            Type = HyperedgeConstraintType.Cardinality,
            MinCardinality = 3,
            MaxCardinality = 10
        };

        // Assert
        constraint.MinCardinality.Should().Be(3);
        constraint.MaxCardinality.Should().Be(10);
    }

    [Fact]
    public void HyperedgeConstraint_RolePresenceConstraint()
    {
        // Arrange & Act
        var constraint = new HyperedgeConstraint
        {
            Type = HyperedgeConstraintType.RolePresence,
            RequiredRoles = new List<string> { "leader", "reviewer", "approver" }
        };

        // Assert
        constraint.RequiredRoles.Should().HaveCount(3);
    }

    [Fact]
    public void HyperedgeConstraint_PropertyConstraint()
    {
        // Arrange & Act
        var constraint = new HyperedgeConstraint
        {
            Type = HyperedgeConstraintType.PropertyValue,
            PropertyConstraints = new Dictionary<string, object>
            {
                ["status"] = "active",
                ["minScore"] = 50
            }
        };

        // Assert
        constraint.PropertyConstraints.Should().ContainKey("status");
    }

    #endregion

    #region HyperedgeMergeResult Tests

    [Fact]
    public void HyperedgeMergeResult_Success()
    {
        // Arrange & Act
        var result = new HyperedgeMergeResult
        {
            Success = true,
            NewCardinality = 15,
            MembersAdded = 8,
            NewVersion = 10,
            TimestampNanos = 14000000
        };

        // Assert
        result.Success.Should().BeTrue();
        result.NewCardinality.Should().Be(15);
        result.MembersAdded.Should().Be(8);
        result.NewVersion.Should().Be(10);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public void HyperedgeMergeResult_Failure_WouldExceedMaxCardinality()
    {
        // Arrange & Act
        var result = new HyperedgeMergeResult
        {
            Success = false,
            NewCardinality = 50,
            MembersAdded = 0,
            NewVersion = 5,
            TimestampNanos = 15000000,
            ErrorMessage = "Merge would exceed maximum cardinality (50)"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.MembersAdded.Should().Be(0);
        result.ErrorMessage.Should().Contain("maximum cardinality");
    }

    #endregion

    #region HyperedgeSplitPredicate Tests

    [Fact]
    public void HyperedgeSplitPredicate_ByRole()
    {
        // Arrange & Act
        var predicate = new HyperedgeSplitPredicate
        {
            SplitRoles = new List<string> { "observer", "inactive" }
        };

        // Assert
        predicate.SplitRoles.Should().HaveCount(2);
        predicate.SplitPropertyName.Should().BeNull();
        predicate.SplitThreshold.Should().BeNull();
    }

    [Fact]
    public void HyperedgeSplitPredicate_ByPropertyThreshold()
    {
        // Arrange & Act
        var predicate = new HyperedgeSplitPredicate
        {
            SplitPropertyName = "score",
            SplitThreshold = 50.0
        };

        // Assert
        predicate.SplitPropertyName.Should().Be("score");
        predicate.SplitThreshold.Should().Be(50.0);
    }

    #endregion

    #region HyperedgeSplitResult Tests

    [Fact]
    public void HyperedgeSplitResult_Success()
    {
        // Arrange & Act
        var result = new HyperedgeSplitResult
        {
            Success = true,
            NewHyperedgeId = "he-split-001",
            MembersMoved = 5,
            MembersRemaining = 10,
            TimestampNanos = 16000000
        };

        // Assert
        result.Success.Should().BeTrue();
        result.NewHyperedgeId.Should().Be("he-split-001");
        result.MembersMoved.Should().Be(5);
        result.MembersRemaining.Should().Be(10);
        result.ErrorMessage.Should().BeNull();
    }

    [Fact]
    public void HyperedgeSplitResult_Failure_WouldViolateMinCardinality()
    {
        // Arrange & Act
        var result = new HyperedgeSplitResult
        {
            Success = false,
            NewHyperedgeId = "",
            MembersMoved = 0,
            MembersRemaining = 3,
            TimestampNanos = 17000000,
            ErrorMessage = "Split would leave original hyperedge below minimum cardinality"
        };

        // Assert
        result.Success.Should().BeFalse();
        result.ErrorMessage.Should().Contain("minimum cardinality");
    }

    #endregion

    #region HyperedgeHistory Tests

    [Fact]
    public void HyperedgeHistory_ShouldContainEvents()
    {
        // Arrange
        var events = new List<HyperedgeHistoryEvent>
        {
            new HyperedgeHistoryEvent
            {
                EventType = HyperedgeEventType.Created,
                TimestampNanos = 1000,
                HlcTimestamp = 1000,
                Version = 1,
                Details = new Dictionary<string, object> { ["initialMembers"] = 3 }
            },
            new HyperedgeHistoryEvent
            {
                EventType = HyperedgeEventType.VertexAdded,
                TimestampNanos = 2000,
                HlcTimestamp = 2000,
                Version = 2,
                VertexId = "v-new"
            },
            new HyperedgeHistoryEvent
            {
                EventType = HyperedgeEventType.PropertiesUpdated,
                TimestampNanos = 3000,
                HlcTimestamp = 3000,
                Version = 3,
                Details = new Dictionary<string, object> { ["changedProps"] = new[] { "title" } }
            }
        };

        // Act
        var history = new HyperedgeHistory
        {
            HyperedgeId = "he-history-test",
            Events = events,
            IsComplete = true
        };

        // Assert
        history.HyperedgeId.Should().Be("he-history-test");
        history.Events.Should().HaveCount(3);
        history.IsComplete.Should().BeTrue();
    }

    [Fact]
    public void HyperedgeHistory_TruncatedHistory()
    {
        // Arrange & Act
        var history = new HyperedgeHistory
        {
            HyperedgeId = "he-truncated",
            Events = Enumerable.Range(0, 100)
                .Select(i => new HyperedgeHistoryEvent
                {
                    EventType = HyperedgeEventType.MessageBroadcast,
                    TimestampNanos = i * 1000,
                    HlcTimestamp = i * 1000,
                    Version = i + 1
                }).ToList(),
            IsComplete = false
        };

        // Assert
        history.IsComplete.Should().BeFalse();
        history.Events.Should().HaveCount(100);
    }

    #endregion

    #region HyperedgeHistoryEvent Tests

    [Theory]
    [InlineData(HyperedgeEventType.Created)]
    [InlineData(HyperedgeEventType.VertexAdded)]
    [InlineData(HyperedgeEventType.VertexRemoved)]
    [InlineData(HyperedgeEventType.PropertiesUpdated)]
    [InlineData(HyperedgeEventType.MessageBroadcast)]
    [InlineData(HyperedgeEventType.Merged)]
    [InlineData(HyperedgeEventType.Split)]
    public void HyperedgeHistoryEvent_AllEventTypesSupported(HyperedgeEventType eventType)
    {
        // Arrange & Act
        var evt = new HyperedgeHistoryEvent
        {
            EventType = eventType,
            TimestampNanos = 18000000,
            HlcTimestamp = 18000000,
            Version = 1
        };

        // Assert
        evt.EventType.Should().Be(eventType);
    }

    [Fact]
    public void HyperedgeHistoryEvent_MembershipEvent_ShouldHaveVertexId()
    {
        // Arrange & Act
        var addEvent = new HyperedgeHistoryEvent
        {
            EventType = HyperedgeEventType.VertexAdded,
            TimestampNanos = 19000000,
            HlcTimestamp = 19000000,
            Version = 5,
            VertexId = "v-joined"
        };

        var removeEvent = new HyperedgeHistoryEvent
        {
            EventType = HyperedgeEventType.VertexRemoved,
            TimestampNanos = 20000000,
            HlcTimestamp = 20000000,
            Version = 6,
            VertexId = "v-left"
        };

        // Assert
        addEvent.VertexId.Should().Be("v-joined");
        removeEvent.VertexId.Should().Be("v-left");
    }

    #endregion

    #region HyperedgeMetrics Tests

    [Fact]
    public void HyperedgeMetrics_ShouldTrackAllPerformanceIndicators()
    {
        // Arrange & Act
        var metrics = new HyperedgeMetrics
        {
            HyperedgeId = "he-metrics-test",
            Cardinality = 10,
            MessagesBroadcast = 500,
            MembershipChanges = 25,
            AvgBroadcastLatencyNanos = 150000.5,
            GpuMemoryBytes = 8192,
            AggregationsPerformed = 100
        };

        // Assert
        metrics.HyperedgeId.Should().Be("he-metrics-test");
        metrics.Cardinality.Should().Be(10);
        metrics.MessagesBroadcast.Should().Be(500);
        metrics.MembershipChanges.Should().Be(25);
        metrics.AvgBroadcastLatencyNanos.Should().Be(150000.5);
        metrics.GpuMemoryBytes.Should().Be(8192);
        metrics.AggregationsPerformed.Should().Be(100);
    }

    #endregion

    #region Integration Scenario Tests

    [Fact]
    public void CompleteWorkflow_CreateHyperedgeBroadcastMessage()
    {
        // This test simulates a complete workflow scenario using DTOs

        // 1. Initialize hyperedge (Meeting)
        var initRequest = new HyperedgeInitRequest
        {
            HyperedgeType = "Meeting",
            Properties = new Dictionary<string, object>
            {
                ["title"] = "Weekly Standup",
                ["room"] = "Conference A"
            },
            InitialMembers = new List<HyperedgeMemberInit>
            {
                new HyperedgeMemberInit { VertexId = "user-alice", Role = "organizer" },
                new HyperedgeMemberInit { VertexId = "user-bob", Role = "attendee" },
                new HyperedgeMemberInit { VertexId = "user-charlie", Role = "attendee" }
            },
            MinCardinality = 2,
            MaxCardinality = 20
        };

        var initResult = new HyperedgeInitResult
        {
            Success = true,
            HyperedgeId = "meeting-weekly-standup",
            Version = 1,
            MemberCount = 3,
            CreatedAtNanos = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000
        };

        // 2. Add another attendee
        var addResult = new HyperedgeMutationResult
        {
            Success = true,
            Operation = "Add",
            VertexId = "user-diana",
            NewVersion = 2,
            CurrentCardinality = 4,
            TimestampNanos = initResult.CreatedAtNanos + 1000000
        };

        // 3. Broadcast message to all attendees
        var message = new HyperedgeMessage
        {
            MessageId = Guid.NewGuid().ToString(),
            MessageType = "MeetingNotification",
            SourceVertexId = "user-alice",
            Payload = new Dictionary<string, object>
            {
                ["content"] = "Meeting starting in 5 minutes",
                ["urgent"] = true
            },
            HlcTimestamp = addResult.TimestampNanos + 1000000
        };

        var broadcastResult = new HyperedgeBroadcastResult
        {
            Success = true,
            TargetCount = 3, // 4 members - 1 sender
            DeliveredCount = 3,
            FailedCount = 0,
            TotalLatencyNanos = 300000
        };

        // Assertions
        initResult.Success.Should().BeTrue();
        initResult.MemberCount.Should().Be(3);
        addResult.Success.Should().BeTrue();
        addResult.CurrentCardinality.Should().Be(4);
        broadcastResult.DeliveredCount.Should().Be(broadcastResult.TargetCount);
        broadcastResult.AvgLatencyNanos.Should().Be(100000); // 300000 / 3
    }

    [Fact]
    public void CompleteWorkflow_SplitAndMergeHyperedges()
    {
        // Simulate splitting a large team hyperedge and merging it back

        // 1. Original hyperedge with 20 members
        var originalState = new HyperedgeState
        {
            HyperedgeId = "team-large",
            HyperedgeType = "Team",
            Version = 5,
            Properties = new Dictionary<string, object> { ["department"] = "Engineering" },
            Members = Enumerable.Range(0, 20)
                .Select(i => new HyperedgeMember
                {
                    VertexId = $"user-{i}",
                    Role = i < 3 ? "lead" : "member",
                    JoinedAtNanos = i * 1000000
                }).ToList(),
            MinCardinality = 2,
            MaxCardinality = 50,
            IsDirected = false,
            CreatedAtNanos = 0,
            ModifiedAtNanos = 5000000,
            HlcTimestamp = 5000000
        };

        // 2. Split by role (move all non-leads to new hyperedge)
        var splitPredicate = new HyperedgeSplitPredicate
        {
            SplitRoles = new List<string> { "member" }
        };

        var splitResult = new HyperedgeSplitResult
        {
            Success = true,
            NewHyperedgeId = "team-large-members",
            MembersMoved = 17, // 20 - 3 leads
            MembersRemaining = 3, // leads
            TimestampNanos = 6000000
        };

        // 3. Merge them back together
        var mergeResult = new HyperedgeMergeResult
        {
            Success = true,
            NewCardinality = 20,
            MembersAdded = 17,
            NewVersion = 7,
            TimestampNanos = 7000000
        };

        // Assertions
        originalState.Cardinality.Should().Be(20);
        splitResult.MembersMoved.Should().Be(17);
        splitResult.MembersRemaining.Should().Be(3);
        mergeResult.NewCardinality.Should().Be(20);
    }

    [Fact]
    public void ConstraintValidation_MeetingRequiresOrganizer()
    {
        // A meeting hyperedge must have at least one organizer role
        var constraint = new HyperedgeConstraint
        {
            Type = HyperedgeConstraintType.RolePresence,
            RequiredRoles = new List<string> { "organizer" }
        };

        // Valid meeting state
        var validState = new HyperedgeState
        {
            HyperedgeId = "meeting-valid",
            HyperedgeType = "Meeting",
            Version = 1,
            Properties = new Dictionary<string, object>(),
            Members = new List<HyperedgeMember>
            {
                new HyperedgeMember { VertexId = "u1", Role = "organizer", JoinedAtNanos = 0 },
                new HyperedgeMember { VertexId = "u2", Role = "attendee", JoinedAtNanos = 1000 }
            },
            MinCardinality = 2,
            MaxCardinality = 50,
            IsDirected = false,
            CreatedAtNanos = 0,
            ModifiedAtNanos = 0,
            HlcTimestamp = 0
        };

        // Check constraint satisfaction
        var hasOrganizer = validState.Members.Any(m => m.Role == "organizer");
        hasOrganizer.Should().BeTrue();

        // Invalid meeting state (no organizer)
        var invalidState = new HyperedgeState
        {
            HyperedgeId = "meeting-invalid",
            HyperedgeType = "Meeting",
            Version = 1,
            Properties = new Dictionary<string, object>(),
            Members = new List<HyperedgeMember>
            {
                new HyperedgeMember { VertexId = "u1", Role = "attendee", JoinedAtNanos = 0 },
                new HyperedgeMember { VertexId = "u2", Role = "attendee", JoinedAtNanos = 1000 }
            },
            MinCardinality = 2,
            MaxCardinality = 50,
            IsDirected = false,
            CreatedAtNanos = 0,
            ModifiedAtNanos = 0,
            HlcTimestamp = 0
        };

        var invalidHasOrganizer = invalidState.Members.Any(m => m.Role == "organizer");
        invalidHasOrganizer.Should().BeFalse();
    }

    #endregion

    #region Temporal Ordering Tests

    [Fact]
    public void HyperedgeHistory_EventsShouldBeTemporallyOrdered()
    {
        // Events should maintain causal ordering via HLC timestamps
        var events = new List<HyperedgeHistoryEvent>
        {
            new HyperedgeHistoryEvent
            {
                EventType = HyperedgeEventType.Created,
                TimestampNanos = 1000000,
                HlcTimestamp = 1000000,
                Version = 1
            },
            new HyperedgeHistoryEvent
            {
                EventType = HyperedgeEventType.VertexAdded,
                TimestampNanos = 2000000,
                HlcTimestamp = 2000000,
                Version = 2,
                VertexId = "v1"
            },
            new HyperedgeHistoryEvent
            {
                EventType = HyperedgeEventType.VertexAdded,
                TimestampNanos = 2000000, // Same physical time
                HlcTimestamp = 2000001,   // But different HLC (logical counter incremented)
                Version = 3,
                VertexId = "v2"
            }
        };

        // HLC timestamps should be monotonically increasing
        for (int i = 1; i < events.Count; i++)
        {
            events[i].HlcTimestamp.Should().BeGreaterThan(events[i - 1].HlcTimestamp);
        }

        // Versions should be monotonically increasing
        for (int i = 1; i < events.Count; i++)
        {
            events[i].Version.Should().BeGreaterThan(events[i - 1].Version);
        }
    }

    #endregion
}
