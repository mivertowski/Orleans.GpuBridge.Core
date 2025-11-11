using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Temporal.Tests.Unit;

/// <summary>
/// Unit tests for Hybrid Logical Clock (HLC) timestamp implementation.
/// </summary>
public sealed class HybridTimestampTests
{
    [Fact]
    public void HybridTimestamp_CreatesWithCorrectValues()
    {
        // Arrange
        long physicalTime = 1000000000L;
        long logicalCounter = 42;
        ushort nodeId = 1;

        // Act
        var timestamp = new HybridTimestamp(physicalTime, logicalCounter, nodeId);

        // Assert
        timestamp.PhysicalTime.Should().Be(physicalTime);
        timestamp.LogicalCounter.Should().Be(logicalCounter);
        timestamp.NodeId.Should().Be(nodeId);
    }

    [Fact]
    public void HybridTimestamp_NowCreatesTimestampWithCurrentTime()
    {
        // Arrange
        long beforeNanos = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();

        // Act
        var timestamp = HybridTimestamp.Now();

        // Assert
        long afterNanos = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        timestamp.PhysicalTime.Should().BeGreaterThanOrEqualTo(beforeNanos);
        timestamp.PhysicalTime.Should().BeLessThanOrEqualTo(afterNanos);
        timestamp.LogicalCounter.Should().Be(0);
    }

    [Fact]
    public void HybridTimestamp_UpdateWithReceivedTimestampAhead()
    {
        // Arrange
        var local = new HybridTimestamp(1000, 5, 1);
        var received = new HybridTimestamp(2000, 10, 2); // Received time is ahead
        long physicalTime = 1500;

        // Act
        var updated = HybridTimestamp.Update(local, received, physicalTime);

        // Assert
        updated.PhysicalTime.Should().Be(2000); // Max of all three
        updated.LogicalCounter.Should().Be(11);  // received.Logical + 1
        updated.NodeId.Should().Be(1);           // Keeps local node ID
    }

    [Fact]
    public void HybridTimestamp_UpdateWithLocalTimestampAhead()
    {
        // Arrange
        var local = new HybridTimestamp(3000, 5, 1);
        var received = new HybridTimestamp(2000, 10, 2);
        long physicalTime = 2500;

        // Act
        var updated = HybridTimestamp.Update(local, received, physicalTime);

        // Assert
        updated.PhysicalTime.Should().Be(3000); // Max = local
        updated.LogicalCounter.Should().Be(6);   // local.Logical + 1
    }

    [Fact]
    public void HybridTimestamp_UpdateWithPhysicalTimeAhead()
    {
        // Arrange
        var local = new HybridTimestamp(1000, 5, 1);
        var received = new HybridTimestamp(2000, 10, 2);
        long physicalTime = 3000; // Physical time is ahead of both

        // Act
        var updated = HybridTimestamp.Update(local, received, physicalTime);

        // Assert
        updated.PhysicalTime.Should().Be(3000);
        updated.LogicalCounter.Should().Be(0); // Reset logical counter
    }

    [Fact]
    public void HybridTimestamp_UpdateWithSamePhysicalTime()
    {
        // Arrange
        var local = new HybridTimestamp(2000, 5, 1);
        var received = new HybridTimestamp(2000, 8, 2); // Same physical time
        long physicalTime = 2000;

        // Act
        var updated = HybridTimestamp.Update(local, received, physicalTime);

        // Assert
        updated.PhysicalTime.Should().Be(2000);
        updated.LogicalCounter.Should().Be(9); // Max(5, 8) + 1
    }

    [Fact]
    public void HybridTimestamp_IncrementWithAdvancedPhysicalTime()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 5, 1);
        long newPhysicalTime = 2000;

        // Act
        var incremented = timestamp.Increment(newPhysicalTime);

        // Assert
        incremented.PhysicalTime.Should().Be(2000);
        incremented.LogicalCounter.Should().Be(0); // Reset when physical time advances
    }

    [Fact]
    public void HybridTimestamp_IncrementWithSamePhysicalTime()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 5, 1);
        long samePhysicalTime = 1000;

        // Act
        var incremented = timestamp.Increment(samePhysicalTime);

        // Assert
        incremented.PhysicalTime.Should().Be(1000);
        incremented.LogicalCounter.Should().Be(6); // Increment logical counter
    }

    [Fact]
    public void HybridTimestamp_CompareToReturnCorrectOrdering()
    {
        // Arrange
        var earlier = new HybridTimestamp(1000, 5, 1);
        var later = new HybridTimestamp(2000, 3, 2);

        // Act & Assert
        earlier.CompareTo(later).Should().BeLessThan(0);
        later.CompareTo(earlier).Should().BeGreaterThan(0);
        earlier.CompareTo(earlier).Should().Be(0);
    }

    [Fact]
    public void HybridTimestamp_CompareToUsesLogicalCounterForTieBreaking()
    {
        // Arrange
        var lower = new HybridTimestamp(1000, 5, 1);
        var higher = new HybridTimestamp(1000, 10, 1); // Same physical, higher logical

        // Act & Assert
        lower.CompareTo(higher).Should().BeLessThan(0);
        higher.CompareTo(lower).Should().BeGreaterThan(0);
    }

    [Fact]
    public void HybridTimestamp_CompareToUsesNodeIdForFinalTieBreaking()
    {
        // Arrange
        var node1 = new HybridTimestamp(1000, 5, 1);
        var node2 = new HybridTimestamp(1000, 5, 2); // Same physical and logical

        // Act & Assert
        node1.CompareTo(node2).Should().BeLessThan(0);
        node2.CompareTo(node1).Should().BeGreaterThan(0);
    }

    [Fact]
    public void HybridTimestamp_EqualityWorks()
    {
        // Arrange
        var ts1 = new HybridTimestamp(1000, 5, 1);
        var ts2 = new HybridTimestamp(1000, 5, 1);
        var ts3 = new HybridTimestamp(2000, 5, 1);

        // Act & Assert
        ts1.Equals(ts2).Should().BeTrue();
        (ts1 == ts2).Should().BeTrue();
        ts1.Equals(ts3).Should().BeFalse();
        (ts1 != ts3).Should().BeTrue();
    }

    [Fact]
    public void HybridTimestamp_ComparisonOperatorsWork()
    {
        // Arrange
        var earlier = new HybridTimestamp(1000, 0, 0);
        var later = new HybridTimestamp(2000, 0, 0);

        // Act & Assert
        (earlier < later).Should().BeTrue();
        (earlier <= later).Should().BeTrue();
        (later > earlier).Should().BeTrue();
        (later >= earlier).Should().BeTrue();
    }

    [Fact]
    public void HybridTimestamp_MaintainsMonotonicityOverManyUpdates()
    {
        // Arrange
        var timestamp = HybridTimestamp.Now();
        const int iterations = 1000;

        // Act & Assert
        for (int i = 0; i < iterations; i++)
        {
            long physicalTime = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
            var received = HybridTimestamp.Now();

            var newTimestamp = HybridTimestamp.Update(timestamp, received, physicalTime);

            // Assert monotonicity
            (newTimestamp >= timestamp).Should().BeTrue(
                $"Timestamp should be monotonically increasing at iteration {i}");

            timestamp = newTimestamp;
        }
    }

    [Fact]
    public void HybridTimestamp_PreservesCausalOrderingInDistributedScenario()
    {
        // Simulate distributed scenario with 3 actors
        var actorA = HybridTimestamp.Now(nodeId: 1);
        var actorB = HybridTimestamp.Now(nodeId: 2);
        var actorC = HybridTimestamp.Now(nodeId: 3);

        // Actor A sends message to B
        var msgAtoB = actorA.Increment(DateTimeOffset.UtcNow.ToUnixTimeNanoseconds());
        actorB = HybridTimestamp.Update(actorB, msgAtoB, DateTimeOffset.UtcNow.ToUnixTimeNanoseconds());

        // Actor B sends message to C (causal chain: A → B → C)
        var msgBtoC = actorB.Increment(DateTimeOffset.UtcNow.ToUnixTimeNanoseconds());
        actorC = HybridTimestamp.Update(actorC, msgBtoC, DateTimeOffset.UtcNow.ToUnixTimeNanoseconds());

        // Assert causal ordering: A → B → C
        (msgAtoB < msgBtoC).Should().BeTrue("Message A→B should happen-before B→C");
        (msgAtoB < actorC).Should().BeTrue("Actor A's event should happen-before Actor C");
        (actorB < actorC).Should().BeTrue("Actor B should happen-before Actor C");
    }
}
