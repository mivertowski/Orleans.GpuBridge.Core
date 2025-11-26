using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Abstractions.Tests.Temporal;

/// <summary>
/// Edge case tests for HybridTimestamp to ensure correctness under extreme conditions.
/// </summary>
public sealed class HybridTimestampEdgeCaseTests
{
    #region Overflow and Boundary Tests

    /// <summary>
    /// Tests that logical counter handles near-overflow values correctly.
    /// </summary>
    [Fact]
    public void LogicalCounter_NearMaxValue_IncrementsSafely()
    {
        // Arrange
        const long physicalTime = 1000000000L; // 1 second
        var timestamp = new HybridTimestamp(physicalTime, long.MaxValue - 10, nodeId: 1);

        // Act - Increment multiple times
        var incremented = timestamp.Increment(physicalTime);

        // Assert - Should increment without overflow
        incremented.LogicalCounter.Should().Be(long.MaxValue - 9);
    }

    /// <summary>
    /// Tests behavior when physical time is at maximum value.
    /// </summary>
    [Fact]
    public void PhysicalTime_AtMaxValue_ComparesCorrectly()
    {
        // Arrange
        var maxTimestamp = new HybridTimestamp(long.MaxValue, 0, nodeId: 1);
        var nearMaxTimestamp = new HybridTimestamp(long.MaxValue - 1, long.MaxValue, nodeId: 1);

        // Act & Assert - MaxValue physical time should be greater
        maxTimestamp.Should().BeGreaterThan(nearMaxTimestamp);
    }

    /// <summary>
    /// Tests that zero values are handled correctly.
    /// </summary>
    [Fact]
    public void ZeroTimestamp_IsValidMinimum()
    {
        // Arrange
        var zero = new HybridTimestamp(0, 0, nodeId: 0);
        var positive = new HybridTimestamp(1, 0, nodeId: 0);

        // Act & Assert
        zero.Should().BeLessThan(positive);
        zero.PhysicalTime.Should().Be(0);
        zero.LogicalCounter.Should().Be(0);
    }

    /// <summary>
    /// Tests negative physical time (before Unix epoch - shouldn't happen but test anyway).
    /// </summary>
    [Fact]
    public void NegativePhysicalTime_ComparesCorrectly()
    {
        // Arrange
        var negative = new HybridTimestamp(-1000, 5, nodeId: 1);
        var zero = new HybridTimestamp(0, 0, nodeId: 1);

        // Act & Assert - Negative should be less than zero
        negative.Should().BeLessThan(zero);
    }

    #endregion

    #region Update Algorithm Tests

    /// <summary>
    /// Tests Update when local and received have identical timestamps.
    /// </summary>
    [Fact]
    public void Update_IdenticalTimestamps_IncrementsLogicalCounter()
    {
        // Arrange
        var local = new HybridTimestamp(1000, 5, nodeId: 1);
        var received = new HybridTimestamp(1000, 5, nodeId: 2);

        // Act
        var result = HybridTimestamp.Update(local, received, physicalTime: 1000);

        // Assert - Should increment the max logical counter
        result.PhysicalTime.Should().Be(1000);
        result.LogicalCounter.Should().Be(6); // max(5, 5) + 1
        result.NodeId.Should().Be(1); // Preserves local node ID
    }

    /// <summary>
    /// Tests Update when physical time jumps forward significantly.
    /// </summary>
    [Fact]
    public void Update_PhysicalTimeJump_ResetsLogicalCounter()
    {
        // Arrange
        var local = new HybridTimestamp(1000, 100, nodeId: 1);
        var received = new HybridTimestamp(1000, 200, nodeId: 2);

        // Act - Physical time jumps ahead
        var result = HybridTimestamp.Update(local, received, physicalTime: 2000);

        // Assert - Logical counter should reset to 0
        result.PhysicalTime.Should().Be(2000);
        result.LogicalCounter.Should().Be(0);
    }

    /// <summary>
    /// Tests Update when received timestamp is significantly ahead.
    /// </summary>
    [Fact]
    public void Update_ReceivedAhead_TakesReceivedPhysicalTime()
    {
        // Arrange
        var local = new HybridTimestamp(1000, 5, nodeId: 1);
        var received = new HybridTimestamp(5000, 10, nodeId: 2);

        // Act
        var result = HybridTimestamp.Update(local, received, physicalTime: 1500);

        // Assert - Should use received physical time and increment its counter
        result.PhysicalTime.Should().Be(5000);
        result.LogicalCounter.Should().Be(11);
    }

    /// <summary>
    /// Tests Update when local timestamp is ahead of received.
    /// </summary>
    [Fact]
    public void Update_LocalAhead_TakesLocalPhysicalTime()
    {
        // Arrange
        var local = new HybridTimestamp(5000, 15, nodeId: 1);
        var received = new HybridTimestamp(1000, 100, nodeId: 2);

        // Act
        var result = HybridTimestamp.Update(local, received, physicalTime: 3000);

        // Assert - Should use local physical time and increment its counter
        result.PhysicalTime.Should().Be(5000);
        result.LogicalCounter.Should().Be(16);
    }

    #endregion

    #region Increment Tests

    /// <summary>
    /// Tests Increment when physical time hasn't changed.
    /// </summary>
    [Fact]
    public void Increment_SamePhysicalTime_IncrementsLogicalCounter()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 5, nodeId: 1);

        // Act
        var result = timestamp.Increment(physicalTime: 1000);

        // Assert
        result.PhysicalTime.Should().Be(1000);
        result.LogicalCounter.Should().Be(6);
        result.NodeId.Should().Be(1);
    }

    /// <summary>
    /// Tests Increment when physical time goes backward (shouldn't reset).
    /// </summary>
    [Fact]
    public void Increment_PhysicalTimeBackward_KeepsCurrentPhysicalTime()
    {
        // Arrange
        var timestamp = new HybridTimestamp(2000, 5, nodeId: 1);

        // Act - Physical time goes backward (clock drift)
        var result = timestamp.Increment(physicalTime: 1000);

        // Assert - Should keep the higher physical time and increment counter
        result.PhysicalTime.Should().Be(2000);
        result.LogicalCounter.Should().Be(6);
    }

    /// <summary>
    /// Tests Increment when physical time advances.
    /// </summary>
    [Fact]
    public void Increment_PhysicalTimeForward_ResetsLogicalCounter()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 100, nodeId: 1);

        // Act
        var result = timestamp.Increment(physicalTime: 2000);

        // Assert
        result.PhysicalTime.Should().Be(2000);
        result.LogicalCounter.Should().Be(0); // Reset when time advances
    }

    #endregion

    #region Comparison Tests

    /// <summary>
    /// Tests total ordering with node ID tie-breaking.
    /// </summary>
    [Fact]
    public void CompareTo_SameTimeAndCounter_UsesNodeIdForOrdering()
    {
        // Arrange
        var timestamp1 = new HybridTimestamp(1000, 5, nodeId: 1);
        var timestamp2 = new HybridTimestamp(1000, 5, nodeId: 2);
        var timestamp3 = new HybridTimestamp(1000, 5, nodeId: 1);

        // Act & Assert
        timestamp1.Should().BeLessThan(timestamp2);
        timestamp1.Should().BeEquivalentTo(timestamp3);
        (timestamp1 == timestamp3).Should().BeTrue();
    }

    /// <summary>
    /// Tests comparison chain for transitivity.
    /// </summary>
    [Fact]
    public void CompareTo_IsTransitive()
    {
        // Arrange
        var a = new HybridTimestamp(1000, 0, nodeId: 0);
        var b = new HybridTimestamp(1000, 5, nodeId: 0);
        var c = new HybridTimestamp(1000, 10, nodeId: 0);

        // Act & Assert - If a < b and b < c, then a < c
        a.Should().BeLessThan(b);
        b.Should().BeLessThan(c);
        a.Should().BeLessThan(c);
    }

    /// <summary>
    /// Tests comparison operators.
    /// </summary>
    [Fact]
    public void ComparisonOperators_AreConsistent()
    {
        // Arrange
        var earlier = new HybridTimestamp(1000, 5, nodeId: 1);
        var later = new HybridTimestamp(2000, 0, nodeId: 1);
        var same = new HybridTimestamp(1000, 5, nodeId: 1);

        // Act & Assert
        (earlier < later).Should().BeTrue();
        (earlier <= later).Should().BeTrue();
        (later > earlier).Should().BeTrue();
        (later >= earlier).Should().BeTrue();
        (earlier == same).Should().BeTrue();
        (earlier != later).Should().BeTrue();
        (earlier <= same).Should().BeTrue();
        (earlier >= same).Should().BeTrue();
    }

    #endregion

    #region Equality and Hash Code Tests

    /// <summary>
    /// Tests that equal timestamps have equal hash codes.
    /// </summary>
    [Fact]
    public void GetHashCode_EqualTimestamps_HaveEqualHashCodes()
    {
        // Arrange
        var timestamp1 = new HybridTimestamp(1000, 5, nodeId: 1);
        var timestamp2 = new HybridTimestamp(1000, 5, nodeId: 1);

        // Act & Assert
        timestamp1.GetHashCode().Should().Be(timestamp2.GetHashCode());
    }

    /// <summary>
    /// Tests hash code distribution for different timestamps.
    /// </summary>
    [Fact]
    public void GetHashCode_DifferentTimestamps_HaveDifferentHashCodes()
    {
        // Arrange
        var hashCodes = new HashSet<int>();

        // Act - Generate many timestamps
        for (int i = 0; i < 1000; i++)
        {
            var timestamp = new HybridTimestamp(i * 1000, i, nodeId: (ushort)(i % 256));
            hashCodes.Add(timestamp.GetHashCode());
        }

        // Assert - Should have high collision resistance (not all same)
        hashCodes.Count.Should().BeGreaterThan(900); // Allow some collisions
    }

    /// <summary>
    /// Tests Equals with null object.
    /// </summary>
    [Fact]
    public void Equals_WithNullObject_ReturnsFalse()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 5, nodeId: 1);

        // Act & Assert
        timestamp.Equals(null).Should().BeFalse();
    }

    /// <summary>
    /// Tests Equals with different type.
    /// </summary>
    [Fact]
    public void Equals_WithDifferentType_ReturnsFalse()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 5, nodeId: 1);

        // Act & Assert
        timestamp.Equals("not a timestamp").Should().BeFalse();
        timestamp.Equals(1000L).Should().BeFalse();
    }

    #endregion

    #region ToString Tests

    /// <summary>
    /// Tests ToString format.
    /// </summary>
    [Fact]
    public void ToString_ReturnsExpectedFormat()
    {
        // Arrange
        var timestamp = new HybridTimestamp(1000, 5, nodeId: 42);

        // Act
        var result = timestamp.ToString();

        // Assert
        result.Should().Be("HLC(1000, 5, Node=42)");
    }

    #endregion

    #region Now Factory Tests

    /// <summary>
    /// Tests that Now creates monotonically increasing timestamps.
    /// </summary>
    [Fact]
    public void Now_CalledMultipleTimes_ReturnsIncreasingTimestamps()
    {
        // Arrange & Act
        var timestamps = new List<HybridTimestamp>();
        for (int i = 0; i < 100; i++)
        {
            timestamps.Add(HybridTimestamp.Now(nodeId: 1));
        }

        // Assert - Physical time should be non-decreasing
        for (int i = 1; i < timestamps.Count; i++)
        {
            timestamps[i].PhysicalTime.Should().BeGreaterThanOrEqualTo(timestamps[i - 1].PhysicalTime);
        }
    }

    /// <summary>
    /// Tests Now with different node IDs.
    /// </summary>
    [Fact]
    public void Now_WithNodeId_SetsNodeIdCorrectly()
    {
        // Arrange & Act
        var timestamp1 = HybridTimestamp.Now(nodeId: 0);
        var timestamp2 = HybridTimestamp.Now(nodeId: 100);
        var timestamp3 = HybridTimestamp.Now(nodeId: ushort.MaxValue);

        // Assert
        timestamp1.NodeId.Should().Be(0);
        timestamp2.NodeId.Should().Be(100);
        timestamp3.NodeId.Should().Be(ushort.MaxValue);
    }

    #endregion
}
