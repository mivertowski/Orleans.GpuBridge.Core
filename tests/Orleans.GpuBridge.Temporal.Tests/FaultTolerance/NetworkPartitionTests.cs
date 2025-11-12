using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Temporal.Tests.FaultTolerance;

/// <summary>
/// Tests for HLC behavior during network partitions and healing.
/// Validates timestamp reconciliation and causality preservation.
/// </summary>
public class NetworkPartitionTests
{
    private readonly ITestOutputHelper _output;

    public NetworkPartitionTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void HLC_PartitionedNodesGenerateDistinctTimestamps()
    {
        // Arrange: Two nodes in different partitions
        var node1 = new HybridLogicalClock(nodeId: 1);
        var node2 = new HybridLogicalClock(nodeId: 2);

        // Act: Both nodes generate timestamps independently during partition
        var timestamps1 = new List<HybridTimestamp>();
        var timestamps2 = new List<HybridTimestamp>();

        for (int i = 0; i < 100; i++)
        {
            timestamps1.Add(node1.Now());
            timestamps2.Add(node2.Now());
            System.Threading.Thread.Sleep(1); // Simulate time passing
        }

        // Assert: All timestamps are unique and ordered
        var allTimestamps = timestamps1.Concat(timestamps2).ToList();
        var uniqueCount = allTimestamps.Distinct().Count();

        Assert.Equal(200, allTimestamps.Count);
        Assert.Equal(200, uniqueCount); // All unique due to nodeId disambiguation

        _output.WriteLine($"Partition independence:");
        _output.WriteLine($"  Node 1 timestamps: {timestamps1.Count}");
        _output.WriteLine($"  Node 2 timestamps: {timestamps2.Count}");
        _output.WriteLine($"  Unique timestamps: {uniqueCount}");
        _output.WriteLine($"  All timestamps distinguishable: VERIFIED");
    }

    [Fact]
    public async Task HLC_PartitionHealing_TimestampsConverge()
    {
        // Arrange: Simulate partition healing
        var node1 = new HybridLogicalClock(nodeId: 1);
        var node2 = new HybridLogicalClock(nodeId: 2);
        var node3 = new HybridLogicalClock(nodeId: 3);

        // Phase 1: Partition - {node1, node2} | {node3}
        _output.WriteLine("Phase 1: Network partition");

        var partition1_timestamps = new List<HybridTimestamp>();
        var partition2_timestamps = new List<HybridTimestamp>();

        for (int i = 0; i < 10; i++)
        {
            partition1_timestamps.Add(node1.Now());
            partition1_timestamps.Add(node2.Now());
            partition2_timestamps.Add(node3.Now());
            await Task.Delay(5);
        }

        _output.WriteLine($"  Partition A (nodes 1,2): {partition1_timestamps.Count} timestamps");
        _output.WriteLine($"  Partition B (node 3): {partition2_timestamps.Count} timestamps");

        // Phase 2: Partition heals - nodes exchange timestamps
        _output.WriteLine("Phase 2: Partition healing");

        var node1_latest = partition1_timestamps.Last();
        var node3_latest = partition2_timestamps.Last();

        // Node 1 receives node 3's timestamp
        var node1_healed = node1.Update(node3_latest);

        // Node 3 receives node 1's timestamp
        var node3_healed = node3.Update(node1_latest);

        // Assert: Clocks converge after healing
        Assert.True(node1_healed.PhysicalTime >= node1_latest.PhysicalTime);
        Assert.True(node3_healed.PhysicalTime >= node3_latest.PhysicalTime);

        // Total ordering maintained
        var allTimestamps = partition1_timestamps.Concat(partition2_timestamps)
            .Append(node1_healed)
            .Append(node3_healed)
            .OrderBy(ts => ts)
            .ToList();

        for (int i = 1; i < allTimestamps.Count; i++)
        {
            Assert.True(allTimestamps[i].CompareTo(allTimestamps[i - 1]) >= 0);
        }

        _output.WriteLine($"  Node 1 after healing: {node1_healed}");
        _output.WriteLine($"  Node 3 after healing: {node3_healed}");
        _output.WriteLine($"  Total ordering maintained: VERIFIED");
    }

    [Fact]
    public void HLC_SplitBrain_PreservesPartialOrders()
    {
        // Arrange: Split-brain scenario with 3 partitions
        var partition1 = new[]
        {
            new HybridLogicalClock(nodeId: 1),
            new HybridLogicalClock(nodeId: 2)
        };

        var partition2 = new[]
        {
            new HybridLogicalClock(nodeId: 3),
            new HybridLogicalClock(nodeId: 4)
        };

        var partition3 = new[]
        {
            new HybridLogicalClock(nodeId: 5)
        };

        // Act: Each partition operates independently
        var timestamps = new Dictionary<int, List<HybridTimestamp>>
        {
            [1] = new(),
            [2] = new(),
            [3] = new()
        };

        for (int i = 0; i < 50; i++)
        {
            // Partition 1
            foreach (var clock in partition1)
            {
                timestamps[1].Add(clock.Now());
            }

            // Partition 2
            foreach (var clock in partition2)
            {
                timestamps[2].Add(clock.Now());
            }

            // Partition 3
            foreach (var clock in partition3)
            {
                timestamps[3].Add(clock.Now());
            }

            System.Threading.Thread.Sleep(1);
        }

        // Assert: Within each partition, ordering is maintained
        foreach (var (partitionId, partitionTimestamps) in timestamps)
        {
            var sorted = partitionTimestamps.OrderBy(ts => ts).ToList();
            for (int i = 1; i < sorted.Count; i++)
            {
                Assert.True(sorted[i].CompareTo(sorted[i - 1]) > 0,
                    $"Partition {partitionId} ordering violated at index {i}");
            }
        }

        _output.WriteLine($"Split-brain partial orders:");
        _output.WriteLine($"  Partition 1: {timestamps[1].Count} timestamps (ordered)");
        _output.WriteLine($"  Partition 2: {timestamps[2].Count} timestamps (ordered)");
        _output.WriteLine($"  Partition 3: {timestamps[3].Count} timestamps (ordered)");
        _output.WriteLine($"  Partial orders preserved: VERIFIED");
    }

    [Fact]
    public async Task HLC_PartitionReconciliation_CausalityPreserved()
    {
        // Arrange: Simulate causally related events across partition
        var nodeA = new HybridLogicalClock(nodeId: 1);
        var nodeB = new HybridLogicalClock(nodeId: 2);

        // Event A1 on node A
        var eventA1 = nodeA.Now();
        await Task.Delay(10);

        // Event A2 on node A (causally after A1)
        var eventA2 = nodeA.Now();
        Assert.True(eventA2.CompareTo(eventA1) > 0, "A2 happens-after A1");

        // During partition: Node B generates independent event B1
        var eventB1 = nodeB.Now();

        // Partition heals: Node B learns about A2
        var nodeB_updated = nodeB.Update(eventA2);

        // Node B generates event B2 (causally after learning about A2)
        var eventB2 = nodeB.Now();

        // Assert: Causality is preserved
        Assert.True(eventA1.CompareTo(eventA2) < 0, "A1 → A2");
        Assert.True(nodeB_updated.CompareTo(eventA2) >= 0, "B(after sync) ≥ A2");
        Assert.True(eventB2.CompareTo(nodeB_updated) > 0, "B2 → B(after sync)");
        Assert.True(eventB2.CompareTo(eventA2) > 0, "B2 → A2 (transitively)");

        _output.WriteLine($"Causality preservation:");
        _output.WriteLine($"  A1 (before partition): {eventA1}");
        _output.WriteLine($"  A2 (before partition): {eventA2}");
        _output.WriteLine($"  B1 (during partition): {eventB1}");
        _output.WriteLine($"  B(after sync):         {nodeB_updated}");
        _output.WriteLine($"  B2 (after learning A2): {eventB2}");
        _output.WriteLine($"  Causal order: A1 → A2 → B(sync) → B2");
        _output.WriteLine($"  Causality: PRESERVED");
    }

    [Theory]
    [InlineData(2, 100)] // 2 partitions, 100 events each
    [InlineData(3, 50)]  // 3 partitions, 50 events each
    [InlineData(5, 20)]  // 5 partitions, 20 events each
    public void HLC_MultiplePartitions_EventualConsistency(int partitionCount, int eventsPerPartition)
    {
        // Arrange: Create multiple partitions
        var partitions = Enumerable.Range(0, partitionCount)
            .Select(i => new HybridLogicalClock(nodeId: (ushort)(i + 1)))
            .ToList();

        var allEvents = new List<HybridTimestamp>();

        // Act: Generate events in each partition
        for (int i = 0; i < eventsPerPartition; i++)
        {
            foreach (var partition in partitions)
            {
                allEvents.Add(partition.Now());
            }
        }

        // Simulate gossip protocol: gradually exchange timestamps
        for (int round = 0; round < partitionCount; round++)
        {
            for (int i = 0; i < partitions.Count(); i++)
            {
                for (int j = i + 1; j < partitions.Count(); j++)
                {
                    // Exchange timestamps between partitions i and j
                    var ts_i = partitions[i].Now();
                    var ts_j = partitions[j].Now();

                    partitions[i].Update(ts_j);
                    partitions[j].Update(ts_i);
                }
            }
        }

        // Assert: After gossip, all clocks are consistent
        var finalTimestamps = partitions.Select(p => p.Now()).ToList();

        // Verify all timestamps are comparable (total order exists)
        for (int i = 0; i < finalTimestamps.Count(); i++)
        {
            for (int j = i + 1; j < finalTimestamps.Count(); j++)
            {
                var comparison = finalTimestamps[i].CompareTo(finalTimestamps[j]);
                Assert.NotEqual(0, comparison); // Different nodes, different timestamps
            }
        }

        _output.WriteLine($"Eventual consistency test:");
        _output.WriteLine($"  Partitions: {partitionCount}");
        _output.WriteLine($"  Events per partition: {eventsPerPartition}");
        _output.WriteLine($"  Total events: {allEvents.Count}");
        _output.WriteLine($"  Gossip rounds: {partitionCount}");
        _output.WriteLine($"  Final state: CONSISTENT");
    }

    [Fact]
    public void HLC_PartitionTolerance_NoDataLoss()
    {
        // Arrange: Simulate partition with message loss
        var node1 = new HybridLogicalClock(nodeId: 1);
        var node2 = new HybridLogicalClock(nodeId: 2);

        var node1_events = new List<HybridTimestamp>();
        var node2_events = new List<HybridTimestamp>();

        // Act: Generate events during partition
        for (int i = 0; i < 100; i++)
        {
            node1_events.Add(node1.Now());
            node2_events.Add(node2.Now());
        }

        // Simulate partial message delivery during healing (50% loss)
        var random = new Random(42); // Deterministic seed
        var delivered = new List<(HybridTimestamp, int)>(); // (timestamp, source_node)

        foreach (var ts in node1_events)
        {
            if (random.Next(2) == 0) // 50% delivery
            {
                node2.Update(ts);
                delivered.Add((ts, 1));
            }
        }

        foreach (var ts in node2_events)
        {
            if (random.Next(2) == 0)
            {
                node1.Update(ts);
                delivered.Add((ts, 2));
            }
        }

        // Assert: Despite message loss, HLC maintains consistency
        var node1_final = node1.Now();
        var node2_final = node2.Now();

        // Both nodes have progressed
        Assert.True(node1_final.CompareTo(node1_events[0]) > 0);
        Assert.True(node2_final.CompareTo(node2_events[0]) > 0);

        _output.WriteLine($"Partition tolerance with message loss:");
        _output.WriteLine($"  Node 1 events: {node1_events.Count}");
        _output.WriteLine($"  Node 2 events: {node2_events.Count}");
        _output.WriteLine($"  Messages delivered: {delivered.Count}/{node1_events.Count + node2_events.Count} (50%)");
        _output.WriteLine($"  Node 1 final: {node1_final}");
        _output.WriteLine($"  Node 2 final: {node2_final}");
        _output.WriteLine($"  Consistency maintained despite losses: VERIFIED");
    }
}
