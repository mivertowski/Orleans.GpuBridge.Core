using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Orleans.GpuBridge.Abstractions.Temporal;
using Xunit;

namespace Orleans.GpuBridge.Tests.Temporal;

public class VectorClockTests
{
    #region Basic Operations Tests

    [Fact]
    public void VectorClock_Create_StartsEmpty()
    {
        var vc = new VectorClock();

        Assert.Equal(0, vc.Count);
        Assert.Equal(0, vc[1]); // Non-existent actors return 0
    }

    [Fact]
    public void VectorClock_CreateSingle_InitializesCorrectly()
    {
        var vc = VectorClock.Create(actorId: 1, value: 5);

        Assert.Equal(1, vc.Count);
        Assert.Equal(5, vc[1]);
        Assert.Equal(0, vc[2]); // Other actors return 0
    }

    [Fact]
    public void VectorClock_Increment_IncreasesValue()
    {
        var vc1 = VectorClock.Create(1, 0);
        var vc2 = vc1.Increment(1);
        var vc3 = vc2.Increment(1);

        Assert.Equal(0, vc1[1]);
        Assert.Equal(1, vc2[1]);
        Assert.Equal(2, vc3[1]);
    }

    [Fact]
    public void VectorClock_Increment_AddsNewActor()
    {
        var vc1 = VectorClock.Create(1, 5);
        var vc2 = vc1.Increment(2); // New actor

        Assert.Equal(5, vc2[1]); // Existing actor unchanged
        Assert.Equal(1, vc2[2]); // New actor initialized to 1
    }

    [Fact]
    public void VectorClock_Merge_TakesMaximum()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 3,
            [3] = 7
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 3,
            [2] = 8,
            [4] = 2
        });

        var merged = vc1.Merge(vc2);

        Assert.Equal(5, merged[1]); // max(5, 3)
        Assert.Equal(8, merged[2]); // max(3, 8)
        Assert.Equal(7, merged[3]); // max(7, 0)
        Assert.Equal(2, merged[4]); // max(0, 2)
    }

    [Fact]
    public void VectorClock_Update_MergesAndIncrements()
    {
        var local = VectorClock.Create(1, 5);
        var received = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 3,
            [2] = 10
        });

        var updated = local.Update(received, localActorId: 1);

        Assert.Equal(6, updated[1]); // max(5, 3) + 1
        Assert.Equal(10, updated[2]); // max(0, 10)
    }

    #endregion

    #region Causal Relationship Tests

    [Fact]
    public void VectorClock_Equal_SameValues()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 3
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 3
        });

        var relationship = vc1.CompareTo(vc2);

        Assert.Equal(CausalRelationship.Equal, relationship);
        Assert.True(vc1.Equals(vc2));
    }

    [Fact]
    public void VectorClock_HappensBefore_AllLessOrEqual()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 3,
            [2] = 2
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 4
        });

        var relationship = vc1.CompareTo(vc2);

        Assert.Equal(CausalRelationship.HappensBefore, relationship);
        Assert.True(vc1.HappensBefore(vc2));
        Assert.False(vc1.HappensAfter(vc2));
        Assert.False(vc1.IsConcurrentWith(vc2));
    }

    [Fact]
    public void VectorClock_HappensAfter_AllGreaterOrEqual()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 10,
            [2] = 8
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 4
        });

        var relationship = vc1.CompareTo(vc2);

        Assert.Equal(CausalRelationship.HappensAfter, relationship);
        Assert.True(vc1.HappensAfter(vc2));
        Assert.False(vc1.HappensBefore(vc2));
    }

    [Fact]
    public void VectorClock_Concurrent_PartialOrdering()
    {
        // vc1: A=5, B=2
        // vc2: A=3, B=7
        // Neither dominates the other → concurrent
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 2
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 3,
            [2] = 7
        });

        var relationship = vc1.CompareTo(vc2);

        Assert.Equal(CausalRelationship.Concurrent, relationship);
        Assert.True(vc1.IsConcurrentWith(vc2));
        Assert.True(vc2.IsConcurrentWith(vc1));
    }

    [Fact]
    public void VectorClock_IsDominatedBy_CorrectBehavior()
    {
        var vc1 = VectorClock.Create(1, 3);
        var vc2 = VectorClock.Create(1, 5);
        var vc3 = VectorClock.Create(1, 3); // Equal to vc1

        Assert.True(vc1.IsDominatedBy(vc2)); // 3 ≤ 5
        Assert.False(vc2.IsDominatedBy(vc1)); // 5 not ≤ 3
        Assert.True(vc1.IsDominatedBy(vc3)); // Equal counts as dominated
    }

    #endregion

    #region Causality Scenario Tests

    [Fact]
    public void VectorClock_LinearCausality_Sequential()
    {
        // Scenario: Actor 1 generates 3 events sequentially
        var e1 = VectorClock.Create(1, 0).Increment(1); // [1:1]
        var e2 = e1.Increment(1);                        // [1:2]
        var e3 = e2.Increment(1);                        // [1:3]

        Assert.Equal(CausalRelationship.HappensBefore, e1.CompareTo(e2));
        Assert.Equal(CausalRelationship.HappensBefore, e2.CompareTo(e3));
        Assert.Equal(CausalRelationship.HappensBefore, e1.CompareTo(e3));
    }

    [Fact]
    public void VectorClock_MessagePassing_CausalityPreserved()
    {
        // Actor 1 sends message to Actor 2
        var actor1 = VectorClock.Create(1, 0);
        var e1 = actor1.Increment(1); // [1:1] - Actor 1 sends

        var actor2 = VectorClock.Create(2, 0);
        var e2 = actor2.Update(e1, localActorId: 2); // [1:1, 2:1] - Actor 2 receives

        Assert.True(e1.HappensBefore(e2)); // Send happens-before receive
    }

    [Fact]
    public void VectorClock_DiamondPattern_ConcurrentBranches()
    {
        // Initial event
        var e0 = VectorClock.Create(1, 1); // [1:1]

        // Two actors process concurrently
        var e1 = e0.Increment(2); // [1:1, 2:1]
        var e2 = e0.Increment(3); // [1:1, 3:1]

        // e1 and e2 are concurrent (both depend on e0, but independent)
        Assert.True(e1.IsConcurrentWith(e2));

        // Join: Actor 4 receives both messages
        var e3 = new VectorClock()
            .Merge(e1)
            .Merge(e2)
            .Increment(4); // [1:1, 2:1, 3:1, 4:1]

        Assert.True(e1.HappensBefore(e3));
        Assert.True(e2.HappensBefore(e3));
    }

    [Fact]
    public void VectorClock_ThreeActors_ComplexCausality()
    {
        // Actor 1 generates event
        var vc1_e1 = VectorClock.Create(1, 0).Increment(1); // [1:1]

        // Actor 2 receives and generates event
        var vc2_e1 = VectorClock.Create(2, 0).Update(vc1_e1, 2); // [1:1, 2:1]
        var vc2_e2 = vc2_e1.Increment(2); // [1:1, 2:2]

        // Actor 3 receives from Actor 2
        var vc3_e1 = VectorClock.Create(3, 0).Update(vc2_e2, 3); // [1:1, 2:2, 3:1]

        // Verify causal chain: e1 → e2 → e3
        Assert.True(vc1_e1.HappensBefore(vc2_e1));
        Assert.True(vc2_e2.HappensAfter(vc2_e1));
        Assert.True(vc3_e1.HappensAfter(vc2_e2));
        Assert.True(vc1_e1.HappensBefore(vc3_e1)); // Transitivity
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void VectorClock_ToBytes_FromBytes_RoundTrip()
    {
        var original = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 100,
            [5] = 250,
            [10] = 500
        });

        var bytes = original.ToBytes();
        var deserialized = VectorClock.FromBytes(bytes);

        Assert.Equal(original, deserialized);
        Assert.Equal(100, deserialized[1]);
        Assert.Equal(250, deserialized[5]);
        Assert.Equal(500, deserialized[10]);
    }

    [Fact]
    public void VectorClock_ToBytes_EmptyClock()
    {
        var vc = new VectorClock();
        var bytes = vc.ToBytes();
        var deserialized = VectorClock.FromBytes(bytes);

        Assert.Equal(vc, deserialized);
        Assert.Equal(0, deserialized.Count);
    }

    [Fact]
    public void VectorClock_FromBytes_InvalidData_ThrowsException()
    {
        var invalidBytes = new byte[] { 1 }; // Too short

        Assert.Throws<ArgumentException>(() => VectorClock.FromBytes(invalidBytes));
    }

    #endregion

    #region Equality and Comparison Tests

    [Fact]
    public void VectorClock_Equality_CorrectBehavior()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long> { [1] = 5, [2] = 3 });
        var vc2 = new VectorClock(new Dictionary<ushort, long> { [1] = 5, [2] = 3 });
        var vc3 = new VectorClock(new Dictionary<ushort, long> { [1] = 5, [2] = 4 });

        Assert.True(vc1.Equals(vc2));
        Assert.True(vc1 == vc2);
        Assert.False(vc1.Equals(vc3));
        Assert.True(vc1 != vc3);
    }

    [Fact]
    public void VectorClock_HashCode_ConsistentForEqual()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long> { [1] = 5, [2] = 3 });
        var vc2 = new VectorClock(new Dictionary<ushort, long> { [1] = 5, [2] = 3 });

        Assert.Equal(vc1.GetHashCode(), vc2.GetHashCode());
    }

    [Fact]
    public void VectorClock_IComparable_ProvidesTotalOrder()
    {
        var vc1 = VectorClock.Create(1, 1);
        var vc2 = VectorClock.Create(1, 5);
        var vc3 = VectorClock.Create(1, 3);

        var list = new List<VectorClock> { vc2, vc1, vc3 };
        list.Sort();

        Assert.Equal(vc1, list[0]);
        Assert.Equal(vc3, list[1]);
        Assert.Equal(vc2, list[2]);
    }

    #endregion

    #region Performance Tests

    [Fact]
    public void VectorClock_Increment_Fast()
    {
        var vc = VectorClock.Create(1, 0);
        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < 10_000; i++)
        {
            vc = vc.Increment(1);
        }

        sw.Stop();

        var avgNanos = sw.Elapsed.TotalNanoseconds / 10_000;
        Assert.True(avgNanos < 1000, $"Increment too slow: {avgNanos:F0}ns (target: <1000ns)");
    }

    [Fact]
    public void VectorClock_Merge_Fast()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 100, [2] = 200, [3] = 300, [4] = 400, [5] = 500
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 150, [2] = 150, [3] = 350, [4] = 350, [5] = 550
        });

        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < 10_000; i++)
        {
            _ = vc1.Merge(vc2);
        }

        sw.Stop();

        var avgNanos = sw.Elapsed.TotalNanoseconds / 10_000;
        Assert.True(avgNanos < 5000, $"Merge too slow: {avgNanos:F0}ns (target: <5000ns)");
    }

    [Fact]
    public void VectorClock_CompareTo_Fast()
    {
        var vc1 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 100, [2] = 200, [3] = 300
        });

        var vc2 = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 150, [2] = 250, [3] = 350
        });

        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < 100_000; i++)
        {
            _ = vc1.CompareTo(vc2);
        }

        sw.Stop();

        var avgNanos = sw.Elapsed.TotalNanoseconds / 100_000;
        Assert.True(avgNanos < 500, $"CompareTo too slow: {avgNanos:F0}ns (target: <500ns)");
    }

    #endregion

    #region ToString Tests

    [Fact]
    public void VectorClock_ToString_EmptyVectorClock()
    {
        var vc = new VectorClock();
        var str = vc.ToString();

        Assert.Equal("VC{}", str);
    }

    [Fact]
    public void VectorClock_ToString_MultipleActors()
    {
        var vc = new VectorClock(new Dictionary<ushort, long>
        {
            [1] = 5,
            [2] = 3,
            [3] = 7
        });

        var str = vc.ToString();

        Assert.Contains("A1:5", str);
        Assert.Contains("A2:3", str);
        Assert.Contains("A3:7", str);
    }

    #endregion
}
