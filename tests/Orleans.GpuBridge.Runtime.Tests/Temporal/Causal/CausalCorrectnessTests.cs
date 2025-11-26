// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Xunit;

namespace Orleans.GpuBridge.Runtime.Tests.Temporal.Causal;

/// <summary>
/// Comprehensive tests for Phase 4: Causal Correctness implementation.
/// </summary>
public class CausalCorrectnessTests
{
    #region CausalGraphAnalyzer Tests

    [Fact]
    public void CausalGraphAnalyzer_AddEvent_TracksEvent()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);
        var timestamp = clock.Now();

        var evt = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = timestamp,
            CausalDependencies = Array.Empty<Guid>()
        };

        analyzer.AddEvent(evt);

        Assert.Equal(1, analyzer.EventCount);
    }

    [Fact]
    public void CausalGraphAnalyzer_GetCausalChain_ReturnsAllAncestors()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        // Create chain: e1 → e2 → e3
        var t1 = clock.Now();
        var e1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock.Now();
        var e2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t2,
            CausalDependencies = new[] { e1.EventId }
        };

        var t3 = clock.Now();
        var e3 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t3,
            CausalDependencies = new[] { e2.EventId }
        };

        analyzer.AddEvents(new[] { e1, e2, e3 });

        var chain = analyzer.GetCausalChain(e3.EventId).ToList();

        Assert.Equal(2, chain.Count);
        Assert.Contains(chain, e => e.EventId == e1.EventId);
        Assert.Contains(chain, e => e.EventId == e2.EventId);
    }

    [Fact]
    public void CausalGraphAnalyzer_GetConcurrentEvents_IdentifiesIndependentEvents()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        // Two independent events from different actors
        var t1 = clock1.Now();
        var e1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock2.Now();
        var e2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 2,
            Timestamp = t2,
            CausalDependencies = Array.Empty<Guid>()
        };

        analyzer.AddEvents(new[] { e1, e2 });

        var concurrent = analyzer.GetConcurrentEvents(e1.EventId).ToList();

        Assert.Single(concurrent);
        Assert.Equal(e2.EventId, concurrent[0].EventId);
    }

    [Fact]
    public void CausalGraphAnalyzer_IsCausallyRelated_CorrectForChain()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var e1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock.Now();
        var e2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t2,
            CausalDependencies = new[] { e1.EventId }
        };

        analyzer.AddEvents(new[] { e1, e2 });

        Assert.True(analyzer.IsCausallyRelated(e1.EventId, e2.EventId));
    }

    [Fact]
    public void CausalGraphAnalyzer_GetRelationship_HappensBefore()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var e1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock.Now();
        var e2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t2,
            CausalDependencies = new[] { e1.EventId }
        };

        analyzer.AddEvents(new[] { e1, e2 });

        Assert.Equal(CausalRelationship.HappensBefore, analyzer.GetRelationship(e1.EventId, e2.EventId));
        Assert.Equal(CausalRelationship.HappensAfter, analyzer.GetRelationship(e2.EventId, e1.EventId));
    }

    [Fact]
    public void CausalGraphAnalyzer_FindCommonAncestors_DiamondPattern()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var clock3 = new HybridCausalClock(actorId: 3);

        // Diamond pattern: root → branch1, branch2 → join
        var t1 = clock1.Now();
        var root = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        _ = clock2.Update(t1);
        var t2 = clock2.Now();
        var branch1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 2,
            Timestamp = t2,
            CausalDependencies = new[] { root.EventId }
        };

        _ = clock3.Update(t1);
        var t3 = clock3.Now();
        var branch2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 3,
            Timestamp = t3,
            CausalDependencies = new[] { root.EventId }
        };

        analyzer.AddEvents(new[] { root, branch1, branch2 });

        var commonAncestors = analyzer.FindCommonAncestors(branch1.EventId, branch2.EventId).ToList();

        Assert.Single(commonAncestors);
        Assert.Equal(root.EventId, commonAncestors[0].EventId);
    }

    [Fact]
    public void CausalGraphAnalyzer_GetDependentEvents_FindsAllDownstream()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var root = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock.Now();
        var child1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t2,
            CausalDependencies = new[] { root.EventId }
        };

        var t3 = clock.Now();
        var child2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t3,
            CausalDependencies = new[] { root.EventId }
        };

        analyzer.AddEvents(new[] { root, child1, child2 });

        var dependents = analyzer.GetDependentEvents(root.EventId).ToList();

        Assert.Equal(2, dependents.Count);
        Assert.Contains(dependents, e => e.EventId == child1.EventId);
        Assert.Contains(dependents, e => e.EventId == child2.EventId);
    }

    [Fact]
    public void CausalGraphAnalyzer_GetCausalDepth_CalculatesCorrectly()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        // Chain of depth 3: e1 → e2 → e3 → e4
        var ids = new List<Guid>();
        var events = new List<CausalEvent>();

        for (int i = 0; i < 4; i++)
        {
            var t = clock.Now();
            var evt = new CausalEvent
            {
                EventId = Guid.NewGuid(),
                ActorId = 1,
                Timestamp = t,
                CausalDependencies = ids.Count > 0 ? new[] { ids[^1] } : Array.Empty<Guid>()
            };
            events.Add(evt);
            ids.Add(evt.EventId);
        }

        analyzer.AddEvents(events);

        Assert.Equal(0, analyzer.GetCausalDepth(events[0].EventId));
        Assert.Equal(1, analyzer.GetCausalDepth(events[1].EventId));
        Assert.Equal(2, analyzer.GetCausalDepth(events[2].EventId));
        Assert.Equal(3, analyzer.GetCausalDepth(events[3].EventId));
    }

    [Fact]
    public void CausalGraphAnalyzer_GetCausalSources_FindsRoots()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        // Two roots
        var t1 = clock1.Now();
        var root1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock2.Now();
        var root2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 2,
            Timestamp = t2,
            CausalDependencies = Array.Empty<Guid>()
        };

        // One dependent
        var t3 = clock1.Now();
        var child = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t3,
            CausalDependencies = new[] { root1.EventId }
        };

        analyzer.AddEvents(new[] { root1, root2, child });

        var sources = analyzer.GetCausalSources().ToList();

        Assert.Equal(2, sources.Count);
    }

    [Fact]
    public void CausalGraphAnalyzer_HasCausalCycle_DetectsNoCycle()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var e1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        var t2 = clock.Now();
        var e2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t2,
            CausalDependencies = new[] { e1.EventId }
        };

        analyzer.AddEvents(new[] { e1, e2 });

        Assert.False(analyzer.HasCausalCycle());
    }

    [Fact]
    public void CausalGraphAnalyzer_Statistics_TracksCorrectly()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        // Add 5 events with chain
        var ids = new List<Guid>();
        for (int i = 0; i < 5; i++)
        {
            var t = clock.Now();
            var evt = new CausalEvent
            {
                EventId = Guid.NewGuid(),
                ActorId = 1,
                Timestamp = t,
                CausalDependencies = ids.Count > 0 ? new[] { ids[^1] } : Array.Empty<Guid>()
            };
            analyzer.AddEvent(evt);
            ids.Add(evt.EventId);
        }

        // Perform queries
        analyzer.GetCausalChain(ids[4]);
        analyzer.IsCausallyRelated(ids[0], ids[4]);

        var stats = analyzer.GetStatistics();

        Assert.Equal(5, stats.EventCount);
        Assert.Equal(4, stats.TotalDependencies);
        Assert.True(stats.TotalQueries > 0);
    }

    #endregion

    #region DeadlockDetector Tests

    [Fact]
    public void DeadlockDetector_NoDeadlock_WhenNoPendingMessages()
    {
        var detector = new DeadlockDetector();

        var result = detector.DetectDeadlock(Array.Empty<IPendingMessage>());

        Assert.False(result.HasDeadlock);
    }

    [Fact]
    public void DeadlockDetector_NoDeadlock_WhenNoCircularDependency()
    {
        var detector = new DeadlockDetector();
        var clock = new HybridCausalClock(actorId: 1);

        var t1 = clock.Now();
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "msg1"
        };

        var t2 = clock.Now();
        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t2,
            Payload = "msg2"
        };

        var result = detector.DetectDeadlock(new[] { msg1, msg2 });

        Assert.False(result.HasDeadlock);
    }

    [Fact]
    public void DeadlockDetector_DetectsDeadlock_WithExplicitCircularDependency()
    {
        var detector = new DeadlockDetector();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        var msg1Id = Guid.NewGuid();
        var msg2Id = Guid.NewGuid();

        var t1 = clock1.Now();
        var msg1 = new CausalMessage
        {
            MessageId = msg1Id,
            SenderId = 1,
            Timestamp = t1,
            Payload = "msg1",
            Dependencies = new[] { msg2Id } // msg1 waits for msg2
        };

        var t2 = clock2.Now();
        var msg2 = new CausalMessage
        {
            MessageId = msg2Id,
            SenderId = 2,
            Timestamp = t2,
            Payload = "msg2",
            Dependencies = new[] { msg1Id } // msg2 waits for msg1 → cycle!
        };

        var result = detector.DetectDeadlock(new[] { msg1, msg2 });

        Assert.True(result.HasDeadlock);
        Assert.Equal(2, result.DeadlockedMessageCount);
    }

    [Fact]
    public void DeadlockDetector_FindDeadlockCycles_ReturnsAllCycles()
    {
        var detector = new DeadlockDetector();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        var msg1Id = Guid.NewGuid();
        var msg2Id = Guid.NewGuid();

        var t1 = clock1.Now();
        var msg1 = new CausalMessage
        {
            MessageId = msg1Id,
            SenderId = 1,
            Timestamp = t1,
            Payload = "msg1",
            Dependencies = new[] { msg2Id }
        };

        var t2 = clock2.Now();
        var msg2 = new CausalMessage
        {
            MessageId = msg2Id,
            SenderId = 2,
            Timestamp = t2,
            Payload = "msg2",
            Dependencies = new[] { msg1Id }
        };

        var cycles = detector.FindDeadlockCycles(new[] { msg1, msg2 });

        Assert.NotEmpty(cycles);
        Assert.Equal(2, cycles[0].Length);
    }

    [Fact]
    public void DeadlockDetector_ResolveDeadlock_DropOldest()
    {
        var detector = new DeadlockDetector();

        var cycle = new DeadlockCycle
        {
            CycleId = Guid.NewGuid(),
            MessageIds = new[] { Guid.NewGuid(), Guid.NewGuid() },
            ActorIds = new ushort[] { 1, 2 }
        };

        var resolution = detector.ResolveDeadlock(cycle, DeadlockResolutionStrategy.DropOldest);

        Assert.True(resolution.Success);
        Assert.Equal(DeadlockResolutionStrategy.DropOldest, resolution.Strategy);
        Assert.Single(resolution.AffectedMessageIds);
        Assert.Single(resolution.UnblockedMessageIds);
    }

    [Fact]
    public void DeadlockDetector_ResolveDeadlock_ForceDelivery()
    {
        var detector = new DeadlockDetector();

        var cycle = new DeadlockCycle
        {
            CycleId = Guid.NewGuid(),
            MessageIds = new[] { Guid.NewGuid(), Guid.NewGuid(), Guid.NewGuid() },
            ActorIds = new ushort[] { 1, 2, 3 }
        };

        var resolution = detector.ResolveDeadlock(cycle, DeadlockResolutionStrategy.ForceDelivery);

        Assert.True(resolution.Success);
        Assert.Equal(DeadlockResolutionStrategy.ForceDelivery, resolution.Strategy);
        Assert.Empty(resolution.AffectedMessageIds);
        Assert.Equal(3, resolution.UnblockedMessageIds.Count);
    }

    [Fact]
    public void DeadlockDetector_ResolveDeadlock_DropAll()
    {
        var detector = new DeadlockDetector();

        var cycle = new DeadlockCycle
        {
            CycleId = Guid.NewGuid(),
            MessageIds = new[] { Guid.NewGuid(), Guid.NewGuid() },
            ActorIds = new ushort[] { 1, 2 }
        };

        var resolution = detector.ResolveDeadlock(cycle, DeadlockResolutionStrategy.DropAll);

        Assert.True(resolution.Success);
        Assert.Equal(DeadlockResolutionStrategy.DropAll, resolution.Strategy);
        Assert.Equal(2, resolution.AffectedMessageIds.Count);
        Assert.Empty(resolution.UnblockedMessageIds);
    }

    [Fact]
    public void DeadlockDetector_ResolveDeadlock_DeterministicVictim()
    {
        var detector = new DeadlockDetector();

        var messageIds = new[] { Guid.NewGuid(), Guid.NewGuid() };
        var cycle = new DeadlockCycle
        {
            CycleId = Guid.NewGuid(),
            MessageIds = messageIds,
            ActorIds = new ushort[] { 1, 2 }
        };

        // Should always select the same victim
        var resolution1 = detector.ResolveDeadlock(cycle, DeadlockResolutionStrategy.DeterministicVictim);
        var resolution2 = detector.ResolveDeadlock(cycle, DeadlockResolutionStrategy.DeterministicVictim);

        Assert.Equal(resolution1.AffectedMessageIds[0], resolution2.AffectedMessageIds[0]);
    }

    [Fact]
    public void DeadlockDetector_Statistics_TracksCorrectly()
    {
        var detector = new DeadlockDetector();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        // Create a deadlock
        var msg1Id = Guid.NewGuid();
        var msg2Id = Guid.NewGuid();

        var t1 = clock1.Now();
        var msg1 = new CausalMessage
        {
            MessageId = msg1Id,
            SenderId = 1,
            Timestamp = t1,
            Payload = "msg1",
            Dependencies = new[] { msg2Id }
        };

        var t2 = clock2.Now();
        var msg2 = new CausalMessage
        {
            MessageId = msg2Id,
            SenderId = 2,
            Timestamp = t2,
            Payload = "msg2",
            Dependencies = new[] { msg1Id }
        };

        // Detect and resolve
        detector.DetectDeadlock(new[] { msg1, msg2 });
        var cycles = detector.FindDeadlockCycles(new[] { msg1, msg2 });
        if (cycles.Count > 0)
        {
            detector.ResolveDeadlock(cycles[0], DeadlockResolutionStrategy.DropOldest);
        }

        var stats = detector.GetStatistics();

        Assert.True(stats.TotalDetections > 0);
        Assert.True(stats.DeadlocksFound > 0);
        Assert.True(stats.SuccessfulResolutions > 0);
    }

    [Fact]
    public void DeadlockDetector_ThreeActorCycle_Detected()
    {
        var detector = new DeadlockDetector();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var clock3 = new HybridCausalClock(actorId: 3);

        // Create 3-way cycle: msg1 → msg2 → msg3 → msg1
        var msg1Id = Guid.NewGuid();
        var msg2Id = Guid.NewGuid();
        var msg3Id = Guid.NewGuid();

        var t1 = clock1.Now();
        var msg1 = new CausalMessage
        {
            MessageId = msg1Id,
            SenderId = 1,
            Timestamp = t1,
            Payload = "msg1",
            Dependencies = new[] { msg3Id }
        };

        var t2 = clock2.Now();
        var msg2 = new CausalMessage
        {
            MessageId = msg2Id,
            SenderId = 2,
            Timestamp = t2,
            Payload = "msg2",
            Dependencies = new[] { msg1Id }
        };

        var t3 = clock3.Now();
        var msg3 = new CausalMessage
        {
            MessageId = msg3Id,
            SenderId = 3,
            Timestamp = t3,
            Payload = "msg3",
            Dependencies = new[] { msg2Id }
        };

        var result = detector.DetectDeadlock(new[] { msg1, msg2, msg3 });

        Assert.True(result.HasDeadlock);
        Assert.Equal(3, result.DeadlockedMessageCount);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public async Task Integration_CausalOrderingQueue_WithDeadlockDetector()
    {
        var clock = new HybridCausalClock(actorId: 99);
        var queue = new CausalOrderingQueue(clock);
        var detector = new DeadlockDetector();

        // Create a deadlocked scenario
        var msg1Id = Guid.NewGuid();
        var msg2Id = Guid.NewGuid();

        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        var t1 = clock1.Now();
        var msg1 = new CausalMessage
        {
            MessageId = msg1Id,
            SenderId = 1,
            Timestamp = t1,
            Payload = "msg1",
            Dependencies = new[] { msg2Id }
        };

        var t2 = clock2.Now();
        var msg2 = new CausalMessage
        {
            MessageId = msg2Id,
            SenderId = 2,
            Timestamp = t2,
            Payload = "msg2",
            Dependencies = new[] { msg1Id }
        };

        await queue.EnqueueAsync(msg1);
        await queue.EnqueueAsync(msg2);

        // Both should be pending due to circular dependency
        Assert.Equal(2, queue.PendingCount);

        // Detect deadlock
        var pending = new[] { msg1, msg2 };
        var deadlockResult = detector.DetectDeadlock(pending);

        Assert.True(deadlockResult.HasDeadlock);
    }

    [Fact]
    public async Task Integration_CausalGraphAnalyzer_TracksDeliveredMessages()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);
        var receiverClock = new HybridCausalClock(actorId: 3);
        var queue = new CausalOrderingQueue(receiverClock);

        // Actor 1 sends message
        var t1 = clock1.Now();
        var msg1 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 1,
            Timestamp = t1,
            Payload = "From Actor 1"
        };

        // Actor 2 receives from 1, then sends
        _ = clock2.Update(t1);
        var t2 = clock2.Now();
        var msg2 = new CausalMessage
        {
            MessageId = Guid.NewGuid(),
            SenderId = 2,
            Timestamp = t2,
            Payload = "From Actor 2"
        };

        // Deliver in correct order
        var delivered1 = await queue.EnqueueAsync(msg1);
        var delivered2 = await queue.EnqueueAsync(msg2);

        // Track in analyzer
        foreach (var msg in delivered1.Concat(delivered2))
        {
            var evt = new CausalEvent
            {
                EventId = msg.MessageId,
                ActorId = msg.SenderId,
                Timestamp = msg.Timestamp,
                CausalDependencies = msg.Dependencies?.ToList() ?? new List<Guid>()
            };
            analyzer.AddEvent(evt);
        }

        Assert.True(analyzer.EventCount > 0);
    }

    [Fact]
    public void Integration_CausalRelationship_VectorClockConsistency()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock1 = new HybridCausalClock(actorId: 1);
        var clock2 = new HybridCausalClock(actorId: 2);

        // Actor 1 generates event
        var t1 = clock1.Now();
        var e1 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 1,
            Timestamp = t1,
            CausalDependencies = Array.Empty<Guid>()
        };

        // Actor 2 receives and generates event
        _ = clock2.Update(t1);
        var t2 = clock2.Now();
        var e2 = new CausalEvent
        {
            EventId = Guid.NewGuid(),
            ActorId = 2,
            Timestamp = t2,
            CausalDependencies = new[] { e1.EventId }
        };

        analyzer.AddEvents(new[] { e1, e2 });

        // Vector clock relationship should match graph relationship
        var vcRelationship = e1.Timestamp.GetCausalRelationship(e2.Timestamp);
        var graphRelationship = analyzer.GetRelationship(e1.EventId, e2.EventId);

        Assert.Equal(CausalRelationship.HappensBefore, vcRelationship);
        Assert.Equal(CausalRelationship.HappensBefore, graphRelationship);
    }

    #endregion

    #region Performance Tests

    [Fact]
    public void Performance_CausalGraphAnalyzer_HandlesLargeGraph()
    {
        var analyzer = new CausalGraphAnalyzer();
        var clock = new HybridCausalClock(actorId: 1);

        // Create 1000 events in a chain
        var ids = new List<Guid>();
        for (int i = 0; i < 1000; i++)
        {
            var t = clock.Now();
            var evt = new CausalEvent
            {
                EventId = Guid.NewGuid(),
                ActorId = 1,
                Timestamp = t,
                CausalDependencies = ids.Count > 0 ? new[] { ids[^1] } : Array.Empty<Guid>()
            };
            analyzer.AddEvent(evt);
            ids.Add(evt.EventId);
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();

        var chain = analyzer.GetCausalChain(ids[^1]).ToList();
        var depth = analyzer.GetCausalDepth(ids[^1]);

        sw.Stop();

        Assert.Equal(999, chain.Count);
        Assert.Equal(999, depth);
        Assert.True(sw.ElapsedMilliseconds < 5000, $"Query took too long: {sw.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void Performance_DeadlockDetector_HandlesManyMessages()
    {
        var detector = new DeadlockDetector();
        var messages = new List<CausalMessage>();

        // Create 100 independent messages (no deadlock)
        for (int i = 0; i < 100; i++)
        {
            var clock = new HybridCausalClock((ushort)i);
            var t = clock.Now();
            messages.Add(new CausalMessage
            {
                MessageId = Guid.NewGuid(),
                SenderId = (ushort)i,
                Timestamp = t,
                Payload = $"Message {i}"
            });
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();

        var result = detector.DetectDeadlock(messages);

        sw.Stop();

        Assert.False(result.HasDeadlock);
        Assert.True(sw.ElapsedMilliseconds < 1000, $"Detection took too long: {sw.ElapsedMilliseconds}ms");
    }

    #endregion
}
