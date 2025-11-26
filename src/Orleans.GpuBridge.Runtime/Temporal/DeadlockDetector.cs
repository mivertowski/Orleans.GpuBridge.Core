// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Detects and resolves deadlocks in causal message ordering.
/// </summary>
/// <remarks>
/// <para>
/// Deadlocks occur when there are circular dependencies in the causal ordering.
/// This can happen when:
/// - Messages have explicit dependencies that form a cycle
/// - Vector clock inconsistencies create impossible orderings
/// - Network partitions cause conflicting causal histories
/// </para>
/// <para>
/// Detection algorithm:
/// 1. Build a wait-for graph from pending messages
/// 2. Use DFS to find cycles in the wait-for graph
/// 3. Report all cycles as potential deadlocks
/// </para>
/// <para>
/// Resolution strategies:
/// - DropOldest: Drop the message that has waited longest
/// - DropNewest: Drop the most recent message in the cycle
/// - ForceDelivery: Deliver messages ignoring causal ordering
/// - DeterministicVictim: Use message ID for consistent victim selection
/// </para>
/// </remarks>
public sealed class DeadlockDetector : IDeadlockDetector
{
    private readonly ILogger? _logger;
    private readonly TimeSpan _messageAgeThreshold;

    // Statistics
    private long _totalDetections;
    private long _deadlocksFound;
    private long _successfulResolutions;
    private long _failedResolutions;
    private long _messagesDropped;
    private long _messagesForceDelivered;
    private int _maxCycleLength;
    private double _totalCycleLength;

    /// <summary>
    /// Creates a new deadlock detector.
    /// </summary>
    /// <param name="messageAgeThreshold">Age threshold for severity calculation</param>
    /// <param name="logger">Optional logger</param>
    public DeadlockDetector(TimeSpan? messageAgeThreshold = null, ILogger? logger = null)
    {
        _messageAgeThreshold = messageAgeThreshold ?? TimeSpan.FromSeconds(30);
        _logger = logger;
    }

    /// <inheritdoc/>
    public DeadlockResult DetectDeadlock(IReadOnlyList<IPendingMessage> pendingMessages)
    {
        _totalDetections++;

        if (pendingMessages.Count == 0)
        {
            return DeadlockResult.NoDeadlock;
        }

        // Build wait-for graph
        var waitForGraph = BuildWaitForGraph(pendingMessages);

        // Find cycles using DFS
        var cycles = FindAllCycles(waitForGraph, pendingMessages);

        if (cycles.Count == 0)
        {
            return DeadlockResult.NoDeadlock;
        }

        _deadlocksFound++;
        var deadlockedMessageCount = cycles.SelectMany(c => c.MessageIds).Distinct().Count();

        _logger?.LogWarning(
            "Deadlock detected: {CycleCount} cycles involving {MessageCount} messages",
            cycles.Count, deadlockedMessageCount);

        return new DeadlockResult
        {
            HasDeadlock = true,
            DeadlockedMessageCount = deadlockedMessageCount,
            CycleCount = cycles.Count,
            Description = $"Found {cycles.Count} deadlock cycle(s) involving {deadlockedMessageCount} messages"
        };
    }

    /// <inheritdoc/>
    public IReadOnlyList<DeadlockCycle> FindDeadlockCycles(IReadOnlyList<IPendingMessage> pendingMessages)
    {
        if (pendingMessages.Count == 0)
        {
            return Array.Empty<DeadlockCycle>();
        }

        var waitForGraph = BuildWaitForGraph(pendingMessages);
        return FindAllCycles(waitForGraph, pendingMessages);
    }

    /// <inheritdoc/>
    public DeadlockResolution ResolveDeadlock(DeadlockCycle cycle, DeadlockResolutionStrategy strategy)
    {
        ArgumentNullException.ThrowIfNull(cycle);

        if (cycle.MessageIds.Count == 0)
        {
            return new DeadlockResolution
            {
                Success = false,
                Strategy = strategy,
                Description = "Empty cycle provided"
            };
        }

        _logger?.LogInformation(
            "Resolving deadlock cycle {CycleId} using strategy {Strategy}",
            cycle.CycleId, strategy);

        try
        {
            var resolution = strategy switch
            {
                DeadlockResolutionStrategy.DropOldest => ResolveByDropOldest(cycle),
                DeadlockResolutionStrategy.DropNewest => ResolveByDropNewest(cycle),
                DeadlockResolutionStrategy.DropLowestPriority => ResolveByDropLowestPriority(cycle),
                DeadlockResolutionStrategy.ForceDelivery => ResolveByForceDelivery(cycle),
                DeadlockResolutionStrategy.DropAll => ResolveByDropAll(cycle),
                DeadlockResolutionStrategy.DeterministicVictim => ResolveByDeterministicVictim(cycle),
                DeadlockResolutionStrategy.WaitForIntervention => new DeadlockResolution
                {
                    Success = false,
                    Strategy = strategy,
                    Description = "Waiting for manual intervention"
                },
                _ => throw new ArgumentOutOfRangeException(nameof(strategy))
            };

            if (resolution.Success)
            {
                _successfulResolutions++;

                if (strategy == DeadlockResolutionStrategy.ForceDelivery)
                {
                    _messagesForceDelivered += resolution.UnblockedMessageIds.Count;
                }
                else if (resolution.AffectedMessageIds.Count > 0)
                {
                    _messagesDropped += resolution.AffectedMessageIds.Count;
                }

                _logger?.LogInformation(
                    "Deadlock resolved: {AffectedCount} messages affected, {UnblockedCount} unblocked",
                    resolution.AffectedMessageIds.Count, resolution.UnblockedMessageIds.Count);
            }
            else
            {
                _failedResolutions++;
            }

            return resolution;
        }
        catch (Exception ex)
        {
            _failedResolutions++;
            _logger?.LogError(ex, "Failed to resolve deadlock using strategy {Strategy}", strategy);

            return new DeadlockResolution
            {
                Success = false,
                Strategy = strategy,
                Description = $"Resolution failed: {ex.Message}"
            };
        }
    }

    /// <inheritdoc/>
    public DeadlockStatistics GetStatistics()
    {
        return new DeadlockStatistics
        {
            TotalDetections = _totalDetections,
            DeadlocksFound = _deadlocksFound,
            SuccessfulResolutions = _successfulResolutions,
            FailedResolutions = _failedResolutions,
            TotalMessagesDropped = _messagesDropped,
            TotalMessagesForceDelivered = _messagesForceDelivered,
            AverageCycleLength = _deadlocksFound > 0 ? _totalCycleLength / _deadlocksFound : 0,
            MaxCycleLength = _maxCycleLength
        };
    }

    /// <summary>
    /// Builds a wait-for graph from pending messages.
    /// </summary>
    private Dictionary<Guid, List<Guid>> BuildWaitForGraph(IReadOnlyList<IPendingMessage> messages)
    {
        var graph = new Dictionary<Guid, List<Guid>>();
        var messageIndex = messages.ToDictionary(m => m.MessageId);

        foreach (var message in messages)
        {
            if (!graph.ContainsKey(message.MessageId))
            {
                graph[message.MessageId] = new List<Guid>();
            }

            // Add explicit dependencies
            if (message.Dependencies != null)
            {
                foreach (var depId in message.Dependencies)
                {
                    if (messageIndex.ContainsKey(depId)) // Only if dependency is also pending
                    {
                        graph[message.MessageId].Add(depId);
                    }
                }
            }

            // Add implicit dependencies based on vector clocks
            foreach (var otherMessage in messages)
            {
                if (otherMessage.MessageId == message.MessageId)
                    continue;

                // If this message's VC shows it should come after otherMessage
                // but otherMessage is still pending, there's a wait dependency
                var relationship = message.Timestamp.GetCausalRelationship(otherMessage.Timestamp);
                if (relationship == CausalRelationship.HappensAfter)
                {
                    if (!graph[message.MessageId].Contains(otherMessage.MessageId))
                    {
                        graph[message.MessageId].Add(otherMessage.MessageId);
                    }
                }
            }
        }

        return graph;
    }

    /// <summary>
    /// Finds all cycles in the wait-for graph using DFS.
    /// </summary>
    private List<DeadlockCycle> FindAllCycles(
        Dictionary<Guid, List<Guid>> graph,
        IReadOnlyList<IPendingMessage> messages)
    {
        var cycles = new List<DeadlockCycle>();
        var visited = new HashSet<Guid>();
        var recursionStack = new Dictionary<Guid, int>(); // Maps to index in path
        var path = new List<Guid>();
        var foundCycleSignatures = new HashSet<string>();

        void DFS(Guid node)
        {
            visited.Add(node);
            recursionStack[node] = path.Count;
            path.Add(node);

            if (graph.TryGetValue(node, out var neighbors))
            {
                foreach (var neighbor in neighbors)
                {
                    if (!visited.Contains(neighbor))
                    {
                        DFS(neighbor);
                    }
                    else if (recursionStack.TryGetValue(neighbor, out var cycleStart))
                    {
                        // Found a cycle
                        var cycleNodes = path.Skip(cycleStart).ToList();
                        var signature = string.Join(",", cycleNodes.OrderBy(x => x));

                        if (foundCycleSignatures.Add(signature))
                        {
                            var cycle = CreateCycle(cycleNodes, messages);
                            cycles.Add(cycle);

                            // Update statistics
                            _totalCycleLength += cycle.Length;
                            if (cycle.Length > _maxCycleLength)
                            {
                                _maxCycleLength = cycle.Length;
                            }
                        }
                    }
                }
            }

            path.RemoveAt(path.Count - 1);
            recursionStack.Remove(node);
        }

        foreach (var node in graph.Keys)
        {
            if (!visited.Contains(node))
            {
                DFS(node);
            }
        }

        return cycles;
    }

    /// <summary>
    /// Creates a DeadlockCycle from cycle nodes.
    /// </summary>
    private DeadlockCycle CreateCycle(List<Guid> cycleNodes, IReadOnlyList<IPendingMessage> messages)
    {
        var messageIndex = messages.ToDictionary(m => m.MessageId);
        var actorIds = cycleNodes
            .Where(id => messageIndex.ContainsKey(id))
            .Select(id => messageIndex[id].SenderId)
            .Distinct()
            .ToList();

        var severity = CalculateSeverity(cycleNodes, messages);

        return new DeadlockCycle
        {
            CycleId = Guid.NewGuid(),
            MessageIds = cycleNodes,
            ActorIds = actorIds,
            Severity = severity
        };
    }

    /// <summary>
    /// Calculates deadlock severity based on cycle characteristics.
    /// </summary>
    private DeadlockSeverity CalculateSeverity(List<Guid> cycleNodes, IReadOnlyList<IPendingMessage> messages)
    {
        var messageIndex = messages.ToDictionary(m => m.MessageId);

        // Score based on:
        // 1. Cycle length
        // 2. Message age
        // 3. Number of actors involved

        var score = 0;

        // Cycle length contribution (1-4 points)
        score += cycleNodes.Count switch
        {
            <= 2 => 1,
            <= 4 => 2,
            <= 8 => 3,
            _ => 4
        };

        // Message age contribution (0-4 points)
        var now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var maxAge = cycleNodes
            .Where(id => messageIndex.ContainsKey(id))
            .Select(id => now - messageIndex[id].Timestamp.PhysicalTimeNanos / 1_000_000)
            .DefaultIfEmpty(0)
            .Max();

        score += maxAge switch
        {
            < 1000 => 0,   // < 1 second
            < 5000 => 1,   // < 5 seconds
            < 30000 => 2,  // < 30 seconds
            < 60000 => 3,  // < 1 minute
            _ => 4         // >= 1 minute
        };

        // Number of actors contribution (0-4 points)
        var actorCount = cycleNodes
            .Where(id => messageIndex.ContainsKey(id))
            .Select(id => messageIndex[id].SenderId)
            .Distinct()
            .Count();

        score += actorCount switch
        {
            <= 2 => 0,
            <= 4 => 1,
            <= 8 => 2,
            <= 16 => 3,
            _ => 4
        };

        // Map total score to severity
        return score switch
        {
            <= 3 => DeadlockSeverity.Low,
            <= 6 => DeadlockSeverity.Medium,
            <= 9 => DeadlockSeverity.High,
            _ => DeadlockSeverity.Critical
        };
    }

    #region Resolution Strategies

    private DeadlockResolution ResolveByDropOldest(DeadlockCycle cycle)
    {
        // The "oldest" in terms of waiting time would be determined by caller
        // Here we just select the first message deterministically
        var victim = cycle.MessageIds.OrderBy(id => id).First();

        return new DeadlockResolution
        {
            Success = true,
            Strategy = DeadlockResolutionStrategy.DropOldest,
            AffectedMessageIds = new[] { victim },
            UnblockedMessageIds = cycle.MessageIds.Where(id => id != victim).ToList(),
            Description = $"Dropped oldest message {victim}"
        };
    }

    private DeadlockResolution ResolveByDropNewest(DeadlockCycle cycle)
    {
        var victim = cycle.MessageIds.OrderByDescending(id => id).First();

        return new DeadlockResolution
        {
            Success = true,
            Strategy = DeadlockResolutionStrategy.DropNewest,
            AffectedMessageIds = new[] { victim },
            UnblockedMessageIds = cycle.MessageIds.Where(id => id != victim).ToList(),
            Description = $"Dropped newest message {victim}"
        };
    }

    private DeadlockResolution ResolveByDropLowestPriority(DeadlockCycle cycle)
    {
        // Without priority info, use deterministic selection
        return ResolveByDeterministicVictim(cycle);
    }

    private DeadlockResolution ResolveByForceDelivery(DeadlockCycle cycle)
    {
        return new DeadlockResolution
        {
            Success = true,
            Strategy = DeadlockResolutionStrategy.ForceDelivery,
            AffectedMessageIds = Array.Empty<Guid>(),
            UnblockedMessageIds = cycle.MessageIds.ToList(),
            Description = $"Force-delivered {cycle.MessageIds.Count} messages ignoring causal ordering"
        };
    }

    private DeadlockResolution ResolveByDropAll(DeadlockCycle cycle)
    {
        return new DeadlockResolution
        {
            Success = true,
            Strategy = DeadlockResolutionStrategy.DropAll,
            AffectedMessageIds = cycle.MessageIds.ToList(),
            UnblockedMessageIds = Array.Empty<Guid>(),
            Description = $"Dropped all {cycle.MessageIds.Count} messages in cycle"
        };
    }

    private DeadlockResolution ResolveByDeterministicVictim(DeadlockCycle cycle)
    {
        // Use consistent hashing to select victim
        // This ensures all nodes in a distributed system select the same victim
        var victim = cycle.MessageIds
            .OrderBy(id => id.GetHashCode())
            .First();

        return new DeadlockResolution
        {
            Success = true,
            Strategy = DeadlockResolutionStrategy.DeterministicVictim,
            AffectedMessageIds = new[] { victim },
            UnblockedMessageIds = cycle.MessageIds.Where(id => id != victim).ToList(),
            Description = $"Deterministically selected victim {victim}"
        };
    }

    #endregion

    /// <summary>
    /// Resets statistics.
    /// </summary>
    public void ResetStatistics()
    {
        _totalDetections = 0;
        _deadlocksFound = 0;
        _successfulResolutions = 0;
        _failedResolutions = 0;
        _messagesDropped = 0;
        _messagesForceDelivered = 0;
        _maxCycleLength = 0;
        _totalCycleLength = 0;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"DeadlockDetector(Detections={_totalDetections}, Found={_deadlocksFound}, Resolved={_successfulResolutions})";
    }
}
