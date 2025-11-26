// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Analyzes causal relationships between events in a distributed system.
/// </summary>
/// <remarks>
/// <para>
/// Provides comprehensive causal analysis including:
/// - Causal chain reconstruction (all events leading to a specific event)
/// - Concurrent event detection (events with no causal relationship)
/// - Common ancestor finding (join points in causal history)
/// - Dependent event discovery (events caused by a specific event)
/// - Causal depth calculation (maximum chain length)
/// </para>
/// <para>
/// Performance characteristics:
/// - GetCausalChain: O(D × E) where D is depth and E is events per level
/// - GetConcurrentEvents: O(N × V) where N is events and V is vector clock size
/// - IsCausallyRelated: O(V) where V is vector clock size
/// - GetCausalDepth: O(D × E) using BFS
/// </para>
/// </remarks>
public sealed class CausalGraphAnalyzer : ICausalGraphAnalyzer
{
    private readonly Dictionary<Guid, CausalEvent> _events = new();
    private readonly Dictionary<Guid, List<Guid>> _dependencyGraph = new(); // eventId → dependencies
    private readonly Dictionary<Guid, List<Guid>> _reverseDependencyGraph = new(); // eventId → dependents
    private readonly ILogger? _logger;

    private long _totalQueries;
    private long _totalEventsAnalyzed;

    /// <summary>
    /// Gets the number of events in the analyzer.
    /// </summary>
    public int EventCount => _events.Count;

    /// <summary>
    /// Gets the total number of queries performed.
    /// </summary>
    public long TotalQueries => _totalQueries;

    /// <summary>
    /// Creates a new causal graph analyzer.
    /// </summary>
    /// <param name="logger">Optional logger</param>
    public CausalGraphAnalyzer(ILogger? logger = null)
    {
        _logger = logger;
    }

    /// <summary>
    /// Adds an event to the analyzer.
    /// </summary>
    /// <param name="evt">Event to add</param>
    public void AddEvent(CausalEvent evt)
    {
        ArgumentNullException.ThrowIfNull(evt);

        _events[evt.EventId] = evt;

        // Build dependency graph
        if (!_dependencyGraph.ContainsKey(evt.EventId))
        {
            _dependencyGraph[evt.EventId] = new List<Guid>();
        }

        foreach (var depId in evt.CausalDependencies)
        {
            _dependencyGraph[evt.EventId].Add(depId);

            // Build reverse graph
            if (!_reverseDependencyGraph.ContainsKey(depId))
            {
                _reverseDependencyGraph[depId] = new List<Guid>();
            }
            _reverseDependencyGraph[depId].Add(evt.EventId);
        }

        _logger?.LogTrace("Added event {EventId} with {DepCount} dependencies",
            evt.EventId, evt.CausalDependencies.Count);
    }

    /// <summary>
    /// Adds multiple events to the analyzer.
    /// </summary>
    /// <param name="events">Events to add</param>
    public void AddEvents(IEnumerable<CausalEvent> events)
    {
        foreach (var evt in events)
        {
            AddEvent(evt);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<CausalEvent> GetCausalChain(Guid eventId)
    {
        _totalQueries++;

        if (!_events.TryGetValue(eventId, out var targetEvent))
        {
            _logger?.LogWarning("Event {EventId} not found in analyzer", eventId);
            return Enumerable.Empty<CausalEvent>();
        }

        var chain = new List<CausalEvent>();
        var visited = new HashSet<Guid>();
        var queue = new Queue<Guid>();

        // Start with the target event's direct dependencies
        foreach (var depId in targetEvent.CausalDependencies)
        {
            if (visited.Add(depId))
            {
                queue.Enqueue(depId);
            }
        }

        // BFS to find all ancestors
        while (queue.Count > 0)
        {
            var currentId = queue.Dequeue();

            if (_events.TryGetValue(currentId, out var currentEvent))
            {
                chain.Add(currentEvent);
                _totalEventsAnalyzed++;

                // Add dependencies to queue
                if (_dependencyGraph.TryGetValue(currentId, out var deps))
                {
                    foreach (var depId in deps)
                    {
                        if (visited.Add(depId))
                        {
                            queue.Enqueue(depId);
                        }
                    }
                }
            }
        }

        // Sort by timestamp (oldest first)
        return chain.OrderBy(e => e.Timestamp.HLC.CompareTo(targetEvent.Timestamp.HLC)).ToList();
    }

    /// <inheritdoc/>
    public IEnumerable<CausalEvent> GetConcurrentEvents(Guid eventId)
    {
        _totalQueries++;

        if (!_events.TryGetValue(eventId, out var targetEvent))
        {
            return Enumerable.Empty<CausalEvent>();
        }

        var concurrent = new List<CausalEvent>();

        foreach (var evt in _events.Values)
        {
            if (evt.EventId == eventId)
                continue;

            _totalEventsAnalyzed++;

            var relationship = targetEvent.Timestamp.GetCausalRelationship(evt.Timestamp);
            if (relationship == CausalRelationship.Concurrent)
            {
                concurrent.Add(evt);
            }
        }

        return concurrent;
    }

    /// <inheritdoc/>
    public bool IsCausallyRelated(Guid eventId1, Guid eventId2)
    {
        _totalQueries++;

        if (!_events.TryGetValue(eventId1, out var event1) ||
            !_events.TryGetValue(eventId2, out var event2))
        {
            return false;
        }

        var relationship = event1.Timestamp.GetCausalRelationship(event2.Timestamp);
        return relationship != CausalRelationship.Concurrent &&
               relationship != CausalRelationship.Equal;
    }

    /// <inheritdoc/>
    public CausalRelationship GetRelationship(Guid eventId1, Guid eventId2)
    {
        _totalQueries++;

        if (!_events.TryGetValue(eventId1, out var event1) ||
            !_events.TryGetValue(eventId2, out var event2))
        {
            return CausalRelationship.Concurrent; // Unknown events treated as concurrent
        }

        return event1.Timestamp.GetCausalRelationship(event2.Timestamp);
    }

    /// <inheritdoc/>
    public IEnumerable<CausalEvent> FindCommonAncestors(params Guid[] eventIds)
    {
        _totalQueries++;

        if (eventIds.Length == 0)
            return Enumerable.Empty<CausalEvent>();

        if (eventIds.Length == 1)
            return GetCausalChain(eventIds[0]);

        // Get causal chains for all events
        var allChains = eventIds.Select(id => GetCausalChain(id).Select(e => e.EventId).ToHashSet()).ToList();

        // Find intersection
        var commonIds = allChains[0];
        for (int i = 1; i < allChains.Count; i++)
        {
            commonIds.IntersectWith(allChains[i]);
        }

        // Return events that are in all chains
        return commonIds
            .Where(id => _events.ContainsKey(id))
            .Select(id => _events[id])
            .OrderBy(e => e.Timestamp.HLC)
            .ToList();
    }

    /// <inheritdoc/>
    public IEnumerable<CausalEvent> GetDependentEvents(Guid eventId)
    {
        _totalQueries++;

        if (!_events.ContainsKey(eventId))
        {
            return Enumerable.Empty<CausalEvent>();
        }

        var dependents = new List<CausalEvent>();
        var visited = new HashSet<Guid>();
        var queue = new Queue<Guid>();

        // Start with direct dependents
        if (_reverseDependencyGraph.TryGetValue(eventId, out var directDependents))
        {
            foreach (var depId in directDependents)
            {
                if (visited.Add(depId))
                {
                    queue.Enqueue(depId);
                }
            }
        }

        // BFS to find all dependents
        while (queue.Count > 0)
        {
            var currentId = queue.Dequeue();

            if (_events.TryGetValue(currentId, out var currentEvent))
            {
                dependents.Add(currentEvent);
                _totalEventsAnalyzed++;

                // Add dependents to queue
                if (_reverseDependencyGraph.TryGetValue(currentId, out var moreDependents))
                {
                    foreach (var depId in moreDependents)
                    {
                        if (visited.Add(depId))
                        {
                            queue.Enqueue(depId);
                        }
                    }
                }
            }
        }

        return dependents.OrderBy(e => e.Timestamp.HLC).ToList();
    }

    /// <inheritdoc/>
    public int GetCausalDepth(Guid eventId)
    {
        _totalQueries++;

        if (!_events.ContainsKey(eventId))
        {
            return 0;
        }

        // BFS with level tracking
        var visited = new HashSet<Guid> { eventId };
        var currentLevel = new List<Guid> { eventId };
        int depth = 0;

        while (currentLevel.Count > 0)
        {
            var nextLevel = new List<Guid>();

            foreach (var currentId in currentLevel)
            {
                if (_dependencyGraph.TryGetValue(currentId, out var deps))
                {
                    foreach (var depId in deps)
                    {
                        if (visited.Add(depId))
                        {
                            nextLevel.Add(depId);
                            _totalEventsAnalyzed++;
                        }
                    }
                }
            }

            if (nextLevel.Count > 0)
            {
                depth++;
            }
            currentLevel = nextLevel;
        }

        return depth;
    }

    /// <summary>
    /// Gets events that are causal "sinks" (have no dependents).
    /// </summary>
    /// <returns>Events with no dependents</returns>
    public IEnumerable<CausalEvent> GetCausalSinks()
    {
        return _events.Values
            .Where(e => !_reverseDependencyGraph.ContainsKey(e.EventId) ||
                        _reverseDependencyGraph[e.EventId].Count == 0)
            .ToList();
    }

    /// <summary>
    /// Gets events that are causal "sources" (have no dependencies).
    /// </summary>
    /// <returns>Events with no dependencies</returns>
    public IEnumerable<CausalEvent> GetCausalSources()
    {
        return _events.Values
            .Where(e => e.CausalDependencies.Count == 0)
            .ToList();
    }

    /// <summary>
    /// Detects if there's a causal cycle in the graph.
    /// </summary>
    /// <returns>True if a cycle exists</returns>
    /// <remarks>
    /// A well-formed causal graph should never have cycles.
    /// This method can help detect corruption or bugs.
    /// </remarks>
    public bool HasCausalCycle()
    {
        var visited = new HashSet<Guid>();
        var recursionStack = new HashSet<Guid>();

        bool HasCycleFromNode(Guid nodeId)
        {
            visited.Add(nodeId);
            recursionStack.Add(nodeId);

            if (_dependencyGraph.TryGetValue(nodeId, out var deps))
            {
                foreach (var depId in deps)
                {
                    if (!visited.Contains(depId))
                    {
                        if (HasCycleFromNode(depId))
                            return true;
                    }
                    else if (recursionStack.Contains(depId))
                    {
                        return true; // Back edge found → cycle
                    }
                }
            }

            recursionStack.Remove(nodeId);
            return false;
        }

        foreach (var eventId in _events.Keys)
        {
            if (!visited.Contains(eventId))
            {
                if (HasCycleFromNode(eventId))
                {
                    _logger?.LogError("Causal cycle detected starting from event {EventId}", eventId);
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Gets statistics about the causal graph.
    /// </summary>
    public CausalGraphStatistics GetStatistics()
    {
        var depths = _events.Keys.Select(GetCausalDepth).ToList();
        var avgDepth = depths.Any() ? depths.Average() : 0;
        var maxDepth = depths.Any() ? depths.Max() : 0;

        return new CausalGraphStatistics
        {
            EventCount = _events.Count,
            TotalDependencies = _dependencyGraph.Values.Sum(d => d.Count),
            TotalQueries = _totalQueries,
            TotalEventsAnalyzed = _totalEventsAnalyzed,
            AverageCausalDepth = avgDepth,
            MaxCausalDepth = maxDepth,
            SourceCount = GetCausalSources().Count(),
            SinkCount = GetCausalSinks().Count()
        };
    }

    /// <summary>
    /// Clears all events from the analyzer.
    /// </summary>
    public void Clear()
    {
        _events.Clear();
        _dependencyGraph.Clear();
        _reverseDependencyGraph.Clear();
        _totalQueries = 0;
        _totalEventsAnalyzed = 0;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"CausalGraphAnalyzer(Events={_events.Count}, Queries={_totalQueries})";
    }
}

/// <summary>
/// Statistics about the causal graph.
/// </summary>
public sealed record CausalGraphStatistics
{
    /// <summary>
    /// Total number of events in the graph.
    /// </summary>
    public int EventCount { get; init; }

    /// <summary>
    /// Total number of dependency edges.
    /// </summary>
    public int TotalDependencies { get; init; }

    /// <summary>
    /// Total queries performed.
    /// </summary>
    public long TotalQueries { get; init; }

    /// <summary>
    /// Total events analyzed across all queries.
    /// </summary>
    public long TotalEventsAnalyzed { get; init; }

    /// <summary>
    /// Average causal depth of events.
    /// </summary>
    public double AverageCausalDepth { get; init; }

    /// <summary>
    /// Maximum causal depth in the graph.
    /// </summary>
    public int MaxCausalDepth { get; init; }

    /// <summary>
    /// Number of causal sources (events with no dependencies).
    /// </summary>
    public int SourceCount { get; init; }

    /// <summary>
    /// Number of causal sinks (events with no dependents).
    /// </summary>
    public int SinkCount { get; init; }

    /// <summary>
    /// Average dependencies per event.
    /// </summary>
    public double AverageDependencies => EventCount > 0 ? (double)TotalDependencies / EventCount : 0;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"CausalGraphStats(Events={EventCount}, Deps={TotalDependencies}, " +
               $"AvgDepth={AverageCausalDepth:F1}, MaxDepth={MaxCausalDepth})";
    }
}
