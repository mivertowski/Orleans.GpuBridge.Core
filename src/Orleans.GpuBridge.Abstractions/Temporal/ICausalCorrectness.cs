// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Temporal;

/// <summary>
/// Interface for causal graph analysis operations.
/// </summary>
/// <remarks>
/// Provides methods for analyzing causal relationships between events in a distributed system.
/// </remarks>
public interface ICausalGraphAnalyzer
{
    /// <summary>
    /// Gets the complete causal chain leading to a specific event.
    /// </summary>
    /// <param name="eventId">Event to trace back from</param>
    /// <returns>All causally preceding events, in causal order</returns>
    IEnumerable<CausalEvent> GetCausalChain(Guid eventId);

    /// <summary>
    /// Gets all events that are concurrent with a specific event.
    /// </summary>
    /// <param name="eventId">Event to find concurrent events for</param>
    /// <returns>All concurrent events (neither before nor after)</returns>
    IEnumerable<CausalEvent> GetConcurrentEvents(Guid eventId);

    /// <summary>
    /// Checks if two events are causally related.
    /// </summary>
    /// <param name="eventId1">First event</param>
    /// <param name="eventId2">Second event</param>
    /// <returns>True if one event causally precedes the other</returns>
    bool IsCausallyRelated(Guid eventId1, Guid eventId2);

    /// <summary>
    /// Gets the causal relationship between two events.
    /// </summary>
    /// <param name="eventId1">First event</param>
    /// <param name="eventId2">Second event</param>
    /// <returns>The causal relationship between events</returns>
    CausalRelationship GetRelationship(Guid eventId1, Guid eventId2);

    /// <summary>
    /// Finds common causal ancestors of multiple events.
    /// </summary>
    /// <param name="eventIds">Events to find ancestors for</param>
    /// <returns>Common causal ancestors</returns>
    IEnumerable<CausalEvent> FindCommonAncestors(params Guid[] eventIds);

    /// <summary>
    /// Finds all events that depend on a specific event.
    /// </summary>
    /// <param name="eventId">Event to find dependents for</param>
    /// <returns>All events that causally follow this event</returns>
    IEnumerable<CausalEvent> GetDependentEvents(Guid eventId);

    /// <summary>
    /// Gets the causal depth (maximum chain length) of an event.
    /// </summary>
    /// <param name="eventId">Event to analyze</param>
    /// <returns>Maximum causal chain length</returns>
    int GetCausalDepth(Guid eventId);
}

/// <summary>
/// Interface for deadlock detection and resolution.
/// </summary>
/// <remarks>
/// Detects circular dependencies in causal message ordering and provides resolution strategies.
/// </remarks>
public interface IDeadlockDetector
{
    /// <summary>
    /// Detects if a deadlock exists in the given message queue.
    /// </summary>
    /// <param name="pendingMessages">Messages awaiting delivery</param>
    /// <returns>Deadlock detection result</returns>
    DeadlockResult DetectDeadlock(IReadOnlyList<IPendingMessage> pendingMessages);

    /// <summary>
    /// Finds all messages involved in a deadlock cycle.
    /// </summary>
    /// <param name="pendingMessages">Messages awaiting delivery</param>
    /// <returns>Messages involved in deadlock cycles</returns>
    IReadOnlyList<DeadlockCycle> FindDeadlockCycles(IReadOnlyList<IPendingMessage> pendingMessages);

    /// <summary>
    /// Resolves a deadlock using the specified strategy.
    /// </summary>
    /// <param name="cycle">Deadlock cycle to resolve</param>
    /// <param name="strategy">Resolution strategy</param>
    /// <returns>Resolution result</returns>
    DeadlockResolution ResolveDeadlock(DeadlockCycle cycle, DeadlockResolutionStrategy strategy);

    /// <summary>
    /// Gets deadlock statistics.
    /// </summary>
    DeadlockStatistics GetStatistics();
}

/// <summary>
/// Interface for pending message in deadlock detection.
/// </summary>
public interface IPendingMessage
{
    /// <summary>
    /// Unique message identifier.
    /// </summary>
    Guid MessageId { get; }

    /// <summary>
    /// Actor that sent the message.
    /// </summary>
    ushort SenderId { get; }

    /// <summary>
    /// Message timestamp for causal ordering.
    /// </summary>
    HybridCausalTimestamp Timestamp { get; }

    /// <summary>
    /// Explicit dependencies (message IDs that must be delivered first).
    /// </summary>
    IReadOnlyList<Guid>? Dependencies { get; }
}

/// <summary>
/// Represents a causal event in the system.
/// </summary>
public sealed record CausalEvent
{
    /// <summary>
    /// Unique event identifier.
    /// </summary>
    public required Guid EventId { get; init; }

    /// <summary>
    /// Actor that generated the event.
    /// </summary>
    public required ushort ActorId { get; init; }

    /// <summary>
    /// Event timestamp.
    /// </summary>
    public required HybridCausalTimestamp Timestamp { get; init; }

    /// <summary>
    /// Causal dependencies (events that must happen before this one).
    /// </summary>
    public IReadOnlyList<Guid> CausalDependencies { get; init; } = Array.Empty<Guid>();

    /// <summary>
    /// Event payload/data.
    /// </summary>
    public object? Payload { get; init; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"CausalEvent({EventId}, Actor={ActorId}, Deps={CausalDependencies.Count})";
    }
}

/// <summary>
/// Result of deadlock detection.
/// </summary>
public sealed record DeadlockResult
{
    /// <summary>
    /// Whether a deadlock was detected.
    /// </summary>
    public required bool HasDeadlock { get; init; }

    /// <summary>
    /// Number of messages involved in deadlocks.
    /// </summary>
    public int DeadlockedMessageCount { get; init; }

    /// <summary>
    /// Number of distinct deadlock cycles.
    /// </summary>
    public int CycleCount { get; init; }

    /// <summary>
    /// Detection timestamp.
    /// </summary>
    public DateTimeOffset DetectionTime { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Description of the deadlock.
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Creates a result indicating no deadlock.
    /// </summary>
    public static DeadlockResult NoDeadlock => new() { HasDeadlock = false };

    /// <inheritdoc/>
    public override string ToString()
    {
        return HasDeadlock
            ? $"Deadlock(Messages={DeadlockedMessageCount}, Cycles={CycleCount})"
            : "NoDeadlock";
    }
}

/// <summary>
/// Represents a cycle in the dependency graph causing a deadlock.
/// </summary>
public sealed record DeadlockCycle
{
    /// <summary>
    /// Unique cycle identifier.
    /// </summary>
    public required Guid CycleId { get; init; }

    /// <summary>
    /// Messages involved in the cycle (in cycle order).
    /// </summary>
    public required IReadOnlyList<Guid> MessageIds { get; init; }

    /// <summary>
    /// Actor IDs involved in the cycle.
    /// </summary>
    public required IReadOnlyList<ushort> ActorIds { get; init; }

    /// <summary>
    /// Length of the cycle.
    /// </summary>
    public int Length => MessageIds.Count;

    /// <summary>
    /// Estimated severity (based on cycle length and message age).
    /// </summary>
    public DeadlockSeverity Severity { get; init; } = DeadlockSeverity.Medium;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"DeadlockCycle({CycleId}, Length={Length}, Severity={Severity})";
    }
}

/// <summary>
/// Severity level of a deadlock.
/// </summary>
public enum DeadlockSeverity
{
    /// <summary>
    /// Low severity - short cycle, recent messages.
    /// </summary>
    Low,

    /// <summary>
    /// Medium severity - moderate cycle length or message age.
    /// </summary>
    Medium,

    /// <summary>
    /// High severity - long cycle or aged messages.
    /// </summary>
    High,

    /// <summary>
    /// Critical severity - system-wide impact.
    /// </summary>
    Critical
}

/// <summary>
/// Strategies for resolving deadlocks.
/// </summary>
public enum DeadlockResolutionStrategy
{
    /// <summary>
    /// Drop the oldest message in the cycle.
    /// </summary>
    DropOldest,

    /// <summary>
    /// Drop the newest message in the cycle.
    /// </summary>
    DropNewest,

    /// <summary>
    /// Drop the message with lowest priority.
    /// </summary>
    DropLowestPriority,

    /// <summary>
    /// Force delivery ignoring causal ordering.
    /// </summary>
    ForceDelivery,

    /// <summary>
    /// Wait for external intervention.
    /// </summary>
    WaitForIntervention,

    /// <summary>
    /// Drop all messages in the cycle.
    /// </summary>
    DropAll,

    /// <summary>
    /// Use a deterministic victim selection based on message ID.
    /// </summary>
    DeterministicVictim
}

/// <summary>
/// Result of deadlock resolution.
/// </summary>
public sealed record DeadlockResolution
{
    /// <summary>
    /// Whether resolution was successful.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Strategy that was applied.
    /// </summary>
    public required DeadlockResolutionStrategy Strategy { get; init; }

    /// <summary>
    /// Messages that were dropped/modified to resolve the deadlock.
    /// </summary>
    public IReadOnlyList<Guid> AffectedMessageIds { get; init; } = Array.Empty<Guid>();

    /// <summary>
    /// Messages that can now be delivered.
    /// </summary>
    public IReadOnlyList<Guid> UnblockedMessageIds { get; init; } = Array.Empty<Guid>();

    /// <summary>
    /// Description of the resolution.
    /// </summary>
    public string? Description { get; init; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return Success
            ? $"Resolution(Strategy={Strategy}, Affected={AffectedMessageIds.Count}, Unblocked={UnblockedMessageIds.Count})"
            : $"ResolutionFailed(Strategy={Strategy})";
    }
}

/// <summary>
/// Statistics about deadlock detection and resolution.
/// </summary>
public sealed record DeadlockStatistics
{
    /// <summary>
    /// Total number of deadlock detections performed.
    /// </summary>
    public long TotalDetections { get; init; }

    /// <summary>
    /// Number of deadlocks found.
    /// </summary>
    public long DeadlocksFound { get; init; }

    /// <summary>
    /// Number of successful resolutions.
    /// </summary>
    public long SuccessfulResolutions { get; init; }

    /// <summary>
    /// Number of failed resolutions.
    /// </summary>
    public long FailedResolutions { get; init; }

    /// <summary>
    /// Total messages dropped due to deadlock resolution.
    /// </summary>
    public long TotalMessagesDropped { get; init; }

    /// <summary>
    /// Total messages force-delivered.
    /// </summary>
    public long TotalMessagesForceDelivered { get; init; }

    /// <summary>
    /// Average cycle length of detected deadlocks.
    /// </summary>
    public double AverageCycleLength { get; init; }

    /// <summary>
    /// Maximum cycle length observed.
    /// </summary>
    public int MaxCycleLength { get; init; }

    /// <summary>
    /// Detection rate (deadlocks per detection attempt).
    /// </summary>
    public double DetectionRate => TotalDetections > 0 ? (double)DeadlocksFound / TotalDetections : 0;

    /// <summary>
    /// Resolution success rate.
    /// </summary>
    public double ResolutionSuccessRate => DeadlocksFound > 0 ? (double)SuccessfulResolutions / DeadlocksFound : 0;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"DeadlockStats(Found={DeadlocksFound}, Resolved={SuccessfulResolutions}, Rate={DetectionRate:P1})";
    }
}
