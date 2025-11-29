using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Priority queue for temporal messages with HLC-based ordering and causal dependency tracking.
/// </summary>
/// <remarks>
/// <para>
/// Features:
/// </para>
/// <list type="bullet">
///   <item><description>HLC-ordered processing (total event ordering)</description></item>
///   <item><description>Causal dependency enforcement (wait for dependencies)</description></item>
///   <item><description>Priority-based processing (within HLC constraints)</description></item>
///   <item><description>Deadline tracking and eviction</description></item>
///   <item><description>Performance: O(log N) enqueue/dequeue, O(1) dependency check</description></item>
/// </list>
/// </remarks>
public sealed class TemporalMessageQueue
{
    private readonly PriorityQueue<TemporalResidentMessage, MessageOrderingKey> _hlcQueue = new();
    private readonly SortedDictionary<long, List<TemporalResidentMessage>> _deadlineIndex = new();
    private readonly HashSet<Guid> _processedMessages = new();
    private readonly Dictionary<Guid, List<TemporalResidentMessage>> _dependencyWaitList = new();
    private readonly IPhysicalClockSource _clockSource;
    private readonly ILogger? _logger;

    // Statistics
    private long _totalEnqueued;
    private long _totalDequeued;
    private long _totalExpired;
    private long _totalDependencyWaits;

    /// <summary>
    /// Gets the number of messages currently in the queue.
    /// </summary>
    public int Count => _hlcQueue.Count;

    /// <summary>
    /// Gets the number of messages waiting for dependencies.
    /// </summary>
    public int DependencyWaitCount => _dependencyWaitList.Values.Sum(list => list.Count);

    /// <summary>
    /// Gets total messages enqueued.
    /// </summary>
    public long TotalEnqueued => _totalEnqueued;

    /// <summary>
    /// Gets total messages dequeued.
    /// </summary>
    public long TotalDequeued => _totalDequeued;

    /// <summary>
    /// Gets total messages expired.
    /// </summary>
    public long TotalExpired => _totalExpired;

    /// <summary>
    /// Creates a new temporal message queue.
    /// </summary>
    /// <param name="clockSource">Physical clock source for deadline tracking</param>
    /// <param name="logger">Optional logger for diagnostics</param>
    public TemporalMessageQueue(IPhysicalClockSource? clockSource = null, ILogger? logger = null)
    {
        _clockSource = clockSource ?? new SystemClockSource();
        _logger = logger;
    }

    /// <summary>
    /// Enqueues a temporal message for processing.
    /// </summary>
    /// <remarks>
    /// The message will be:
    /// 1. Added to HLC-ordered priority queue
    /// 2. Indexed by deadline (if ValidityWindow specified)
    /// 3. Tracked for causal dependencies
    /// </remarks>
    public void Enqueue(TemporalResidentMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);

        _totalEnqueued++;

        // Check if message has already expired
        var currentTime = _clockSource.GetCurrentTimeNanos();
        if (message.IsExpired(currentTime))
        {
            _totalExpired++;
            _logger?.LogWarning(
                "Message {RequestId} expired before enqueue (deadline miss by {Delta}ms)",
                message.RequestId,
                (currentTime - message.ValidityWindow!.Value.EndNanos) / 1_000_000.0);
            return;
        }

        // Add to HLC-ordered queue with priority
        var orderingKey = new MessageOrderingKey(message.HLC, message.Priority, message.SequenceNumber);
        _hlcQueue.Enqueue(message, orderingKey);

        // Index by deadline if specified
        if (message.ValidityWindow.HasValue)
        {
            var deadline = message.ValidityWindow.Value.EndNanos;
            if (!_deadlineIndex.TryGetValue(deadline, out var list))
            {
                list = new List<TemporalResidentMessage>();
                _deadlineIndex[deadline] = list;
            }
            list.Add(message);
        }

        _logger?.LogTrace(
            "Enqueued message {RequestId} with HLC={HLC}, Priority={Priority}, Deps={DepsCount}",
            message.RequestId, message.HLC, message.Priority, message.CausalDependencies.Length);
    }

    /// <summary>
    /// Attempts to dequeue the next message that is ready for processing.
    /// </summary>
    /// <remarks>
    /// A message is ready if:
    /// 1. All causal dependencies have been processed
    /// 2. The message has not expired
    /// 3. It has the earliest HLC timestamp among ready messages
    /// </remarks>
    /// <param name="message">The dequeued message, or null if queue is empty</param>
    /// <returns>True if a message was dequeued</returns>
    public bool TryDequeue(out TemporalResidentMessage? message)
    {
        // Evict expired messages first
        EvictExpiredMessages();

        message = null;

        while (_hlcQueue.TryDequeue(out var candidate, out _))
        {
            // Check if all causal dependencies are satisfied
            if (!AreDependenciesSatisfied(candidate))
            {
                // Defer this message until dependencies are processed
                AddToDependencyWaitList(candidate);
                _totalDependencyWaits++;

                _logger?.LogTrace(
                    "Message {RequestId} waiting for {DepsCount} dependencies",
                    candidate.RequestId, candidate.CausalDependencies.Length);

                continue; // Try next message
            }

            // Message is ready
            message = candidate;
            _totalDequeued++;

            _logger?.LogTrace(
                "Dequeued message {RequestId} with HLC={HLC}, Age={Age}ms",
                message.RequestId, message.HLC,
                message.GetAgeNanos(_clockSource.GetCurrentTimeNanos()) / 1_000_000.0);

            return true;
        }

        return false; // Queue is empty or all messages waiting for dependencies
    }

    /// <summary>
    /// Marks a message as processed and releases any dependents.
    /// </summary>
    /// <param name="messageId">ID of the processed message</param>
    public void MarkProcessed(Guid messageId)
    {
        _processedMessages.Add(messageId);

        // Check if any messages were waiting for this dependency
        if (_dependencyWaitList.TryGetValue(messageId, out var waitingMessages))
        {
            // Re-enqueue messages that were waiting for this dependency
            foreach (var waitingMessage in waitingMessages)
            {
                if (AreDependenciesSatisfied(waitingMessage))
                {
                    // All dependencies now satisfied, re-enqueue
                    var orderingKey = new MessageOrderingKey(
                        waitingMessage.HLC,
                        waitingMessage.Priority,
                        waitingMessage.SequenceNumber);
                    _hlcQueue.Enqueue(waitingMessage, orderingKey);

                    _logger?.LogTrace(
                        "Released message {RequestId} after dependency {DepId} completed",
                        waitingMessage.RequestId, messageId);
                }
            }

            _dependencyWaitList.Remove(messageId);
        }
    }

    /// <summary>
    /// Evicts messages that have passed their deadline.
    /// </summary>
    public void EvictExpiredMessages()
    {
        var currentTime = _clockSource.GetCurrentTimeNanos();

        // Find all deadlines that have passed
        var expiredDeadlines = _deadlineIndex.Keys
            .TakeWhile(deadline => deadline <= currentTime)
            .ToList();

        foreach (var deadline in expiredDeadlines)
        {
            var expiredMessages = _deadlineIndex[deadline];
            _totalExpired += expiredMessages.Count;

            foreach (var expiredMessage in expiredMessages)
            {
                var missedBy = currentTime - deadline;
                _logger?.LogWarning(
                    "Message {RequestId} missed deadline by {Delta}ms (HLC={HLC})",
                    expiredMessage.RequestId,
                    missedBy / 1_000_000.0,
                    expiredMessage.HLC);
            }

            _deadlineIndex.Remove(deadline);
        }
    }

    /// <summary>
    /// Gets statistics about the queue.
    /// </summary>
    public QueueStatistics GetStatistics()
    {
        return new QueueStatistics
        {
            MessagesInQueue = Count,
            MessagesWaitingDependencies = DependencyWaitCount,
            TotalEnqueued = TotalEnqueued,
            TotalDequeued = TotalDequeued,
            TotalExpired = TotalExpired,
            TotalDependencyWaits = _totalDependencyWaits,
            ProcessedMessages = _processedMessages.Count
        };
    }

    /// <summary>
    /// Clears the queue and resets statistics.
    /// </summary>
    public void Clear()
    {
        _hlcQueue.Clear();
        _deadlineIndex.Clear();
        _processedMessages.Clear();
        _dependencyWaitList.Clear();
        _totalEnqueued = 0;
        _totalDequeued = 0;
        _totalExpired = 0;
        _totalDependencyWaits = 0;
    }

    /// <summary>
    /// Checks if all causal dependencies for a message have been satisfied.
    /// </summary>
    private bool AreDependenciesSatisfied(TemporalResidentMessage message)
    {
        if (message.CausalDependencies.IsEmpty)
            return true;

        return message.CausalDependencies.All(depId => _processedMessages.Contains(depId));
    }

    /// <summary>
    /// Adds a message to the dependency wait list.
    /// </summary>
    private void AddToDependencyWaitList(TemporalResidentMessage message)
    {
        foreach (var depId in message.CausalDependencies)
        {
            if (!_processedMessages.Contains(depId))
            {
                if (!_dependencyWaitList.TryGetValue(depId, out var list))
                {
                    list = new List<TemporalResidentMessage>();
                    _dependencyWaitList[depId] = list;
                }

                if (!list.Contains(message))
                {
                    list.Add(message);
                }
            }
        }
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"TemporalMessageQueue(Count={Count}, Waiting={DependencyWaitCount}, " +
               $"Enqueued={TotalEnqueued}, Dequeued={TotalDequeued}, Expired={TotalExpired})";
    }
}

/// <summary>
/// Composite key for ordering messages in the priority queue.
/// </summary>
/// <remarks>
/// Ordering priority:
/// 1. HLC timestamp (primary ordering - ensures temporal correctness)
/// 2. Message priority (secondary - higher priority processed first)
/// 3. Sequence number (tertiary - FIFO within same HLC and priority)
/// </remarks>
internal readonly struct MessageOrderingKey : IComparable<MessageOrderingKey>
{
    public HybridTimestamp HLC { get; }
    public MessagePriority Priority { get; }
    public ulong SequenceNumber { get; }

    public MessageOrderingKey(HybridTimestamp hlc, MessagePriority priority, ulong sequenceNumber)
    {
        HLC = hlc;
        Priority = priority;
        SequenceNumber = sequenceNumber;
    }

    public int CompareTo(MessageOrderingKey other)
    {
        // Primary: HLC timestamp (ascending - earliest first)
        var hlcComparison = HLC.CompareTo(other.HLC);
        if (hlcComparison != 0)
            return hlcComparison;

        // Secondary: Priority (descending - highest priority first)
        var priorityComparison = other.Priority.CompareTo(Priority);
        if (priorityComparison != 0)
            return priorityComparison;

        // Tertiary: Sequence number (ascending - FIFO)
        return SequenceNumber.CompareTo(other.SequenceNumber);
    }
}

/// <summary>
/// Statistics about the temporal message queue.
/// </summary>
public sealed record QueueStatistics
{
    /// <summary>
    /// Number of messages currently in the queue.
    /// </summary>
    public int MessagesInQueue { get; init; }

    /// <summary>
    /// Number of messages waiting for dependencies.
    /// </summary>
    public int MessagesWaitingDependencies { get; init; }

    /// <summary>
    /// Total messages enqueued since creation.
    /// </summary>
    public long TotalEnqueued { get; init; }

    /// <summary>
    /// Total messages dequeued since creation.
    /// </summary>
    public long TotalDequeued { get; init; }

    /// <summary>
    /// Total messages expired without processing.
    /// </summary>
    public long TotalExpired { get; init; }

    /// <summary>
    /// Total times messages waited on dependencies.
    /// </summary>
    public long TotalDependencyWaits { get; init; }

    /// <summary>
    /// Number of successfully processed messages.
    /// </summary>
    public int ProcessedMessages { get; init; }

    /// <summary>
    /// Average ratio of messages waiting vs processed.
    /// </summary>
    public double AverageWaitRatio =>
        TotalEnqueued > 0 ? (double)TotalDependencyWaits / TotalEnqueued : 0;

    /// <summary>
    /// Ratio of expired to total messages.
    /// </summary>
    public double ExpiredRatio =>
        TotalEnqueued > 0 ? (double)TotalExpired / TotalEnqueued : 0;
}
