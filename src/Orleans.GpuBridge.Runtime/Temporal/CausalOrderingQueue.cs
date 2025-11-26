using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Temporal;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Message queue that enforces causal ordering delivery using vector clocks.
/// </summary>
/// <remarks>
/// <para>
/// This queue ensures that messages are delivered in an order consistent with
/// causality. A message is only delivered when all its causal dependencies
/// (messages that happened-before it) have been delivered.
/// </para>
/// <para>
/// Features:
/// - Causal dependency tracking using vector clocks
/// - Automatic message buffering until dependencies satisfied
/// - Deadlock detection (circular dependencies)
/// - Message timeout and eviction
/// - Statistics tracking
/// </para>
/// <para>
/// Use cases:
/// - Multi-actor distributed systems
/// - Event sourcing with causal consistency
/// - Collaborative editing (conflict detection)
/// - Distributed transactions
/// </para>
/// </remarks>
public sealed class CausalOrderingQueue
{
    private readonly HybridCausalClock _localClock;
    private readonly List<CausalMessage> _pendingMessages = new();
    private readonly List<CausalMessage> _deliveredMessages = new();
    private readonly ILogger? _logger;

    private readonly long _maxPendingMessages;
    private readonly long _messageTimeoutNanos;

    // Statistics
    private long _totalEnqueued;
    private long _totalDelivered;
    private long _totalTimedOut;
    private long _totalCausalViolations;

    /// <summary>
    /// Gets the number of pending messages.
    /// </summary>
    public int PendingCount => _pendingMessages.Count;

    /// <summary>
    /// Gets the number of delivered messages (retained for history).
    /// </summary>
    public int DeliveredCount => _deliveredMessages.Count;

    /// <summary>
    /// Gets the local actor's vector clock.
    /// </summary>
    public VectorClock CurrentVectorClock => _localClock.VectorClock;

    /// <summary>
    /// Creates a new causal ordering queue.
    /// </summary>
    /// <param name="localClock">Local actor's hybrid causal clock</param>
    /// <param name="maxPendingMessages">Maximum messages to buffer (prevents memory issues)</param>
    /// <param name="messageTimeoutNanos">Message timeout in nanoseconds (0 = no timeout)</param>
    /// <param name="logger">Optional logger</param>
    public CausalOrderingQueue(
        HybridCausalClock localClock,
        long maxPendingMessages = 10_000,
        long messageTimeoutNanos = 60_000_000_000, // 60 seconds default
        ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(localClock);

        _localClock = localClock;
        _maxPendingMessages = maxPendingMessages;
        _messageTimeoutNanos = messageTimeoutNanos;
        _logger = logger;
    }

    /// <summary>
    /// Enqueues a message for causal delivery.
    /// </summary>
    /// <param name="message">Message to enqueue</param>
    /// <returns>
    /// List of messages ready for delivery (may include this message and others
    /// that were waiting for dependencies).
    /// </returns>
    /// <remarks>
    /// The queue will attempt to deliver this message immediately if dependencies
    /// are satisfied. If not, it will be buffered until dependencies arrive.
    /// </remarks>
    public async Task<IReadOnlyList<CausalMessage>> EnqueueAsync(
        CausalMessage message,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(message);

        _totalEnqueued++;

        // Check for timeout (message too old)
        if (_messageTimeoutNanos > 0)
        {
            var age = _localClock.VectorClock[_localClock.ActorId] -
                      message.Timestamp.VectorClock[message.SenderId];

            if (age > _messageTimeoutNanos)
            {
                _totalTimedOut++;
                _logger?.LogWarning(
                    "Message from actor {SenderId} timed out (age: {AgeMs}ms)",
                    message.SenderId, age / 1_000_000);
                return Array.Empty<CausalMessage>();
            }
        }

        // Add to pending queue
        _pendingMessages.Add(message);

        // Check if we exceed max pending
        if (_pendingMessages.Count > _maxPendingMessages)
        {
            // Remove oldest message by HLC
            var oldest = _pendingMessages.OrderBy(m => m.Timestamp.HLC).First();
            _pendingMessages.Remove(oldest);
            _totalTimedOut++;

            _logger?.LogWarning(
                "Pending queue full, evicted oldest message from actor {ActorId}",
                oldest.SenderId);
        }

        // Try to deliver ready messages
        var delivered = await TryDeliverReadyMessagesAsync(ct);

        return delivered;
    }

    /// <summary>
    /// Attempts to deliver all messages whose dependencies are satisfied.
    /// </summary>
    /// <returns>List of messages that were delivered</returns>
    private Task<IReadOnlyList<CausalMessage>> TryDeliverReadyMessagesAsync(
        CancellationToken ct = default)
    {
        var delivered = new List<CausalMessage>();
        bool progress = true;

        while (progress && _pendingMessages.Count > 0)
        {
            progress = false;

            for (int i = _pendingMessages.Count - 1; i >= 0; i--)
            {
                ct.ThrowIfCancellationRequested();

                var message = _pendingMessages[i];

                // Check if dependencies are satisfied
                if (CanDeliver(message))
                {
                    // Remove from pending
                    _pendingMessages.RemoveAt(i);

                    // Update local clock
                    var receiveTimestamp = _localClock.Update(message.Timestamp);

                    // Mark as delivered
                    message.SetDelivered(receiveTimestamp);
                    _deliveredMessages.Add(message);
                    delivered.Add(message);
                    _totalDelivered++;

                    _logger?.LogDebug(
                        "Delivered message from actor {SenderId} (VC: {VC})",
                        message.SenderId, message.Timestamp.VectorClock);

                    progress = true;
                }
            }
        }

        // Trim delivered messages history if too large
        if (_deliveredMessages.Count > 1000)
        {
            _deliveredMessages.RemoveRange(0, _deliveredMessages.Count - 1000);
        }

        return Task.FromResult<IReadOnlyList<CausalMessage>>(delivered);
    }

    /// <summary>
    /// Checks if a message can be delivered based on causal dependencies.
    /// </summary>
    /// <param name="message">Message to check</param>
    /// <returns>True if all dependencies are satisfied</returns>
    private bool CanDeliver(CausalMessage message)
    {
        // Message can be delivered if we've observed all events in its vector clock
        // (except the sender's current event)
        var messageVC = message.Timestamp.VectorClock;
        var currentVC = _localClock.VectorClock;

        foreach (var actorId in messageVC.ActorIds)
        {
            // For the sender: we should have seen all events *before* this one
            if (actorId == message.SenderId)
            {
                var senderClock = messageVC[actorId];
                var ourKnowledge = currentVC[actorId];

                // We should know about senderClock - 1 events from the sender
                if (ourKnowledge < senderClock - 1)
                {
                    return false; // Missing prior messages from sender
                }
            }
            else
            {
                // For other actors: we should have seen all events the sender saw
                if (currentVC[actorId] < messageVC[actorId])
                {
                    return false; // Missing causal dependencies
                }
            }
        }

        // Check explicit dependencies if any
        if (message.Dependencies != null && message.Dependencies.Count > 0)
        {
            foreach (var depId in message.Dependencies)
            {
                if (!_deliveredMessages.Any(m => m.MessageId == depId))
                {
                    return false; // Explicit dependency not yet delivered
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Detects potential deadlocks (circular dependencies).
    /// </summary>
    /// <returns>True if a deadlock is detected</returns>
    public bool DetectDeadlock()
    {
        // Simple deadlock detection: if all pending messages have unsatisfied dependencies
        // and none can be delivered, we have a deadlock

        if (_pendingMessages.Count == 0)
            return false;

        // Check if any message can be delivered
        foreach (var message in _pendingMessages)
        {
            if (CanDeliver(message))
                return false; // At least one can be delivered - no deadlock
        }

        // All messages are blocked - potential deadlock
        _logger?.LogError(
            "Potential deadlock detected: {Count} messages pending, none deliverable",
            _pendingMessages.Count);

        return true;
    }

    /// <summary>
    /// Gets statistics about the queue.
    /// </summary>
    public CausalQueueStatistics GetStatistics()
    {
        return new CausalQueueStatistics
        {
            TotalEnqueued = _totalEnqueued,
            TotalDelivered = _totalDelivered,
            TotalTimedOut = _totalTimedOut,
            TotalCausalViolations = _totalCausalViolations,
            PendingCount = _pendingMessages.Count,
            DeliveredCount = _deliveredMessages.Count,
            CurrentVectorClock = _localClock.VectorClock
        };
    }

    /// <summary>
    /// Clears all messages.
    /// </summary>
    public void Clear()
    {
        _pendingMessages.Clear();
        _deliveredMessages.Clear();
        _totalEnqueued = 0;
        _totalDelivered = 0;
        _totalTimedOut = 0;
        _totalCausalViolations = 0;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"CausalOrderingQueue(Pending={_pendingMessages.Count}, " +
               $"Delivered={_totalDelivered}, VC={_localClock.VectorClock})";
    }
}

/// <summary>
/// Represents a message with causal ordering information.
/// </summary>
public sealed class CausalMessage : IPendingMessage
{
    /// <summary>
    /// Unique message identifier.
    /// </summary>
    public required Guid MessageId { get; init; }

    /// <summary>
    /// Actor that sent the message.
    /// </summary>
    public required ushort SenderId { get; init; }

    /// <summary>
    /// Timestamp when message was sent.
    /// </summary>
    public required HybridCausalTimestamp Timestamp { get; init; }

    /// <summary>
    /// Message payload.
    /// </summary>
    public required object Payload { get; init; }

    /// <summary>
    /// Explicit causal dependencies (message IDs that must be delivered first).
    /// </summary>
    public IReadOnlyList<Guid>? Dependencies { get; init; }

    /// <summary>
    /// Timestamp when message was delivered (null if not yet delivered).
    /// </summary>
    public HybridCausalTimestamp? DeliveryTimestamp { get; private set; }

    /// <summary>
    /// Whether the message has been delivered.
    /// </summary>
    public bool IsDelivered => DeliveryTimestamp != null;

    /// <summary>
    /// Delivery latency in nanoseconds (available after delivery).
    /// </summary>
    public long? DeliveryLatencyNanos => DeliveryTimestamp != null
        ? DeliveryTimestamp.PhysicalTimeNanos - Timestamp.PhysicalTimeNanos
        : null;

    /// <summary>
    /// Marks the message as delivered.
    /// </summary>
    /// <param name="deliveryTimestamp">Timestamp at delivery</param>
    internal void SetDelivered(HybridCausalTimestamp deliveryTimestamp)
    {
        DeliveryTimestamp = deliveryTimestamp;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var status = IsDelivered ? "Delivered" : "Pending";
        return $"CausalMessage({MessageId}, Sender={SenderId}, {status})";
    }
}

/// <summary>
/// Statistics about causal ordering queue.
/// </summary>
public sealed record CausalQueueStatistics
{
    public long TotalEnqueued { get; init; }
    public long TotalDelivered { get; init; }
    public long TotalTimedOut { get; init; }
    public long TotalCausalViolations { get; init; }
    public int PendingCount { get; init; }
    public int DeliveredCount { get; init; }
    public required VectorClock CurrentVectorClock { get; init; }

    public double DeliveryRate => TotalEnqueued > 0
        ? (double)TotalDelivered / TotalEnqueued
        : 0.0;

    public override string ToString()
    {
        return $"CausalQueueStats(Enqueued={TotalEnqueued}, " +
               $"Delivered={TotalDelivered}, DeliveryRate={DeliveryRate:P0})";
    }
}
