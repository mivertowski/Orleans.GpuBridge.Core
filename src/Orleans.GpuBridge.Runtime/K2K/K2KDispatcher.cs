// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.K2K;
using Orleans.GpuBridge.Runtime.Routing;

namespace Orleans.GpuBridge.Runtime.K2K;

/// <summary>
/// Implementation of K2K dispatcher for GPU-to-GPU message passing.
/// This enables sub-microsecond latency communication between GPU-resident actors.
/// </summary>
/// <remarks>
/// <para>
/// The K2K dispatcher uses GPU-native atomic operations for lock-free message queuing.
/// Queue memory layout matches <see cref="RingKernelQueueHeader"/>:
/// </para>
/// <code>
/// [WriteIndex:4][ReadIndex:4][WriteCommitted:4][Capacity:4][MessageSize:4][Reserved:12][Data...]
/// </code>
/// <para>
/// Writers use atomic increment on WriteIndex to claim a slot, write data,
/// then atomic increment WriteCommitted to signal completion.
/// Readers check WriteCommitted > ReadIndex before reading.
/// </para>
/// </remarks>
public sealed class K2KDispatcher : IK2KDispatcher, IDisposable
{
    private readonly ILogger<K2KDispatcher> _logger;
    private readonly IGpuPeerToPeerMemory? _p2pMemory;
    private readonly ConcurrentDictionary<K2KQueueKey, IntPtr> _queueRegistry;
    private readonly ConcurrentDictionary<string, List<long>> _actorsByType;
    private readonly ConcurrentDictionary<K2KQueueKey, int> _actorDevices;
    private readonly ConcurrentDictionary<(int, int), P2PCapabilityInfo?> _p2pCapabilityCache;
    private readonly ReaderWriterLockSlim _registryLock;
    private bool _disposed;

    // Statistics counters (thread-safe)
    private long _totalDispatches;
    private long _p2pDispatches;
    private long _cpuRoutedDispatches;
    private long _failedDispatches;
    private long _broadcastOperations;
    private long _requestResponseOperations;
    private long _totalLatencyTicks;
    private long _maxLatencyTicks;

    /// <summary>
    /// Completion flag value indicating response is ready.
    /// </summary>
    private const int ResponseReady = 1;

    /// <summary>
    /// Completion flag value indicating response is pending.
    /// </summary>
    private const int ResponsePending = 0;

    /// <summary>
    /// Queue header offset constants (matches RingKernelQueueHeader layout).
    /// </summary>
    private static class QueueOffsets
    {
        /// <summary>Write index offset in bytes.</summary>
        public const int WriteIndex = 0;
        /// <summary>Read index offset in bytes.</summary>
        public const int ReadIndex = 4;
        /// <summary>Write committed offset in bytes.</summary>
        public const int WriteCommitted = 8;
        /// <summary>Capacity offset in bytes.</summary>
        public const int Capacity = 12;
        /// <summary>Message size offset in bytes.</summary>
        public const int MessageSize = 16;
        /// <summary>Total header size in bytes.</summary>
        public const int HeaderSize = 32;
    }

    /// <summary>
    /// Initializes a new instance of the K2K dispatcher.
    /// </summary>
    /// <param name="logger">Logger for dispatcher operations.</param>
    /// <param name="p2pMemory">Optional P2P memory provider for direct GPU-to-GPU transfers.</param>
    public K2KDispatcher(ILogger<K2KDispatcher> logger, IGpuPeerToPeerMemory? p2pMemory = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _p2pMemory = p2pMemory;
        _queueRegistry = new ConcurrentDictionary<K2KQueueKey, IntPtr>();
        _actorsByType = new ConcurrentDictionary<string, List<long>>();
        _actorDevices = new ConcurrentDictionary<K2KQueueKey, int>();
        _p2pCapabilityCache = new ConcurrentDictionary<(int, int), P2PCapabilityInfo?>();
        _registryLock = new ReaderWriterLockSlim();

        if (_p2pMemory != null)
        {
            _logger.LogInformation("K2K dispatcher initialized with P2P memory provider");
        }
        else
        {
            _logger.LogInformation("K2K dispatcher initialized (CPU-routed mode only)");
        }
    }

    /// <inheritdoc />
    public ValueTask DispatchAsync<TMessage>(
        long sourceActorId,
        string targetActorType,
        string targetMethod,
        long targetActorId,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var startTicks = Stopwatch.GetTimestamp();
        var queuePtr = GetTargetQueuePointer(targetActorType, targetActorId);

        if (queuePtr == IntPtr.Zero)
        {
            Interlocked.Increment(ref _failedDispatches);
            _logger.LogWarning(
                "K2K dispatch failed: target queue not found for {ActorType}:{ActorId}",
                targetActorType, targetActorId);
            return ValueTask.CompletedTask;
        }

        // Determine routing path based on device placement
        var sourceKey = new K2KQueueKey(targetActorType, sourceActorId);
        var targetKey = new K2KQueueKey(targetActorType, targetActorId);
        var sourceDevice = _actorDevices.GetValueOrDefault(sourceKey, -1);
        var targetDevice = _actorDevices.GetValueOrDefault(targetKey, -1);

        var usedP2P = false;

        // If both actors are on known devices and we have P2P capability, consider P2P path
        if (sourceDevice >= 0 && targetDevice >= 0 && sourceDevice != targetDevice && _p2pMemory != null)
        {
            var capability = GetCachedP2PCapability(sourceDevice, targetDevice);
            if (capability?.IsEnabled == true)
            {
                // Use P2P path - direct memory write
                usedP2P = true;
                _logger.LogTrace(
                    "K2K P2P dispatch: {SourceId}[dev{SourceDev}] -> {TargetType}:{TargetId}[dev{TargetDev}].{Method}",
                    sourceActorId, sourceDevice, targetActorType, targetActorId, targetDevice, targetMethod);
            }
        }

        // Enqueue message (works for both local and P2P accessible memory)
        EnqueueToGpuQueue(queuePtr, ref message);

        // Update statistics
        Interlocked.Increment(ref _totalDispatches);
        if (usedP2P)
        {
            Interlocked.Increment(ref _p2pDispatches);
        }
        else
        {
            Interlocked.Increment(ref _cpuRoutedDispatches);
        }

        RecordLatency(startTicks);

        _logger.LogTrace(
            "K2K dispatch: {SourceId} -> {TargetType}:{TargetId}.{Method} (P2P: {UsedP2P})",
            sourceActorId, targetActorType, targetActorId, targetMethod, usedP2P);

        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    public async ValueTask<TResponse> DispatchWithResponseAsync<TRequest, TResponse>(
        long sourceActorId,
        string targetActorType,
        string targetMethod,
        long targetActorId,
        TRequest request,
        CancellationToken cancellationToken = default)
        where TRequest : unmanaged
        where TResponse : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        Interlocked.Increment(ref _requestResponseOperations);
        var startTicks = Stopwatch.GetTimestamp();

        var queuePtr = GetTargetQueuePointer(targetActorType, targetActorId);
        if (queuePtr == IntPtr.Zero)
        {
            Interlocked.Increment(ref _failedDispatches);
            _logger.LogWarning(
                "K2K dispatch with response failed: target queue not found for {ActorType}:{ActorId}",
                targetActorType, targetActorId);
            return default;
        }

        // Allocate response slot: [CompletionFlag:4 bytes][Response:sizeof(TResponse)]
        var responseSlotSize = 4 + Unsafe.SizeOf<TResponse>();
        var responseSlot = new byte[responseSlotSize];
        var responseHandle = GCHandle.Alloc(responseSlot, GCHandleType.Pinned);

        try
        {
            var responsePtr = responseHandle.AddrOfPinnedObject();

            // Initialize completion flag to pending (0)
            unsafe
            {
                *(int*)responsePtr = ResponsePending;
            }

            // Enqueue request with response pointer
            EnqueueRequestToGpuQueue(queuePtr, ref request, responsePtr);

            // Wait for response using completion flag
            var spinWait = new SpinWait();
            var startTime = Environment.TickCount64;
            const int timeoutMs = 5000;

            while (!HasResponse(responsePtr))
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    throw new OperationCanceledException(cancellationToken);
                }

                if (Environment.TickCount64 - startTime > timeoutMs)
                {
                    _logger.LogWarning("K2K request-response timeout after {Timeout}ms", timeoutMs);
                    return default;
                }

                spinWait.SpinOnce();

                if (spinWait.NextSpinWillYield)
                {
                    await Task.Yield();
                }
            }

            // Read response from slot (after completion flag)
            Interlocked.Increment(ref _totalDispatches);
            Interlocked.Increment(ref _cpuRoutedDispatches);
            RecordLatency(startTicks);

            unsafe
            {
                var responseDataPtr = (byte*)responsePtr + 4;
                return Unsafe.ReadUnaligned<TResponse>(responseDataPtr);
            }
        }
        finally
        {
            responseHandle.Free();
        }
    }

    /// <inheritdoc />
    public ValueTask BroadcastAsync<TMessage>(
        long sourceActorId,
        string targetActorType,
        string targetMethod,
        ReadOnlySpan<long> targetActorIds,
        TMessage message,
        CancellationToken cancellationToken = default)
        where TMessage : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        Interlocked.Increment(ref _broadcastOperations);
        var startTicks = Stopwatch.GetTimestamp();
        var dispatchedCount = 0;

        foreach (var targetId in targetActorIds)
        {
            var queuePtr = GetTargetQueuePointer(targetActorType, targetId);
            if (queuePtr != IntPtr.Zero)
            {
                EnqueueToGpuQueue(queuePtr, ref message);
                dispatchedCount++;
                Interlocked.Increment(ref _totalDispatches);
                Interlocked.Increment(ref _cpuRoutedDispatches);
            }
            else
            {
                Interlocked.Increment(ref _failedDispatches);
            }
        }

        RecordLatency(startTicks);

        _logger.LogTrace(
            "K2K broadcast from {SourceId} to {Count}/{Total} {TargetType} actors",
            sourceActorId, dispatchedCount, targetActorIds.Length, targetActorType);

        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    public IntPtr GetTargetQueuePointer(string targetActorType, long targetActorId)
    {
        var key = new K2KQueueKey(targetActorType, targetActorId);
        return _queueRegistry.TryGetValue(key, out var ptr) ? ptr : IntPtr.Zero;
    }

    /// <inheritdoc />
    public void RegisterQueue(string actorType, long actorId, IntPtr queuePointer)
    {
        ArgumentException.ThrowIfNullOrEmpty(actorType);

        if (queuePointer == IntPtr.Zero)
        {
            throw new ArgumentException("Queue pointer cannot be zero", nameof(queuePointer));
        }

        var key = new K2KQueueKey(actorType, actorId);

        _registryLock.EnterWriteLock();
        try
        {
            _queueRegistry[key] = queuePointer;

            // Track actors by type for broadcast/ring routing
            var actorList = _actorsByType.GetOrAdd(actorType, _ => new List<long>());
            lock (actorList)
            {
                if (!actorList.Contains(actorId))
                {
                    actorList.Add(actorId);
                }
            }
        }
        finally
        {
            _registryLock.ExitWriteLock();
        }

        _logger.LogDebug("Registered K2K queue for {ActorType}:{ActorId}", actorType, actorId);
    }

    /// <inheritdoc />
    public void UnregisterQueue(string actorType, long actorId)
    {
        var key = new K2KQueueKey(actorType, actorId);

        _registryLock.EnterWriteLock();
        try
        {
            _queueRegistry.TryRemove(key, out _);

            if (_actorsByType.TryGetValue(actorType, out var actorList))
            {
                lock (actorList)
                {
                    actorList.Remove(actorId);
                }
            }
        }
        finally
        {
            _registryLock.ExitWriteLock();
        }

        _logger.LogDebug("Unregistered K2K queue for {ActorType}:{ActorId}", actorType, actorId);
    }

    /// <inheritdoc />
    public void RegisterActorDevice(string actorType, long actorId, int deviceId)
    {
        ArgumentException.ThrowIfNullOrEmpty(actorType);

        if (deviceId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceId), "Device ID must be non-negative");
        }

        var key = new K2KQueueKey(actorType, actorId);
        _actorDevices[key] = deviceId;

        _logger.LogDebug(
            "Registered actor device: {ActorType}:{ActorId} on device {DeviceId}",
            actorType, actorId, deviceId);

        // Preemptively check P2P capabilities for this device
        if (_p2pMemory != null)
        {
            foreach (var existingDevice in _actorDevices.Values.Distinct())
            {
                if (existingDevice != deviceId)
                {
                    // Cache P2P capability in both directions
                    _ = GetCachedP2PCapability(deviceId, existingDevice);
                    _ = GetCachedP2PCapability(existingDevice, deviceId);
                }
            }
        }
    }

    /// <inheritdoc />
    public int GetActorDevice(string actorType, long actorId)
    {
        var key = new K2KQueueKey(actorType, actorId);
        return _actorDevices.TryGetValue(key, out var deviceId) ? deviceId : -1;
    }

    /// <inheritdoc />
    public K2KRoutingStats GetRoutingStats()
    {
        var total = Interlocked.Read(ref _totalDispatches);
        var avgLatencyNs = total > 0
            ? (double)Interlocked.Read(ref _totalLatencyTicks) / total * 1_000_000_000.0 / Stopwatch.Frequency
            : 0.0;

        var maxLatencyNs = (double)Interlocked.Read(ref _maxLatencyTicks) * 1_000_000_000.0 / Stopwatch.Frequency;

        // Count P2P-enabled device pairs
        var p2pEnabledPairs = _p2pCapabilityCache.Values
            .Count(c => c?.IsEnabled == true);

        return new K2KRoutingStats
        {
            TotalDispatches = total,
            P2PDispatches = Interlocked.Read(ref _p2pDispatches),
            CpuRoutedDispatches = Interlocked.Read(ref _cpuRoutedDispatches),
            FailedDispatches = Interlocked.Read(ref _failedDispatches),
            BroadcastOperations = Interlocked.Read(ref _broadcastOperations),
            RequestResponseOperations = Interlocked.Read(ref _requestResponseOperations),
            AverageLatencyNs = avgLatencyNs,
            P99LatencyNs = maxLatencyNs, // Simplified - true P99 would need histogram
            RegisteredQueues = _queueRegistry.Count,
            P2PEnabledPairs = p2pEnabledPairs
        };
    }

    /// <summary>
    /// Gets all registered actor IDs for a given type (for broadcast/ring routing).
    /// </summary>
    public ReadOnlySpan<long> GetActorIdsByType(string actorType)
    {
        if (_actorsByType.TryGetValue(actorType, out var actorList))
        {
            lock (actorList)
            {
                return actorList.ToArray();
            }
        }

        return ReadOnlySpan<long>.Empty;
    }

    /// <summary>
    /// Gets the next actor ID in ring order.
    /// </summary>
    public long GetNextInRing(string actorType, long currentActorId)
    {
        if (!_actorsByType.TryGetValue(actorType, out var actorList))
        {
            return -1;
        }

        lock (actorList)
        {
            if (actorList.Count == 0)
            {
                return -1;
            }

            var index = actorList.IndexOf(currentActorId);
            if (index < 0)
            {
                return actorList[0]; // Not in ring, return first
            }

            return actorList[(index + 1) % actorList.Count];
        }
    }

    /// <summary>
    /// Computes hash-routed target actor ID.
    /// </summary>
    public long GetHashRoutedTarget(string actorType, int messageHash)
    {
        if (!_actorsByType.TryGetValue(actorType, out var actorList))
        {
            return -1;
        }

        lock (actorList)
        {
            if (actorList.Count == 0)
            {
                return -1;
            }

            // Simple consistent hashing - production would use jump hash or similar
            var index = Math.Abs(messageHash) % actorList.Count;
            return actorList[index];
        }
    }

    /// <summary>
    /// Enqueues a message to the GPU queue using atomic operations.
    /// </summary>
    /// <remarks>
    /// Uses the RingKernelQueueHeader layout:
    /// [WriteIndex:4][ReadIndex:4][WriteCommitted:4][Capacity:4][MessageSize:4][Reserved:12][Data...]
    ///
    /// Algorithm:
    /// 1. Atomically increment WriteIndex to claim a slot
    /// 2. Check for queue full condition
    /// 3. Write message data to the claimed slot
    /// 4. Memory barrier to ensure data visibility
    /// 5. Atomically increment WriteCommitted to signal completion
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void EnqueueToGpuQueue<TMessage>(IntPtr queuePtr, ref TMessage message)
        where TMessage : unmanaged
    {
        if (queuePtr == IntPtr.Zero)
        {
            return;
        }

        var header = (int*)queuePtr;
        var capacity = header[QueueOffsets.Capacity / 4];
        var messageSlotSize = header[QueueOffsets.MessageSize / 4];

        if (capacity <= 0)
        {
            return; // Invalid queue
        }

        // Atomically claim a write slot by incrementing WriteIndex
        var claimed = Interlocked.Increment(ref header[QueueOffsets.WriteIndex / 4]) - 1;
        var readIndex = Volatile.Read(ref header[QueueOffsets.ReadIndex / 4]);

        // Check if queue is full (claimed slot would exceed capacity)
        if (claimed - readIndex >= capacity)
        {
            // Queue full - release the claimed slot
            Interlocked.Decrement(ref header[QueueOffsets.WriteIndex / 4]);
            return;
        }

        // Calculate write position using bitwise AND for power-of-2 capacity
        var slotIndex = claimed & (capacity - 1);
        var dataStart = (byte*)queuePtr + QueueOffsets.HeaderSize;
        var writePtr = dataStart + (slotIndex * messageSlotSize);

        // Copy message data to the slot
        var messageSize = sizeof(TMessage);
        var copySize = Math.Min(messageSize, messageSlotSize);
        Unsafe.CopyBlockUnaligned(writePtr, Unsafe.AsPointer(ref message), (uint)copySize);

        // Memory barrier to ensure message data is visible before commit
        Thread.MemoryBarrier();

        // Atomically increment WriteCommitted to signal message is ready
        Interlocked.Increment(ref header[QueueOffsets.WriteCommitted / 4]);
    }

    /// <summary>
    /// Enqueues a request with response pointer to the GPU queue.
    /// </summary>
    /// <remarks>
    /// Creates a combined message that includes both the request data and
    /// the response pointer. The GPU kernel will write the response to
    /// the specified address when processing is complete.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void EnqueueRequestToGpuQueue<TRequest>(
        IntPtr queuePtr,
        ref TRequest request,
        IntPtr responsePtr)
        where TRequest : unmanaged
    {
        // Create a combined request message with envelope header
        var requestWithEnvelope = new K2KRequestWithEnvelope<TRequest>
        {
            Envelope = new K2KRequestEnvelope
            {
                ResponsePtr = (long)responsePtr,
                RequestSize = sizeof(TRequest)
            },
            Request = request
        };

        // Enqueue the combined message
        EnqueueToGpuQueue(queuePtr, ref requestWithEnvelope);
    }

    /// <summary>
    /// Dequeues a message from the GPU queue using atomic operations.
    /// </summary>
    /// <remarks>
    /// Uses the RingKernelQueueHeader layout for lock-free dequeue.
    /// Returns false if queue is empty or if dequeue fails due to contention.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool TryDequeueFromGpuQueue<TMessage>(IntPtr queuePtr, out TMessage message)
        where TMessage : unmanaged
    {
        message = default;

        if (queuePtr == IntPtr.Zero)
        {
            return false;
        }

        var header = (int*)queuePtr;
        var capacity = header[QueueOffsets.Capacity / 4];
        var messageSlotSize = header[QueueOffsets.MessageSize / 4];

        // Check if messages are available
        var writeCommitted = Volatile.Read(ref header[QueueOffsets.WriteCommitted / 4]);
        var readIndex = Volatile.Read(ref header[QueueOffsets.ReadIndex / 4]);

        if (writeCommitted <= readIndex)
        {
            return false; // Queue empty
        }

        // Atomically claim a message by incrementing ReadIndex
        var claimed = Interlocked.Increment(ref header[QueueOffsets.ReadIndex / 4]) - 1;

        // Verify we got a valid message (could race with other readers)
        writeCommitted = Volatile.Read(ref header[QueueOffsets.WriteCommitted / 4]);
        if (claimed >= writeCommitted)
        {
            // No message at this slot - release
            Interlocked.Decrement(ref header[QueueOffsets.ReadIndex / 4]);
            return false;
        }

        // Memory barrier to ensure we see latest message data
        Thread.MemoryBarrier();

        // Calculate read position
        var slotIndex = claimed & (capacity - 1);
        var dataStart = (byte*)queuePtr + QueueOffsets.HeaderSize;
        var readPtr = dataStart + (slotIndex * messageSlotSize);

        // Read message data
        message = Unsafe.ReadUnaligned<TMessage>(readPtr);

        return true;
    }

    /// <summary>
    /// Gets the number of messages available in the queue.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe int GetQueueMessageCount(IntPtr queuePtr)
    {
        if (queuePtr == IntPtr.Zero)
        {
            return 0;
        }

        var header = (int*)queuePtr;
        var writeCommitted = Volatile.Read(ref header[QueueOffsets.WriteCommitted / 4]);
        var readIndex = Volatile.Read(ref header[QueueOffsets.ReadIndex / 4]);

        return Math.Max(0, writeCommitted - readIndex);
    }

    /// <summary>
    /// Initializes a queue header at the specified memory location.
    /// </summary>
    /// <param name="queuePtr">Pointer to the queue memory.</param>
    /// <param name="capacity">Queue capacity (must be power of 2).</param>
    /// <param name="messageSize">Size of each message slot in bytes.</param>
    public static unsafe void InitializeQueue(IntPtr queuePtr, int capacity, int messageSize)
    {
        if (queuePtr == IntPtr.Zero)
        {
            throw new ArgumentException("Queue pointer cannot be zero", nameof(queuePtr));
        }

        if (capacity <= 0 || (capacity & (capacity - 1)) != 0)
        {
            throw new ArgumentException("Capacity must be a positive power of 2", nameof(capacity));
        }

        if (messageSize <= 0 || messageSize % 4 != 0)
        {
            throw new ArgumentException("Message size must be a positive multiple of 4", nameof(messageSize));
        }

        var header = (int*)queuePtr;

        // Initialize header atomically
        Volatile.Write(ref header[QueueOffsets.WriteIndex / 4], 0);
        Volatile.Write(ref header[QueueOffsets.ReadIndex / 4], 0);
        Volatile.Write(ref header[QueueOffsets.WriteCommitted / 4], 0);
        header[QueueOffsets.Capacity / 4] = capacity;
        header[QueueOffsets.MessageSize / 4] = messageSize;

        // Clear reserved fields
        header[5] = 0;
        header[6] = 0;
        header[7] = 0;

        // Memory barrier to ensure initialization is visible
        Thread.MemoryBarrier();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe bool HasResponse(IntPtr responsePtr)
    {
        // Response slot layout: [CompletionFlag:4 bytes][Response data...]
        // Check if completion flag has been set to ResponseReady (1)
        var completionFlag = Volatile.Read(ref *(int*)responsePtr);
        return completionFlag == ResponseReady;
    }

    /// <summary>
    /// Writes a response to a K2K response slot.
    /// Called by GPU kernel (via CPU fallback) or ring kernel runtime.
    /// </summary>
    /// <typeparam name="TResponse">The response type.</typeparam>
    /// <param name="responsePtr">Pointer to the response slot.</param>
    /// <param name="response">The response data.</param>
    public static unsafe void WriteResponse<TResponse>(IntPtr responsePtr, TResponse response)
        where TResponse : unmanaged
    {
        // Response slot layout: [CompletionFlag:4 bytes][Response data...]
        var dataPtr = (byte*)responsePtr + 4;

        // Write response data first
        Unsafe.WriteUnaligned(dataPtr, response);

        // Memory barrier to ensure response data is visible before completion flag
        Thread.MemoryBarrier();

        // Set completion flag to signal response is ready
        Volatile.Write(ref *(int*)responsePtr, ResponseReady);
    }

    /// <summary>
    /// Envelope for K2K request messages that include response pointer.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct K2KRequestEnvelope
    {
        /// <summary>
        /// Pointer to the response slot where GPU kernel should write the response.
        /// </summary>
        public long ResponsePtr;

        /// <summary>
        /// Size of the request data that follows this envelope.
        /// </summary>
        public int RequestSize;

        /// <summary>
        /// Reserved for alignment.
        /// </summary>
        private int _reserved;
    }

    /// <summary>
    /// Combined request message with envelope header for K2K request-response pattern.
    /// </summary>
    /// <typeparam name="TRequest">The request type (must be unmanaged).</typeparam>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct K2KRequestWithEnvelope<TRequest>
        where TRequest : unmanaged
    {
        /// <summary>
        /// The envelope containing response pointer and metadata.
        /// </summary>
        public K2KRequestEnvelope Envelope;

        /// <summary>
        /// The actual request data.
        /// </summary>
        public TRequest Request;
    }

    #region Private Helper Methods

    /// <summary>
    /// Gets P2P capability from cache, or queries the P2P memory provider and caches the result.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private P2PCapabilityInfo? GetCachedP2PCapability(int sourceDevice, int targetDevice)
    {
        if (_p2pMemory == null)
        {
            return null;
        }

        var key = (sourceDevice, targetDevice);
        return _p2pCapabilityCache.GetOrAdd(key, k => _p2pMemory.GetP2PCapability(k.Item1, k.Item2));
    }

    /// <summary>
    /// Records latency for statistics tracking.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void RecordLatency(long startTicks)
    {
        var elapsedTicks = Stopwatch.GetTimestamp() - startTicks;
        Interlocked.Add(ref _totalLatencyTicks, elapsedTicks);

        // Update max latency using compare-and-swap pattern
        long currentMax;
        do
        {
            currentMax = Interlocked.Read(ref _maxLatencyTicks);
            if (elapsedTicks <= currentMax)
            {
                break;
            }
        } while (Interlocked.CompareExchange(ref _maxLatencyTicks, elapsedTicks, currentMax) != currentMax);
    }

    #endregion

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _registryLock.Dispose();
        _queueRegistry.Clear();
        _actorsByType.Clear();
        _actorDevices.Clear();
        _p2pCapabilityCache.Clear();

        _logger.LogInformation("K2K dispatcher disposed");
    }

    /// <summary>
    /// Key for identifying K2K queue registrations.
    /// </summary>
    private readonly record struct K2KQueueKey(string ActorType, long ActorId);
}
