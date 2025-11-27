// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Runtime.Routing;

namespace Orleans.GpuBridge.Runtime.K2K;

/// <summary>
/// Implementation of K2K dispatcher for GPU-to-GPU message passing.
/// This enables sub-microsecond latency communication between GPU-resident actors.
/// </summary>
public sealed class K2KDispatcher : IK2KDispatcher, IDisposable
{
    private readonly ILogger<K2KDispatcher> _logger;
    private readonly ConcurrentDictionary<K2KQueueKey, IntPtr> _queueRegistry;
    private readonly ConcurrentDictionary<string, List<long>> _actorsByType;
    private readonly ReaderWriterLockSlim _registryLock;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the K2K dispatcher.
    /// </summary>
    public K2KDispatcher(ILogger<K2KDispatcher> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _queueRegistry = new ConcurrentDictionary<K2KQueueKey, IntPtr>();
        _actorsByType = new ConcurrentDictionary<string, List<long>>();
        _registryLock = new ReaderWriterLockSlim();
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

        var queuePtr = GetTargetQueuePointer(targetActorType, targetActorId);
        if (queuePtr == IntPtr.Zero)
        {
            _logger.LogWarning(
                "K2K dispatch failed: target queue not found for {ActorType}:{ActorId}",
                targetActorType, targetActorId);
            return ValueTask.CompletedTask;
        }

        // Enqueue message directly to GPU memory
        EnqueueToGpuQueue(queuePtr, ref message);

        _logger.LogTrace(
            "K2K dispatch: {SourceId} -> {TargetType}:{TargetId}.{Method}",
            sourceActorId, targetActorType, targetActorId, targetMethod);

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

        var queuePtr = GetTargetQueuePointer(targetActorType, targetActorId);
        if (queuePtr == IntPtr.Zero)
        {
            _logger.LogWarning(
                "K2K dispatch with response failed: target queue not found for {ActorType}:{ActorId}",
                targetActorType, targetActorId);
            return default;
        }

        // For request-response, we need to set up a response channel
        // This is a simplified implementation - production would use GPU-side completion
        var responseBuffer = new TResponse[1];
        var responseHandle = GCHandle.Alloc(responseBuffer, GCHandleType.Pinned);

        try
        {
            // Enqueue request with response pointer
            EnqueueRequestToGpuQueue(queuePtr, ref request, responseHandle.AddrOfPinnedObject());

            // Wait for response (in production, this would use GPU events)
            // For now, use a simple spin-wait with yield
            var spinWait = new SpinWait();
            var startTime = Environment.TickCount64;
            const int timeoutMs = 5000;

            while (!HasResponse(responseHandle.AddrOfPinnedObject()))
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

            return responseBuffer[0];
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

        var dispatchedCount = 0;

        foreach (var targetId in targetActorIds)
        {
            var queuePtr = GetTargetQueuePointer(targetActorType, targetId);
            if (queuePtr != IntPtr.Zero)
            {
                EnqueueToGpuQueue(queuePtr, ref message);
                dispatchedCount++;
            }
        }

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void EnqueueToGpuQueue<TMessage>(IntPtr queuePtr, ref TMessage message)
        where TMessage : unmanaged
    {
        // In production, this would use GPU atomics to enqueue
        // For now, we write directly to the queue memory
        // The actual GPU kernel will pick up the message

        // Queue format: [head:4][tail:4][capacity:4][data...]
        // We increment tail atomically and write at tail position
        var queueBase = (int*)queuePtr;
        var capacity = queueBase[2];

        if (capacity <= 0)
        {
            return; // Invalid queue
        }

        // Atomic increment of tail (simplified - real impl uses Interlocked on GPU memory)
        var tail = Interlocked.Increment(ref queueBase[1]) - 1;
        var writeIndex = tail % capacity;

        // Write message at calculated position
        var dataStart = (byte*)(queueBase + 3);
        var messageSize = sizeof(TMessage);
        var writePtr = dataStart + (writeIndex * messageSize);

        Unsafe.CopyBlockUnaligned(writePtr, Unsafe.AsPointer(ref message), (uint)messageSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void EnqueueRequestToGpuQueue<TRequest>(
        IntPtr queuePtr,
        ref TRequest request,
        IntPtr responsePtr)
        where TRequest : unmanaged
    {
        // Similar to EnqueueToGpuQueue but includes response pointer in message header
        EnqueueToGpuQueue(queuePtr, ref request);

        // In production, the response pointer would be part of the message envelope
        // The GPU kernel would write the response to this address
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe bool HasResponse(IntPtr responsePtr)
    {
        // Check if response has been written (simplified)
        // In production, this would check a completion flag
        return false; // Placeholder - actual implementation checks GPU memory
    }

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

        _logger.LogInformation("K2K dispatcher disposed");
    }

    /// <summary>
    /// Key for identifying K2K queue registrations.
    /// </summary>
    private readonly record struct K2KQueueKey(string ActorType, long ActorId);
}
