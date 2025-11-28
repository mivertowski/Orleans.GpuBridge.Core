# Kernel Attributes Guide

**Complete reference for DotCompute kernel attributes in Orleans.GpuBridge.Core.**

## Overview

DotCompute provides declarative attributes for GPU kernel configuration. This guide covers all available attributes, their parameters, and usage patterns for temporal actor development.

## Core Attributes

### [Kernel] Attribute

The `[Kernel]` attribute marks a method for GPU execution:

```csharp
[Kernel("kernel-id", PreferredWorkGroupSize = 256)]
public static void MyKernel(ReadOnlySpan<float> input, Span<float> output, int size)
{
    // Kernel implementation
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Id` | `string` | Required | Unique kernel identifier |
| `PreferredWorkGroupSize` | `int` | 256 | Optimal threads per work group |
| `RequiresSharedMemory` | `bool` | false | Enable shared memory allocation |
| `EnableTimestamps` | `bool` | false | Auto-inject GPU timestamps |
| `EnableBarriers` | `bool` | false | Enable device-wide barriers |
| `MemoryOrdering` | `MemoryOrderingMode` | Relaxed | Memory consistency model |
| `SharedMemorySize` | `int` | 0 | Shared memory allocation in bytes |

### [RingKernel] Attribute

The `[RingKernel]` attribute marks a method as a persistent ring kernel:

```csharp
[RingKernel(
    MessageQueueSize = 4096,
    ProcessingMode = RingProcessingMode.Continuous,
    EnableTimestamps = true,
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void ActorDispatchLoop(
    Span<long> timestamps,
    Span<GpuMessage> messageQueue,
    Span<int> queueHead,
    Span<int> queueTail,
    Span<ActorState> states,
    Span<bool> stopSignal)
{
    // Infinite dispatch loop
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MessageQueueSize` | `int` | 4096 | Ring buffer capacity (power of 2) |
| `ProcessingMode` | `RingProcessingMode` | Continuous | Kernel execution mode |
| `MaxMessagesPerIteration` | `int` | 4 | Message batch size |
| `EnableTimestamps` | `bool` | false | Auto-inject GPU timestamps |
| `EnableBarriers` | `bool` | false | Enable device-wide barriers |
| `BarrierScope` | `BarrierScope` | WorkGroup | Barrier synchronization scope |
| `MemoryOrdering` | `MemoryOrderingMode` | Relaxed | Memory consistency model |

## Temporal Attributes

### EnableTimestamps

When `EnableTimestamps = true`, DotCompute automatically injects GPU hardware timestamps:

```csharp
[Kernel(EnableTimestamps = true)]
public static void TimestampedKernel(
    Span<long> timestamps,    // First parameter: auto-injected timestamps
    Span<ActorState> states)
{
    int tid = GetGlobalId(0);
    long gpuTime = timestamps[tid];  // Nanosecond-resolution GPU clock

    // Use for HLC updates
    UpdateHLC(ref states[tid], gpuTime);
}
```

**Key Points:**
- First parameter **must** be `Span<long> timestamps`
- Timestamps are in nanoseconds (1ns resolution)
- Minimal overhead: ~10ns per timestamp query
- Essential for temporal actors

### MemoryOrdering Modes

```csharp
public enum MemoryOrderingMode
{
    /// <summary>
    /// No ordering guarantees. Fastest mode.
    /// Use for independent operations.
    /// </summary>
    Relaxed,

    /// <summary>
    /// Release-acquire semantics. Causal ordering.
    /// RECOMMENDED for temporal actors.
    /// ~15% overhead vs Relaxed.
    /// </summary>
    ReleaseAcquire,

    /// <summary>
    /// Sequential consistency. Total order.
    /// Use only when absolutely required.
    /// ~40% overhead vs Relaxed.
    /// </summary>
    Sequential
}
```

**When to Use Each Mode:**

| Mode | Use Case | Example |
|------|----------|---------|
| `Relaxed` | Independent parallel operations | Matrix multiplication |
| `ReleaseAcquire` | Causal message passing | Actor messaging |
| `Sequential` | Total ordering required | Global consensus |

### EnableBarriers

Device-wide barriers enable multi-actor coordination:

```csharp
[Kernel(EnableBarriers = true, BarrierScope = BarrierScope.Device)]
public static void CoordinatedKernel(
    Span<ActorState> states,
    Span<int> globalCounter)
{
    int actorId = GetGlobalId(0);

    // Phase 1: Local computation
    states[actorId].Value = ComputeUpdate(states[actorId]);

    // DEVICE BARRIER: Wait for ALL threads across GPU
    DeviceBarrier();

    // Phase 2: Global aggregation (only thread 0)
    if (actorId == 0)
    {
        int sum = 0;
        for (int i = 0; i < states.Length; i++)
            sum += states[i].Value;
        globalCounter[0] = sum;
    }

    // DEVICE BARRIER: Wait for aggregation
    DeviceBarrier();

    // Phase 3: All actors read global result
    states[actorId].GlobalSnapshot = globalCounter[0];
}
```

**Barrier Scopes:**

| Scope | Description | Overhead |
|-------|-------------|----------|
| `WorkGroup` | Synchronize threads in same work group | ~1μs |
| `Device` | Synchronize all threads on GPU | ~10μs |

## Pattern Catalog

### Pattern 1: Simple GPU Kernel

```csharp
[Kernel("vector/add", PreferredWorkGroupSize = 256)]
public static void VectorAdd(
    ReadOnlySpan<float> a,
    ReadOnlySpan<float> b,
    Span<float> result,
    int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
}
```

### Pattern 2: Temporal Kernel with HLC

```csharp
[Kernel(
    EnableTimestamps = true,
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void HLCUpdateKernel(
    Span<long> timestamps,
    Span<long> localPhysical,
    Span<long> localLogical,
    Span<ActorMessage> messages)
{
    int actorId = GetGlobalId(0);
    long gpuTime = timestamps[actorId];
    var message = messages[actorId];

    // HLC update algorithm
    long maxPhysical = Max(localPhysical[actorId], message.HLCPhysical, gpuTime);

    if (maxPhysical == localPhysical[actorId] && maxPhysical == message.HLCPhysical)
        localLogical[actorId] = Max(localLogical[actorId], message.HLCLogical) + 1;
    else if (maxPhysical == localPhysical[actorId])
        localLogical[actorId]++;
    else if (maxPhysical == message.HLCPhysical)
        localLogical[actorId] = message.HLCLogical + 1;
    else
        localLogical[actorId] = 0;

    localPhysical[actorId] = maxPhysical;
}
```

### Pattern 3: Ring Kernel Actor

```csharp
[RingKernel(
    MessageQueueSize = 4096,
    ProcessingMode = RingProcessingMode.Continuous,
    EnableTimestamps = true,
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire,
    MaxMessagesPerIteration = 4)]
public static void ActorRingKernel(
    Span<long> timestamps,
    Span<GpuMessage> messageQueue,
    Span<int> queueHead,
    Span<int> queueTail,
    Span<ActorState> states,
    Span<bool> stopSignal)
{
    int actorId = GetGlobalId(0);
    ref var state = ref states[actorId];

    while (!stopSignal[0])
    {
        // Process batch of messages
        for (int i = 0; i < 4; i++)
        {
            int head = AtomicLoad(ref queueHead[actorId]);
            int tail = queueTail[actorId];

            if (head == tail)
                break;

            int idx = tail % 4096;
            var msg = messageQueue[actorId * 4096 + idx];
            long gpuTime = timestamps[actorId];

            ProcessMessage(ref state, msg, gpuTime);

            queueTail[actorId] = tail + 1;
        }

        Yield();
    }
}
```

### Pattern 4: Matrix Operations with Shared Memory

```csharp
[Kernel("matrix/multiply",
    PreferredWorkGroupSize = 16,
    RequiresSharedMemory = true,
    SharedMemorySize = 16 * 16 * 8)]
public static void MatrixMultiply(
    ReadOnlySpan<float> a,
    ReadOnlySpan<float> b,
    Span<float> c,
    int m, int n, int k)
{
    // Tiled matrix multiplication with shared memory
    var tileA = AllocateShared<float>(16 * 16);
    var tileB = AllocateShared<float>(16 * 16);

    int row = GetLocalId(0);
    int col = GetLocalId(1);
    int globalRow = GetGlobalId(0);
    int globalCol = GetGlobalId(1);

    float sum = 0.0f;

    for (int t = 0; t < k; t += 16)
    {
        // Load tile to shared memory
        tileA[row * 16 + col] = a[globalRow * k + t + col];
        tileB[row * 16 + col] = b[(t + row) * n + globalCol];

        Barrier();  // Wait for tile load

        // Compute partial result
        for (int i = 0; i < 16; i++)
            sum += tileA[row * 16 + i] * tileB[i * 16 + col];

        Barrier();  // Wait before loading next tile
    }

    c[globalRow * n + globalCol] = sum;
}
```

### Pattern 5: Causal Message Passing

```csharp
[Kernel(MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void CausalSendKernel(
    Span<ActorMessage> buffer,
    Span<long> timestamps,
    Span<bool> ready,
    int msgId,
    ActorMessage message,
    long timestamp)
{
    // Write message (will be visible before ready flag due to RELEASE)
    buffer[msgId] = message;
    timestamps[msgId] = timestamp;

    // RELEASE fence: Ensure writes complete before setting ready
    ready[msgId] = true;
}

[Kernel(MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void CausalReceiveKernel(
    Span<ActorMessage> buffer,
    Span<long> timestamps,
    Span<bool> ready,
    Span<ActorState> states,
    int actorId)
{
    // ACQUIRE: Check ready flag
    if (!ready[actorId])
        return;

    // ACQUIRE fence: Reads after ready flag will see writes before ready was set
    long timestamp = timestamps[actorId];
    var message = buffer[actorId];

    // Causal ordering guaranteed
    ProcessMessage(ref states[actorId], message, timestamp);
}
```

## Performance Guidelines

### Work Group Size Selection

| Operation Type | Recommended Size | Rationale |
|---------------|------------------|-----------|
| Vector operations | 256-512 | High parallelism, simple ops |
| Matrix operations | 16×16 (256) | Shared memory tiling |
| Reduction | 128-256 | Warp-level primitives |
| Ring kernels | 256 | Balance throughput/latency |
| Image processing | 16×16 or 32×32 | 2D data locality |

### Memory Ordering Overhead

| Mode | Relative Performance | Use Case |
|------|---------------------|----------|
| Relaxed | 100% | Default for independent ops |
| ReleaseAcquire | ~85% | Temporal actor messaging |
| Sequential | ~60% | Avoid unless necessary |

### Barrier Costs

| Barrier Type | Typical Latency | Use Sparingly |
|--------------|----------------|---------------|
| WorkGroup | 1μs | Yes |
| Device | 10μs | Very sparingly |

## Best Practices

### DO

- Use `EnableTimestamps = true` for all temporal actors
- Use `ReleaseAcquire` ordering for causal message passing
- Set `PreferredWorkGroupSize` based on operation type
- Use `MaxMessagesPerIteration` for batched ring kernel processing
- Profile with different configurations

### DON'T

- Don't use `Sequential` ordering unless truly required
- Don't use device barriers in tight loops
- Don't exceed shared memory limits
- Don't ignore timestamp injection for temporal ordering
- Don't mix ordering modes in related operations

## See Also

- [GPU-Native Actors Introduction](introduction/README.md) - Foundational concepts
- [DotCompute Developer Guide](../../developer-guide/GPU-NATIVE-ACTORS.md) - Complete integration guide
- [Ring Kernel Integration](../../architecture/RING-KERNEL-INTEGRATION.md) - Architecture details
- [Temporal Performance](performance/README.md) - Performance benchmarks

---

*Kernel Attributes Guide: Declarative GPU programming for temporal actors.*

**Version**: 0.1.0
**DotCompute Version**: 0.5.1
**Last Updated**: 2025-11-28
