# DotCompute Kernel Attributes Guide
## [Kernel] and [RingKernel] Attribute Patterns for Temporal Correctness

## Overview

DotCompute 0.4.2-rc2 introduces powerful attribute-based configuration for GPU kernels, enabling temporal correctness features through declarative syntax. This guide covers all attribute patterns for temporal actor systems.

---

## Table of Contents

1. [[Kernel] Attribute](#kernel-attribute)
2. [[RingKernel] Attribute](#ringkernel-attribute)
3. [Timing Features](#timing-features)
4. [Barrier Synchronization](#barrier-synchronization)
5. [Memory Ordering](#memory-ordering)
6. [Complete Patterns Reference](#complete-patterns-reference)
7. [Best Practices](#best-practices)

---

## [Kernel] Attribute

The `[Kernel]` attribute configures GPU kernels with temporal correctness features.

### Basic Syntax

```csharp
[Kernel]
public static void MyKernel(Span<float> data)
{
    // Basic GPU kernel
}
```

### Available Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `EnableTimestamps` | `bool` | `false` | Auto-inject GPU timestamps at kernel entry |
| `EnableBarriers` | `bool` | `false` | Enable device-wide barriers |
| `BarrierScope` | `BarrierScope` | `ThreadBlock` | Scope of barrier synchronization |
| `MemoryOrdering` | `MemoryOrderingMode` | `Relaxed` | Memory consistency model |
| `SharedMemorySize` | `int` | `0` | Shared memory allocation (bytes) |
| `PreferredWorkGroupSize` | `int` | `null` | Hint for work group size optimization |

---

## [RingKernel] Attribute

The `[RingKernel]` attribute creates **persistent GPU threads** that run in an infinite loop, processing messages as they arrive.

### Basic Syntax

```csharp
[RingKernel(MessageQueueSize = 1024)]
public static void MyRingKernel(
    Span<Message> messageQueue,
    Span<int> queueHead,
    Span<int> queueTail)
{
    // Infinite dispatch loop - runs forever until stopped
    while (!stopSignal)
    {
        ProcessNextMessage();
    }
}
```

### Available Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `MessageQueueSize` | `int` | `1024` | Size of ring buffer for message queue |
| `ProcessingMode` | `RingProcessingMode` | `Continuous` | Message processing strategy |
| `EnableTimestamps` | `bool` | `false` | Auto-inject timestamps for message timing |
| `EnableBarriers` | `bool` | `false` | Enable barriers for synchronized processing |
| `MemoryOrdering` | `MemoryOrderingMode` | `ReleaseAcquire` | Memory consistency (default: causal) |
| `YieldStrategy` | `YieldStrategy` | `Spin` | CPU/GPU yield strategy when queue empty |
| `MaxMessagesPerIteration` | `int` | `1` | Batch size for message processing |

### Processing Modes

```csharp
public enum RingProcessingMode
{
    /// <summary>
    /// Process messages continuously until stopped (infinite loop).
    /// </summary>
    Continuous,

    /// <summary>
    /// Process messages in batches with periodic yields.
    /// </summary>
    Batched,

    /// <summary>
    /// Process single message and return (for testing/debugging).
    /// </summary>
    SingleShot
}
```

---

## Timing Features

### Pattern 1: Automatic Timestamp Injection

**Use Case**: Record GPU entry time for every kernel invocation.

```csharp
[Kernel(EnableTimestamps = true)]
public static void TimestampedKernel(
    Span<long> timestamps,    // Auto-injected by DotCompute
    Span<float> data)         // Your data
{
    int tid = GetGlobalId(0);

    // timestamps[tid] contains GPU entry time in nanoseconds
    long gpuTime = timestamps[tid];

    // Use timestamp for temporal ordering
    ProcessWithTimestamp(data[tid], gpuTime);
}
```

**Key Points**:
- First parameter MUST be `Span<long> timestamps`
- DotCompute automatically records `%%globaltimer` (CUDA) or equivalent
- Resolution: 1ns (CUDA), 1μs (OpenCL), 100ns (CPU)

### Pattern 2: HLC Update on GPU

**Use Case**: Update Hybrid Logical Clocks entirely on GPU.

```csharp
[Kernel(EnableTimestamps = true)]
public static void HLCUpdateKernel(
    Span<long> timestamps,        // Auto-injected
    Span<long> localPhysical,     // Actor's HLC physical time
    Span<long> localLogical,      // Actor's HLC logical counter
    Span<ActorMessage> messages)  // Incoming messages
{
    int actorId = GetGlobalId(0);

    long gpuTime = timestamps[actorId];
    var message = messages[actorId];

    // Update HLC with GPU timestamp
    long maxPhysical = Max(localPhysical[actorId], message.HLCPhysical, gpuTime);

    if (maxPhysical == localPhysical[actorId] && maxPhysical == message.HLCPhysical)
    {
        localLogical[actorId] = Max(localLogical[actorId], message.HLCLogical) + 1;
    }
    else if (maxPhysical == localPhysical[actorId])
    {
        localLogical[actorId]++;
    }
    else if (maxPhysical == message.HLCPhysical)
    {
        localLogical[actorId] = message.HLCLogical + 1;
    }
    else
    {
        localLogical[actorId] = 0;
    }

    localPhysical[actorId] = maxPhysical;
}
```

### Pattern 3: Clock Calibration

**Use Case**: Synchronize GPU and CPU clocks for cross-device temporal alignment.

```csharp
[Kernel(EnableTimestamps = true)]
public static void CalibrationSampleKernel(
    Span<long> gpuTimestamps,     // Auto-injected GPU times
    Span<long> cpuTimestamps,     // CPU times (passed from host)
    Span<long> offsetSamples)     // Output: GPU - CPU offsets
{
    int sampleId = GetGlobalId(0);

    long gpuTime = gpuTimestamps[sampleId];
    long cpuTime = cpuTimestamps[sampleId];

    // Calculate offset for this sample
    offsetSamples[sampleId] = gpuTime - cpuTime;
}

// Host side: Statistical analysis of offset samples
var medianOffset = CalculateMedian(offsetSamples);
var driftPPM = CalculateDrift(offsetSamples, duration);
```

---

## Barrier Synchronization

### Pattern 4: Device-Wide Barrier for Multi-Actor Coordination

**Use Case**: Synchronize all actors across entire GPU device.

```csharp
[Kernel(
    EnableBarriers = true,
    BarrierScope = BarrierScope.Device)]
public static void MultiActorBarrierKernel(
    Span<ActorState> states,
    Span<int> globalCounter)
{
    int actorId = GetGlobalId(0);

    // Phase 1: Each actor does local computation
    states[actorId].Value = ComputeLocalUpdate(states[actorId]);

    // BARRIER: Wait for ALL actors across entire device
    DeviceBarrier();

    // Phase 2: Global aggregation (all local updates complete)
    if (actorId == 0)
    {
        // Only actor 0 does global work
        int sum = 0;
        for (int i = 0; i < states.Length; i++)
        {
            sum += states[i].Value;
        }
        globalCounter[0] = sum;
    }

    // BARRIER: Wait for global aggregation
    DeviceBarrier();

    // Phase 3: All actors can now read the global result
    int globalValue = globalCounter[0];
    states[actorId].GlobalSnapshot = globalValue;
}
```

**Barrier Scopes**:

```csharp
public enum BarrierScope
{
    /// <summary>
    /// Thread block scope - synchronizes threads within a block.
    /// Fastest (~1μs), supports up to 1024 threads.
    /// </summary>
    ThreadBlock,

    /// <summary>
    /// Device scope - synchronizes all threads on GPU.
    /// Moderate (~10μs), supports up to 1M threads (CUDA Cooperative Groups).
    /// </summary>
    Device,

    /// <summary>
    /// System scope - synchronizes across multiple GPUs.
    /// Slowest (~100μs), uses host-side coordination.
    /// </summary>
    System
}
```

### Pattern 5: Temporal Pattern Detection with Barriers

**Use Case**: Detect patterns requiring coordination across multiple actors.

```csharp
[Kernel(
    EnableBarriers = true,
    BarrierScope = BarrierScope.Device,
    EnableTimestamps = true)]
public static void PatternDetectionKernel(
    Span<long> timestamps,
    Span<TemporalEvent> events,
    Span<bool> localMatches,
    Span<bool> globalPatternDetected,
    long timeWindowNanos)
{
    int eventId = GetGlobalId(0);
    long eventTime = timestamps[eventId];

    // Step 1: Local pattern check
    localMatches[eventId] = CheckLocalPattern(events[eventId], timeWindowNanos);

    // BARRIER: All local checks must complete
    DeviceBarrier();

    // Step 2: Global pattern analysis
    if (eventId == 0)
    {
        // Analyze all local matches for global pattern
        bool globalPattern = AnalyzePattern(localMatches, events);
        globalPatternDetected[0] = globalPattern;
    }

    // BARRIER: Wait for global analysis
    DeviceBarrier();

    // Step 3: React to pattern detection
    if (globalPatternDetected[0])
    {
        HandlePatternDetected(ref events[eventId], eventTime);
    }
}
```

---

## Memory Ordering

### Pattern 6: Causal Message Send (Release Semantics)

**Use Case**: Send message with guarantee that writes complete before timestamp is visible.

```csharp
[Kernel(MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void SendCausalMessage(
    Span<ActorMessage> messageBuffer,
    Span<long> messageTimestamps,
    Span<bool> messageReady,
    int messageId,
    ActorMessage message,
    long timestamp)
{
    // Write message data
    messageBuffer[messageId] = message;

    // RELEASE fence: Ensure message write completes before timestamp write
    // (DotCompute inserts this automatically with ReleaseAcquire mode)

    // Write timestamp (signals message is ready)
    messageTimestamps[messageId] = timestamp;

    // RELEASE fence: Ensure timestamp write completes before ready flag
    messageReady[messageId] = true;
}
```

### Pattern 7: Causal Message Receive (Acquire Semantics)

**Use Case**: Receive message with guarantee that timestamp is read before message data.

```csharp
[Kernel(MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void ReceiveCausalMessage(
    Span<ActorMessage> messageBuffer,
    Span<long> messageTimestamps,
    Span<bool> messageReady,
    Span<ActorState> actorStates,
    int actorId)
{
    int messageId = actorId;

    // ACQUIRE: Check if message is ready
    if (!messageReady[messageId])
        return; // Message not yet available

    // ACQUIRE fence: Ensure ready flag read completes before timestamp read
    long timestamp = messageTimestamps[messageId];

    // ACQUIRE fence: Ensure timestamp read completes before message read
    var message = messageBuffer[messageId];

    // Now safe to process - causal ordering guaranteed
    ProcessMessage(ref actorStates[actorId], message, timestamp);
}
```

### Pattern 8: Ring Kernel with Causal Ordering

**Use Case**: Persistent GPU thread processing messages with causal correctness.

```csharp
[RingKernel(
    MessageQueueSize = 4096,
    ProcessingMode = RingProcessingMode.Continuous,
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire,
    EnableTimestamps = true)]
public static void CausalRingKernel(
    Span<long> timestamps,
    Span<ActorMessage> messageQueue,
    Span<int> queueHead,
    Span<int> queueTail,
    Span<ActorState> actorStates,
    Span<bool> stopSignal)
{
    int actorId = GetGlobalId(0);

    // Infinite dispatch loop
    while (!stopSignal[0])
    {
        // ACQUIRE: Check queue head (producer index)
        int head = AtomicLoad(ref queueHead[0]);
        int tail = queueTail[actorId];

        if (head != tail)
        {
            // Message available
            int messageIndex = tail % messageQueue.Length;

            // ACQUIRE: Read message with causal ordering
            var message = messageQueue[messageIndex];
            long gpuTime = timestamps[actorId];

            // Process message
            ProcessActorMessage(ref actorStates[actorId], message, gpuTime);

            // RELEASE: Advance tail (release message slot)
            queueTail[actorId] = tail + 1;
        }
        else
        {
            // No messages - yield to reduce GPU power
            Yield();
        }
    }
}
```

**Memory Ordering Modes**:

```csharp
public enum MemoryOrderingMode
{
    /// <summary>
    /// Relaxed consistency - no ordering guarantees.
    /// Fastest, best for independent operations.
    /// </summary>
    Relaxed,

    /// <summary>
    /// Release-acquire consistency - causal ordering.
    /// Release: Writes visible before subsequent operations.
    /// Acquire: Reads visible after prior operations.
    /// Recommended for temporal actors.
    /// </summary>
    ReleaseAcquire,

    /// <summary>
    /// Sequential consistency - total order across all threads.
    /// Slowest, strongest guarantees.
    /// Use only when absolutely necessary.
    /// </summary>
    Sequential
}
```

---

## Complete Patterns Reference

### Pattern Matrix

| Pattern | [Kernel] | [RingKernel] | Timing | Barriers | Ordering |
|---------|----------|--------------|--------|----------|----------|
| **Timestamp Injection** | ✅ | ✅ | `EnableTimestamps=true` | - | - |
| **HLC Update on GPU** | ✅ | ✅ | `EnableTimestamps=true` | - | - |
| **Clock Calibration** | ✅ | ❌ | `EnableTimestamps=true` | - | - |
| **Device Barrier** | ✅ | ✅ | - | `EnableBarriers=true, BarrierScope=Device` | - |
| **Pattern Detection** | ✅ | ❌ | ✅ | `Device` | - |
| **Causal Send/Receive** | ✅ | ✅ | - | - | `ReleaseAcquire` |
| **Persistent Ring** | ❌ | ✅ | ✅ | ✅ | `ReleaseAcquire` |

### Full-Featured Example

**Pattern**: GPU-resident temporal actor with all features enabled.

```csharp
[RingKernel(
    MessageQueueSize = 8192,
    ProcessingMode = RingProcessingMode.Continuous,
    EnableTimestamps = true,
    EnableBarriers = true,
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire,
    MaxMessagesPerIteration = 4)]
public static void FullFeaturedTemporalActorRing(
    Span<long> timestamps,              // Auto-injected
    Span<ActorMessage> messageQueue,    // Ring buffer
    Span<int> queueHead,                // Producer index
    Span<int> queueTail,                // Consumer index per actor
    Span<ActorState> actorStates,       // Actor state
    Span<long> hlcPhysical,             // HLC physical time
    Span<long> hlcLogical,              // HLC logical counter
    Span<int> globalBarrierCounter,     // Coordination state
    Span<bool> stopSignal)              // Stop flag
{
    int actorId = GetGlobalId(0);

    while (!stopSignal[0])
    {
        // Process up to 4 messages per iteration
        for (int i = 0; i < 4; i++)
        {
            int head = AtomicLoad(ref queueHead[0]);
            int tail = queueTail[actorId];

            if (head == tail)
                break; // No more messages

            // Dequeue message with ACQUIRE semantics
            int messageIndex = tail % messageQueue.Length;
            var message = messageQueue[messageIndex];
            long gpuTime = timestamps[actorId];

            // Update HLC
            var localHlc = new HybridTimestamp(hlcPhysical[actorId], hlcLogical[actorId]);
            var updatedHlc = UpdateHLC(localHlc, message.Timestamp, gpuTime);
            hlcPhysical[actorId] = updatedHlc.PhysicalTime;
            hlcLogical[actorId] = updatedHlc.LogicalCounter;

            // Process message
            ProcessActorMessage(ref actorStates[actorId], message, updatedHlc);

            // Advance tail with RELEASE semantics
            queueTail[actorId] = tail + 1;
        }

        // Periodic device-wide barrier for coordination
        if (ShouldSynchronize(actorStates[actorId]))
        {
            DeviceBarrier();

            if (actorId == 0)
            {
                // Global coordination work
                CoordinateActors(actorStates, globalBarrierCounter);
            }

            DeviceBarrier();
        }

        // No messages and no coordination - yield
        Yield();
    }
}
```

---

## Best Practices

### 1. Timestamp Injection

✅ **DO**:
- Always place `Span<long> timestamps` as first parameter
- Use timestamps for HLC updates and causal ordering
- Batch timestamp queries when possible

❌ **DON'T**:
- Don't manually query timestamps in hot paths
- Don't assume timestamp parameter index (always first)

### 2. Ring Kernels

✅ **DO**:
- Use ring kernels for high-frequency message processing
- Set `MessageQueueSize` to power of 2 for efficiency
- Use `ReleaseAcquire` ordering for causal correctness
- Implement graceful shutdown with `stopSignal`

❌ **DON'T**:
- Don't use ring kernels for one-shot computations
- Don't forget to handle queue overflow
- Don't use `Sequential` ordering unless absolutely necessary

### 3. Barriers

✅ **DO**:
- Use `ThreadBlock` scope when possible (fastest)
- Use `Device` scope for actor coordination
- Minimize barrier frequency (expensive)

❌ **DON'T**:
- Don't use barriers inside tight loops
- Don't mix barrier scopes within same kernel
- Don't exceed hardware limits (check GPU capabilities)

### 4. Memory Ordering

✅ **DO**:
- Use `Relaxed` for independent operations
- Use `ReleaseAcquire` for causal message passing
- Use `Sequential` only when total order required

❌ **DON'T**:
- Don't over-use `Sequential` (40% performance penalty)
- Don't assume ordering without explicit fences
- Don't mix ordering modes within same workflow

### 5. Performance

✅ **DO**:
- Batch operations to amortize overhead
- Profile with `nvprof` or vendor tools
- Monitor queue depths for ring kernels
- Calibrate clocks periodically (not per message)

❌ **DON'T**:
- Don't over-calibrate clocks (<5min intervals)
- Don't use device barriers for thread-block work
- Don't allocate GPU memory in hot paths

---

## Performance Characteristics

| Feature | Overhead | Best Use Case |
|---------|----------|---------------|
| Timestamp injection | ~10ns | All temporal kernels |
| Thread block barrier | ~1μs | Work group sync |
| Device barrier | ~10μs | Multi-actor coordination |
| System barrier | ~100μs | Multi-GPU sync |
| Relaxed ordering | 0% | Independent ops |
| Release-acquire ordering | ~15% | Causal messaging |
| Sequential ordering | ~40% | Total order requirements |
| Ring kernel (no messages) | ~50ns/iteration | Idle GPU thread |
| Ring kernel (active) | ~100-500ns/message | High-frequency messaging |

---

## References

- **DotCompute Documentation**: https://mivertowski.github.io/DotCompute/docs/
- **Timing API Guide**: https://mivertowski.github.io/DotCompute/docs/articles/guides/timing-api.html
- **Ring Kernels Guide**: https://mivertowski.github.io/DotCompute/docs/articles/guides/ring-kernels-introduction.html
- **Barriers & Memory Ordering**: https://mivertowski.github.io/DotCompute/docs/articles/advanced/barriers-and-memory-ordering.html

---

*Guide Version: 1.0*
*Last Updated: 2025-01-11*
*DotCompute Version: 0.4.2-rc2*
