# Temporal Correctness

**Ensuring causal consistency and temporal ordering in GPU-accelerated distributed systems.**

## Overview

Temporal Correctness in Orleans.GpuBridge.Core provides mechanisms for maintaining **causal ordering**, **happened-before relationships**, and **consistent timestamps** across distributed GPU-Native Actors.

## Section Contents

### [Introduction](introduction/README.md)
Foundational concepts of temporal correctness, why it matters for distributed systems, and how GPU acceleration changes the paradigm.

### [Hybrid Logical Clocks (HLC)](hlc/README.md)
Implementation of HLC on GPU with 20ns resolution, combining physical and logical time for causally consistent timestamps.

### [Vector Clocks](vector-clocks/README.md)
GPU-accelerated vector clocks for tracking causal dependencies across distributed actors.

### [Pattern Detection](pattern-detection/README.md)
Real-time detection of temporal patterns, causal anomalies, and ordering violations.

### [Architecture](architecture/README.md)
Technical deep-dive into GPU clock sources, synchronization protocols, and memory ordering semantics.

### [Performance](performance/README.md)
Benchmarks, optimization strategies, and performance characteristics of temporal operations.

### [Software PTP Synchronization](../software-ptp-distributed-time-synchronization.md)
Software-based Precision Time Protocol for microsecond-accurate synchronization across nodes.

## Key Concepts

### Hybrid Logical Clocks (HLC)
```csharp
// HLC timestamp: physical + logical component
public readonly struct HybridTimestamp
{
    public long PhysicalTime { get; }  // Nanoseconds since epoch
    public int LogicalCounter { get; } // Lamport-style counter
    public int NodeId { get; }         // Source node identifier
}
```

### Temporal Guarantees

| Guarantee | Description | GPU Support |
|-----------|-------------|-------------|
| **Causality** | If A→B, then T(A) < T(B) | HLC on GPU |
| **Monotonicity** | Timestamps always increase | Atomic operations |
| **Bounded Skew** | Max clock drift < ε | PTP synchronization |
| **Total Order** | All events globally ordered | Vector clocks |

### GPU Temporal Performance

| Operation | CPU | GPU | Improvement |
|-----------|-----|-----|-------------|
| HLC update | 50ns | 20ns | 2.5× |
| Vector clock merge | 200ns | 80ns | 2.5× |
| Causal check | 100ns | 40ns | 2.5× |
| Pattern detection | 10μs | 500ns | 20× |

## Quick Start

```csharp
// DotCompute kernel with temporal features
[Kernel(
    EnableTimestamps = true,              // Auto-inject GPU timestamps
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]  // Causal consistency
public static void TemporalKernel(
    Span<long> timestamps,                // Auto-injected by DotCompute
    Span<ActorState> states)
{
    int tid = GetGlobalId(0);
    long gpuTime = timestamps[tid];       // 20ns GPU clock

    // Update HLC with GPU timestamp
    UpdateHLC(ref states[tid], gpuTime);
}
```

## Related Documentation

- [GPU-Native Actors](../gpu-actors/introduction/README.md) - Actors that live on GPU
- [Ring Kernel Integration](../../architecture/RING-KERNEL-INTEGRATION.md) - Temporal alignment in ring kernels
- [DotCompute Guide](../../developer-guide/GPU-NATIVE-ACTORS.md) - Temporal attributes and patterns

---

*Temporal Correctness: Causality at the speed of light.*
