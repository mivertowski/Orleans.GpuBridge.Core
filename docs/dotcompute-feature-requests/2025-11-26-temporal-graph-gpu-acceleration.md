# DotCompute Feature Requests: Temporal Graph GPU Acceleration

**Date**: November 26, 2025
**Context**: Orleans.GpuBridge.Core - Temporal Graph and Interval Tree GPU Acceleration
**DotCompute Version**: 0.4.2+
**Priority**: Medium-High for distributed temporal analytics

---

## Executive Summary

During test validation of Orleans.GpuBridge.Core's Temporal subsystem, we identified several opportunities where DotCompute could provide GPU-accelerated implementations for temporal graph operations. Currently, these operations run on CPU with some concurrency limitations.

---

## 1. Thread-Safe GPU IntervalTree

### 1.1 Problem Statement

The current `IntervalTree<TKey, TValue>` implementation is **not thread-safe**. Tests had to be skipped:

```csharp
[Fact(Skip = "IntervalTree is not thread-safe by design - use TemporalGraphStorage for concurrent access")]
public async Task IntervalTree_ConcurrentOperations() { ... }

[Fact(Skip = "IntervalTree in TemporalGraphStorage needs synchronization for concurrent edge insertion")]
public async Task TemporalGraph_ConcurrentEdgeInsertion() { ... }
```

Concurrent edge insertions cause tree corruption (stack overflow from unbalanced tree).

### 1.2 Feature Request

GPU-accelerated, lock-free interval tree implementation using parallel primitives:

```csharp
public interface IGpuIntervalTree<TKey, TValue> where TKey : unmanaged, IComparable<TKey>
{
    /// <summary>
    /// Add interval using GPU atomics for thread safety.
    /// </summary>
    void AddGpu(TKey start, TKey end, TValue value);

    /// <summary>
    /// Parallel batch query - process thousands of point queries simultaneously.
    /// </summary>
    IEnumerable<TValue> QueryBatch(ReadOnlySpan<TKey> queryPoints);

    /// <summary>
    /// GPU-parallel range query over interval tree.
    /// </summary>
    IEnumerable<TValue> QueryRangeGpu(TKey start, TKey end);
}
```

### 1.3 Expected Benefits

| Operation | CPU (current) | GPU (proposed) | Speedup |
|-----------|---------------|----------------|---------|
| Single insert | O(log N) | O(log N) | 1x (but lock-free) |
| Batch insert (1M) | ~500ms | ~5ms | 100x |
| Range query (10K) | ~10ms | ~0.1ms | 100x |
| Point query batch | ~1ms per query | ~0.01ms per 1000 | 100x |

### 1.4 GPU Data Structure Design

```cuda
// GPU-friendly interval tree node (cache-aligned)
struct GpuIntervalNode {
    long start;       // Interval start
    long end;         // Interval end
    long max;         // Max end in subtree (for overlap queries)
    int left;         // Left child index (-1 if none)
    int right;        // Right child index (-1 if none)
    int height;       // AVL height for balancing
    uint32_t value;   // Payload or index to value array
};

// Lock-free insertion using CUDA atomics
__device__ void insertIntervalGpu(
    GpuIntervalNode* tree,
    int* nodeCount,
    long start, long end, uint32_t value
) {
    // Use atomicCAS for lock-free node allocation
    int newNode = atomicAdd(nodeCount, 1);
    // ... AVL insertion with atomic updates
}
```

---

## 2. GPU-Accelerated Temporal Path Finding

### 2.1 Problem Statement

The `FindTemporalPaths` algorithm currently runs on CPU using BFS:

```csharp
public IEnumerable<TemporalPath> FindTemporalPaths(
    ulong startNode, ulong endNode, long maxTimeSpanNanos)
{
    // CPU BFS with temporal constraints
    // O(V + E) per query
}
```

For large graphs (millions of edges), this becomes a bottleneck.

### 2.2 Feature Request

GPU-parallel temporal path finding:

```csharp
public interface IGpuTemporalPathFinder
{
    /// <summary>
    /// Find all temporal paths using GPU-parallel BFS.
    /// </summary>
    Task<IEnumerable<TemporalPath>> FindPathsGpuAsync(
        ulong startNode,
        ulong endNode,
        long maxTimeSpanNanos,
        int maxPathLength = 10);

    /// <summary>
    /// Batch path finding - multiple source-destination pairs.
    /// </summary>
    Task<Dictionary<(ulong, ulong), TemporalPath>> FindPathsBatchAsync(
        IEnumerable<(ulong source, ulong target)> queries,
        long maxTimeSpanNanos);
}
```

### 2.3 GPU Algorithm

```cuda
// GPU-parallel temporal BFS
__global__ void temporalBfsKernel(
    const TemporalEdge* edges,
    const int* adjacencyOffsets,  // CSR format
    const int* adjacencyList,
    long* distances,              // Distance from source
    int* predecessors,            // Path reconstruction
    long maxTimeSpan,
    int currentLevel
) {
    int nodeId = blockIdx.x * blockDim.x + threadIdx.x;
    if (distances[nodeId] != currentLevel) return;

    // Process all outgoing edges in parallel
    int start = adjacencyOffsets[nodeId];
    int end = adjacencyOffsets[nodeId + 1];

    for (int i = start; i < end; i++) {
        TemporalEdge edge = edges[adjacencyList[i]];

        // Check temporal constraint
        if (edge.validFrom - distances[nodeId] <= maxTimeSpan) {
            atomicMin(&distances[edge.targetId], currentLevel + 1);
            predecessors[edge.targetId] = nodeId;
        }
    }
}
```

---

## 3. GPU Hybrid Logical Clock (HLC) Operations

### 3.1 Problem Statement

The CPU `HybridLogicalClock` is thread-safe using `Interlocked` operations, but GPU actors need HLC on-GPU:

```csharp
// Current CPU implementation
public HybridTimestamp Now()
{
    while (true)
    {
        var current = _timestamp;
        long physicalTime = _clockSource.GetCurrentTimeNanos();
        // ... atomic update
    }
}
```

### 3.2 Feature Request

GPU-resident HLC with system-scope atomics (for native Linux deployment):

```csharp
public interface IGpuHybridLogicalClock
{
    /// <summary>
    /// Generate timestamp entirely on GPU (no CPU roundtrip).
    /// </summary>
    [RingKernelMethod]
    HybridTimestamp Now();

    /// <summary>
    /// Update HLC from received message timestamp (GPU-to-GPU).
    /// </summary>
    [RingKernelMethod]
    HybridTimestamp Update(HybridTimestamp received);

    /// <summary>
    /// Batch timestamp generation for high-throughput scenarios.
    /// </summary>
    void GenerateBatch(Span<HybridTimestamp> timestamps);
}
```

### 3.3 GPU Implementation

```cuda
// GPU HLC state (system-scope for CPU visibility on native Linux)
struct GpuHlcState {
    cuda::atomic<long, cuda::memory_scope_system> physicalTime;
    cuda::atomic<long, cuda::memory_scope_system> logicalCounter;
    ushort nodeId;
};

__device__ HybridTimestamp hlcNow(GpuHlcState* state) {
    long gpuTime = clock64() / GPU_CLOCK_RATIO;  // Convert to nanoseconds

    long oldPhysical = state->physicalTime.load();
    long newPhysical = max(gpuTime, oldPhysical);

    if (newPhysical == oldPhysical) {
        long counter = state->logicalCounter.fetch_add(1);
        return HybridTimestamp{newPhysical, counter + 1, state->nodeId};
    }

    state->physicalTime.store(newPhysical);
    state->logicalCounter.store(0);
    return HybridTimestamp{newPhysical, 0, state->nodeId};
}
```

---

## 4. GPU Vector Clock Synchronization

### 4.1 Problem Statement

Vector clocks for N nodes require O(N) storage and O(N) comparison operations. For large clusters, this becomes expensive.

### 4.2 Feature Request

GPU-accelerated vector clock operations:

```csharp
public interface IGpuVectorClock
{
    /// <summary>
    /// Parallel merge of two vector clocks.
    /// </summary>
    void Merge(Span<long> target, ReadOnlySpan<long> source);

    /// <summary>
    /// Batch comparison - happens-before relationship for multiple events.
    /// </summary>
    void CompareBatch(
        ReadOnlySpan<long> clocks,  // N clocks, each M entries
        Span<CausalRelation> results);

    /// <summary>
    /// Increment local component and return new clock.
    /// </summary>
    void Tick(Span<long> clock, int nodeId);
}
```

---

## 5. Concurrent TemporalGraphStorage

### 5.1 Problem Statement

Multiple tests were skipped due to concurrent access issues:

```csharp
[Fact(Skip = "IntervalTree in TemporalGraphStorage needs synchronization - queries during insertions")]
public async Task TemporalGraph_ConcurrentQueries() { ... }

[Fact(Skip = "IntervalTree/TemporalGraphStorage is not thread-safe - concurrent writes cause stack overflow")]
public async Task StressTest_HighContentionScenario() { ... }
```

### 5.2 Feature Request

GPU-backed temporal graph storage with lock-free operations:

```csharp
public interface IGpuTemporalGraphStorage
{
    /// <summary>
    /// Lock-free edge insertion with GPU atomics.
    /// </summary>
    void AddEdge(ulong source, ulong target, long validFrom, long validTo, HybridTimestamp hlc);

    /// <summary>
    /// Concurrent-safe time range query.
    /// </summary>
    IEnumerable<TemporalEdge> GetEdgesInTimeRange(ulong nodeId, long startTime, long endTime);

    /// <summary>
    /// GPU-parallel graph snapshot at specific time.
    /// </summary>
    IEnumerable<TemporalEdge> GetSnapshotAtTimeGpu(long timestamp);

    /// <summary>
    /// Batch edge insertion (high throughput).
    /// </summary>
    void AddEdgesBatch(ReadOnlySpan<TemporalEdge> edges);
}
```

---

## 6. Implementation Priority

| Feature | Priority | Complexity | Impact |
|---------|----------|------------|--------|
| Thread-safe GPU IntervalTree | High | High | Enables concurrent temporal queries |
| GPU Temporal Path Finding | Medium | Medium | 100x speedup for graph analytics |
| GPU HLC Operations | Medium | Low | Essential for GPU-native actors |
| GPU Vector Clock | Low | Medium | Large cluster optimization |
| Concurrent TemporalGraphStorage | High | High | Production requirement |

---

## 7. Testing Requirements

All GPU implementations should pass equivalent tests to CPU versions:

```csharp
[Theory]
[InlineData(ExecutionMode.Cpu)]
[InlineData(ExecutionMode.Gpu)]
public async Task IntervalTree_ConcurrentOperations(ExecutionMode mode)
{
    var tree = mode == ExecutionMode.Gpu
        ? new GpuIntervalTree<long, string>()
        : new IntervalTree<long, string>();

    // Same test logic, validated on both CPU and GPU
}
```

---

## 8. Related DotCompute Issues

- Ring kernel message queue improvements (see `2025-11-24-messagequeue-struct-interlocked-exchange.md`)
- WSL2 system-scope atomics limitations (see `2025-11-24-wsl2-ring-kernel-blocking.md`)
- GPU ring buffer bridge integration (see `2025-11-24-gpu-ring-buffer-bridge.md`)

---

## 9. Next Steps

1. Prototype GPU IntervalTree using CUDA atomics
2. Benchmark against CPU implementation on large datasets
3. Integrate with existing Orleans.GpuBridge.Core temporal subsystem
4. Add GPU fallback detection for systems without CUDA

---

*Document created based on Orleans.GpuBridge.Core test validation findings (November 26, 2025)*
