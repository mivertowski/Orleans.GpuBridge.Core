# Temporal Correctness for GPU-Native Actors
## Design Document v1.0

## Executive Summary

This document outlines a comprehensive architecture for achieving temporal correctness in GPU-native actors for behavioral analytics in graph spaces and physics simulations. The design addresses three critical requirements:

1. **Temporal Pattern Detection**: Capture time-based patterns in financial transaction graphs
2. **Causal Ordering**: Maintain message causality across distributed GPU actors
3. **Physical Timing**: Provide nanosecond-precision timing for physics simulations

---

## 1. Problem Statement

### Use Case 1: Financial Transaction Graph Analytics
```
Node A → $1000 → Node B (T₀)
         ↓
    Node B → $500 → Node C (T₀ + 1.2s)
         ↓
    Node B → $500 → Node D (T₀ + 1.8s)
```

**Challenges**:
- Detect temporal patterns (rapid splitting, circular flows)
- Maintain causality: C and D events depend on B receiving funds
- Sub-second precision required for pattern detection
- Distributed execution across multiple GPU actors

### Use Case 2: Spatial Quantized Wave Propagation
```
Wave source at (x₀, y₀) → propagates to neighbors
Each cell computes: ψ(x,y,t+Δt) = f(ψ(neighbors, t))
```

**Challenges**:
- Strict temporal ordering: step N must complete before N+1 starts
- Synchronization barrier across 1M+ GPU actors
- Nanosecond timing precision for numerical stability
- Guaranteed message delivery order

---

## 2. Architectural Components

### 2.1 Hybrid Logical Clocks (HLC)

**Why HLC?**
- Combines physical time (wall clock) with logical ordering
- Provides total ordering of events across distributed system
- Bounded clock drift from physical time
- Compatible with GPU execution model

**Implementation**:
```csharp
public readonly struct HybridTimestamp : IComparable<HybridTimestamp>
{
    // Physical time component (nanoseconds since epoch)
    public readonly long PhysicalTime;

    // Logical counter (increments on concurrent events)
    public readonly long LogicalCounter;

    // Node/actor identifier for tie-breaking
    public readonly ushort NodeId;

    public int CompareTo(HybridTimestamp other)
    {
        if (PhysicalTime != other.PhysicalTime)
            return PhysicalTime.CompareTo(other.PhysicalTime);
        if (LogicalCounter != other.LogicalCounter)
            return LogicalCounter.CompareTo(other.LogicalCounter);
        return NodeId.CompareTo(other.NodeId);
    }
}
```

**Clock Update Rules**:
```csharp
// On local event (send message)
HLC_local.PhysicalTime = max(HLC_local.PhysicalTime, SystemClock.Now)
HLC_local.LogicalCounter++

// On receive message
HLC_local.PhysicalTime = max(HLC_local.PhysicalTime,
                             SystemClock.Now,
                             HLC_message.PhysicalTime)
HLC_local.LogicalCounter = max(HLC_local.LogicalCounter,
                               HLC_message.LogicalCounter) + 1
```

**Benefits**:
- ✅ Total ordering of all events
- ✅ Causality preservation (if A → B, then HLC(A) < HLC(B))
- ✅ Bounded drift from physical time
- ✅ GPU-friendly (64-bit integer comparison)

### 2.2 Vector Clocks for Causal Dependencies

**Use Case**: Track causal relationships in transaction graphs

```csharp
public sealed class VectorClock
{
    // actor_id → logical_timestamp
    private readonly Dictionary<ushort, long> _clocks;

    public void Increment(ushort actorId)
    {
        _clocks[actorId] = _clocks.GetValueOrDefault(actorId) + 1;
    }

    public void Merge(VectorClock other)
    {
        foreach (var (actorId, timestamp) in other._clocks)
        {
            _clocks[actorId] = Math.Max(_clocks.GetValueOrDefault(actorId),
                                         timestamp);
        }
    }

    // Returns: -1 (before), 0 (concurrent), 1 (after)
    public int CompareTo(VectorClock other)
    {
        bool anyLess = false, anyGreater = false;

        foreach (var actorId in _clocks.Keys.Union(other._clocks.Keys))
        {
            var thisTime = _clocks.GetValueOrDefault(actorId);
            var otherTime = other._clocks.GetValueOrDefault(actorId);

            if (thisTime < otherTime) anyLess = true;
            if (thisTime > otherTime) anyGreater = true;
        }

        if (anyLess && !anyGreater) return -1;  // Happens-before
        if (anyGreater && !anyLess) return 1;   // Happens-after
        return 0;                                // Concurrent
    }
}
```

**Benefits**:
- ✅ Detect concurrent events (transaction splitting)
- ✅ Establish causal chains (A caused B caused C)
- ✅ Pattern matching: "find all transactions concurrent with X"

### 2.3 Physical Time Synchronization

**Options**:

#### Option A: NTP (Network Time Protocol)
- **Accuracy**: ±1-10ms
- **Use Case**: Financial analytics (sufficient for second-scale patterns)
- **Implementation**: Built into OS

#### Option B: PTP (Precision Time Protocol / IEEE 1588)
- **Accuracy**: ±10-100ns
- **Use Case**: Physics simulations requiring nanosecond precision
- **Implementation**: Requires hardware support (NIC with PTP)

#### Option C: GPS Time Sync
- **Accuracy**: ±100ns
- **Use Case**: Cross-datacenter temporal correctness
- **Implementation**: GPS receiver per node

**Recommendation**: Start with NTP, upgrade to PTP for physics simulations

**Clock Synchronization API**:
```csharp
public interface IPhysicalClockSource
{
    // Get current time in nanoseconds since epoch
    long GetCurrentTimeNanos();

    // Get estimated clock error bound (±nanoseconds)
    long GetErrorBound();

    // Check if clock is synchronized
    bool IsSynchronized { get; }

    // Get clock drift rate (PPM - parts per million)
    double GetClockDrift();
}

public sealed class PtpClockSource : IPhysicalClockSource
{
    // Uses Linux PTP API (ptp_clock_gettime)
    // Or Windows PTP API (IOCTL_PTP_GET_TIME)
}
```

### 2.4 Temporal Message Extensions

**Enhanced Message Type**:
```csharp
public abstract record TemporalResidentMessage : ResidentMessage
{
    // Hybrid Logical Clock timestamp
    public HybridTimestamp HLC { get; init; }

    // Vector clock for causal dependencies (optional, for transaction graphs)
    public VectorClock? VectorClock { get; init; }

    // Physical timestamp with error bound
    public long PhysicalTimeNanos { get; init; }
    public long TimestampErrorBoundNanos { get; init; }

    // Causality chain: IDs of messages that happened-before this one
    public ImmutableArray<Guid> CausalDependencies { get; init; }

    // Temporal window: valid processing time range
    public TimeRange? ValidityWindow { get; init; }

    // Message sequence number (per sender)
    public ulong SequenceNumber { get; init; }
}

public readonly record struct TimeRange
{
    public long StartNanos { get; init; }
    public long EndNanos { get; init; }

    public bool Contains(long timeNanos)
        => timeNanos >= StartNanos && timeNanos <= EndNanos;
}
```

### 2.5 Temporal Priority Queue

**Requirements**:
- Sort messages by HLC timestamp
- Support deadline-based eviction
- O(log N) insert/remove
- GPU-friendly data structure

**Implementation**:
```csharp
public sealed class TemporalMessageQueue
{
    // Min-heap sorted by HLC timestamp
    private readonly PriorityQueue<TemporalResidentMessage, HybridTimestamp> _hlcQueue;

    // Deadline tracking: deadline → message
    private readonly SortedDictionary<long, List<TemporalResidentMessage>> _deadlineIndex;

    // Causality tracking: message_id → dependents
    private readonly Dictionary<Guid, List<TemporalResidentMessage>> _causalGraph;

    public void Enqueue(TemporalResidentMessage message)
    {
        // Add to HLC-ordered queue
        _hlcQueue.Enqueue(message, message.HLC);

        // Track deadline if specified
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

        // Track causal dependencies
        foreach (var depId in message.CausalDependencies)
        {
            if (!_causalGraph.TryGetValue(depId, out var dependents))
            {
                dependents = new List<TemporalResidentMessage>();
                _causalGraph[depId] = dependents;
            }
            dependents.Add(message);
        }
    }

    public bool TryDequeue(out TemporalResidentMessage? message)
    {
        // Check deadlines and evict expired messages
        EvictExpiredMessages();

        // Get message with earliest HLC
        if (!_hlcQueue.TryDequeue(out message, out _))
            return false;

        // Check if causal dependencies are satisfied
        foreach (var depId in message.CausalDependencies)
        {
            if (_causalGraph.ContainsKey(depId))
            {
                // Dependency not yet processed, re-enqueue
                _hlcQueue.Enqueue(message, message.HLC);
                return TryDequeue(out message);  // Try next message
            }
        }

        return true;
    }

    private void EvictExpiredMessages()
    {
        var now = _clockSource.GetCurrentTimeNanos();

        foreach (var (deadline, messages) in _deadlineIndex)
        {
            if (deadline > now) break;  // No more expired deadlines

            foreach (var msg in messages)
            {
                // Remove from queue and log deadline miss
                _logger.LogWarning("Message {RequestId} missed deadline by {Delta}ns",
                    msg.RequestId, now - deadline);
            }

            _deadlineIndex.Remove(deadline);
        }
    }
}
```

---

## 3. Temporal Graph Storage

### 3.1 Time-Indexed Adjacency Structure

**Goal**: Efficiently query "What edges existed between time T₁ and T₂?"

**Data Structure**:
```csharp
public sealed class TemporalGraphStorage
{
    // Node ID → temporal edge list
    private readonly Dictionary<ulong, TemporalEdgeList> _adjacency;

    // Global time-index for fast temporal queries
    private readonly IntervalTree<long, TemporalEdge> _timeIndex;

    public void AddEdge(ulong sourceId, ulong targetId, TemporalEdge edge)
    {
        // Add to adjacency list
        if (!_adjacency.TryGetValue(sourceId, out var edgeList))
        {
            edgeList = new TemporalEdgeList();
            _adjacency[sourceId] = edgeList;
        }
        edgeList.Add(edge);

        // Index by time range for fast temporal queries
        _timeIndex.Add(edge.ValidFrom, edge.ValidTo, edge);
    }

    public IEnumerable<TemporalEdge> GetEdgesInTimeRange(
        ulong sourceId, long startTimeNanos, long endTimeNanos)
    {
        if (!_adjacency.TryGetValue(sourceId, out var edgeList))
            return Enumerable.Empty<TemporalEdge>();

        return edgeList.Query(startTimeNanos, endTimeNanos);
    }

    // Find all paths that occurred within a time window
    public IEnumerable<TemporalPath> FindTemporalPaths(
        ulong startNode, ulong endNode, long maxTimeSpanNanos)
    {
        // BFS with temporal constraints
        var queue = new Queue<(ulong node, TemporalPath path, long latestTime)>();
        queue.Enqueue((startNode, new TemporalPath(), 0));

        while (queue.Count > 0)
        {
            var (node, path, latestTime) = queue.Dequeue();

            if (node == endNode)
            {
                yield return path;
                continue;
            }

            // Explore edges that start after latestTime
            foreach (var edge in GetEdgesInTimeRange(node, latestTime, long.MaxValue))
            {
                if (edge.ValidFrom - latestTime > maxTimeSpanNanos)
                    continue;  // Time constraint violated

                var newPath = path.Append(edge);
                queue.Enqueue((edge.TargetId, newPath, edge.ValidFrom));
            }
        }
    }
}

public readonly struct TemporalEdge
{
    public ulong SourceId { get; init; }
    public ulong TargetId { get; init; }

    // Time range when edge was valid
    public long ValidFrom { get; init; }
    public long ValidTo { get; init; }

    // Edge metadata (e.g., transaction amount)
    public ImmutableDictionary<string, object> Properties { get; init; }

    // Causality information
    public HybridTimestamp HLC { get; init; }
    public VectorClock? VectorClock { get; init; }
}

public sealed class TemporalEdgeList
{
    // Store edges sorted by ValidFrom time
    private readonly SortedList<long, List<TemporalEdge>> _edges = new();

    public void Add(TemporalEdge edge)
    {
        if (!_edges.TryGetValue(edge.ValidFrom, out var list))
        {
            list = new List<TemporalEdge>();
            _edges[edge.ValidFrom] = list;
        }
        list.Add(edge);
    }

    public IEnumerable<TemporalEdge> Query(long startTime, long endTime)
    {
        // Binary search for start index
        var startIdx = BinarySearchFloor(_edges.Keys, startTime);

        for (int i = startIdx; i < _edges.Count; i++)
        {
            var (time, edges) = _edges.ElementAt(i);
            if (time > endTime) break;

            foreach (var edge in edges)
            {
                if (edge.ValidTo >= startTime && edge.ValidFrom <= endTime)
                    yield return edge;
            }
        }
    }
}
```

### 3.2 GPU-Resident Temporal Graph

**Goal**: Store temporal graph directly in GPU memory for fast pattern matching

**Memory Layout**:
```
[Adjacency Offsets]  [Edge Data]  [Time Index]
0: offset=0          Edge 0       TimeRange 0
1: offset=5          Edge 1       TimeRange 1
2: offset=12         Edge 2       TimeRange 2
...                  ...          ...
```

**CUDA Kernel for Temporal BFS**:
```cuda
__global__ void temporal_bfs_kernel(
    const ulong* adjacency_offsets,
    const TemporalEdge* edges,
    const ulong start_node,
    const long time_window_ns,
    TemporalPath* output_paths,
    int* output_count)
{
    // Shared memory for wavefront
    __shared__ ulong wavefront[256];
    __shared__ long wavefront_times[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize wavefront
    if (tid == 0) {
        wavefront[0] = start_node;
        wavefront_times[0] = 0;
    }
    __syncthreads();

    // BFS iterations
    for (int depth = 0; depth < MAX_DEPTH; depth++)
    {
        if (tid >= wavefront_size) return;

        ulong node = wavefront[tid];
        long latest_time = wavefront_times[tid];

        // Load adjacency list
        ulong edge_start = adjacency_offsets[node];
        ulong edge_end = adjacency_offsets[node + 1];

        // Explore edges
        for (ulong i = edge_start; i < edge_end; i++)
        {
            TemporalEdge edge = edges[i];

            // Check temporal constraint
            if (edge.valid_from >= latest_time &&
                edge.valid_from - latest_time <= time_window_ns)
            {
                // Add to next wavefront
                int next_idx = atomicAdd(&next_wavefront_size, 1);
                next_wavefront[next_idx] = edge.target_id;
                next_wavefront_times[next_idx] = edge.valid_from;
            }
        }

        __syncthreads();
    }
}
```

---

## 4. Temporal Pattern Detection Engine

### 4.1 Sliding Time Windows

```csharp
public sealed class TemporalPatternDetector
{
    // Time window configuration
    private readonly long _windowSizeNanos;
    private readonly long _slideIntervalNanos;

    // Event buffer for current window
    private readonly List<TemporalEvent> _currentWindow = new();

    // Pattern definitions
    private readonly List<ITemporalPattern> _patterns = new();

    public async Task ProcessEventAsync(TemporalEvent evt, CancellationToken ct)
    {
        // Add event to current window
        _currentWindow.Add(evt);

        // Remove events outside window
        var windowStart = evt.HLC.PhysicalTime - _windowSizeNanos;
        _currentWindow.RemoveAll(e => e.HLC.PhysicalTime < windowStart);

        // Check all patterns
        foreach (var pattern in _patterns)
        {
            if (await pattern.MatchAsync(_currentWindow, ct))
            {
                await OnPatternDetectedAsync(pattern, _currentWindow, ct);
            }
        }
    }
}

public interface ITemporalPattern
{
    string Name { get; }
    Task<bool> MatchAsync(IReadOnlyList<TemporalEvent> window, CancellationToken ct);
}

// Example: Rapid transaction splitting pattern
public sealed class RapidSplitPattern : ITemporalPattern
{
    public string Name => "RapidTransactionSplit";

    public async Task<bool> MatchAsync(
        IReadOnlyList<TemporalEvent> window, CancellationToken ct)
    {
        // Find: A→B followed by B→C and B→D within 5 seconds
        var transactions = window.OfType<TransactionEvent>().ToList();

        foreach (var inbound in transactions)
        {
            var targetNode = inbound.TargetId;
            var inboundTime = inbound.HLC.PhysicalTime;

            var outbounds = transactions
                .Where(t => t.SourceId == targetNode)
                .Where(t => t.HLC.PhysicalTime > inboundTime)
                .Where(t => t.HLC.PhysicalTime - inboundTime <= 5_000_000_000) // 5 seconds
                .ToList();

            if (outbounds.Count >= 2)
            {
                // Pattern matched!
                return true;
            }
        }

        return false;
    }
}
```

### 4.2 GPU-Accelerated Pattern Matching

**Goal**: Run pattern detection on GPU for high throughput

```csharp
public sealed class GpuPatternMatcher
{
    private readonly IGpuBackendProvider _backend;
    private readonly CompiledKernel _patternKernel;

    public async Task<PatternMatch[]> FindPatternsAsync(
        TemporalEvent[] events,
        ITemporalPattern[] patterns,
        CancellationToken ct)
    {
        // Allocate GPU memory
        var eventsGpu = await _backend.Memory.AllocateAsync<TemporalEvent>(
            events.Length, ct);
        var resultsGpu = await _backend.Memory.AllocateAsync<PatternMatch>(
            events.Length * patterns.Length, ct);

        // Transfer events to GPU
        await _backend.Memory.WriteAsync(eventsGpu, events, ct);

        // Execute pattern matching kernel
        await _patternKernel.ExecuteAsync(new object[]
        {
            eventsGpu,
            events.Length,
            resultsGpu,
            _windowSizeNanos,
            patterns.Length
        }, ct);

        // Read results
        var results = new PatternMatch[events.Length * patterns.Length];
        await _backend.Memory.ReadAsync(resultsGpu, results, ct);

        return results.Where(m => m.Matched).ToArray();
    }
}
```

---

## 5. DotCompute Backend Extensions

### 5.1 GPU-Side Timestamp Injection

**Problem**: CPU-generated timestamps have latency (10-100μs)

**Solution**: Inject timestamps directly on GPU using GPU clock

**CUDA Implementation**:
```cuda
// Get GPU nanosecond timer
__device__ __forceinline__ long gpu_nanotime()
{
    long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));

    // Convert GPU clock cycles to nanoseconds
    // Assumes NVIDIA GPU clock runs at 1 GHz
    return time;
}

__global__ void process_message_with_timestamp(
    Message* messages,
    long* timestamps,
    int count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    // Inject GPU timestamp
    timestamps[tid] = gpu_nanotime();

    // Process message
    process_message(&messages[tid]);
}
```

**DotCompute Extension API**:
```csharp
public interface ITimestampInjector
{
    // Enable GPU-side timestamp injection
    void EnableGpuTimestamps();

    // Get GPU clock frequency (Hz)
    long GetGpuClockFrequency();

    // Calibrate GPU clock offset from CPU clock
    (long offsetNanos, long driftPPM) CalibrateGpuClock();
}

public sealed class CudaTimestampInjector : ITimestampInjector
{
    public void EnableGpuTimestamps()
    {
        // Inject timestamp instruction at kernel entry
        _kernelPatcher.InjectPrologue(
            "mov.u64 %timestamp, %%globaltimer;");
    }
}
```

### 5.2 GPU Synchronization Barriers

**Problem**: Wave propagation requires all actors to complete step N before step N+1

**Solution**: GPU-wide barrier using device-level synchronization

**CUDA Cooperative Groups Implementation**:
```cuda
#include <cooperative_groups.h>

__global__ void wave_propagation_step(
    float* state_current,
    float* state_next,
    int grid_size,
    long* step_timestamp)
{
    namespace cg = cooperative_groups;

    // Device-wide grid synchronization
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= grid_size) return;

    // Compute next state
    state_next[tid] = compute_wave_propagation(state_current, tid);

    // BARRIER: Wait for ALL threads across ALL blocks
    grid.sync();

    // Record timestamp after barrier
    if (tid == 0)
    {
        *step_timestamp = gpu_nanotime();
    }
}
```

**DotCompute API Extension**:
```csharp
public interface IDeviceBarrier
{
    // Create a device-wide barrier for kernel synchronization
    IBarrierHandle CreateDeviceBarrier(int participantCount);

    // Launch kernel with device-wide synchronization
    Task ExecuteWithBarrierAsync(
        CompiledKernel kernel,
        IBarrierHandle barrier,
        object[] arguments,
        CancellationToken ct);
}
```

### 5.3 Causal Memory Ordering

**Problem**: GPU memory operations are weakly ordered

**Solution**: Enforce causal memory ordering with acquire-release semantics

**CUDA Implementation**:
```cuda
// Causal write: ensure write is visible to all readers
__device__ void causal_write(volatile long* addr, long value)
{
    // Release fence: make all prior writes visible
    __threadfence_system();

    // Atomic write with release semantics
    atomicExch((unsigned long long*)addr, (unsigned long long)value);
}

// Causal read: ensure read observes all prior writes
__device__ long causal_read(volatile long* addr)
{
    // Atomic read with acquire semantics
    long value = atomicAdd((unsigned long long*)addr, 0);

    // Acquire fence: make read visible to all subsequent operations
    __threadfence_system();

    return value;
}
```

---

## 6. System Architecture

### 6.1 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                 Orleans Grain Layer                      │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │ Temporal Actor  │  │  Temporal Pattern Detector   │  │
│  │   - HLC clock   │  │   - Sliding windows          │  │
│  │   - Vector clock│  │   - Pattern matching         │  │
│  └────────┬────────┘  └──────────────┬───────────────┘  │
└───────────┼───────────────────────────┼──────────────────┘
            │                           │
            │ Temporal                  │ Pattern
            │ Messages                  │ Events
            │                           │
┌───────────▼───────────────────────────▼──────────────────┐
│              Ring Kernel Layer                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │       Temporal Message Queue                       │  │
│  │  - Priority queue (HLC-ordered)                    │  │
│  │  - Causal dependency tracking                      │  │
│  │  - Deadline-based eviction                         │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                   │
│  ┌────────────────────▼───────────────────────────────┐  │
│  │     GPU-Resident Temporal Graph                    │  │
│  │  - Adjacency list with time-indexing               │  │
│  │  - Temporal BFS/DFS kernels                        │  │
│  │  - Pattern matching kernels                        │  │
│  └────────────────────┬───────────────────────────────┘  │
└───────────────────────┼───────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────┐
│              DotCompute Backend                            │
│  ┌──────────────────┐  ┌────────────────────────────┐    │
│  │ Timestamp        │  │  Device Barrier Manager    │    │
│  │ Injector         │  │  - Cooperative groups      │    │
│  │ - GPU clock      │  │  - Grid-wide sync          │    │
│  │ - Clock calib.   │  │  - Step coordination       │    │
│  └──────────────────┘  └────────────────────────────┘    │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │     Causal Memory Manager                        │    │
│  │  - Acquire-release semantics                     │    │
│  │  - Memory fences                                 │    │
│  └──────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Message Flow with Temporal Correctness

```
Actor A                    Actor B                    Actor C
  │                          │                          │
  ├─ Send(msg₁)              │                          │
  │  HLC: (t=100, l=1, n=A)  │                          │
  │  Seq: 1                  │                          │
  ├────────────────────────>│                          │
  │                          │                          │
  │                          ├─ Receive(msg₁)          │
  │                          │  Update HLC:             │
  │                          │  (t=101, l=2, n=B)      │
  │                          │                          │
  │                          ├─ Send(msg₂) ───────────>│
  │                          │  HLC: (t=101, l=3, n=B) │
  │                          │  Causal: [msg₁]         │
  │                          │  Seq: 1                  │
  │                          │                          │
  │                          ├─ Send(msg₃) ───────────>│
  │                          │  HLC: (t=101, l=4, n=B) │
  │                          │  Causal: [msg₁]         │
  │                          │  Seq: 2                  │
  │                          │                          │
  │                          │                          ├─ Receive(msg₂)
  │                          │                          │  Queue: [msg₂]
  │                          │                          │
  │                          │                          ├─ Receive(msg₃)
  │                          │                          │  Queue: [msg₂, msg₃]
  │                          │                          │  (ordered by HLC)
  │                          │                          │
  │                          │                          ├─ Process(msg₂)
  │                          │                          │  Check causality ✓
  │                          │                          │
  │                          │                          ├─ Process(msg₃)
  │                          │                          │  Check causality ✓
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Basic temporal infrastructure

#### Tasks:
1. **Hybrid Logical Clocks**
   - Implement `HybridTimestamp` struct
   - Clock update rules (send/receive)
   - Clock synchronization with physical time
   - Unit tests for clock monotonicity

2. **Temporal Message Extensions**
   - Extend `ResidentMessage` with HLC
   - Add sequence numbers
   - Add causal dependency tracking
   - Message serialization/deserialization

3. **Physical Time Synchronization**
   - Implement NTP clock source
   - Clock error bound estimation
   - Clock drift detection
   - Platform-specific implementations (Linux/Windows)

4. **Temporal Priority Queue**
   - HLC-ordered priority queue
   - Deadline tracking
   - Basic causal dependency checking
   - Performance benchmarks

**Deliverables**:
- ✅ HLC implementation with tests
- ✅ Temporal message types
- ✅ NTP clock synchronization
- ✅ Priority queue with HLC ordering

**DotCompute Changes**: None (pure Orleans layer)

---

### Phase 2: Graph Storage (Weeks 3-4)
**Goal**: Temporal graph data structures

#### Tasks:
1. **CPU Temporal Graph**
   - Implement `TemporalEdge` and `TemporalEdgeList`
   - Time-indexed adjacency structure
   - Interval tree for temporal queries
   - Temporal BFS/DFS algorithms

2. **GPU Temporal Graph**
   - GPU memory layout for temporal graph
   - Pinned memory transfers
   - CUDA kernels for graph traversal
   - Performance benchmarks vs CPU

3. **Pattern Storage**
   - Event buffer management
   - Sliding window implementation
   - Pattern definition interface
   - Example patterns (rapid split, circular flow)

**Deliverables**:
- ✅ Temporal graph storage (CPU)
- ✅ GPU-resident graph structure
- ✅ Temporal query API
- ✅ Pattern storage framework

**DotCompute Changes**:
- Add graph memory allocator
- Add graph kernel execution support

---

### Phase 3: Pattern Detection (Weeks 5-6)
**Goal**: Real-time pattern matching

#### Tasks:
1. **CPU Pattern Detector**
   - Sliding window implementation
   - Pattern matching engine
   - Pattern composition (AND/OR/NOT)
   - Performance optimization

2. **GPU Pattern Kernels**
   - CUDA pattern matching kernels
   - Parallel pattern evaluation
   - Result aggregation
   - Memory coalescing optimization

3. **Pattern Language**
   - DSL for pattern definition
   - Pattern compiler to GPU kernels
   - Pattern library (common financial patterns)

**Deliverables**:
- ✅ Real-time pattern detector
- ✅ GPU-accelerated matching
- ✅ Pattern definition language
- ✅ Pattern library with examples

**DotCompute Changes**:
- Add pattern kernel templates
- Optimize kernel dispatch for small patterns

---

### Phase 4: Causal Correctness (Weeks 7-8)
**Goal**: Guaranteed causal ordering

#### Tasks:
1. **Vector Clocks**
   - Implement `VectorClock` class
   - Merge algorithm
   - Causality detection (happens-before)
   - Integration with HLC

2. **Causal Message Delivery**
   - Dependency-aware message queue
   - Delayed delivery until dependencies met
   - Deadlock detection
   - Timeout handling

3. **Causal Graph Analysis**
   - Extract causal chains from events
   - Visualize causality graphs
   - Find concurrent events
   - Pattern matching on causal structure

**Deliverables**:
- ✅ Vector clock implementation
- ✅ Causal message ordering
- ✅ Causal graph extraction
- ✅ Causality-based pattern matching

**DotCompute Changes**:
- Add causal memory ordering primitives
- Memory fence support

---

### Phase 5: GPU Timing Extensions (Weeks 9-10)
**Goal**: GPU-native timing and synchronization

#### Tasks:
1. **GPU Timestamp Injection**
   - CUDA `%%globaltimer` integration
   - GPU clock calibration with CPU
   - Clock drift compensation
   - Nanosecond precision validation

2. **Device-Wide Barriers**
   - Cooperative groups integration
   - Grid-wide synchronization
   - Multi-kernel barriers
   - Performance overhead measurement

3. **Causal Memory Ordering**
   - Acquire-release semantics
   - Memory fence primitives
   - Cross-actor memory visibility
   - Correctness validation

**Deliverables**:
- ✅ GPU timestamp injection
- ✅ Device barriers for wave propagation
- ✅ Causal memory ordering
- ✅ Performance benchmarks

**DotCompute Changes**:
- **Major**: Add GPU timing API
- **Major**: Add barrier support
- **Major**: Add memory ordering primitives

---

### Phase 6: Physical Time Precision (Weeks 11-12)
**Goal**: Sub-microsecond timing for physics

#### Tasks:
1. **PTP Clock Support**
   - Linux PTP API integration
   - Windows PTP API integration
   - Hardware timestamp support
   - Clock synchronization validation

2. **GPS Time Sync**
   - GPS receiver integration (optional)
   - Time distribution to all nodes
   - Fault tolerance
   - Accuracy validation

3. **Time-Sensitive Networking**
   - Network latency measurement
   - Compensate for network delays
   - Bounded latency guarantees
   - Cross-datacenter sync

**Deliverables**:
- ✅ PTP clock synchronization
- ✅ GPS time support (optional)
- ✅ Network latency compensation
- ✅ End-to-end timing validation

**DotCompute Changes**: None

---

### Phase 7: Integration & Optimization (Weeks 13-14)
**Goal**: Production-ready system

#### Tasks:
1. **Performance Optimization**
   - Profile end-to-end latency
   - Optimize critical paths
   - Reduce memory allocations
   - GPU kernel optimization

2. **Fault Tolerance**
   - Clock desynchronization recovery
   - Message loss handling
   - Actor failure recovery
   - Temporal consistency validation

3. **Monitoring & Observability**
   - Clock drift metrics
   - Message latency histograms
   - Pattern detection rates
   - Causal violation detection

4. **Documentation**
   - API documentation
   - Pattern writing guide
   - Performance tuning guide
   - Troubleshooting guide

**Deliverables**:
- ✅ Production-grade performance
- ✅ Fault tolerance mechanisms
- ✅ Monitoring dashboards
- ✅ Complete documentation

**DotCompute Changes**: None

---

## 8. DotCompute Backend Modifications Summary

### Required Changes:

#### 8.1 Timing API (Phase 5)
```csharp
namespace DotCompute.Timing;

public interface ITimingProvider
{
    // Get GPU nanosecond timestamp
    Task<long> GetGpuTimestampAsync(CancellationToken ct);

    // Calibrate GPU clock with CPU clock
    Task<ClockCalibration> CalibrateAsync(CancellationToken ct);

    // Enable automatic timestamp injection in kernels
    void EnableTimestampInjection();
}

public readonly struct ClockCalibration
{
    public long OffsetNanos { get; init; }      // GPU - CPU offset
    public double DriftPPM { get; init; }       // Parts per million drift
    public long ErrorBoundNanos { get; init; }  // ± error bound
}
```

#### 8.2 Barrier API (Phase 5)
```csharp
namespace DotCompute.Synchronization;

public interface IBarrierProvider
{
    // Create device-wide barrier
    IBarrierHandle CreateBarrier(int participantCount);

    // Launch kernel with barrier support
    Task ExecuteWithBarrierAsync(
        ICompiledKernel kernel,
        IBarrierHandle barrier,
        LaunchConfiguration config,
        object[] arguments,
        CancellationToken ct);
}

public interface IBarrierHandle : IDisposable
{
    // Wait for all participants
    Task WaitAsync(CancellationToken ct);

    // Check if barrier is ready
    bool IsReady { get; }
}
```

#### 8.3 Memory Ordering API (Phase 5)
```csharp
namespace DotCompute.Memory;

public interface IMemoryOrderingProvider
{
    // Ensure acquire-release semantics for memory operations
    void EnableCausalOrdering();

    // Insert memory fence in kernel
    void InsertFence(FenceType type);
}

public enum FenceType
{
    ThreadBlock,    // __threadfence_block()
    Device,         // __threadfence()
    System          // __threadfence_system()
}
```

### Implementation Strategy:

1. **CUDA Backend** (Primary target)
   - Use `%%globaltimer` for timestamps
   - Use cooperative groups for barriers
   - Use memory fences for ordering

2. **OpenCL Backend** (Secondary)
   - Use `clock()` for timestamps (less precise)
   - Use global barriers
   - Use memory fences

3. **CPU Backend** (Fallback)
   - Use `Stopwatch` for timestamps
   - Use `Barrier` class for synchronization
   - Use volatile/Interlocked for ordering

---

## 9. Performance Targets

### Latency Targets:
- **HLC timestamp generation**: <50ns
- **Message enqueue/dequeue**: <100ns
- **Causal dependency check**: <200ns
- **GPU timestamp injection**: <10ns
- **Device-wide barrier**: <10μs for 1M actors
- **Pattern detection (CPU)**: <1ms per window
- **Pattern detection (GPU)**: <100μs per window

### Throughput Targets:
- **Message processing**: 10M messages/sec per GPU
- **Temporal queries**: 1M queries/sec
- **Pattern matching**: 100K patterns/sec per GPU
- **Graph updates**: 5M edges/sec

### Accuracy Targets:
- **Clock synchronization (NTP)**: ±1ms
- **Clock synchronization (PTP)**: ±100ns
- **GPU clock calibration**: ±50ns
- **Causal ordering**: 100% correctness

---

## 10. Validation Strategy

### 10.1 Correctness Tests

**Temporal Ordering**:
```csharp
[Fact]
public async Task Messages_Are_Processed_In_HLC_Order()
{
    var actor = GetActor();

    // Send messages out of order
    await actor.SendAsync(CreateMessage(hlc: (100, 2, A)));
    await actor.SendAsync(CreateMessage(hlc: (100, 1, A)));
    await actor.SendAsync(CreateMessage(hlc: (100, 3, A)));

    // Verify processing order
    var processed = await actor.GetProcessedMessagesAsync();
    Assert.Equal(new[] { (100,1,A), (100,2,A), (100,3,A) },
                 processed.Select(m => m.HLC));
}
```

**Causal Ordering**:
```csharp
[Fact]
public async Task Dependent_Messages_Are_Processed_After_Dependencies()
{
    var actorA = GetActor("A");
    var actorB = GetActor("B");

    // A sends msg1
    var msg1 = await actorA.SendAsync(new Message { Data = "msg1" });

    // B sends msg2 depending on msg1
    var msg2 = await actorB.SendAsync(new Message
    {
        Data = "msg2",
        CausalDependencies = new[] { msg1.RequestId }
    });

    // Verify msg2 is not processed until msg1 completes
    Assert.False(actorB.HasProcessed(msg2.RequestId));

    await msg1.CompletionTask;
    await Task.Delay(100); // Allow processing

    Assert.True(actorB.HasProcessed(msg2.RequestId));
}
```

**Pattern Detection**:
```csharp
[Fact]
public async Task Rapid_Split_Pattern_Is_Detected()
{
    var detector = GetPatternDetector();

    // Simulate transaction split
    var events = new[]
    {
        new TransactionEvent("A", "B", 1000, t=0),
        new TransactionEvent("B", "C", 500, t=1_000_000_000),  // 1s later
        new TransactionEvent("B", "D", 500, t=1_500_000_000),  // 1.5s later
    };

    foreach (var evt in events)
        await detector.ProcessEventAsync(evt);

    var matches = await detector.GetMatchesAsync();
    Assert.Single(matches);
    Assert.Equal("RapidTransactionSplit", matches[0].PatternName);
}
```

### 10.2 Performance Tests

**Latency Benchmark**:
```csharp
[Fact]
public async Task HLC_Generation_Latency_Under_50ns()
{
    var clock = GetHLC();
    var iterations = 1_000_000;

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        clock.GetTimestamp();
    }
    sw.Stop();

    var avgLatency = sw.Elapsed.TotalNanoseconds / iterations;
    Assert.True(avgLatency < 50, $"Actual: {avgLatency}ns");
}
```

**Throughput Benchmark**:
```csharp
[Fact]
public async Task Message_Processing_Throughput_10M_Per_Second()
{
    var actor = GetActor();
    var messages = Enumerable.Range(0, 10_000_000)
        .Select(i => CreateMessage(i))
        .ToArray();

    var sw = Stopwatch.StartNew();
    await Parallel.ForEachAsync(messages, async (msg, ct) =>
    {
        await actor.SendAsync(msg, ct);
    });
    await actor.WaitForCompletionAsync();
    sw.Stop();

    var throughput = messages.Length / sw.Elapsed.TotalSeconds;
    Assert.True(throughput >= 10_000_000, $"Actual: {throughput:N0} msg/s");
}
```

---

## 11. Example Usage

### Use Case 1: Financial Transaction Graph

```csharp
// Initialize temporal actor system
var siloHost = new SiloHostBuilder()
    .AddTemporalActors(options =>
    {
        options.ClockSource = new PtpClockSource();
        options.EnableVectorClocks = true;
        options.MessageQueueCapacity = 100_000;
    })
    .AddGpuBridge(options =>
    {
        options.EnableTemporalGraph = true;
        options.PatternDetectionMode = PatternDetectionMode.GpuAccelerated;
    })
    .Build();

await siloHost.StartAsync();

// Get temporal actor grain
var accountA = client.GetGrain<ITemporalAccountGrain>("account-A");
var accountB = client.GetGrain<ITemporalAccountGrain>("account-B");
var accountC = client.GetGrain<ITemporalAccountGrain>("account-C");

// Register pattern detector
var detector = client.GetGrain<IPatternDetectorGrain>(0);
await detector.RegisterPatternAsync(new RapidSplitPattern
{
    TimeWindowSeconds = 5,
    MinimumSplits = 2,
    SuspiciousThreshold = 3
});

// Execute transactions
await accountA.TransferAsync("account-B", 1000);  // HLC: (t₀, 1, A)

// These will be temporally ordered and causally tracked
await accountB.TransferAsync("account-C", 500);   // HLC: (t₀+1s, 2, B), depends on A→B
await accountB.TransferAsync("account-D", 500);   // HLC: (t₀+1.5s, 3, B), depends on A→B

// Check for pattern matches
var patterns = await detector.GetDetectedPatternsAsync();
if (patterns.Any(p => p.Name == "RapidTransactionSplit"))
{
    Console.WriteLine("Suspicious transaction pattern detected!");

    // Get temporal path
    var path = await accountA.GetTemporalPathAsync("account-D",
        maxTimeSpan: TimeSpan.FromSeconds(10));

    Console.WriteLine($"Path: {string.Join(" → ", path.Select(e => e.TargetId))}");
    Console.WriteLine($"Total time: {path.TotalTimeSpan.TotalSeconds}s");
}
```

### Use Case 2: Wave Propagation Simulation

```csharp
// Initialize physics simulation
var siloHost = new SiloHostBuilder()
    .AddTemporalActors(options =>
    {
        options.ClockSource = new PtpClockSource();
        options.SynchronizationMode = SynchronizationMode.Lockstep;
        options.TimeStepNanos = 1_000_000; // 1ms time step
    })
    .AddGpuBridge(options =>
    {
        options.EnableDeviceBarriers = true;
        options.EnableGpuTimestamps = true;
    })
    .Build();

await siloHost.StartAsync();

// Create wave simulation grid (1M actors)
var grid = await CreateWaveGridAsync(1000, 1000);

// Initialize wave source
var source = grid[500, 500];
await source.SetWaveAmplitudeAsync(1.0);

// Run simulation with temporal correctness
for (int step = 0; step < 10000; step++)
{
    // All actors compute next state in parallel
    var stepTasks = grid.SelectMany(actor =>
        actor.ComputeNextStateAsync(step));

    await Task.WhenAll(stepTasks);

    // BARRIER: Ensure all actors completed step N
    await grid.SynchronizeAsync();

    // Advance to next time step (strict temporal ordering)
    await grid.AdvanceTimeStepAsync();

    // Query wave state at specific location
    var amplitude = await grid[750, 750].GetAmplitudeAsync();
    Console.WriteLine($"Step {step}: Amplitude at (750,750) = {amplitude}");
}
```

---

## 12. Open Questions

1. **Clock Synchronization Strategy**:
   - Start with NTP or require PTP from day 1?
   - Hardware requirements for PTP?

2. **Vector Clock Overhead**:
   - Vector clocks grow with actor count. Use bloom filters for compression?
   - When to prune old entries?

3. **GPU Memory Limits**:
   - How large can temporal graphs be (100M edges? 1B edges)?
   - Eviction strategy for GPU-resident graph?

4. **Cross-Datacenter Timing**:
   - GPS time sync for multi-region deployments?
   - TrueTime-like API for bounded uncertainty?

5. **DotCompute API Surface**:
   - Should timing/barrier APIs be in DotCompute core or extensions?
   - Backwards compatibility concerns?

---

## 13. References

- [1] Lamport, L. (1978). "Time, Clocks, and the Ordering of Events"
- [2] Kulkarni et al. (2014). "Logical Physical Clocks and Consistent Snapshots"
- [3] Corbett et al. (2013). "Spanner: Google's Globally Distributed Database" (TrueTime)
- [4] NVIDIA CUDA Cooperative Groups Programming Guide
- [5] IEEE 1588 Precision Time Protocol Specification

---

## Conclusion

This design provides a **comprehensive foundation** for temporal correctness in GPU-native actors, addressing both financial graph analytics and physics simulation use cases. The phased implementation plan allows incremental development while maintaining production-grade quality at each stage.

**Key Innovations**:
1. Hybrid Logical Clocks for total event ordering
2. GPU-resident temporal graphs for high-performance queries
3. GPU-side timestamp injection for nanosecond precision
4. Device-wide barriers for lockstep simulation
5. Causal memory ordering for distributed correctness

**Next Steps**:
1. Review and refine this design
2. Prioritize phases based on immediate use cases
3. Prototype HLC implementation (Phase 1)
4. Validate DotCompute API changes with maintainer

---

*Document Version: 1.0*
*Last Updated: 2025-11-10*
*Author: Claude (Anthropic)*
