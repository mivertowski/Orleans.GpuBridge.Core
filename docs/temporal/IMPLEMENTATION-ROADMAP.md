# Temporal Correctness Implementation Roadmap
## 14-Week Implementation Plan

## Quick Reference

**Total Duration**: 14 weeks
**Team Size**: 1-2 developers
**DotCompute Changes Required**: Phases 5-6 (Weeks 9-12)
**First Production-Ready Milestone**: End of Phase 4 (Week 8)

---

## Phase Overview

| Phase | Duration | Focus Area | DotCompute Changes | Priority |
|-------|----------|------------|-------------------|----------|
| Phase 1 | 2 weeks | Foundation (HLC, Messages) | None | 🔴 Critical |
| Phase 2 | 2 weeks | Graph Storage | Minor (memory layout) | 🔴 Critical |
| Phase 3 | 2 weeks | Pattern Detection | Minor (kernel templates) | 🟡 High |
| Phase 4 | 2 weeks | Causal Correctness | Minor (memory fences) | 🟡 High |
| Phase 5 | 2 weeks | GPU Timing | **Major (new API)** | 🟢 Medium |
| Phase 6 | 2 weeks | Physical Time Sync | None | 🔵 Low |
| Phase 7 | 2 weeks | Integration & Polish | None | 🟡 High |

---

## Phase 1: Foundation (Weeks 1-2) 🔴

### Goals
- Implement Hybrid Logical Clocks (HLC)
- Extend message types with temporal metadata
- Basic NTP clock synchronization
- Temporal priority queue

### Deliverables

#### 1.1 Hybrid Logical Clock Implementation
**File**: `src/Orleans.GpuBridge.Abstractions/Temporal/HybridLogicalClock.cs`

```csharp
public readonly struct HybridTimestamp : IComparable<HybridTimestamp>
{
    public long PhysicalTime { get; init; }
    public long LogicalCounter { get; init; }
    public ushort NodeId { get; init; }

    // Clock update methods
    // Comparison operators
    // Serialization support
}

public sealed class HybridLogicalClock
{
    public HybridTimestamp Now();
    public void Update(HybridTimestamp receivedTimestamp);
}
```

**Tests**: `tests/Orleans.GpuBridge.Abstractions.Tests/Temporal/HybridLogicalClockTests.cs`
- Monotonicity tests
- Happens-before relationship tests
- Concurrent event ordering tests

#### 1.2 Temporal Message Types
**File**: `src/Orleans.GpuBridge.Grains/Resident/Messages/TemporalResidentMessage.cs`

```csharp
public abstract record TemporalResidentMessage : ResidentMessage
{
    public HybridTimestamp HLC { get; init; }
    public long PhysicalTimeNanos { get; init; }
    public long TimestampErrorBoundNanos { get; init; }
    public ImmutableArray<Guid> CausalDependencies { get; init; }
    public ulong SequenceNumber { get; init; }
}
```

#### 1.3 Physical Clock Synchronization
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/PhysicalClockSource.cs`

```csharp
public interface IPhysicalClockSource
{
    long GetCurrentTimeNanos();
    long GetErrorBound();
    bool IsSynchronized { get; }
}

public sealed class NtpClockSource : IPhysicalClockSource
{
    // NTP synchronization implementation
}
```

#### 1.4 Temporal Priority Queue
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/TemporalMessageQueue.cs`

```csharp
public sealed class TemporalMessageQueue
{
    public void Enqueue(TemporalResidentMessage message);
    public bool TryDequeue(out TemporalResidentMessage? message);
    public void EvictExpiredMessages();
}
```

### Testing Requirements
- ✅ HLC monotonicity across 1M operations
- ✅ Message ordering correctness
- ✅ Clock synchronization accuracy (±10ms)
- ✅ Queue performance (>1M ops/sec)

### Dependencies
- None (pure Orleans layer)

### DotCompute Changes
- None

---

## Phase 2: Graph Storage (Weeks 3-4) 🔴

### Goals
- CPU-based temporal graph storage
- GPU-resident temporal graph
- Time-indexed queries
- Temporal path finding

### Deliverables

#### 2.1 CPU Temporal Graph
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/TemporalGraphStorage.cs`

```csharp
public sealed class TemporalGraphStorage
{
    public void AddEdge(ulong sourceId, ulong targetId, TemporalEdge edge);
    public IEnumerable<TemporalEdge> GetEdgesInTimeRange(
        ulong sourceId, long startTimeNanos, long endTimeNanos);
    public IEnumerable<TemporalPath> FindTemporalPaths(
        ulong startNode, ulong endNode, long maxTimeSpanNanos);
}

public readonly struct TemporalEdge
{
    public ulong SourceId { get; init; }
    public ulong TargetId { get; init; }
    public long ValidFrom { get; init; }
    public long ValidTo { get; init; }
    public ImmutableDictionary<string, object> Properties { get; init; }
}
```

#### 2.2 GPU Temporal Graph
**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/GpuTemporalGraph.cs`

```csharp
public sealed class GpuTemporalGraph : IDisposable
{
    public Task UploadGraphAsync(TemporalGraphStorage graph, CancellationToken ct);
    public Task<TemporalPath[]> FindPathsAsync(
        ulong startNode, ulong endNode, long maxTimeSpanNanos, CancellationToken ct);
}
```

**CUDA Kernel**: `kernels/temporal_bfs.cu`
```cuda
__global__ void temporal_bfs_kernel(
    const ulong* adjacency_offsets,
    const TemporalEdge* edges,
    ulong start_node,
    long time_window_ns,
    TemporalPath* output_paths,
    int* output_count);
```

#### 2.3 Interval Tree for Time Indexing
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/IntervalTree.cs`

```csharp
public sealed class IntervalTree<TKey, TValue> where TKey : IComparable<TKey>
{
    public void Add(TKey start, TKey end, TValue value);
    public IEnumerable<TValue> Query(TKey start, TKey end);
}
```

### Testing Requirements
- ✅ Graph operations correctness
- ✅ Temporal query correctness
- ✅ GPU vs CPU result equivalence
- ✅ Performance: 1M edges, <1ms queries

### Dependencies
- Phase 1 (for temporal metadata)

### DotCompute Changes
- **Minor**: Add support for custom graph memory layouts
- **Minor**: Optimize kernel dispatch for graph traversal

---

## Phase 3: Pattern Detection (Weeks 5-6) 🟡

### Goals
- Sliding window pattern matching
- GPU-accelerated pattern detection
- Pattern definition language
- Common pattern library

### Deliverables

#### 3.1 Pattern Detection Engine
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/TemporalPatternDetector.cs`

```csharp
public sealed class TemporalPatternDetector
{
    public async Task ProcessEventAsync(TemporalEvent evt, CancellationToken ct);
    public void RegisterPattern(ITemporalPattern pattern);
    public IEnumerable<PatternMatch> GetMatches();
}

public interface ITemporalPattern
{
    string Name { get; }
    Task<bool> MatchAsync(IReadOnlyList<TemporalEvent> window, CancellationToken ct);
}
```

#### 3.2 GPU Pattern Matcher
**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/GpuPatternMatcher.cs`

```csharp
public sealed class GpuPatternMatcher
{
    public async Task<PatternMatch[]> FindPatternsAsync(
        TemporalEvent[] events,
        ITemporalPattern[] patterns,
        CancellationToken ct);
}
```

**CUDA Kernel**: `kernels/pattern_match.cu`
```cuda
__global__ void pattern_match_kernel(
    const TemporalEvent* events,
    int event_count,
    PatternMatch* results,
    long window_size_ns);
```

#### 3.3 Pattern Library
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/Patterns/`
- `RapidSplitPattern.cs` - Transaction splitting detection
- `CircularFlowPattern.cs` - Circular transaction detection
- `HighFrequencyPattern.cs` - High-frequency trading detection
- `TemporalClusterPattern.cs` - Time-clustered activity detection

#### 3.4 Pattern DSL (Optional)
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/PatternLanguage/`

```csharp
// DSL for defining patterns
var pattern = Pattern.Define("RapidSplit")
    .Within(TimeSpan.FromSeconds(5))
    .Match(e => e.Type == "Transaction")
    .Where(e => e.Source == variable("source"))
    .Followed(TimeSpan.FromSeconds(2))
    .By(e => e.Type == "Transaction" && e.Target == variable("target1"))
    .And(e => e.Type == "Transaction" && e.Target == variable("target2"))
    .Build();
```

### Testing Requirements
- ✅ Pattern matching correctness
- ✅ GPU vs CPU equivalence
- ✅ Performance: 100K events/sec on GPU
- ✅ False positive rate < 1%

### Dependencies
- Phase 2 (for temporal graph queries)

### DotCompute Changes
- **Minor**: Add pattern kernel templates
- **Minor**: Optimize for small kernel dispatch

---

## Phase 4: Causal Correctness (Weeks 7-8) 🟡

### Goals
- Vector clock implementation
- Causal message ordering
- Dependency tracking
- Deadlock detection

### Deliverables

#### 4.1 Vector Clocks
**File**: `src/Orleans.GpuBridge.Abstractions/Temporal/VectorClock.cs`

```csharp
public sealed class VectorClock
{
    public void Increment(ushort actorId);
    public void Merge(VectorClock other);
    public int CompareTo(VectorClock other);  // -1 (before), 0 (concurrent), 1 (after)
}
```

#### 4.2 Causal Message Delivery
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/CausalMessageQueue.cs`

```csharp
public sealed class CausalMessageQueue : TemporalMessageQueue
{
    public override bool TryDequeue(out TemporalResidentMessage? message)
    {
        // Check causal dependencies before dequeuing
    }

    public void MarkDependencySatisfied(Guid dependencyId);
    public bool HasUnsatisfiedDependencies(TemporalResidentMessage message);
}
```

#### 4.3 Causal Graph Analysis
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/CausalGraphAnalyzer.cs`

```csharp
public sealed class CausalGraphAnalyzer
{
    public IEnumerable<TemporalEvent> GetCausalChain(Guid eventId);
    public IEnumerable<TemporalEvent> GetConcurrentEvents(Guid eventId);
    public bool IsCausallyRelated(Guid eventId1, Guid eventId2);
}
```

#### 4.4 Deadlock Detection
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/DeadlockDetector.cs`

```csharp
public sealed class DeadlockDetector
{
    public bool DetectDeadlock(CausalMessageQueue queue);
    public IEnumerable<Guid> FindDeadlockedMessages();
    public void ResolveDeadlock(DeadlockResolutionStrategy strategy);
}
```

### Testing Requirements
- ✅ Causal ordering correctness
- ✅ Deadlock detection accuracy
- ✅ Performance overhead < 10%
- ✅ Integration with HLC

### Dependencies
- Phase 1 (for HLC)
- Phase 3 (for pattern detection on causal chains)

### DotCompute Changes
- **Minor**: Add memory fence primitives (prepare for Phase 5)

---

## Phase 5: GPU Timing Extensions (Weeks 9-10) 🟢

### Goals
- GPU-side timestamp injection
- Device-wide barriers
- Causal memory ordering

### Deliverables

#### 5.1 DotCompute Timing API
**File**: `DotCompute/Timing/ITimingProvider.cs` (in DotCompute repo)

```csharp
public interface ITimingProvider
{
    Task<long> GetGpuTimestampAsync(CancellationToken ct);
    Task<ClockCalibration> CalibrateAsync(int sampleCount, CancellationToken ct);
    void EnableTimestampInjection(bool enable = true);
}
```

**Implementation**: `DotCompute/Timing/CudaTimingProvider.cs`
```csharp
public sealed class CudaTimingProvider : ITimingProvider
{
    // CUDA %%globaltimer implementation
}
```

#### 5.2 DotCompute Barrier API
**File**: `DotCompute/Synchronization/IBarrierProvider.cs`

```csharp
public interface IBarrierProvider
{
    IBarrierHandle CreateBarrier(int participantCount);
    Task ExecuteWithBarrierAsync(
        ICompiledKernel kernel,
        IBarrierHandle barrier,
        LaunchConfiguration config,
        object[] arguments,
        CancellationToken ct);
}
```

**Implementation**: `DotCompute/Synchronization/CudaBarrierProvider.cs`
```csharp
public sealed class CudaBarrierProvider : IBarrierProvider
{
    // CUDA Cooperative Groups implementation
}
```

#### 5.3 DotCompute Memory Ordering API
**File**: `DotCompute/Memory/IMemoryOrderingProvider.cs`

```csharp
public interface IMemoryOrderingProvider
{
    void EnableCausalOrdering(bool enable);
    void InsertFence(FenceType type, FenceLocation? location = null);
}
```

#### 5.4 Orleans.GpuBridge Integration
**File**: `src/Orleans.GpuBridge.Backends.DotCompute/TemporalIntegration.cs`

```csharp
public static class TemporalIntegration
{
    public static void EnableGpuTimestamps(this IGpuBackendProvider provider);
    public static Task<ClockCalibration> CalibrateGpuClockAsync(
        this IGpuBackendProvider provider, CancellationToken ct);
}
```

### Testing Requirements
- ✅ GPU timestamp accuracy (±50ns)
- ✅ Barrier correctness (1M threads)
- ✅ Memory ordering correctness
- ✅ Performance overhead < 5%

### Dependencies
- Phase 1-4 (for integration)

### DotCompute Changes
- **MAJOR**: Timing API implementation
- **MAJOR**: Barrier API implementation
- **MAJOR**: Memory ordering primitives

**Coordination Required**: Work with DotCompute maintainer

---

## Phase 6: Physical Time Precision (Weeks 11-12) 🔵

### Goals
- PTP clock synchronization
- Sub-microsecond timing
- Network latency compensation

### Deliverables

#### 6.1 PTP Clock Support
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/PtpClockSource.cs`

```csharp
public sealed class PtpClockSource : IPhysicalClockSource
{
    // Linux: ptp_clock_gettime()
    // Windows: IOCTL_PTP_GET_TIME
}
```

#### 6.2 Network Latency Compensation
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/NetworkLatencyCompensator.cs`

```csharp
public sealed class NetworkLatencyCompensator
{
    public async Task<long> MeasureLatencyAsync(IPEndPoint remote, CancellationToken ct);
    public long CompensateTimestamp(long timestamp, IPEndPoint source);
}
```

#### 6.3 GPS Time Sync (Optional)
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/GpsClockSource.cs`

```csharp
public sealed class GpsClockSource : IPhysicalClockSource
{
    // GPS receiver integration
}
```

### Testing Requirements
- ✅ PTP accuracy (±100ns)
- ✅ Network latency measurement
- ✅ Cross-datacenter sync
- ✅ Fault tolerance

### Dependencies
- Phase 1 (clock synchronization)

### DotCompute Changes
- None

---

## Phase 7: Integration & Optimization (Weeks 13-14) 🟡

### Goals
- End-to-end performance optimization
- Fault tolerance
- Monitoring and observability
- Documentation

### Deliverables

#### 7.1 Performance Optimization
- Profile critical paths
- Optimize memory allocations
- GPU kernel optimization
- Batch processing tuning

**Targets**:
- HLC generation: <50ns
- Message throughput: 10M/sec
- Pattern detection: <100μs per window (GPU)
- Temporal queries: <1ms

#### 7.2 Fault Tolerance
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/TemporalFaultHandler.cs`

```csharp
public sealed class TemporalFaultHandler
{
    public Task HandleClockDesyncAsync();
    public Task HandleMessageLossAsync(Guid messageId);
    public Task HandleActorFailureAsync(string actorId);
}
```

#### 7.3 Monitoring
**File**: `src/Orleans.GpuBridge.Runtime/Temporal/TemporalMetrics.cs`

```csharp
public sealed class TemporalMetrics
{
    public TimeSpan ClockDrift { get; }
    public Distribution MessageLatency { get; }
    public long PatternDetectionRate { get; }
    public int CausalViolationCount { get; }
}
```

**Metrics**:
- Clock drift over time
- Message latency histograms
- Pattern detection rates
- Causal violation counts

#### 7.4 Documentation
- API documentation (XML docs)
- Getting started guide
- Pattern writing tutorial
- Performance tuning guide
- Troubleshooting guide
- Architecture overview

### Testing Requirements
- ✅ Load testing (10M messages/sec)
- ✅ Chaos testing (network failures, clock skew)
- ✅ Long-running stability tests
- ✅ Cross-datacenter tests

### Dependencies
- All previous phases

### DotCompute Changes
- None

---

## Milestone Summary

### Milestone 1: Basic Temporal Correctness (End of Phase 2)
**Week 4**
- ✅ HLC-based message ordering
- ✅ Temporal graph storage
- ✅ Basic time queries
- **Status**: Functional but CPU-bound

### Milestone 2: Pattern Detection (End of Phase 3)
**Week 6**
- ✅ Pattern matching engine
- ✅ GPU-accelerated detection
- ✅ Pattern library
- **Status**: Ready for financial analytics use case

### Milestone 3: Production-Ready (End of Phase 4)
**Week 8**
- ✅ Causal ordering guarantees
- ✅ Deadlock detection
- ✅ Fault tolerance
- **Status**: Production-ready for financial use case (CPU timing)

### Milestone 4: GPU-Native Timing (End of Phase 5)
**Week 10**
- ✅ GPU-side timestamps
- ✅ Device barriers
- ✅ Nanosecond precision
- **Status**: Ready for physics simulation use case

### Milestone 5: High-Precision Timing (End of Phase 6)
**Week 12**
- ✅ PTP synchronization
- ✅ Sub-microsecond accuracy
- **Status**: Production-ready for physics use case

### Milestone 6: Production Hardened (End of Phase 7)
**Week 14**
- ✅ Optimized performance
- ✅ Complete monitoring
- ✅ Full documentation
- **Status**: Enterprise-ready

---

## Resource Allocation

### Developer Effort (Person-Weeks)

| Phase | Coding | Testing | Documentation | DotCompute Work | Total |
|-------|--------|---------|---------------|-----------------|-------|
| Phase 1 | 6 days | 3 days | 1 day | 0 days | 10 days |
| Phase 2 | 7 days | 2 days | 1 day | 0 days | 10 days |
| Phase 3 | 8 days | 2 days | 0 days | 0 days | 10 days |
| Phase 4 | 7 days | 3 days | 0 days | 0 days | 10 days |
| Phase 5 | 5 days | 2 days | 1 day | 2 days | 10 days |
| Phase 6 | 6 days | 2 days | 2 days | 0 days | 10 days |
| Phase 7 | 4 days | 4 days | 2 days | 0 days | 10 days |
| **Total** | **43 days** | **18 days** | **7 days** | **2 days** | **70 days** |

**Note**: DotCompute work can be done in parallel by maintainer

### Hardware Requirements
- 1× NVIDIA GPU (Compute 6.0+) for development
- 1× PTP-capable network card (Phase 6 only)
- Optional: GPS receiver (Phase 6, if cross-datacenter sync needed)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DotCompute API changes delayed | Medium | High | Phases 1-4 don't depend on it |
| GPU barrier performance insufficient | Low | Medium | Fall back to CPU synchronization |
| PTP hardware unavailable | Low | Low | NTP sufficient for financial use case |
| Clock drift too high | Low | Medium | Increase calibration frequency |
| Pattern matching too slow | Medium | Medium | Start with CPU, optimize GPU later |

---

## Success Criteria

### Performance Targets
- ✅ Message throughput: 10M messages/sec
- ✅ HLC generation latency: <50ns
- ✅ Pattern detection: <100μs per window (GPU)
- ✅ Temporal queries: <1ms for 1M edges
- ✅ Clock synchronization: ±1ms (NTP) or ±100ns (PTP)

### Correctness Targets
- ✅ 100% causal ordering correctness
- ✅ Zero missed patterns (0% false negatives)
- ✅ <1% false positive rate
- ✅ No deadlocks in normal operation

### Quality Targets
- ✅ >90% code coverage
- ✅ All APIs documented
- ✅ Performance tests passing
- ✅ Integration tests passing

---

## Next Steps

### Immediate Actions (Week 1)
1. **Set up development branch**
   ```bash
   git checkout -b feature/temporal-correctness
   ```

2. **Create project structure**
   ```bash
   mkdir -p src/Orleans.GpuBridge.Runtime/Temporal
   mkdir -p tests/Orleans.GpuBridge.Runtime.Tests/Temporal
   ```

3. **Start Phase 1 implementation**
   - Implement `HybridTimestamp` struct
   - Write unit tests for clock monotonicity
   - Benchmark HLC performance

4. **Contact DotCompute maintainer**
   - Share `DOTCOMPUTE-API-SPEC.md`
   - Discuss Phase 5 timeline
   - Agree on API surface

### Weekly Check-ins
- Monday: Plan week's work
- Wednesday: Mid-week progress review
- Friday: Demo completed features, update roadmap

---

## Appendix: File Structure

```
Orleans.GpuBridge.Core/
├── src/
│   ├── Orleans.GpuBridge.Abstractions/
│   │   └── Temporal/
│   │       ├── HybridLogicalClock.cs
│   │       ├── VectorClock.cs
│   │       └── IPhysicalClockSource.cs
│   ├── Orleans.GpuBridge.Runtime/
│   │   └── Temporal/
│   │       ├── TemporalMessageQueue.cs
│   │       ├── TemporalGraphStorage.cs
│   │       ├── TemporalPatternDetector.cs
│   │       ├── CausalMessageQueue.cs
│   │       ├── DeadlockDetector.cs
│   │       ├── NtpClockSource.cs
│   │       ├── PtpClockSource.cs
│   │       └── Patterns/
│   │           ├── RapidSplitPattern.cs
│   │           └── CircularFlowPattern.cs
│   ├── Orleans.GpuBridge.Grains/
│   │   └── Resident/
│   │       └── Messages/
│   │           └── TemporalResidentMessage.cs
│   └── Orleans.GpuBridge.Backends.DotCompute/
│       └── Temporal/
│           ├── GpuTemporalGraph.cs
│           ├── GpuPatternMatcher.cs
│           └── TemporalIntegration.cs
├── tests/
│   └── Orleans.GpuBridge.Runtime.Tests/
│       └── Temporal/
│           ├── HybridLogicalClockTests.cs
│           ├── TemporalGraphTests.cs
│           ├── PatternDetectionTests.cs
│           └── CausalOrderingTests.cs
└── docs/
    └── temporal/
        ├── TEMPORAL-CORRECTNESS-DESIGN.md
        ├── DOTCOMPUTE-API-SPEC.md
        ├── IMPLEMENTATION-ROADMAP.md (this file)
        ├── GETTING-STARTED.md
        └── PATTERN-WRITING-GUIDE.md
```

---

*Document Version: 1.0*
*Last Updated: 2025-11-10*
*Author: Claude (Anthropic)*
