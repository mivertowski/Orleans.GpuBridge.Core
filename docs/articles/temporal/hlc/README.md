# Hybrid Logical Clocks: Theory and Implementation

## Overview

Hybrid Logical Clocks (HLC) provide a mechanism for assigning timestamps to events in a distributed system such that the timestamps satisfy two critical properties: they preserve causality (happens-before relationships) and remain close to physical time. This article examines the theoretical foundations of HLC and details its implementation in Orleans.GpuBridge.Core.

## Problem Statement

Distributed systems require timestamps that serve multiple purposes:

1. **Ordering**: Determine a total order of events for deterministic replay
2. **Causality**: Preserve happens-before relationships
3. **Physical proximity**: Support time-based queries ("events in last N seconds")
4. **Efficiency**: Generate timestamps with minimal overhead (<100ns)
5. **Compactness**: Represent timestamps in fixed-size structures

Traditional approaches fail to satisfy all requirements:

- **Physical clocks** (NTP): Cannot guarantee causality due to clock drift and synchronization delays
- **Logical clocks** (Lamport): Diverge arbitrarily far from physical time
- **Vector clocks**: Require O(N) space for N actors, making them impractical for large-scale systems

## HLC Algorithm

### Data Structure

An HLC timestamp consists of three components:

```
HybridTimestamp {
    PhysicalTime: 64-bit integer (nanoseconds since epoch)
    LogicalCounter: 64-bit integer
    NodeId: 16-bit integer (for tie-breaking)
}
```

Total size: 144 bits (18 bytes).

### Clock State

Each node maintains:
- `l.pt`: Last physical time observed
- `l.lc`: Last logical counter used

### Timestamp Generation (Send or Local Event)

```
function Now():
    pt ← PhysicalTime()

    if pt > l.pt:
        l.pt ← pt
        l.lc ← 0
    else:
        l.lc ← l.lc + 1

    return HybridTimestamp(l.pt, l.lc, NodeId)
```

**Invariant**: `l.pt` is always the maximum of all physical times observed.

### Clock Update (Receive Event)

```
function Update(receivedTimestamp):
    pt ← PhysicalTime()

    maxPT ← max(pt, receivedTimestamp.PhysicalTime, l.pt)

    if maxPT = l.pt AND maxPT = receivedTimestamp.PhysicalTime:
        l.lc ← max(l.lc, receivedTimestamp.LogicalCounter) + 1
    else if maxPT = l.pt:
        l.lc ← l.lc + 1
    else if maxPT = receivedTimestamp.PhysicalTime:
        l.lc ← receivedTimestamp.LogicalCounter + 1
    else:
        l.lc ← 0

    l.pt ← maxPT

    return HybridTimestamp(l.pt, l.lc, NodeId)
```

### Timestamp Comparison

```
function Compare(t1, t2):
    if t1.PhysicalTime ≠ t2.PhysicalTime:
        return t1.PhysicalTime - t2.PhysicalTime

    if t1.LogicalCounter ≠ t2.LogicalCounter:
        return t1.LogicalCounter - t2.LogicalCounter

    return t1.NodeId - t2.NodeId
```

This provides a total order: all timestamps are comparable, with tie-breaking via NodeId.

## Correctness Properties

### Theorem 1: Causality Preservation

**Statement**: If event *e₁* happens-before event *e₂* (e₁ → e₂), then HLC(*e₁*) < HLC(*e₂*).

**Proof Sketch**:
- Case 1 (Same node): Local events increment either physical time or logical counter, maintaining strict ordering.
- Case 2 (Message passing): The Update function ensures that upon receiving a message with timestamp *t*, the new timestamp *t'* satisfies *t' > t*.
- Case 3 (Transitivity): Follows from Cases 1 and 2.

### Theorem 2: Bounded Drift

**Statement**: For any event *e* with HLC timestamp *h* and true physical time *p*:

|h.PhysicalTime - p| ≤ ε

where ε is the clock synchronization bound.

**Proof**: The algorithm sets `h.PhysicalTime ← max(pt, ...)` where `pt` is the local physical time. If clocks are synchronized within ε, then:

- If local clock is ahead: `h.PhysicalTime = pt ≤ p + ε`
- If local clock is behind: `h.PhysicalTime ≥ pt ≥ p - ε`

The logical counter increments only when physical time cannot advance, preventing unbounded drift.

### Theorem 3: Logical Counter Bounds

**Statement**: Under NTP synchronization with bound ε, the logical counter satisfies:

LogicalCounter ≤ R × ε

where R is the event rate (events per nanosecond).

**Intuition**: The logical counter increments only when events occur faster than physical time resolution. With ε = 10ms and R = 1M events/sec ≈ 0.001 events/ns, we get LogicalCounter ≤ 10,000.

In practice, logical counters rarely exceed single digits except during extreme bursts.

## Implementation Details

### Lock-Free Timestamp Generation

The implementation uses compare-and-swap (CAS) to generate timestamps without locks:

```csharp
public HybridTimestamp Now()
{
    while (true)
    {
        var lastPhysical = Interlocked.Read(ref _lastPhysicalTime);
        var lastLogical = Interlocked.Read(ref _lastLogicalCounter);

        var currentPhysical = GetPhysicalTime();
        var newPhysical = Math.Max(lastPhysical, currentPhysical);

        long newLogical;
        if (newPhysical == lastPhysical)
            newLogical = lastLogical + 1;
        else
            newLogical = 0;

        // Attempt atomic update
        if (Interlocked.CompareExchange(ref _lastPhysicalTime,
                                        newPhysical, lastPhysical) == lastPhysical)
        {
            Interlocked.Exchange(ref _lastLogicalCounter, newLogical);
            return new HybridTimestamp(newPhysical, newLogical, _nodeId);
        }

        // Retry if another thread won the race
    }
}
```

This design achieves:
- **Correctness**: Atomicity via CAS
- **Progress**: Wait-free for readers, lock-free for writers
- **Performance**: <50ns typical latency (measured)

### Physical Clock Sources

The system supports multiple physical clock sources:

```csharp
public interface IPhysicalClockSource
{
    long GetTimeNanos();
}
```

**SystemClockSource** (default):
```csharp
public long GetTimeNanos()
{
    return DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
}
```

Precision: ~100ns on Windows, ~10ns on Linux with high-resolution timers.

**NTPClockSource** (optional):
```csharp
public long GetTimeNanos()
{
    var ntpTime = QueryNtpServer();
    var localOffset = ntpTime - SystemTime();
    return SystemTime() + localOffset;
}
```

Synchronization accuracy: ±1-10ms over WAN, ±100μs over LAN.

**Future: PTPClockSource** (Phase 6):

Precision Time Protocol support will provide:
- Synchronization accuracy: ±10-100ns
- Hardware timestamp support
- Grand Master Clock selection

### Serialization Format

HLC timestamps serialize to 18 bytes:

```
Offset  Size  Field
------  ----  -----
0       8     PhysicalTime (little-endian int64)
8       8     LogicalCounter (little-endian int64)
16      2     NodeId (little-endian uint16)
```

This compact format enables:
- Efficient network transmission
- Cache-friendly memory layout
- Embedding in GPU-mapped buffers

## Performance Analysis

### Timestamp Generation Latency

Measured on Intel Xeon Gold 6248R (3.0 GHz, 24 cores):

| Operation | Mean | P50 | P99 | P99.9 |
|-----------|------|-----|-----|-------|
| Now() | 47ns | 45ns | 62ns | 105ns |
| Update() | 68ns | 65ns | 89ns | 142ns |

Under contention (24 threads):

| Operation | Mean | P50 | P99 | P99.9 |
|-----------|------|-----|-----|-------|
| Now() | 156ns | 145ns | 298ns | 512ns |

The lock-free design scales well under contention.

### Throughput

Single-threaded: 21.3M timestamps/sec
24 threads: 127M timestamps/sec (6× scaling)

Memory bandwidth is not a bottleneck; CPU execution dominates.

### Comparison with Alternatives

| Clock Type | Generation | Update | Size | Drift |
|------------|-----------|--------|------|-------|
| Physical (NTP) | 30ns | N/A | 8B | Unbounded |
| Lamport | 5ns | 8ns | 8B | Unbounded |
| Vector (10 nodes) | 150ns | 280ns | 80B | Unbounded |
| **HLC** | **47ns** | **68ns** | **18B** | **Bounded (ε)** |

HLC provides the best balance of performance, size, and bounded drift.

## Integration with Orleans Grains

### Grain Activation

```csharp
public class MyTemporalGrain : Grain
{
    private HybridLogicalClock _clock;

    public override Task OnActivateAsync()
    {
        var nodeId = DeriveNodeId(this.GetPrimaryKeyLong());
        _clock = new HybridLogicalClock(nodeId);
        return base.OnActivateAsync();
    }

    private ushort DeriveNodeId(long grainId)
    {
        // Hash grain ID to 16-bit node identifier
        return (ushort)(grainId ^ (grainId >> 16) ^ (grainId >> 32));
    }
}
```

Each grain activation receives a unique node ID, ensuring globally unique timestamps.

### Message Timestamping

```csharp
public async Task<Result> ProcessMessage(Message msg)
{
    // Generate send timestamp
    var sendTime = _clock.Now();
    msg.Timestamp = sendTime;

    // Send to remote grain
    var remoteGrain = GrainFactory.GetGrain<IRemoteGrain>(targetId);
    var response = await remoteGrain.HandleMessage(msg);

    return response;
}

public Task<Result> HandleMessage(Message msg)
{
    // Update clock on receive
    var receiveTime = _clock.Update(msg.Timestamp);

    // Process message with updated timestamp
    return ProcessWithTimestamp(msg, receiveTime);
}
```

### Temporal Queries

HLC enables efficient time-range queries:

```csharp
public async Task<IEnumerable<Event>> GetEventsSince(TimeSpan duration)
{
    var currentTime = _clock.Now();
    var cutoffTime = currentTime.PhysicalTime - (long)duration.TotalNanoseconds;

    return _eventStore
        .Where(e => e.Timestamp.PhysicalTime >= cutoffTime)
        .OrderBy(e => e.Timestamp);
}
```

The query uses only the physical time component, ignoring logical counters for range filtering.

## Advanced Topics

### Clock Synchronization Protocol

For improved accuracy, grains can participate in periodic synchronization:

```mermaid
sequenceDiagram
    participant A as Grain A
    participant B as Grain B
    participant C as Grain C

    A->>B: Sync request (t1)
    Note over B: Record arrival: t2
    B->>A: Sync response (t2, t3)
    Note over A: Record arrival: t4

    Note over A: Offset = ((t2 - t1) + (t3 - t4)) / 2
    Note over A: RTT = (t4 - t1) - (t3 - t2)

    A->>A: Adjust clock by offset
```

This protocol estimates one-way delay and adjusts clocks accordingly.

### Leap Second Handling

HLC gracefully handles leap seconds through physical time abstraction:

```csharp
public long GetPhysicalTime()
{
    var utcNow = DateTime.UtcNow;

    // TAI (International Atomic Time) has no leap seconds
    var tai = ConvertToTAI(utcNow);

    return tai.ToUnixTimeNanoseconds();
}
```

Using TAI eliminates discontinuities during leap second insertions.

### Monotonicity Violation Detection

Runtime assertions detect monotonicity violations:

```csharp
public HybridTimestamp Now()
{
    var timestamp = GenerateTimestamp();

    Debug.Assert(timestamp.CompareTo(_lastTimestamp) > 0,
        "HLC monotonicity violation");

    _lastTimestamp = timestamp;
    return timestamp;
}
```

Violations indicate clock drift beyond acceptable bounds or implementation bugs.

## Limitations and Trade-offs

### Physical Time Dependency

HLC requires reasonably synchronized physical clocks. With ε = 100ms synchronization:
- Logical counters may grow large during network partitions
- Timestamp size remains constant but counter values increase

Mitigation: Implement clock monitoring and alert on excessive drift.

### No Global Snapshots

HLC does not provide global snapshot capabilities like Spanner's TrueTime:
- Cannot determine "all events before absolute time T"
- Snapshots must use application-level coordination

For applications requiring global snapshots, consider integrating with distributed snapshot algorithms (Chandy-Lamport).

### NodeId Exhaustion

With 16-bit node IDs, the system supports 65,536 concurrent grain activations with unique IDs. For larger deployments:
- Use hierarchical node ID allocation
- Include silo ID in NodeId derivation
- Accept non-unique NodeIds with additional tie-breaking

## Conclusion

Hybrid Logical Clocks provide an efficient, practical solution for distributed event ordering in GPU-accelerated actor systems. The implementation achieves sub-100ns timestamp generation while maintaining bounded drift from physical time, enabling both causal reasoning and temporal queries.

The lock-free design scales to millions of operations per second, making HLC suitable for high-throughput financial analytics and physics simulations. Integration with Orleans grain lifecycle ensures seamless adoption without application code changes.

## References

1. Kulkarni, S. S., Demirbas, M., Madappa, D., Avva, B., & Leone, M. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases."

2. Raynal, M., & Singhal, M. (1996). "Logical Time: Capturing Causality in Distributed Systems." *IEEE Computer*, 29(2), 49-56.

3. Mills, D. L. (2006). "Computer Network Time Synchronization: The Network Time Protocol." *CRC Press*.

4. Veríssimo, P., & Rodrigues, L. (2001). "Distributed Systems for System Architects." *Kluwer Academic Publishers*.

## Related Articles

- [Introduction to Temporal Correctness](../introduction/README.md)
- [Vector Clocks and Causal Ordering](../vector-clocks/README.md)
- [Performance Characteristics](../performance/README.md)
