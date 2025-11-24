# DotCompute Bug Report: MessageQueue<T>.TryEnqueue Fails with Struct Types

**Date:** 2025-11-24
**Severity:** Critical / Bug
**Component:** `DotCompute.Core.Messaging.MessageQueue<T>`
**Affects:** All ring kernel message passing using struct message types

## Summary

`MessageQueue<T>.TryEnqueue` throws `NotSupportedException` when `T` is a struct (value type) because it uses `Interlocked.Exchange<T>` which has a constraint `where T : class`.

## Error Observed

```
System.NotSupportedException: The specified type must be a reference type, a primitive type, or an enum type.
   at DotCompute.Core.Messaging.MessageQueue`1.TryEnqueue(T message, CancellationToken cancellationToken)
   at DotCompute.Backends.CUDA.RingKernels.CudaRingKernelRuntime.SendToNamedQueueAsync[T](String queueName, T message, CancellationToken cancellationToken)
```

## Root Cause

In `MessageQueue.cs`, lines 193 and 234:

```csharp
// Line 193 (TryEnqueue):
Interlocked.Exchange(ref _buffer[slotIndex], message);

// Line 234 (TryDequeue):
message = Interlocked.Exchange(ref _buffer[slotIndex], default);
```

**Problem:** `Interlocked.Exchange<T>(ref T, T)` has an implicit constraint `where T : class`. It only works with reference types (classes), not value types (structs).

When `T` is a struct like `VectorAddProcessorRingRequest`, the runtime throws `NotSupportedException`.

## Message Type Definition

The `VectorAddProcessorRingRequest` is a struct implementing `IRingKernelMessage`:

```csharp
[MemoryPackable]
public partial struct VectorAddProcessorRingRequest : IRingKernelMessage
{
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public Guid? CorrelationId { get; set; }
    // ... primitive fields only
}
```

This is the **correct** design for CUDA-compatible messages - structs are required for GPU memory compatibility and efficient serialization.

## Suggested Fix

### Option A: Use lock + direct assignment (Recommended)

Replace `Interlocked.Exchange` with a lock for value types:

```csharp
// In class declaration, add:
private readonly object[] _locks; // One lock per slot

// In constructor:
_locks = new object[options.Capacity];
for (int i = 0; i < options.Capacity; i++)
    _locks[i] = new object();

// In TryEnqueue (line 193):
lock (_locks[slotIndex])
{
    _buffer[slotIndex] = message;
}

// In TryDequeue (line 234):
lock (_locks[slotIndex])
{
    message = _buffer[slotIndex];
    _buffer[slotIndex] = default;
}
```

**Pros:** Works with all types (reference and value types)
**Cons:** Slightly higher overhead due to lock acquisition

### Option B: Use Volatile for value types

For structs, use `Volatile.Write` and `Volatile.Read`:

```csharp
// In TryEnqueue:
if (typeof(T).IsValueType)
{
    Volatile.Write(ref _buffer[slotIndex], message);
}
else
{
    Interlocked.Exchange(ref _buffer[slotIndex], message);
}

// In TryDequeue:
if (typeof(T).IsValueType)
{
    message = Volatile.Read(ref _buffer[slotIndex]);
    Volatile.Write(ref _buffer[slotIndex], default);
}
else
{
    message = Interlocked.Exchange(ref _buffer[slotIndex], default);
}
```

**Pros:** No locks, JIT can optimize the branch
**Cons:** Not truly atomic for large structs (data races possible)

### Option C: Change message types to classes

Change message types from `struct` to `class`:

```csharp
[MemoryPackable]
public partial class VectorAddProcessorRingRequest : IRingKernelMessage
{
    // ... same fields
}
```

**Pros:** Simple fix, `Interlocked.Exchange` works natively
**Cons:**
- Heap allocations per message (GC pressure)
- Breaks CUDA memory compatibility (structs are required for GPU)
- Not recommended for high-performance message passing

## Recommended Solution

**Option A (lock-based)** is the safest fix because:

1. Works with all types (structs and classes)
2. Truly atomic operations (no data races)
3. Minimal impact on existing code
4. Overhead is acceptable for <50ns target (modern locks are ~20ns uncontended)

### Alternative: Striped Lock Pattern

For even better performance, use a striped lock pattern:

```csharp
private readonly object[] _stripedLocks = new object[32]; // 32 stripes

// In constructor:
for (int i = 0; i < 32; i++)
    _stripedLocks[i] = new object();

// Lock by slot index modulo stripe count:
lock (_stripedLocks[slotIndex & 31])
{
    _buffer[slotIndex] = message;
}
```

This reduces lock contention while maintaining correctness.

## Impact Assessment

| Scenario | Current Status |
|----------|---------------|
| Class message types | Works |
| Struct message types | **BROKEN** |
| CUDA ring kernels (need structs) | **BROKEN** |
| High-performance messaging | Blocked |

## Test Environment

- **OS:** WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- **GPU:** NVIDIA RTX (Compute Capability 8.9)
- **CUDA:** 13.x
- **DotCompute:** 0.5.0-alpha
- **Orleans.GpuBridge:** MessagePassingTest

## Workaround

Until fixed, consumers can:

1. Convert struct messages to classes (loses GPU compatibility)
2. Implement custom message queuing without `MessageQueue<T>`
3. Use a different queue implementation

## Priority

**CRITICAL** - This blocks all CUDA ring kernel message passing functionality since struct message types are required for GPU memory compatibility.

## References

- `DotCompute.Core/Messaging/MessageQueue.cs` - Buggy code
- `Orleans.GpuBridge.Backends.DotCompute/Temporal/VectorAddMessages.cs` - Affected message types
- `DotCompute.Backends.CUDA/RingKernels/CudaRingKernelRuntime.cs` - Call site
- Microsoft Docs: [Interlocked.Exchange](https://docs.microsoft.com/en-us/dotnet/api/system.threading.interlocked.exchange)
