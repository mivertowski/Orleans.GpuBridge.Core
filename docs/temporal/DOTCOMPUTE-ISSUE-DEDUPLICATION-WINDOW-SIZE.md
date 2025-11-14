# DotCompute Issue: MessageQueueOptions DeduplicationWindowSize Validation Error

**Date**: November 14, 2025
**DotCompute Version**: v0.5.2-alpha
**Issue Type**: Runtime validation error
**Severity**: Critical blocker
**Status**: 🚧 Blocking message passing tests

---

## Issue Summary

When launching a ring kernel, `CpuRingKernelRuntime.CreateNamedMessageQueueAsync<T>()` sets `MessageQueueOptions.DeduplicationWindowSize = 4096`, but `MessageQueueOptions.Validate()` enforces a maximum value of 1024. This causes an `ArgumentOutOfRangeException` and prevents ring kernels from launching.

---

## Error Details

### Exception
```
System.ArgumentOutOfRangeException: DeduplicationWindowSize must be between 16 and 1024. (Parameter 'DeduplicationWindowSize')
Actual value was 4096.
```

### Stack Trace
```
at DotCompute.Abstractions.Messaging.MessageQueueOptions.Validate()
at DotCompute.Core.Messaging.MessageQueue`1..ctor(MessageQueueOptions options)
at DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.CreateNamedMessageQueueAsync[T](String queueName, MessageQueueOptions options, CancellationToken cancellationToken)
at DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.<>c__DisplayClass6_0.<<LaunchAsync>b__0>d.MoveNext()
at Orleans.GpuBridge.Backends.DotCompute.Temporal.Generated.VectorAddProcessorRingRingKernelWrapper.LaunchAsync(Int32 gridSize, Int32 blockSize, CancellationToken cancellationToken)
at RingKernelValidation.MessagePassingTest.RunAsync(ILoggerFactory loggerFactory, String backend)
```

### Test Output
```
=== Message Passing Validation Test (CPU) ===
Testing: VectorAddRequest → Ring Kernel → VectorAddResponse

info: MessagePassingTest[0]
      Step 1: Creating CPU ring kernel runtime...
info: DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime[0]
      CPU ring kernel runtime initialized
info: MessagePassingTest[0]
      ✓ Runtime created
info: MessagePassingTest[0]
      Step 2: Creating ring kernel wrapper...
info: MessagePassingTest[0]
      ✓ Wrapper created
info: MessagePassingTest[0]
      Step 3: Launching kernel...

=== ❌ TEST FAILED ===
fail: MessagePassingTest[0]
      ❌ Message passing test failed!
      System.Reflection.TargetInvocationException: Exception has been thrown by the target of an invocation.
       ---> System.ArgumentOutOfRangeException: DeduplicationWindowSize must be between 16 and 1024.
```

---

## Root Cause Analysis

### Where the Error Occurs

**File**: `DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.cs`
**Method**: `LaunchAsync()`
**Action**: Creates message queues for ring kernel input/output

When `LaunchAsync()` detects `IRingKernelMessage` types (classes), it calls `CreateNamedMessageQueueAsync<T>()` with `MessageQueueOptions` that includes:
```csharp
var options = new MessageQueueOptions
{
    Capacity = capacity,  // e.g., 4096
    DeduplicationWindowSize = capacity  // ❌ Problem: Sets to 4096
};
```

### Why the Error Occurs

**File**: `DotCompute.Abstractions.Messaging.MessageQueueOptions.cs`
**Method**: `Validate()`
**Constraint**: Enforces `DeduplicationWindowSize` range of 16-1024

```csharp
public void Validate()
{
    if (DeduplicationWindowSize < 16 || DeduplicationWindowSize > 1024)
    {
        throw new ArgumentOutOfRangeException(
            nameof(DeduplicationWindowSize),
            DeduplicationWindowSize,
            "DeduplicationWindowSize must be between 16 and 1024.");
    }
    // ... other validations
}
```

### The Mismatch

- **Ring kernel launch code**: Sets `DeduplicationWindowSize = capacity` (e.g., 4096 for high-throughput queues)
- **MessageQueueOptions validation**: Enforces maximum of 1024
- **Result**: Exception when `capacity > 1024`

---

## Reproduction Steps

1. **Create ring kernel with IRingKernelMessage types** (VectorAddRequestMessage, VectorAddResponseMessage are classes)
2. **Launch kernel** with standard capacity (e.g., 4096 messages)
3. **Observe exception** during queue creation

**Minimal Test Case**:
```csharp
var runtime = new CpuRingKernelRuntime(logger, compiler);
var wrapper = new VectorAddProcessorRingRingKernelWrapper(runtime);

// This will fail if capacity > 1024
await wrapper.LaunchAsync(gridSize: 1, blockSize: 1, cancellationToken);
```

---

## Suggested Fixes

### Option 1: Clamp DeduplicationWindowSize to Maximum (Quick Fix)

**Location**: `CpuRingKernelRuntime.LaunchAsync()`

```csharp
// Before
var options = new MessageQueueOptions
{
    Capacity = capacity,
    DeduplicationWindowSize = capacity  // ❌ Can exceed 1024
};

// After
var options = new MessageQueueOptions
{
    Capacity = capacity,
    DeduplicationWindowSize = Math.Min(capacity, 1024)  // ✅ Respects constraint
};
```

**Pros**:
- Simple fix
- Maintains backwards compatibility
- Allows high-capacity queues with reasonable deduplication window

**Cons**:
- DeduplicationWindowSize doesn't scale with capacity
- May not be ideal for very large queues

### Option 2: Relax Validation Constraint (Design Change)

**Location**: `MessageQueueOptions.Validate()`

```csharp
// Before
if (DeduplicationWindowSize < 16 || DeduplicationWindowSize > 1024)
{
    throw new ArgumentOutOfRangeException(...);
}

// After - Allow larger windows for high-capacity queues
const int MinDeduplicationWindowSize = 16;
int maxDeduplicationWindowSize = Math.Min(Capacity, 4096);  // Scale with capacity

if (DeduplicationWindowSize < MinDeduplicationWindowSize ||
    DeduplicationWindowSize > maxDeduplicationWindowSize)
{
    throw new ArgumentOutOfRangeException(...);
}
```

**Pros**:
- Allows deduplication window to scale with queue capacity
- More flexible for high-throughput scenarios

**Cons**:
- Changes validation semantics
- May impact memory usage for large queues

### Option 3: Make DeduplicationWindowSize Configurable

**Location**: Ring kernel launch API

```csharp
public class RingKernelLaunchOptions
{
    public int QueueCapacity { get; set; } = 4096;
    public int DeduplicationWindowSize { get; set; } = 1024;  // Explicit configuration
    // ... other options
}

await runtime.LaunchAsync(kernelId, launchOptions, cancellationToken);
```

**Pros**:
- Maximum flexibility
- Users can optimize for their use case

**Cons**:
- API complexity increases
- Requires users to understand deduplication window sizing

---

## Recommended Solution

**Immediate Fix**: Option 1 (Clamp DeduplicationWindowSize)
- Apply fix in both `CpuRingKernelRuntime.LaunchAsync()` and `CudaRingKernelRuntime.LaunchAsync()`
- Change: `DeduplicationWindowSize = Math.Min(capacity, 1024)`
- This unblocks message passing tests immediately

**Long-term Enhancement**: Option 2 (Relax Validation)
- Design decision: Should deduplication window scale with capacity?
- Consider memory/performance tradeoffs
- Document deduplication semantics and sizing recommendations

---

## Impact Assessment

### Current Impact
- ✅ Build succeeds (0 errors, 0 warnings)
- ✅ Constraint violation from v0.5.1-alpha is fixed
- ❌ Ring kernels cannot launch (validation error at runtime)
- ❌ Message passing tests blocked
- ❌ Cannot validate 100-500ns latency target
- ❌ Cannot validate 2M+ messages/s throughput

### Files Affected
- `DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.cs` (needs fix)
- `DotCompute.Backends.CUDA.RingKernels.CudaRingKernelRuntime.cs` (needs fix)
- `DotCompute.Abstractions.Messaging.MessageQueueOptions.cs` (may need design change)

---

## Workaround

**None available**. The validation error occurs in DotCompute's internal queue creation code during kernel launch. Orleans.GpuBridge.Core has no control over `MessageQueueOptions` passed to `CreateNamedMessageQueueAsync<T>()`.

---

## Test Case (Once Fixed)

After fix is applied, this test should succeed:

```csharp
// Create CPU ring kernel runtime
var runtime = new CpuRingKernelRuntime(loggerFactory.CreateLogger<CpuRingKernelRuntime>());

// Create wrapper for VectorAddProcessorRing kernel
var wrapper = new VectorAddProcessorRingRingKernelWrapper(runtime);

// Launch kernel with high-capacity queue (should not throw)
await wrapper.LaunchAsync(gridSize: 1, blockSize: 1, CancellationToken.None);

// Send message to named queue
var request = new VectorAddRequestMessage
{
    VectorALength = 10,
    Operation = VectorOperation.Add,
    UseGpuMemory = false
};

var sent = await runtime.SendToNamedQueueAsync("VectorAddProcessor_Input", request, CancellationToken.None);
Assert.True(sent, "Message should be sent successfully");

// Receive response from named queue
var response = await runtime.ReceiveFromNamedQueueAsync<VectorAddResponseMessage>(
    "VectorAddProcessor_Output",
    CancellationToken.None);

Assert.NotNull(response);
Assert.True(response.Success);
Assert.Equal(10, response.ProcessedElements);
```

---

## Related Issues

- **DOTCOMPUTE-ISSUE-NAMED-QUEUE-CONSTRAINT.md** - Constraint violation (fixed in v0.5.2-alpha)
- This validation error was discovered immediately after constraint fix was deployed

---

## References

- **Error Location**: Line 46 of MessagePassingTest.cs (kernel launch)
- **Stack Trace**: See `/tmp/message-test.log`
- **Build Log**: `/tmp/build2.log` (build successful)
- **DotCompute Version**: ec461333 (Phase 1.5 - Real-Time Telemetry APIs)

---

**Status**: 🚧 **Blocking message passing tests** | ⏸️ **Waiting for DotCompute fix**

**Priority**: Critical (blocks Phase 5 completion)

**Estimated Fix Time**: <30 minutes (Option 1 - clamp to 1024)

---

*Issue discovered during Phase 5 Week 15 - DotCompute SDK v0.5.2-alpha integration testing*
