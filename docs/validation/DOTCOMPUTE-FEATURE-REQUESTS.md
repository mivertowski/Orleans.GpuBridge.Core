# DotCompute Feature Requests - Orleans.GpuBridge.Core Integration

**Date**: 2025-11-21
**Requested By**: Orleans.GpuBridge.Core Development Team
**DotCompute Version**: v0.5.0-alpha

---

## 🔴 CRITICAL: Fix Parameter Validation for MemoryPack Structs

### Priority: **IMMEDIATE**
### Severity: **CRITICAL BLOCKER**
### Estimated Fix Time: ~30 minutes

### Problem

The `CudaRingKernelCompiler.IsSupportedCudaType()` method incorrectly rejects `Span<T>` parameters where `T` is a MemoryPack struct, even though these are valid value types for CUDA compilation.

### Current Behavior (BROKEN)

**File**: `DotCompute.Backends.CUDA/RingKernels/CudaRingKernelCompiler.New.cs:260`

```csharp
private bool IsSupportedCudaType(Type type)
{
    // Check for Span<T>
    if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Span<>))
    {
        var elementType = type.GetGenericArguments()[0];
        return elementType.IsPrimitive;  // ❌ PROBLEM: Only accepts primitives!
    }

    return type.IsPrimitive || type.IsValueType;
}
```

**Error Message**:
```
Failed to compile Ring Kernel 'VectorAddProcessor': Parameter 'requestQueue' has unsupported type
'VectorAddRequestMessage' for CUDA compilation. Supported types: primitives, Span<T>, arrays, and value types.
```

### Required Fix

```csharp
private bool IsSupportedCudaType(Type type)
{
    // Check for Span<T>
    if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Span<>))
    {
        var elementType = type.GetGenericArguments()[0];

        // ✅ FIX: Accept ALL value types, not just primitives
        // Value types include: primitives, structs, enums
        return elementType.IsValueType && !elementType.IsPointer;
    }

    return type.IsPrimitive || type.IsValueType;
}
```

### Enhanced Validation (OPTIONAL)

For maximum safety, also validate MemoryPack compatibility:

```csharp
private bool IsSupportedCudaType(Type type)
{
    if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Span<>))
    {
        var elementType = type.GetGenericArguments()[0];

        // Accept value types (structs)
        if (!elementType.IsValueType || elementType.IsPointer)
            return false;

        // OPTIONAL: Verify MemoryPack compatibility
        var hasMemoryPackable = elementType.GetCustomAttributes(false)
            .Any(a => a.GetType().Name == "MemoryPackableAttribute");

        if (!hasMemoryPackable)
        {
            _logger.LogWarning(
                "Type {TypeName} used in Span<T> lacks [MemoryPackable] attribute. " +
                "This may cause GPU memory layout issues.",
                elementType.Name);
        }

        return true;
    }

    return type.IsPrimitive || type.IsValueType;
}
```

### Test Case

**Ring Kernel**:
```csharp
[RingKernel(KernelId = "VectorAddProcessor")]
public static void VectorAddProcessorRing(
    Span<VectorAddRequestMessage> requestQueue,   // Should be VALID
    Span<VectorAddResponseMessage> responseQueue) // Should be VALID
{
    // Message processing...
}
```

**Message Type**:
```csharp
[MemoryPackable]
public readonly partial struct VectorAddRequestMessage : IRingKernelMessage
{
    [MemoryPackOrder(0)] public long Timestamp { get; init; }
    [MemoryPackOrder(1)] public int ActorId { get; init; }
    [MemoryPackOrder(2)] public int Size { get; init; }
}
```

**Expected**: Compilation succeeds
**Actual**: Compilation fails with "unsupported type" error

### Impact

- **Blocks**: All GPU-native actor message passing
- **Affects**: Phase 2 validation of Orleans.GpuBridge.Core
- **Urgency**: Cannot proceed with GPU testing until fixed

---

## 🟡 MEDIUM: Implement Advanced Kernel Configuration Attributes

### Priority: **MEDIUM**
### Severity: **ENHANCEMENT**
### Target Version: v0.5.1 or later

### Overview

Orleans.GpuBridge.Core requires advanced kernel configuration attributes for temporal actor systems, GPU-side synchronization, and performance optimization. These attributes are currently used in the codebase but not yet implemented in DotCompute.Generators.

### Required Attributes (6 Total)

---

### 1. `[EnableTimestamps]`

**Purpose**: Enable GPU hardware timestamp tracking for temporal consistency.

**Usage**:
```csharp
[Kernel("HybridLogicalClock", EnableTimestamps = true)]
public static void HybridLogicalClockKernel(Span<long> timestamps, ...)
```

**Generated PTX**:
```cuda
__global__ void HybridLogicalClockKernel(long* timestamps, ...)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    timestamps[tid] = clock64();  // GPU timestamp capture
    // ... rest of kernel
}
```

**Use Cases**:
- Hybrid Logical Clock (HLC) implementation
- Vector Clock synchronization
- Temporal pattern detection

---

### 2. `[MemoryOrdering(MemoryOrderingMode)]`

**Purpose**: Specify memory ordering semantics for GPU operations (critical for lock-free algorithms).

**Enum**:
```csharp
namespace DotCompute.Generators.Kernel.Attributes;

public enum MemoryOrderingMode
{
    Relaxed,                // No ordering guarantees (fastest)
    Release,                // Store-release semantics
    Acquire,                // Load-acquire semantics
    ReleaseAcquire,         // Both release and acquire
    SequentiallyConsistent  // Full sequential consistency (slowest)
}
```

**Usage**:
```csharp
[RingKernel]
[MemoryOrdering(MemoryOrderingMode.ReleaseAcquire)]
public static void ActorMessageProcessorRing(...)
```

**Generated PTX**:
```cuda
// Relaxed
ld.global.relaxed.u32 %r1, [addr];

// Release
st.global.release.u32 [addr], %r1;

// SequentiallyConsistent
membar.sys;
st.global.u32 [addr], %r1;
```

**Use Cases**:
- Lock-free message queues
- Multi-actor coordination
- Atomic operations with specific guarantees

---

### 3. `[EnableBarriers(BarrierScope)]` / `[EnableBarriers(bool)]`

**Purpose**: Enable GPU synchronization barriers with specified scope.

**Enum**:
```csharp
namespace DotCompute.Generators.Kernel.Attributes;

public enum BarrierScope
{
    Warp,   // Warp-level barrier (__syncwarp)
    Block,  // Block-local barrier (__syncthreads)
    Grid    // Grid-wide barrier (cooperative launch)
}
```

**Usage**:
```csharp
[Kernel]
[EnableBarriers(BarrierScope.Grid)]
public static void TemporalReduceKernel(...)

// OR boolean overload:
[Kernel]
[EnableBarriers(true)]  // Defaults to Block scope
public static void ActorStateManagerRing(...)
```

**Generated CUDA**:
```cuda
// Grid barrier
cooperative_groups::this_grid().sync();

// Block barrier
__syncthreads();

// Warp barrier
__syncwarp();
```

**Use Cases**:
- Temporal reduce operations
- Actor state synchronization
- Multi-phase algorithms

---

### 4. `[MessageQueueSize(int)]`

**Purpose**: Configure ring kernel message queue capacity (overrides default 1024).

**Usage**:
```csharp
[RingKernel]
[MessageQueueSize(4096)]  // High-volume actor
public static void ActorSchedulerRing(...)

[RingKernel]
[MessageQueueSize(8192)]  // Very high volume
public static void ActorMessageProcessorRing(...)
```

**Effect**:
- Allocates GPU memory buffer of specified size
- Generates queue wrapping logic: `index % queueSize`
- Affects cache locality and memory bandwidth

**Use Cases**:
- Tune queue sizes for different actor workloads
- Optimize memory usage vs latency trade-off

---

### 5. `[ProcessingMode(RingProcessingMode)]`

**Purpose**: Specify how ring kernel processes messages from queue.

**Enum**:
```csharp
namespace DotCompute.Generators.Kernel.Attributes;

public enum RingProcessingMode
{
    Continuous,  // Process single message per iteration (low latency)
    Batch,       // Process multiple messages per iteration (high throughput)
    Adaptive     // Switch between batch/continuous based on queue depth
}
```

**Usage**:
```csharp
[RingKernel]
[ProcessingMode(RingProcessingMode.Continuous)]  // Latency-optimized
public static void LowLatencyActorRing(...)

[RingKernel]
[ProcessingMode(RingProcessingMode.Batch)]  // Throughput-optimized
public static void HighThroughputActorRing(...)
```

**Generated Code**:
```cuda
// Continuous mode
if (hasMessage()) {
    process_message();
}

// Batch mode
for (int i = 0; i < batchSize && hasMessage(); i++) {
    process_message();
}

// Adaptive mode
int batchSize = (queueDepth > threshold) ? MAX_BATCH : 1;
```

**Use Cases**:
- Optimize for latency vs throughput
- Dynamic workload adaptation

---

### 6. `[MaxMessagesPerIteration(int)]`

**Purpose**: Limit maximum messages processed per dispatch loop iteration (prevents starvation).

**Usage**:
```csharp
[RingKernel]
[MaxMessagesPerIteration(16)]  // Process up to 16 messages before yielding
public static void FairActorRing(...)
```

**Generated Code**:
```cuda
int messagesProcessed = 0;
while (!stopSignal && messagesProcessed < MAX_MESSAGES_PER_ITERATION)
{
    if (hasMessage()) {
        process_message();
        messagesProcessed++;
    }
    else break;
}
```

**Use Cases**:
- Ensure fairness when multiple actors share GPU
- Prevent high-volume actors from starving others
- Bounded execution time per iteration

---

## Implementation Guidance

### Phase 1: Basic Implementation (v0.5.1)

**Target**: Get basic attribute recognition working

1. Create attribute classes in `DotCompute.Generators.Kernel.Attributes`
2. Add enum definitions (MemoryOrderingMode, BarrierScope, RingProcessingMode)
3. Implement attribute detection in source generator
4. Generate basic CUDA code for each attribute

**Estimated Time**: 2-4 hours per attribute

---

### Phase 2: PTX Generation (v0.5.2)

**Target**: Generate correct PTX instructions

1. Implement memory fence instructions (`membar.sys`, `.release`, `.acquire`)
2. Add barrier synchronization (`__syncthreads`, `__syncwarp`, cooperative groups)
3. Generate timestamp capture code (`clock64()`, `globaltimer()`)

**Estimated Time**: 4-8 hours total

---

### Phase 3: Optimization (v0.6.0)

**Target**: Optimize generated code

1. Adaptive processing mode logic
2. Queue size optimization hints
3. Memory ordering optimization based on hardware

**Estimated Time**: 8-16 hours total

---

## Testing Requirements

### Unit Tests
- Attribute detection and parsing
- Correct enum value handling
- PTX instruction generation

### Integration Tests
- Compile kernels with each attribute
- Verify generated PTX correctness
- Run on actual GPU hardware

### Performance Tests
- Measure overhead of each attribute
- Benchmark memory ordering impact
- Validate latency vs throughput trade-offs

---

## Migration Plan

Once attributes are implemented:

1. **Orleans.GpuBridge.Core** team:
   - Restore disabled temporal kernels
   - Re-enable advanced features
   - Run full validation suite

2. **Breaking Changes**: None - these are new attributes

3. **Backwards Compatibility**: Maintained - existing kernels continue working

---

## Priority Ranking

1. **IMMEDIATE** (v0.5.0-alpha hotfix): Fix `IsSupportedCudaType()` validation bug
2. **HIGH** (v0.5.1): `[EnableTimestamps]` - Required for temporal actors
3. **HIGH** (v0.5.1): `[MemoryOrdering]` - Required for lock-free queues
4. **MEDIUM** (v0.5.2): `[EnableBarriers]` - Required for synchronization
5. **LOW** (v0.6.0): `[MessageQueueSize]` - Performance optimization
6. **LOW** (v0.6.0): `[ProcessingMode]` - Performance optimization
7. **LOW** (v0.6.0): `[MaxMessagesPerIteration]` - Fairness optimization

---

## Contact

**Project**: Orleans.GpuBridge.Core
**Repository**: https://github.com/mivertowski/Orleans.GpuBridge.Core
**Documentation**: `/docs/validation/`

**For Questions**: Create issue in Orleans.GpuBridge.Core repository tagged `DotCompute-integration`

---

**Prepared by**: Orleans.GpuBridge.Core Development Team
**Date**: 2025-11-21
**For**: DotCompute Development Team
