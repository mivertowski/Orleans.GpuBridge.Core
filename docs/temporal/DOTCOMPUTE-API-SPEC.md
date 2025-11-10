# DotCompute API Extensions for Temporal Correctness
## API Specification v1.0

## Overview

This document specifies the API extensions required in DotCompute to support temporal correctness features in Orleans.GpuBridge.Core. These extensions enable GPU-native timing, synchronization barriers, and causal memory ordering.

---

## 1. Timing API

### 1.1 Interface Definitions

```csharp
namespace DotCompute.Timing;

/// <summary>
/// Provides GPU-native timing capabilities for temporal correctness.
/// </summary>
public interface ITimingProvider
{
    /// <summary>
    /// Gets the current GPU timestamp in nanoseconds.
    /// </summary>
    /// <remarks>
    /// Implementation notes:
    /// - CUDA: Uses %%globaltimer register (1ns resolution)
    /// - OpenCL: Uses clock() built-in (microsecond resolution)
    /// - CPU: Uses Stopwatch (100ns resolution)
    /// </remarks>
    Task<long> GetGpuTimestampAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets multiple GPU timestamps in batch (more efficient).
    /// </summary>
    Task<long[]> GetGpuTimestampsBatchAsync(int count, CancellationToken ct = default);

    /// <summary>
    /// Calibrates GPU clock against CPU clock.
    /// </summary>
    /// <remarks>
    /// Performs multiple round-trip measurements to determine:
    /// - Offset: GPU_time - CPU_time
    /// - Drift: Rate of clock divergence (parts per million)
    /// - Error bound: ± uncertainty in offset measurement
    /// </remarks>
    Task<ClockCalibration> CalibrateAsync(
        int sampleCount = 100,
        CancellationToken ct = default);

    /// <summary>
    /// Enables automatic timestamp injection at kernel entry points.
    /// </summary>
    /// <remarks>
    /// When enabled, kernels automatically record entry timestamp in parameter slot 0.
    /// Application must allocate device memory for timestamps.
    /// </remarks>
    void EnableTimestampInjection(bool enable = true);

    /// <summary>
    /// Gets GPU clock frequency in Hz.
    /// </summary>
    long GetGpuClockFrequency();

    /// <summary>
    /// Gets timer resolution in nanoseconds.
    /// </summary>
    long GetTimerResolutionNanos();
}

/// <summary>
/// Results of GPU-CPU clock calibration.
/// </summary>
public readonly struct ClockCalibration
{
    /// <summary>
    /// Clock offset in nanoseconds (GPU_time - CPU_time).
    /// </summary>
    public long OffsetNanos { get; init; }

    /// <summary>
    /// Clock drift rate in parts per million (PPM).
    /// Positive values indicate GPU clock runs faster than CPU clock.
    /// </summary>
    public double DriftPPM { get; init; }

    /// <summary>
    /// Error bound in nanoseconds (±).
    /// Actual offset is within [OffsetNanos - ErrorBoundNanos, OffsetNanos + ErrorBoundNanos].
    /// </summary>
    public long ErrorBoundNanos { get; init; }

    /// <summary>
    /// Number of samples used for calibration.
    /// </summary>
    public int SampleCount { get; init; }

    /// <summary>
    /// Calibration timestamp (CPU time).
    /// </summary>
    public long CalibrationTimestampNanos { get; init; }

    /// <summary>
    /// Converts GPU timestamp to CPU time using calibration data.
    /// </summary>
    public long GpuToCpuTime(long gpuTimeNanos)
    {
        // Compensate for drift since calibration
        var elapsedSinceCalibration = gpuTimeNanos - CalibrationTimestampNanos;
        var driftCorrection = (long)(elapsedSinceCalibration * (DriftPPM / 1_000_000.0));

        return gpuTimeNanos - OffsetNanos - driftCorrection;
    }

    /// <summary>
    /// Gets uncertainty range for timestamp conversion.
    /// </summary>
    public (long min, long max) GetUncertaintyRange(long gpuTimeNanos)
    {
        var cpuTime = GpuToCpuTime(gpuTimeNanos);
        return (cpuTime - ErrorBoundNanos, cpuTime + ErrorBoundNanos);
    }
}
```

### 1.2 Usage Examples

#### Basic Timestamp Query
```csharp
var provider = deviceManager.GetTimingProvider();

// Get single timestamp
var timestamp = await provider.GetGpuTimestampAsync();
Console.WriteLine($"GPU time: {timestamp}ns");

// Get batch of timestamps (more efficient)
var timestamps = await provider.GetGpuTimestampsBatchAsync(1000);
```

#### Clock Calibration
```csharp
// Calibrate GPU clock
var calibration = await provider.CalibrateAsync(sampleCount: 1000);

Console.WriteLine($"Offset: {calibration.OffsetNanos}ns");
Console.WriteLine($"Drift: {calibration.DriftPPM} PPM");
Console.WriteLine($"Error: ±{calibration.ErrorBoundNanos}ns");

// Convert GPU timestamp to CPU time
var gpuTime = await provider.GetGpuTimestampAsync();
var cpuTime = calibration.GpuToCpuTime(gpuTime);
var (minTime, maxTime) = calibration.GetUncertaintyRange(gpuTime);

Console.WriteLine($"CPU time: {cpuTime}ns ±{maxTime - cpuTime}ns");
```

#### Automatic Timestamp Injection
```csharp
// Enable timestamp injection
provider.EnableTimestampInjection(true);

// Allocate memory for timestamps
var timestampsGpu = await memoryAllocator.AllocateAsync<long>(batchSize);

// Compile kernel (timestamps will be injected automatically)
var kernel = await compiler.CompileAsync(kernelSource);

// Execute kernel (timestamp will be written to first parameter)
await executor.ExecuteAsync(kernel, new object[]
{
    timestampsGpu,    // Parameter 0: auto-injected timestamps
    inputBuffer,      // Parameter 1: your actual input
    outputBuffer      // Parameter 2: your actual output
});

// Read timestamps
var timestamps = await memoryAllocator.ReadAsync<long>(timestampsGpu, batchSize);
Console.WriteLine($"First kernel invocation at: {timestamps[0]}ns");
```

### 1.3 CUDA Implementation Details

```cuda
// GPU timestamp register access
__device__ __forceinline__ long gpu_nanotime()
{
    long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

// Automatic injection in kernel prologue
__global__ void user_kernel_with_timestamp(
    long* timestamps,  // Auto-injected parameter
    float* input,      // User parameter
    float* output)     // User parameter
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Timestamp injection (auto-generated by DotCompute)
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        timestamps[blockIdx.x] = gpu_nanotime();
    }
    __syncthreads();

    // User kernel code
    output[tid] = input[tid] * 2.0f;
}
```

### 1.4 Performance Characteristics

| Operation | CUDA | OpenCL | CPU | Notes |
|-----------|------|--------|-----|-------|
| Single timestamp | ~5ns | ~100ns | ~100ns | CUDA uses register |
| Batch timestamps (1000) | ~50ns | ~1μs | ~100μs | Amortized cost |
| Clock calibration | ~10ms | ~10ms | ~10ms | 100 samples |
| Timestamp injection overhead | ~10ns | ~50ns | N/A | Per kernel launch |

---

## 2. Barrier API

### 2.1 Interface Definitions

```csharp
namespace DotCompute.Synchronization;

/// <summary>
/// Provides device-wide synchronization barriers for multi-kernel coordination.
/// </summary>
public interface IBarrierProvider
{
    /// <summary>
    /// Creates a device-wide barrier.
    /// </summary>
    /// <param name="participantCount">Number of threads/blocks participating in barrier</param>
    /// <param name="options">Barrier configuration options</param>
    IBarrierHandle CreateBarrier(int participantCount, BarrierOptions? options = null);

    /// <summary>
    /// Launches kernel with device-wide barrier support.
    /// </summary>
    /// <remarks>
    /// Requires CUDA Cooperative Groups or equivalent OpenCL extension.
    /// Falls back to host-side synchronization if not supported.
    /// </remarks>
    Task ExecuteWithBarrierAsync(
        ICompiledKernel kernel,
        IBarrierHandle barrier,
        LaunchConfiguration config,
        object[] arguments,
        CancellationToken ct = default);

    /// <summary>
    /// Checks if device supports hardware barriers.
    /// </summary>
    bool IsHardwareBarrierSupported { get; }

    /// <summary>
    /// Gets maximum barrier participant count.
    /// </summary>
    int MaxBarrierParticipants { get; }
}

/// <summary>
/// Configuration options for barriers.
/// </summary>
public sealed class BarrierOptions
{
    /// <summary>
    /// Timeout for barrier wait (default: infinite).
    /// </summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.MaxValue;

    /// <summary>
    /// Enable barrier timeout detection.
    /// </summary>
    public bool EnableTimeoutDetection { get; init; } = false;

    /// <summary>
    /// Synchronization scope (block, device, system).
    /// </summary>
    public BarrierScope Scope { get; init; } = BarrierScope.Device;

    /// <summary>
    /// Enable barrier arrival counting for debugging.
    /// </summary>
    public bool EnableArrivalCounting { get; init; } = false;
}

public enum BarrierScope
{
    /// <summary>
    /// Thread block scope (__syncthreads).
    /// </summary>
    ThreadBlock,

    /// <summary>
    /// Device scope (cooperative groups grid.sync()).
    /// </summary>
    Device,

    /// <summary>
    /// System scope (across devices, uses host synchronization).
    /// </summary>
    System
}

/// <summary>
/// Handle to a barrier instance.
/// </summary>
public interface IBarrierHandle : IDisposable
{
    /// <summary>
    /// Waits for all participants to arrive at barrier.
    /// </summary>
    Task WaitAsync(CancellationToken ct = default);

    /// <summary>
    /// Resets barrier for next use.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets number of participants that have arrived.
    /// </summary>
    int ArrivalCount { get; }

    /// <summary>
    /// Gets total participant count.
    /// </summary>
    int ParticipantCount { get; }

    /// <summary>
    /// Checks if all participants have arrived.
    /// </summary>
    bool IsReady { get; }

    /// <summary>
    /// Gets barrier ID for debugging.
    /// </summary>
    Guid BarrierId { get; }
}
```

### 2.2 Usage Examples

#### Basic Device-Wide Barrier
```csharp
var barrierProvider = deviceManager.GetBarrierProvider();

// Create barrier for 1M threads (1000 blocks × 1024 threads)
var barrier = barrierProvider.CreateBarrier(
    participantCount: 1000 * 1024,
    options: new BarrierOptions
    {
        Scope = BarrierScope.Device,
        EnableArrivalCounting = true
    });

try
{
    // Launch kernel with barrier support
    await barrierProvider.ExecuteWithBarrierAsync(
        kernel: waveSimKernel,
        barrier: barrier,
        config: new LaunchConfiguration
        {
            GridDim = (1000, 1, 1),
            BlockDim = (1024, 1, 1)
        },
        arguments: new object[] { stateBuffer, nextStateBuffer });

    Console.WriteLine($"Barrier arrivals: {barrier.ArrivalCount}");
}
finally
{
    barrier.Dispose();
}
```

#### Multi-Step Simulation with Barriers
```csharp
// Create persistent barrier for simulation
var barrier = barrierProvider.CreateBarrier(gridSize * blockSize);

for (int step = 0; step < 10000; step++)
{
    // Step 1: Compute next state (all threads in parallel)
    await barrierProvider.ExecuteWithBarrierAsync(
        computeKernel, barrier, config,
        new object[] { currentState, nextState });

    // Barrier ensures ALL threads completed step before proceeding
    await barrier.WaitAsync();

    // Step 2: Swap buffers
    (currentState, nextState) = (nextState, currentState);

    // Reset barrier for next iteration
    barrier.Reset();
}
```

#### System-Wide Barrier (Multi-GPU)
```csharp
// Barrier across multiple GPUs
var systemBarrier = barrierProvider.CreateBarrier(
    participantCount: gpuCount * threadsPerGpu,
    options: new BarrierOptions { Scope = BarrierScope.System });

// Launch kernel on each GPU
var tasks = gpus.Select(async gpu =>
{
    await gpu.ExecuteWithBarrierAsync(kernel, systemBarrier, config, args);
});

await Task.WhenAll(tasks);
await systemBarrier.WaitAsync();  // All GPUs complete
```

### 2.3 CUDA Implementation Details

```cuda
#include <cooperative_groups.h>

__global__ void wave_step_with_barrier(
    float* state_current,
    float* state_next,
    int grid_size)
{
    namespace cg = cooperative_groups;

    // Get device-wide grid group
    cg::grid_group grid = cg::this_grid();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= grid_size) return;

    // Compute next state
    state_next[tid] = compute_wave_propagation(state_current, tid);

    // BARRIER: Wait for ALL threads across ALL blocks
    grid.sync();

    // Now safe to read from state_next (all writes completed)
    float gradient = state_next[tid] - state_next[(tid + 1) % grid_size];
}
```

**Kernel Launch Requirements:**
```csharp
// Must use cudaLaunchCooperativeKernel for device-wide barriers
cudaLaunchCooperativeKernel(
    kernel,
    gridDim,
    blockDim,
    args,
    0,  // shared memory
    stream);

// Alternative: cudaLaunchCooperativeKernelMultiDevice for multi-GPU
```

### 2.4 Performance Characteristics

| Operation | Hardware Barrier | Software Barrier | Notes |
|-----------|------------------|------------------|-------|
| Block-level barrier | ~1μs | ~10μs | 1K threads |
| Device-level barrier | ~10μs | ~100μs | 1M threads |
| System-level barrier | ~100μs | ~1ms | Multi-GPU |
| Overhead per launch | ~20μs | ~50μs | Cooperative launch overhead |

---

## 3. Memory Ordering API

### 3.1 Interface Definitions

```csharp
namespace DotCompute.Memory;

/// <summary>
/// Provides causal memory ordering primitives for distributed GPU actors.
/// </summary>
public interface IMemoryOrderingProvider
{
    /// <summary>
    /// Enables causal memory ordering for all memory operations.
    /// </summary>
    /// <remarks>
    /// When enabled:
    /// - Writes use release semantics
    /// - Reads use acquire semantics
    /// - Memory fences enforce ordering
    /// </remarks>
    void EnableCausalOrdering(bool enable = true);

    /// <summary>
    /// Inserts memory fence at specified location in kernel.
    /// </summary>
    void InsertFence(FenceType type, FenceLocation? location = null);

    /// <summary>
    /// Configures memory consistency model.
    /// </summary>
    void SetConsistencyModel(MemoryConsistencyModel model);

    /// <summary>
    /// Gets current consistency model.
    /// </summary>
    MemoryConsistencyModel ConsistencyModel { get; }

    /// <summary>
    /// Checks if device supports acquire-release semantics.
    /// </summary>
    bool IsAcquireReleaseSupported { get; }
}

/// <summary>
/// Memory fence types (matches CUDA/OpenCL semantics).
/// </summary>
public enum FenceType
{
    /// <summary>
    /// Thread block fence (__threadfence_block / barrier).
    /// </summary>
    ThreadBlock,

    /// <summary>
    /// Device fence (__threadfence / mem_fence).
    /// </summary>
    Device,

    /// <summary>
    /// System fence (__threadfence_system / mem_fence + atomic_work_item_fence).
    /// </summary>
    System
}

/// <summary>
/// Location where fence should be inserted.
/// </summary>
public sealed class FenceLocation
{
    /// <summary>
    /// Insert before specified instruction index.
    /// </summary>
    public int? InstructionIndex { get; init; }

    /// <summary>
    /// Insert at kernel entry.
    /// </summary>
    public bool AtEntry { get; init; }

    /// <summary>
    /// Insert at kernel exit.
    /// </summary>
    public bool AtExit { get; init; }

    /// <summary>
    /// Insert after all writes.
    /// </summary>
    public bool AfterWrites { get; init; }

    /// <summary>
    /// Insert before all reads.
    /// </summary>
    public bool BeforeReads { get; init; }
}

/// <summary>
/// Memory consistency models.
/// </summary>
public enum MemoryConsistencyModel
{
    /// <summary>
    /// Relaxed consistency (default GPU model, fastest).
    /// </summary>
    Relaxed,

    /// <summary>
    /// Release-acquire consistency (causal ordering).
    /// </summary>
    ReleaseAcquire,

    /// <summary>
    /// Sequential consistency (total order, slowest).
    /// </summary>
    Sequential
}
```

### 3.2 Usage Examples

#### Enable Causal Ordering
```csharp
var orderingProvider = deviceManager.GetMemoryOrderingProvider();

// Enable causal memory ordering
orderingProvider.EnableCausalOrdering(true);
orderingProvider.SetConsistencyModel(MemoryConsistencyModel.ReleaseAcquire);

// Now all memory operations use acquire-release semantics
await executor.ExecuteAsync(kernel, args);
```

#### Insert Memory Fences
```csharp
// Insert fence at kernel entry (ensure all prior writes visible)
orderingProvider.InsertFence(
    FenceType.System,
    new FenceLocation { AtEntry = true });

// Insert fence after writes (release semantics)
orderingProvider.InsertFence(
    FenceType.Device,
    new FenceLocation { AfterWrites = true });

// Insert fence before reads (acquire semantics)
orderingProvider.InsertFence(
    FenceType.Device,
    new FenceLocation { BeforeReads = true });
```

### 3.3 CUDA Implementation Details

```cuda
// Causal write with release semantics
__device__ void causal_write_i64(volatile long* addr, long value)
{
    // Release fence: make all prior writes visible
    __threadfence_system();

    // Atomic write (ensures visibility)
    atomicExch((unsigned long long*)addr, (unsigned long long)value);
}

// Causal read with acquire semantics
__device__ long causal_read_i64(volatile long* addr)
{
    // Atomic read
    long value = atomicAdd((unsigned long long*)addr, 0ULL);

    // Acquire fence: make read visible to subsequent operations
    __threadfence_system();

    return value;
}

// Example kernel with causal ordering
__global__ void actor_message_send(
    long* message_buffer,
    long* timestamp_buffer,
    int message_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= message_count) return;

    // Write message data
    message_buffer[tid] = compute_message(tid);

    // RELEASE: Ensure message write completes before timestamp
    __threadfence_system();

    // Write timestamp (signals message is ready)
    causal_write_i64(&timestamp_buffer[tid], gpu_nanotime());
}

__global__ void actor_message_receive(
    long* message_buffer,
    long* timestamp_buffer,
    int message_count)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= message_count) return;

    // ACQUIRE: Read timestamp first
    long timestamp = causal_read_i64(&timestamp_buffer[tid]);

    // Now safe to read message (timestamp read ensures message is visible)
    long message = message_buffer[tid];

    process_message(message, timestamp);
}
```

### 3.4 Performance Impact

| Consistency Model | Relative Performance | Use Case |
|-------------------|---------------------|----------|
| Relaxed | 1.0× (baseline) | High-throughput batch processing |
| Release-Acquire | 0.85× (15% overhead) | Causal ordering for actor messages |
| Sequential | 0.60× (40% overhead) | Strict ordering requirements |

**Fence Overhead**:
- `__threadfence_block()`: ~10ns
- `__threadfence()`: ~100ns
- `__threadfence_system()`: ~200ns

---

## 4. Integration with Existing DotCompute APIs

### 4.1 Device Manager Extension

```csharp
public interface IDeviceManager
{
    // Existing methods...
    IEnumerable<IDevice> GetDevices();
    IDevice GetDevice(int deviceIndex);

    // NEW: Get timing provider
    ITimingProvider GetTimingProvider(IDevice device);

    // NEW: Get barrier provider
    IBarrierProvider GetBarrierProvider(IDevice device);

    // NEW: Get memory ordering provider
    IMemoryOrderingProvider GetMemoryOrderingProvider(IDevice device);
}
```

### 4.2 Device Capabilities Extension

```csharp
public interface IDevice
{
    // Existing properties...
    string Name { get; }
    DeviceType Type { get; }
    long MemorySize { get; }

    // NEW: Timing capabilities
    bool SupportsNanosecondTimers { get; }
    long TimerResolutionNanos { get; }
    long ClockFrequencyHz { get; }

    // NEW: Barrier capabilities
    bool SupportsCooperativeGroups { get; }
    int MaxBarrierParticipants { get; }

    // NEW: Memory ordering capabilities
    bool SupportsAcquireRelease { get; }
    MemoryConsistencyModel DefaultConsistencyModel { get; }
}
```

---

## 5. Platform Support Matrix

| Feature | CUDA (Compute 6.0+) | OpenCL 2.0+ | CPU Fallback |
|---------|---------------------|-------------|--------------|
| **Nanosecond timers** | ✅ (%%globaltimer) | ⚠️ (μs resolution) | ✅ (Stopwatch) |
| **Timestamp injection** | ✅ | ✅ | ✅ |
| **Clock calibration** | ✅ | ✅ | ✅ |
| **Device barriers** | ✅ (Coop. Groups) | ⚠️ (Extension req.) | ✅ (Barrier class) |
| **System barriers** | ✅ | ⚠️ | ✅ |
| **Acquire-release** | ✅ (CUDA 9.0+) | ✅ (2.0+) | ✅ (volatile/Interlocked) |
| **Memory fences** | ✅ | ✅ | ✅ |

**Legend**:
- ✅ Full support
- ⚠️ Partial support or requires extensions
- ❌ Not supported

---

## 6. Backwards Compatibility

### 6.1 API Design Principles

1. **Opt-in features**: All temporal features are opt-in via configuration
2. **Graceful degradation**: Falls back to CPU synchronization if GPU features unavailable
3. **Capability detection**: Runtime checks for hardware support
4. **No breaking changes**: All new APIs are additions, not modifications

### 6.2 Feature Detection Example

```csharp
var device = deviceManager.GetDevice(0);
var timingProvider = deviceManager.GetTimingProvider(device);

if (device.SupportsNanosecondTimers)
{
    // Use GPU-native timestamps
    var timestamp = await timingProvider.GetGpuTimestampAsync();
}
else
{
    // Fall back to CPU timestamps
    var timestamp = Stopwatch.GetTimestamp();
}
```

---

## 7. Testing Requirements

### 7.1 Unit Tests

```csharp
[Fact]
public async Task TimingProvider_ReturnsMonotonicTimestamps()
{
    var provider = GetTimingProvider();
    var t1 = await provider.GetGpuTimestampAsync();
    await Task.Delay(10);
    var t2 = await provider.GetGpuTimestampAsync();

    Assert.True(t2 > t1, "Timestamps must be monotonically increasing");
}

[Fact]
public async Task ClockCalibration_HasBoundedError()
{
    var provider = GetTimingProvider();
    var calibration = await provider.CalibrateAsync(sampleCount: 1000);

    Assert.True(calibration.ErrorBoundNanos < 1_000_000,
                "Error bound should be < 1ms for 1000 samples");
}

[Fact]
public async Task Barrier_SynchronizesAllThreads()
{
    var barrierProvider = GetBarrierProvider();
    var barrier = barrierProvider.CreateBarrier(participantCount: 1024);

    await barrierProvider.ExecuteWithBarrierAsync(
        testKernel, barrier, config, args);

    Assert.Equal(1024, barrier.ArrivalCount);
    Assert.True(barrier.IsReady);
}
```

### 7.2 Integration Tests

```csharp
[Fact]
public async Task TemporalActors_MaintainCausalOrder()
{
    // Enable temporal features
    var ordering = deviceManager.GetMemoryOrderingProvider(device);
    ordering.EnableCausalOrdering(true);

    // Test actor message passing
    var actorA = GetActor("A");
    var actorB = GetActor("B");

    var msg1 = await actorA.SendAsync("message-1");
    var msg2 = await actorB.SendAsync("message-2", dependencies: new[] { msg1.Id });

    // Verify msg2 processed after msg1
    Assert.True(msg2.Timestamp > msg1.Timestamp);
}
```

---

## 8. Performance Benchmarks

### 8.1 Required Benchmarks

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| Single timestamp query | < 10ns | Latency |
| Batch timestamp query (1000) | < 1μs | Amortized latency |
| Clock calibration (100 samples) | < 10ms | Total time |
| Timestamp injection overhead | < 20ns | Per kernel launch |
| Device barrier (1M threads) | < 100μs | Synchronization time |
| System fence overhead | < 500ns | Per fence |
| Memory ordering impact | < 20% | Throughput reduction |

---

## 9. Documentation Requirements

### 9.1 API Documentation

- XML doc comments for all public APIs
- Code examples for common scenarios
- Performance characteristics table
- Platform support matrix

### 9.2 Migration Guide

- How to enable temporal features in existing code
- Performance impact analysis
- Troubleshooting common issues

---

## 10. Implementation Checklist

### Phase 1: Timing API
- [ ] Define interfaces (ITimingProvider, ClockCalibration)
- [ ] Implement CUDA backend (%%globaltimer)
- [ ] Implement OpenCL backend (clock())
- [ ] Implement CPU fallback (Stopwatch)
- [ ] Add clock calibration algorithm
- [ ] Add timestamp injection
- [ ] Write unit tests
- [ ] Write performance benchmarks
- [ ] Document API

### Phase 2: Barrier API
- [ ] Define interfaces (IBarrierProvider, IBarrierHandle)
- [ ] Implement CUDA backend (Cooperative Groups)
- [ ] Implement OpenCL backend (work-group barriers)
- [ ] Implement CPU fallback (Barrier class)
- [ ] Add multi-device support
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Document API

### Phase 3: Memory Ordering API
- [ ] Define interfaces (IMemoryOrderingProvider)
- [ ] Implement CUDA backend (fences, acquire-release)
- [ ] Implement OpenCL backend (mem_fence)
- [ ] Implement CPU fallback (volatile/Interlocked)
- [ ] Add consistency model configuration
- [ ] Write correctness tests
- [ ] Write performance benchmarks
- [ ] Document API

---

## 11. Open Questions for DotCompute Maintainer

1. **API Surface Location**:
   - Should these APIs be in `DotCompute.Core` or separate package `DotCompute.Temporal`?
   - Preference for package organization?

2. **Versioning**:
   - Target DotCompute version for these features (0.5.0, 1.0.0)?
   - Semantic versioning strategy?

3. **Platform Priorities**:
   - Priority order: CUDA → OpenCL → CPU?
   - Should we support older CUDA compute capabilities (< 6.0)?

4. **Testing Infrastructure**:
   - Preferred testing framework (xUnit, NUnit)?
   - CI/CD integration requirements?

5. **Performance Targets**:
   - Are the performance targets in this spec acceptable?
   - Any specific benchmarks you'd like to see?

6. **Breaking Changes**:
   - Any concerns about backwards compatibility?
   - Deprecation strategy if needed?

---

## Conclusion

These API extensions provide the foundation for temporal correctness in GPU-native actors while maintaining backwards compatibility and graceful degradation. The phased implementation approach allows for iterative development and validation.

**Key Design Principles**:
1. **Opt-in**: Features are enabled explicitly
2. **Graceful degradation**: Falls back to CPU when GPU features unavailable
3. **Performance-conscious**: Minimal overhead when disabled
4. **Platform-agnostic**: Works across CUDA, OpenCL, and CPU

**Next Steps**:
1. Review and approve API design
2. Prioritize implementation phases
3. Set up development branch
4. Begin Phase 1 implementation (Timing API)

---

*Document Version: 1.0*
*Last Updated: 2025-11-10*
*Author: Claude (Anthropic)*
