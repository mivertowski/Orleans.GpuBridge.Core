# DotCompute Feature Request: WSL2 Ring Kernel Blocking Issue

**Date:** 2025-11-24
**Severity:** Critical / Architectural Limitation
**Component:** `DotCompute.Backends.CUDA.RingKernels.CudaRingKernelRuntime`
**Affects:** Ring kernel activation/deactivation on WSL2

## Summary

In WSL2, when a persistent ring kernel is running an infinite dispatch loop, CUDA API calls from the host (including `cudaSetDevice`) block indefinitely. This prevents host-to-device communication for control block updates, making ring kernels unusable on WSL2.

## Root Cause

### The Ring Kernel Pattern
Ring kernels are designed to:
1. Launch once and run forever (persistent kernel)
2. Poll a control block in device memory for `IsActive` flag
3. Process messages from input queue when active
4. Respond via output queue

### WSL2 Behavior
In WSL2 (Windows Subsystem for Linux 2):
- Pinned memory allocation fails (`cudaHostAlloc` returns `InvalidValue`)
- Unified memory has concurrent access issues
- When a kernel is running (even non-cooperative), CUDA API calls like `cudaSetDevice`, `cudaMemcpy` block waiting for the kernel to complete

### Diagnostic Evidence
```
[DIAG] Step 10: Using NON-COOPERATIVE kernel (WSL2 compatibility mode)
[DIAG] Step 10: Kernel launch returned Success
[DIAG] LaunchAsync COMPLETED SUCCESSFULLY!
info: MessagePassingTest[0]
      Step 3: Activating kernel...
[DIAG] ActivateAsync: Using async control block (WSL2 mode)
[DIAG] WriteNonBlocking: Starting (IsActive=1, ShouldTerminate=0)
[DIAG] WriteNonBlocking: Writing to staging buffer at 0x781F9CEDBDC0
[DIAG] WriteNonBlocking: Staging buffer written
[DIAG] WriteNonBlocking: IsStagingPinned=False
[DIAG] WriteNonBlocking: Using SYNC copy via Driver API...
[DIAG] WriteNonBlocking: cuMemcpyHtoD returned: InvalidValue
[DIAG] WriteNonBlocking: Falling back to cudaMemcpy with cudaSetDevice...
<HANGS HERE - cudaSetDevice blocks forever>
```

## Attempted Solutions

### 1. Async Control Block Pattern
Created `AsyncControlBlock` with staging buffer + CUDA events for non-blocking communication.
**Result:** Staging buffer falls back to regular `malloc` (not pinned), still blocks.

### 2. Driver API Instead of Runtime API
Tried `cuMemcpyHtoD` to avoid `cudaSetDevice`.
**Result:** Returns `InvalidValue` because device memory was allocated via Runtime API (`cudaMalloc`).

### 3. Non-Cooperative Kernel Launch
Use regular `cuLaunchKernel` instead of `cuLaunchCooperativeKernel`.
**Result:** Kernel launches successfully but CUDA calls still block.

## Proposed Solutions

### Option A: Finite Kernel Iterations (Recommended for WSL2)

Instead of infinite loop, run kernel for limited iterations:

```cuda
__global__ void ring_kernel_wsl2(ControlBlock* cb) {
    // Process up to MAX_ITERATIONS, then exit
    for (int i = 0; i < MAX_ITERATIONS && !cb->ShouldTerminate; i++) {
        if (cb->IsActive) {
            process_message();
        }
        __threadfence(); // Memory barrier
    }
    // Kernel exits, allowing host to relaunch with updated control block
}
```

Pros:
- Works on WSL2
- Host can update control block between launches

Cons:
- Kernel launch overhead between batches
- Not truly "persistent" kernels

### Option B: CUDA Graph-Based Execution

Use CUDA Graphs to capture and replay kernel execution:

```csharp
// Capture once
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStreamBeginCapture(stream);
launchRingKernel<<<grid, block, 0, stream>>>(controlBlock);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph);

// Replay many times without blocking host
for (int i = 0; i < batches; i++) {
    UpdateControlBlockCPU(controlBlock); // Direct memory if unified
    cudaGraphLaunch(graphExec, stream);
}
```

### Option C: Hybrid Mode Detection

Auto-detect WSL2 and use appropriate mode:

```csharp
if (IsRunningInWsl2())
{
    // Use finite-iteration kernel mode
    await LaunchFiniteKernelAsync(kernelId, iterations: 1000);
}
else
{
    // Use persistent kernel mode
    await LaunchPersistentKernelAsync(kernelId);
}
```

### Option D: Message Queue via Separate Stream

Use a separate CUDA stream for message queue operations that doesn't block on kernel completion:

```csharp
// Launch kernel on stream 0
cuLaunchKernel(kernel, stream0, ...);

// Use stream 1 for message queue DMA (non-blocking)
cudaMemcpyAsync(deviceQueue, hostData, size, stream1);
```

**Note:** This requires CUDA stream concurrency which may be limited on WSL2.

## Impact Assessment

| Platform | Persistent Kernels | Ring Kernel Pattern |
|----------|-------------------|---------------------|
| Native Linux | Works | Works |
| Native Windows | Works | Works |
| WSL2 | Blocks | **BROKEN** |

## DotCompute Analysis: Existing Capabilities and Gaps

### What Exists (Investigation Results)

**1. RingKernelMode.EventDriven** (in `RingKernelEnums.cs`):
```csharp
/// <summary>
/// Event-driven kernel that is launched on-demand.
/// </summary>
/// <remarks>
/// The kernel is launched when messages are available and terminates after
/// processing the current batch. This mode conserves GPU resources but incurs
/// kernel launch overhead for each batch.
/// </remarks>
EventDriven
```
**STATUS: DEFINED BUT NOT IMPLEMENTED** - The stub generator (`CudaRingKernelStubGenerator.cs`) does not check `kernel.Mode` and always generates persistent infinite-loop kernels.

**2. MaxMessagesPerIteration** (in `RingKernelAttribute.cs`):
```csharp
public int MaxMessagesPerIteration { get; set; }
```
**STATUS: IMPLEMENTED BUT INSUFFICIENT** - Limits messages per iteration but kernel still runs infinite `while (control_block->should_terminate == 0)` loop. Doesn't help WSL2 because the kernel never exits.

**3. ProcessingMode (Batch/Continuous/Adaptive)**:
```csharp
public enum RingProcessingMode { Continuous, Batch, Adaptive }
```
**STATUS: FULLY IMPLEMENTED** - Controls how many messages are processed per iteration, but doesn't affect kernel lifetime.

**4. PersistentKernelConfig.MaxIterations** (in `Persistent/Types/PersistentKernelConfig.cs`):
```csharp
public int MaxIterations { get; set; } = 1000;
```
**STATUS: DIFFERENT SYSTEM** - This is for the wave equation solver, not ring kernels.

### What's Missing for WSL2 Support

1. **EventDriven mode implementation** in `CudaRingKernelStubGenerator.cs`:
   - Need to generate finite-iteration loop instead of infinite loop
   - Add iteration counter with configurable max
   - Kernel exits after max iterations or when queue empty

2. **Runtime support for kernel relaunching**:
   - Host monitors kernel completion
   - Updates control block between launches
   - Relaunches kernel automatically

3. **WSL2 auto-detection** in runtime:
   - Already exists: `IsRunningInWsl2()` helper
   - Missing: Mode selection based on platform

## Recommended Action

1. **Short-term:** Document WSL2 as unsupported for ring kernels
2. **Medium-term:** Implement Option A (finite iterations) as WSL2 fallback
3. **Long-term:** Implement Option C (hybrid mode detection) for best experience

### Implementation Priority

**HIGH PRIORITY - EventDriven Mode Implementation:**

The `RingKernelMode.EventDriven` is already defined in the abstractions. Implementation requires:

**File: `CudaRingKernelStubGenerator.cs` (around line 824)**

Current (Persistent mode only):
```csharp
_ = builder.AppendLine("    while (control_block->should_terminate == 0)");
```

Proposed (EventDriven support):
```csharp
if (kernel.Mode == RingKernelMode.EventDriven)
{
    _ = builder.AppendLine("    // Event-driven mode: exit after processing batch");
    _ = builder.AppendLine("    int iterations = 0;");
    _ = builder.AppendLine("    const int MAX_ITERATIONS = 1000;");
    _ = builder.AppendLine("    while (control_block->should_terminate == 0 && iterations < MAX_ITERATIONS)");
    _ = builder.AppendLine("    {");
    _ = builder.AppendLine("        iterations++;");
}
else
{
    _ = builder.AppendLine("    // Persistent mode: run until termination");
    _ = builder.AppendLine("    while (control_block->should_terminate == 0)");
    _ = builder.AppendLine("    {");
}
```

**File: `CudaRingKernelRuntime.cs`**

Add kernel relaunching logic for EventDriven mode:
```csharp
if (kernelInfo.Mode == RingKernelMode.EventDriven && IsRunningInWsl2())
{
    // Use relaunch loop pattern
    while (!cts.IsCancellationRequested)
    {
        await LaunchKernelOnceAsync(kernelState);
        await UpdateControlBlockAsync(kernelState);
        await Task.Delay(10); // Small delay to allow processing
    }
}
```

## Test Environment

- **OS:** WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- **GPU:** NVIDIA RTX (Compute Capability 8.9)
- **CUDA:** 13.x
- **DotCompute:** 0.5.0-alpha

## References

- DotCompute WSL2 Limitations Guide: `/docs/guides/wsl2-cuda-limitations.md`
- NVIDIA CUDA WSL User Guide: https://docs.nvidia.com/cuda/wsl-user-guide/
- Orleans.GpuBridge Ring Kernel Test: `tests/RingKernelValidation/MessagePassingTest.cs`
