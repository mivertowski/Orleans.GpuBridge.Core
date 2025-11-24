# Phase 2: GPU Execution Timing Design

**Date**: 2025-01-18
**Status**: 📐 DESIGN PHASE
**Prerequisites**: DotCompute kernel launch implementation complete
**Target**: Measure true GPU-only latency (100-500ns goal)

---

## Overview

Once DotCompute implements actual kernel launch, we need precise GPU-side timing to validate:

1. **Kernel execution latency** (target: 100-500ns)
2. **Message processing throughput** (target: 2M msgs/s per actor)
3. **Memory transfer overhead** (Host→Device, Device→Host)
4. **End-to-end latency breakdown**

This document designs the timing instrumentation strategy.

---

## Architecture: Multi-Layer Timing Strategy

### Layer 1: CUDA Event Timing (Kernel-Level)

**Purpose**: Measure GPU-only execution time with nanosecond precision

**Implementation**: Native CUDA events in DotCompute runtime

```csharp
// CudaRingKernelRuntime.cs - Add timing to kernel execution
public class GpuKernelTimer
{
    private IntPtr _startEvent;
    private IntPtr _stopEvent;
    private IntPtr _stream;

    public async Task<GpuTimingMetrics> MeasureKernelExecutionAsync(
        Func<Task> kernelOperation,
        CancellationToken cancellationToken)
    {
        // Create CUDA events with precise timing flags
        var createFlags = CudaEventFlags.BlockingSync | CudaEventFlags.Default;
        CudaRuntime.cudaEventCreate(out _startEvent, createFlags);
        CudaRuntime.cudaEventCreate(out _stopEvent, createFlags);

        // Record start event
        CudaRuntime.cudaEventRecord(_startEvent, _stream);

        // Execute kernel operation
        await kernelOperation();

        // Record stop event
        CudaRuntime.cudaEventRecord(_stopEvent, _stream);

        // Wait for completion
        CudaRuntime.cudaEventSynchronize(_stopEvent);

        // Calculate elapsed time (milliseconds with microsecond precision)
        float elapsedMs = 0;
        CudaRuntime.cudaEventElapsedTime(out elapsedMs, _startEvent, _stopEvent);

        return new GpuTimingMetrics
        {
            GpuLatencyNs = (long)(elapsedMs * 1_000_000),  // Convert ms → ns
            GpuLatencyUs = elapsedMs * 1000,                 // Convert ms → μs
            GpuLatencyMs = elapsedMs,
            MeasuredAt = DateTime.UtcNow
        };
    }
}
```

**Key Features**:
- `cudaEventCreate`: Creates timing events
- `cudaEventRecord`: Marks GPU timeline points
- `cudaEventElapsedTime`: Calculates GPU-only time (no host overhead)
- Nanosecond precision: CUDA events have ~500ns resolution

### Layer 2: Telemetry Buffer Integration

**Purpose**: Real-time GPU metrics without CPU round-trips

DotCompute already has `CudaTelemetryBuffer` infrastructure. We'll enhance it:

```csharp
// Enhanced telemetry structure
public struct RingKernelTelemetryV2
{
    // Existing fields
    public long MessagesProcessed { get; set; }
    public long MessagesDropped { get; set; }
    public long InputQueueDepth { get; set; }
    public long OutputQueueDepth { get; set; }

    // NEW: GPU timing fields (written by kernel)
    public long LastMessageLatencyNs { get; set; }     // Last message processing time
    public long MinMessageLatencyNs { get; set; }       // Minimum latency observed
    public long MaxMessageLatencyNs { get; set; }       // Maximum latency observed
    public long AvgMessageLatencyNs { get; set; }       // Rolling average (last 1000)

    // NEW: GPU timestamp fields
    public long KernelStartCycles { get; set; }         // GPU clock at kernel start
    public long LastMessageCycles { get; set; }         // GPU clock at last message

    // NEW: Performance counters
    public long CacheHits { get; set; }                 // L1/L2 cache hits
    public long CacheMisses { get; set; }               // Cache misses
    public long AtomicContentions { get; set; }         // Lock-free queue contentions
}
```

**GPU-Side Updates**:

```cuda
// In VectorAddKernel.cu - Add timing to message processing loop
__global__ void VectorAddActor_kernel(
    MessageQueue<OrleansGpuMessage>* input_queue,
    MessageQueue<OrleansGpuMessage>* output_queue,
    KernelControl* control,
    RingKernelTelemetryV2* telemetry,  // NEW: Telemetry pointer
    float* workspace,
    int workspace_size)
{
    while (!control->terminate.load()) {
        OrleansGpuMessage msg;

        if (input_queue->try_dequeue(msg)) {
            // Record start time (GPU clock cycles)
            long long start_cycles = clock64();

            // Process message
            VectorAddRequest* req = (VectorAddRequest*)msg.payload;
            // ... vector addition logic ...

            // Record end time
            long long end_cycles = clock64();
            long long latency_cycles = end_cycles - start_cycles;

            // Convert cycles to nanoseconds
            // RTX 4090: 2.52 GHz base clock = 0.397 ns/cycle
            long long latency_ns = latency_cycles * 397 / 1000;  // Approximate conversion

            // Update telemetry (atomic operations)
            atomicAdd(&telemetry->MessagesProcessed, 1);
            atomicMax(&telemetry->MaxMessageLatencyNs, latency_ns);
            atomicMin(&telemetry->MinMessageLatencyNs, latency_ns);

            // Store last message latency
            telemetry->LastMessageLatencyNs = latency_ns;
            telemetry->LastMessageCycles = end_cycles;

            // Enqueue response...
        }
    }
}
```

**Host-Side Polling**:

```csharp
// Poll telemetry from pinned host memory (zero-copy)
public async Task<GpuPerformanceSnapshot> GetRealtimeMetricsAsync(
    string kernelId,
    CancellationToken cancellationToken)
{
    var telemetry = await _runtime.GetTelemetryAsync(kernelId, cancellationToken);

    return new GpuPerformanceSnapshot
    {
        MessagesProcessed = telemetry.MessagesProcessed,
        LastMessageLatencyNs = telemetry.LastMessageLatencyNs,
        MinLatencyNs = telemetry.MinMessageLatencyNs,
        MaxLatencyNs = telemetry.MaxMessageLatencyNs,
        AvgLatencyNs = telemetry.AvgMessageLatencyNs,
        Throughput = CalculateThroughput(telemetry),
        Timestamp = DateTime.UtcNow
    };
}
```

### Layer 3: MessagePassingTest Integration

**Purpose**: Automated latency measurement in tests

**Enhanced Test Structure**:

```csharp
public static async Task<TestResults> RunWithGpuTimingAsync(
    ILoggerFactory loggerFactory,
    string backend = "CUDA")
{
    var logger = loggerFactory.CreateLogger("MessagePassingTest");
    var gpuTimer = new GpuKernelTimer();  // NEW
    var latencyMetrics = new List<LatencyBreakdown>();  // NEW

    // ... kernel launch and activation ...

    foreach (var (name, size, a, b) in testCases)
    {
        logger.LogInformation($"Test: {name}");

        // Create request
        var request = new VectorAddRequestMessage { /* ... */ };

        // LAYER 1: Host timestamp
        var hostSendStart = DateTime.UtcNow;

        // LAYER 2: GPU event timing (measure queue transfer)
        var transferMetrics = await gpuTimer.MeasureKernelExecutionAsync(async () =>
        {
            await runtime.SendToNamedQueueAsync(inputQueueName, request, CancellationToken.None);
        }, CancellationToken.None);

        var hostSendEnd = DateTime.UtcNow;

        // Wait for response
        var hostReceiveStart = DateTime.UtcNow;
        var responseMsg = await WaitForResponseAsync(runtime, outputQueueName, timeout);
        var hostReceiveEnd = DateTime.UtcNow;

        // LAYER 3: GPU telemetry (kernel execution time)
        var telemetry = await runtime.GetTelemetryAsync("VectorAddProcessor", CancellationToken.None);

        // LAYER 4: Calculate breakdown
        var breakdown = new LatencyBreakdown
        {
            // Host measurements
            HostSendLatencyUs = (hostSendEnd - hostSendStart).TotalMicroseconds,
            HostReceiveLatencyUs = (hostReceiveEnd - hostReceiveStart).TotalMicroseconds,
            HostTotalLatencyUs = (hostReceiveEnd - hostSendStart).TotalMicroseconds,

            // GPU measurements
            GpuTransferLatencyNs = transferMetrics.GpuLatencyNs,
            GpuKernelLatencyNs = telemetry.LastMessageLatencyNs,

            // Calculated overhead
            BridgeOverheadUs = CalculateBridgeOverhead(hostSendLatencyUs, transferMetrics),
            SerializationOverheadUs = CalculateSerializationOverhead(),

            // Validation
            ComputationCorrect = ValidateResult(responseMsg, expected),
            TestCase = name,
            VectorSize = size
        };

        latencyMetrics.Add(breakdown);

        // Log detailed breakdown
        logger.LogInformation($"  Latency Breakdown:");
        logger.LogInformation($"    Host total: {breakdown.HostTotalLatencyUs:F2}μs");
        logger.LogInformation($"    GPU transfer: {breakdown.GpuTransferLatencyNs:F0}ns");
        logger.LogInformation($"    GPU kernel: {breakdown.GpuKernelLatencyNs:F0}ns ← CRITICAL METRIC");
        logger.LogInformation($"    Bridge overhead: {breakdown.BridgeOverheadUs:F2}μs");
    }

    return new TestResults
    {
        AllPassed = latencyMetrics.All(m => m.ComputationCorrect),
        Metrics = latencyMetrics,
        Summary = GenerateSummary(latencyMetrics)
    };
}
```

---

## Implementation Plan

### Step 1: Create GPU Timer Infrastructure

**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Telemetry/GpuKernelTimer.cs`

```csharp
namespace Orleans.GpuBridge.Backends.DotCompute.Telemetry;

/// <summary>
/// CUDA event-based GPU kernel timing with nanosecond precision.
/// </summary>
public sealed class GpuKernelTimer : IAsyncDisposable
{
    private readonly ILogger<GpuKernelTimer> _logger;
    private IntPtr _context;
    private IntPtr _stream;
    private IntPtr _startEvent;
    private IntPtr _stopEvent;
    private bool _initialized;
    private bool _disposed;

    public GpuKernelTimer(IntPtr context, IntPtr stream, ILogger<GpuKernelTimer> logger)
    {
        _context = context;
        _stream = stream;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Initializes CUDA events for timing.
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        await Task.Run(() =>
        {
            // Set CUDA context
            var ctxResult = CudaRuntimeCore.cuCtxSetCurrent(_context);
            if (ctxResult != CudaError.Success)
            {
                throw new InvalidOperationException($"Failed to set CUDA context: {ctxResult}");
            }

            // Create timing events with blocking sync
            var flags = 0x01;  // cudaEventBlockingSync
            var createStartResult = CudaRuntime.cudaEventCreate(out _startEvent, flags);
            if (createStartResult != CudaError.Success)
            {
                throw new InvalidOperationException($"Failed to create start event: {createStartResult}");
            }

            var createStopResult = CudaRuntime.cudaEventCreate(out _stopEvent, flags);
            if (createStopResult != CudaError.Success)
            {
                CudaRuntime.cudaEventDestroy(_startEvent);
                throw new InvalidOperationException($"Failed to create stop event: {createStopResult}");
            }

            _initialized = true;
            _logger.LogDebug("GPU kernel timer initialized (stream={Stream:X16})", _stream.ToInt64());
        }, cancellationToken);
    }

    /// <summary>
    /// Measures GPU-only execution time for a kernel operation.
    /// </summary>
    /// <param name="kernelOperation">Async operation to time.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>GPU timing metrics with nanosecond precision.</returns>
    public async Task<GpuTimingMetrics> MeasureAsync(
        Func<Task> kernelOperation,
        CancellationToken cancellationToken = default)
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Timer not initialized. Call InitializeAsync() first.");
        }

        ObjectDisposedException.ThrowIf(_disposed, this);

        return await Task.Run(async () =>
        {
            // Set CUDA context
            CudaRuntimeCore.cuCtxSetCurrent(_context);

            // Record start event on stream
            var recordStartResult = CudaRuntime.cudaEventRecord(_startEvent, _stream);
            if (recordStartResult != CudaError.Success)
            {
                throw new InvalidOperationException($"Failed to record start event: {recordStartResult}");
            }

            // Execute kernel operation
            await kernelOperation();

            // Record stop event on stream
            var recordStopResult = CudaRuntime.cudaEventRecord(_stopEvent, _stream);
            if (recordStopResult != CudaError.Success)
            {
                throw new InvalidOperationException($"Failed to record stop event: {recordStopResult}");
            }

            // Wait for stop event (blocking)
            var syncResult = CudaRuntime.cudaEventSynchronize(_stopEvent);
            if (syncResult != CudaError.Success)
            {
                throw new InvalidOperationException($"Failed to synchronize stop event: {syncResult}");
            }

            // Calculate elapsed time
            float elapsedMs = 0;
            var elapsedResult = CudaRuntime.cudaEventElapsedTime(out elapsedMs, _startEvent, _stopEvent);
            if (elapsedResult != CudaError.Success)
            {
                throw new InvalidOperationException($"Failed to get elapsed time: {elapsedResult}");
            }

            return new GpuTimingMetrics
            {
                GpuLatencyNs = (long)(elapsedMs * 1_000_000),    // ms → ns
                GpuLatencyUs = elapsedMs * 1000,                  // ms → μs
                GpuLatencyMs = elapsedMs,
                MeasuredAt = DateTime.UtcNow
            };
        }, cancellationToken);
    }

    /// <summary>
    /// Measures multiple iterations and returns statistical summary.
    /// </summary>
    public async Task<TimingStatistics> MeasureIterationsAsync(
        Func<Task> kernelOperation,
        int iterations,
        CancellationToken cancellationToken = default)
    {
        var measurements = new List<long>(iterations);

        for (int i = 0; i < iterations; i++)
        {
            var metrics = await MeasureAsync(kernelOperation, cancellationToken);
            measurements.Add(metrics.GpuLatencyNs);
        }

        return new TimingStatistics
        {
            MinLatencyNs = measurements.Min(),
            MaxLatencyNs = measurements.Max(),
            AvgLatencyNs = (long)measurements.Average(),
            MedianLatencyNs = CalculateMedian(measurements),
            P95LatencyNs = CalculatePercentile(measurements, 0.95),
            P99LatencyNs = CalculatePercentile(measurements, 0.99),
            StdDevNs = CalculateStdDev(measurements),
            Iterations = iterations
        };
    }

    private static long CalculateMedian(List<long> values)
    {
        var sorted = values.OrderBy(x => x).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
    }

    private static long CalculatePercentile(List<long> values, double percentile)
    {
        var sorted = values.OrderBy(x => x).ToList();
        int index = (int)Math.Ceiling(percentile * sorted.Count) - 1;
        return sorted[Math.Max(0, Math.Min(index, sorted.Count - 1))];
    }

    private static double CalculateStdDev(List<long> values)
    {
        double avg = values.Average();
        double sumSquares = values.Sum(v => Math.Pow(v - avg, 2));
        return Math.Sqrt(sumSquares / values.Count);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;

        if (_initialized)
        {
            await Task.Run(() =>
            {
                CudaRuntimeCore.cuCtxSetCurrent(_context);

                if (_startEvent != IntPtr.Zero)
                {
                    CudaRuntime.cudaEventDestroy(_startEvent);
                }

                if (_stopEvent != IntPtr.Zero)
                {
                    CudaRuntime.cudaEventDestroy(_stopEvent);
                }

                _logger.LogDebug("GPU kernel timer disposed");
            });
        }

        _disposed = true;
    }
}

/// <summary>
/// GPU timing metrics with nanosecond precision.
/// </summary>
public record GpuTimingMetrics
{
    public long GpuLatencyNs { get; init; }
    public double GpuLatencyUs { get; init; }
    public double GpuLatencyMs { get; init; }
    public DateTime MeasuredAt { get; init; }
}

/// <summary>
/// Statistical summary of timing measurements.
/// </summary>
public record TimingStatistics
{
    public long MinLatencyNs { get; init; }
    public long MaxLatencyNs { get; init; }
    public long AvgLatencyNs { get; init; }
    public long MedianLatencyNs { get; init; }
    public long P95LatencyNs { get; init; }
    public long P99LatencyNs { get; init; }
    public double StdDevNs { get; init; }
    public int Iterations { get; init; }
}
```

### Step 2: Integrate with CudaRingKernelRuntime

**File**: `DotCompute.Backends.CUDA/RingKernels/CudaRingKernelRuntime.cs`

Add timer to `KernelState`:

```csharp
private sealed class KernelState
{
    // ... existing fields ...

    // NEW: GPU timing infrastructure
    public GpuKernelTimer? Timer { get; set; }
    public ConcurrentQueue<GpuTimingMetrics> RecentTimings { get; set; } = new();
}
```

### Step 3: Enhanced MessagePassingTest

**File**: `tests/RingKernelValidation/MessagePassingTest.cs`

Add GPU timing measurements as shown in Layer 3 design above.

### Step 4: Create Nsight Compute Integration

**File**: `tests/RingKernelValidation/scripts/profile-gpu-kernel.sh`

```bash
#!/bin/bash

# Nsight Compute profiling script for VectorAddProcessor kernel
# Measures kernel execution time, SM efficiency, memory bandwidth

NCU_PATH="/usr/local/cuda/bin/ncu"

# Profile with comprehensive metrics
$NCU_PATH \
  --target-processes all \
  --kernel-name "VectorAddActor_kernel" \
  --launch-skip 0 \
  --launch-count 10 \
  --metrics \
    sm__cycles_elapsed.avg,\
    sm__cycles_elapsed.sum,\
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    smsp__average_warps_active.avg,\
    gpu__time_duration.sum \
  --csv \
  --log-file nsight-metrics.csv \
  dotnet test --filter "FullyQualifiedName~MessagePassingTest" \
  --logger "console;verbosity=detailed"

# Parse results
echo "=== Nsight Compute Results ==="
cat nsight-metrics.csv | grep "VectorAddActor_kernel"
```

---

## Expected Results

### Success Criteria

Once kernel launch is implemented, we expect:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Kernel Execution** | 100-500ns | GPU telemetry (clock64) |
| **End-to-End Latency** | <10μs | Host timestamps |
| **Message Throughput** | 2M msgs/s | Messages processed / uptime |
| **Bridge Overhead** | <5μs | Host latency - GPU latency |

### Validation Checkpoints

✅ **Checkpoint 1**: CUDA events show kernel executing
```
GpuKernelTimer: Measured kernel launch in 150ns
```

✅ **Checkpoint 2**: Telemetry shows message processing
```
Telemetry: LastMessageLatencyNs=350
Telemetry: MessagesProcessed=1000
```

✅ **Checkpoint 3**: Nsight Compute shows kernel in trace
```
VectorAddActor_kernel: 42 SMs, 250ns duration, 128 bytes DRAM
```

✅ **Checkpoint 4**: Sub-microsecond latency achieved
```
Test Result: GPU kernel latency = 425ns ✅ (target: 100-500ns)
```

---

## Usage Example

Once implemented:

```bash
# Run tests with GPU timing
cd tests/RingKernelValidation
dotnet test --filter "MessagePassingTest" --logger "console;verbosity=detailed"

# Expected output:
# === Message Passing Validation Test (CUDA) ===
#
# Test: Small Vector (10 elements, inline)
#   Latency Breakdown:
#     Host total: 8.3μs
#     GPU transfer: 450ns
#     GPU kernel: 350ns ← VALIDATED! ✅
#     Bridge overhead: 4.2μs
#   ✓ PASSED - GPU execution verified
#
# === GPU VALIDATION SUCCESS ===
# Target latency: 100-500ns
# Measured latency: 350ns ✅
# GPU-native actor paradigm PROVEN! 🚀
```

---

## Next Steps

**After DotCompute implements kernel launch**:

1. **Implement GpuKernelTimer** (~2-3 hours)
   - Create timer class
   - Add CUDA event P/Invoke bindings
   - Write unit tests

2. **Integrate with MessagePassingTest** (~1-2 hours)
   - Add timing measurements
   - Create latency breakdown reporting
   - Update logging

3. **Run validation** (~30 minutes)
   - Execute tests with timing
   - Verify sub-microsecond latency
   - Generate validation report

4. **Nsight Compute profiling** (~1 hour)
   - Create profiling scripts
   - Analyze SM utilization
   - Measure memory bandwidth

**Total Estimated Time**: 1 day

---

## Files to Create

1. `src/Orleans.GpuBridge.Backends.DotCompute/Telemetry/GpuKernelTimer.cs` - Timer implementation
2. `src/Orleans.GpuBridge.Backends.DotCompute/Telemetry/GpuTimingMetrics.cs` - Metrics DTOs
3. `tests/RingKernelValidation/MessagePassingTestV2.cs` - Enhanced test with timing
4. `tests/RingKernelValidation/scripts/profile-gpu-kernel.sh` - Nsight Compute script
5. `tests/RingKernelValidation/scripts/analyze-latency.py` - Latency analysis tool

---

**Design Status**: ✅ READY FOR IMPLEMENTATION
**Blocked By**: DotCompute kernel launch (Phase 1.5)
**Estimated Validation Time**: 1 day after unblocked
