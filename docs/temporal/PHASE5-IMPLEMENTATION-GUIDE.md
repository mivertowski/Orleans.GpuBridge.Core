# Phase 5 Implementation Guide: GPU Timing Extensions
## DotCompute 0.4.2-rc2 Integration

## Overview

This guide provides step-by-step instructions for implementing Phase 5 temporal correctness features using **DotCompute 0.4.2-rc2**, which includes all required timing, barrier, and memory ordering APIs.

**Prerequisites**:
- ✅ DotCompute 0.4.2-rc2 (now available!)
- ✅ Phase 1-4 implementation complete
- ✅ Orleans.GpuBridge.Core base infrastructure

---

## Table of Contents

1. [DotCompute 0.4.2-rc2 Features](#dotcompute-042-rc2-features)
2. [Timing API Integration](#timing-api-integration)
3. [Ring Kernel Implementation](#ring-kernel-implementation)
4. [Barrier Synchronization](#barrier-synchronization)
5. [Memory Ordering for Causal Correctness](#memory-ordering-for-causal-correctness)
6. [Complete Integration Example](#complete-integration-example)
7. [Testing and Validation](#testing-and-validation)
8. [Performance Tuning](#performance-tuning)

---

## DotCompute 0.4.2-rc2 Features

### Available APIs

DotCompute 0.4.2-rc2 provides three main feature sets through attribute-based configuration:

#### 1. **Timing API** - GPU Nanosecond Timestamps
```csharp
[Kernel(EnableTimestamps = true)]
public static void TimedKernel(Span<long> timestamps, Span<float> data)
{
    // timestamps[workItemId] automatically contains GPU entry time
    // Resolution: 1ns (CUDA), 1μs (OpenCL), 100ns (CPU)
}
```

**Key Features**:
- Automatic timestamp injection at kernel entry
- Sub-microsecond precision on CUDA (%%globaltimer register)
- Clock calibration for GPU-CPU time synchronization
- Batch timestamp queries for efficiency

#### 2. **Ring Kernels** - Persistent GPU Threads
```csharp
[RingKernel(MessageQueueSize = 1024, ProcessingMode = RingProcessingMode.Continuous)]
public static void MessageProcessorRing(
    Span<Message> messageQueue,
    Span<int> queueHead,
    Span<int> queueTail)
{
    // Infinite dispatch loop - kernel runs forever
    // Processes messages as they arrive
    // Zero kernel launch overhead after initial dispatch
}
```

**Key Features**:
- Persistent GPU threads (launched once, run forever)
- Lock-free message queues in GPU memory
- Sub-microsecond message latency (100-500ns)
- Automatic queue management

#### 3. **Barriers & Memory Ordering** - Causal Correctness
```csharp
[Kernel(
    EnableBarriers = true,
    BarrierScope = BarrierScope.Device,
    MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void CausalKernel(Span<Message> messages)
{
    // Device-wide barriers for synchronization
    // Acquire-release semantics for causal ordering
    // Automatic fence insertion
}
```

**Key Features**:
- Device-wide barriers (CUDA Cooperative Groups)
- Thread block and system-wide barriers
- Acquire-release memory semantics
- Configurable consistency models (Relaxed, ReleaseAcquire, Sequential)

---

## Timing API Integration

### Step 1: Enable Timestamps in Kernels

Create temporal kernels with automatic timestamp injection:

**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/TemporalKernels.cs`

```csharp
using DotCompute.Abstractions;
using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// GPU kernels with temporal timestamp support.
/// </summary>
public static class TemporalKernels
{
    /// <summary>
    /// Actor message processing kernel with GPU-side timestamp injection.
    /// </summary>
    /// <remarks>
    /// DotCompute automatically injects GPU timestamp at kernel entry.
    /// Timestamp resolution: 1ns (CUDA), 1μs (OpenCL), 100ns (CPU).
    /// </remarks>
    [Kernel(EnableTimestamps = true)]
    public static void ProcessActorMessageWithTimestamp(
        Span<long> timestamps,          // Auto-injected: GPU entry time (ns)
        Span<ActorMessage> messages,    // Input: messages to process
        Span<ActorState> states,        // In/Out: actor states
        Span<long> hlcPhysical,         // In/Out: HLC physical time
        Span<long> hlcLogical)          // In/Out: HLC logical counter
    {
        int tid = GetGlobalId(0);

        // Read GPU timestamp (already recorded by DotCompute)
        long gpuTimestamp = timestamps[tid];

        // Update HLC with GPU timestamp
        var message = messages[tid];
        var currentHlc = new HybridTimestamp(hlcPhysical[tid], hlcLogical[tid]);
        var updatedHlc = UpdateHLC(currentHlc, message.Timestamp, gpuTimestamp);

        hlcPhysical[tid] = updatedHlc.PhysicalTime;
        hlcLogical[tid] = updatedHlc.LogicalCounter;

        // Process message with temporal ordering
        ProcessMessage(ref states[tid], message, updatedHlc);
    }

    /// <summary>
    /// HLC update logic (matches Phase 1 implementation).
    /// </summary>
    private static HybridTimestamp UpdateHLC(
        HybridTimestamp local,
        HybridTimestamp received,
        long physicalTime)
    {
        long newPhysical = Math.Max(Math.Max(local.PhysicalTime, received.PhysicalTime), physicalTime);
        long newLogical = 0;

        if (newPhysical == local.PhysicalTime && newPhysical == received.PhysicalTime)
        {
            newLogical = Math.Max(local.LogicalCounter, received.LogicalCounter) + 1;
        }
        else if (newPhysical == local.PhysicalTime)
        {
            newLogical = local.LogicalCounter + 1;
        }
        else if (newPhysical == received.PhysicalTime)
        {
            newLogical = received.LogicalCounter + 1;
        }

        return new HybridTimestamp(newPhysical, newLogical);
    }

    /// <summary>
    /// GPU-side message processing.
    /// </summary>
    private static void ProcessMessage(
        ref ActorState state,
        ActorMessage message,
        HybridTimestamp timestamp)
    {
        // Update actor state with timestamped message
        state.LastProcessedTimestamp = timestamp;
        state.MessageCount++;

        // Apply message payload to state
        switch (message.Type)
        {
            case MessageType.StateUpdate:
                state.Data = message.Payload;
                break;
            case MessageType.Query:
                // Query processing
                break;
        }
    }
}
```

### Step 2: Create Clock Calibration Service

Synchronize GPU and CPU clocks for accurate temporal alignment:

**File**: `src/Orleans.GpuBridge.Runtime/Temporal/GpuClockCalibrator.cs`

```csharp
using DotCompute.Abstractions;
using DotCompute.Timing;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Calibrates GPU clock against CPU time for temporal correctness.
/// </summary>
public sealed class GpuClockCalibrator
{
    private readonly ITimingProvider _timingProvider;
    private readonly ILogger<GpuClockCalibrator> _logger;
    private ClockCalibration? _currentCalibration;

    public GpuClockCalibrator(
        ITimingProvider timingProvider,
        ILogger<GpuClockCalibrator> logger)
    {
        _timingProvider = timingProvider ?? throw new ArgumentNullException(nameof(timingProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Gets the current clock calibration (performs calibration if not cached).
    /// </summary>
    public async Task<ClockCalibration> GetCalibrationAsync(CancellationToken ct = default)
    {
        if (_currentCalibration == null || IsCalibrationStale(_currentCalibration.Value))
        {
            _currentCalibration = await CalibrateAsync(sampleCount: 1000, ct);
        }

        return _currentCalibration.Value;
    }

    /// <summary>
    /// Performs GPU-CPU clock calibration with specified sample count.
    /// </summary>
    public async Task<ClockCalibration> CalibrateAsync(
        int sampleCount = 100,
        CancellationToken ct = default)
    {
        _logger.LogInformation("Starting GPU clock calibration with {SampleCount} samples...", sampleCount);

        var calibration = await _timingProvider.CalibrateAsync(sampleCount, ct);

        _logger.LogInformation(
            "GPU clock calibration complete: Offset={OffsetNs}ns, Drift={DriftPPM}ppm, Error=±{ErrorBoundNs}ns",
            calibration.OffsetNanos,
            calibration.DriftPPM,
            calibration.ErrorBoundNanos);

        _currentCalibration = calibration;
        return calibration;
    }

    /// <summary>
    /// Converts GPU timestamp to CPU time using current calibration.
    /// </summary>
    public long GpuToCpuTime(long gpuTimeNanos)
    {
        if (_currentCalibration == null)
        {
            throw new InvalidOperationException("Clock not calibrated. Call GetCalibrationAsync() first.");
        }

        return _currentCalibration.Value.GpuToCpuTime(gpuTimeNanos);
    }

    /// <summary>
    /// Converts CPU timestamp to GPU time using current calibration.
    /// </summary>
    public long CpuToGpuTime(long cpuTimeNanos)
    {
        if (_currentCalibration == null)
        {
            throw new InvalidOperationException("Clock not calibrated. Call GetCalibrationAsync() first.");
        }

        // Reverse conversion: GPU = CPU + offset + drift correction
        long elapsedSinceCalibration = cpuTimeNanos - _currentCalibration.Value.CalibrationTimestampNanos;
        long driftCorrection = (long)(elapsedSinceCalibration * (_currentCalibration.Value.DriftPPM / 1_000_000.0));

        return cpuTimeNanos + _currentCalibration.Value.OffsetNanos + driftCorrection;
    }

    /// <summary>
    /// Checks if calibration needs refresh (every 5 minutes or if drift > 1000 PPM).
    /// </summary>
    private static bool IsCalibrationStale(ClockCalibration calibration)
    {
        const long FiveMinutesNanos = 5L * 60 * 1_000_000_000;
        const double MaxDriftPPM = 1000.0;

        long now = DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
        long elapsed = now - calibration.CalibrationTimestampNanos;

        return elapsed > FiveMinutesNanos || Math.Abs(calibration.DriftPPM) > MaxDriftPPM;
    }
}
```

### Step 3: Register Timing Services

Add timing services to DI container:

**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/ServiceCollectionExtensions.cs`

```csharp
using DotCompute.Timing;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Runtime.Temporal;
using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds GPU timing services for temporal correctness.
    /// </summary>
    public static IServiceCollection AddGpuTiming(
        this IServiceCollection services,
        Action<GpuTimingOptions>? configure = null)
    {
        var options = new GpuTimingOptions();
        configure?.Invoke(options);

        // Register DotCompute timing provider
        services.AddSingleton<ITimingProvider>(sp =>
        {
            var deviceManager = sp.GetRequiredService<IDeviceManager>();
            var device = deviceManager.GetDevice(options.DeviceIndex);
            return deviceManager.GetTimingProvider(device);
        });

        // Register clock calibrator
        services.AddSingleton<GpuClockCalibrator>();

        // Enable timestamp injection globally if requested
        if (options.EnableTimestampInjection)
        {
            services.AddSingleton(sp =>
            {
                var timingProvider = sp.GetRequiredService<ITimingProvider>();
                timingProvider.EnableTimestampInjection(true);
                return timingProvider;
            });
        }

        return services;
    }
}

public sealed class GpuTimingOptions
{
    public int DeviceIndex { get; set; } = 0;
    public bool EnableTimestampInjection { get; set; } = true;
    public bool AutoCalibrate { get; set; } = true;
    public int CalibrationSampleCount { get; set; } = 1000;
}
```

---

## Ring Kernel Implementation

Ring kernels enable persistent GPU threads for **zero-latency message processing**.

### Step 1: Define Ring Kernel for Actor Message Processing

**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/ActorRingKernels.cs`

```csharp
using DotCompute.Abstractions;
using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

/// <summary>
/// Persistent ring kernels for GPU-resident actor message processing.
/// </summary>
public static class ActorRingKernels
{
    /// <summary>
    /// Persistent ring kernel for processing actor messages on GPU.
    /// </summary>
    /// <remarks>
    /// This kernel runs in an infinite loop, processing messages as they arrive.
    /// Launch once and it runs forever until explicitly stopped.
    ///
    /// Performance: 100-500ns message latency (vs 10-50μs with kernel re-launch).
    /// </remarks>
    [RingKernel(
        MessageQueueSize = 4096,
        ProcessingMode = RingProcessingMode.Continuous,
        EnableTimestamps = true)]
    public static void ActorMessageProcessorRing(
        Span<long> timestamps,              // Auto-injected timestamps
        Span<ActorMessage> messageQueue,    // Ring buffer of messages
        Span<int> queueHead,                // Producer index
        Span<int> queueTail,                // Consumer index
        Span<ActorState> actorStates,       // Actor state (GPU-resident)
        Span<long> hlcPhysical,             // HLC physical time per actor
        Span<long> hlcLogical,              // HLC logical counter per actor
        Span<bool> stopSignal)              // Stop flag for graceful shutdown
    {
        int actorId = GetGlobalId(0);

        // Infinite dispatch loop - only exits when stopSignal is true
        while (!stopSignal[0])
        {
            // Check if message available (lock-free queue)
            int head = AtomicLoad(ref queueHead[0]);
            int tail = AtomicLoad(ref queueTail[actorId]);

            if (head != tail)
            {
                // Dequeue message
                int messageIndex = tail % messageQueue.Length;
                var message = messageQueue[messageIndex];

                // Update HLC with message timestamp
                long gpuTime = timestamps[actorId];
                var localHlc = new HybridTimestamp(hlcPhysical[actorId], hlcLogical[actorId]);
                var updatedHlc = UpdateHLC(localHlc, message.Timestamp, gpuTime);

                hlcPhysical[actorId] = updatedHlc.PhysicalTime;
                hlcLogical[actorId] = updatedHlc.LogicalCounter;

                // Process message
                ProcessActorMessage(ref actorStates[actorId], message, updatedHlc);

                // Advance tail (release message slot)
                AtomicStore(ref queueTail[actorId], tail + 1);
            }
            else
            {
                // No messages - yield briefly to reduce GPU power
                // (DotCompute implements this as a lightweight pause)
                Yield();
            }
        }
    }

    /// <summary>
    /// HLC update with causal ordering.
    /// </summary>
    private static HybridTimestamp UpdateHLC(
        HybridTimestamp local,
        HybridTimestamp received,
        long physicalTime)
    {
        // Same as ProcessActorMessageWithTimestamp
        // (Implementation omitted for brevity)
    }

    /// <summary>
    /// Process actor message on GPU.
    /// </summary>
    private static void ProcessActorMessage(
        ref ActorState state,
        ActorMessage message,
        HybridTimestamp timestamp)
    {
        // Same as ProcessActorMessageWithTimestamp
        // (Implementation omitted for brevity)
    }
}
```

### Step 2: Ring Kernel Lifecycle Management

**File**: `src/Orleans.GpuBridge.Runtime/Temporal/RingKernelManager.cs`

```csharp
using DotCompute.Abstractions;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Manages lifecycle of persistent ring kernels for actor message processing.
/// </summary>
public sealed class RingKernelManager : IDisposable
{
    private readonly IKernelExecutor _executor;
    private readonly ILogger<RingKernelManager> _logger;
    private readonly CancellationTokenSource _stopSignal = new();
    private Task? _ringKernelTask;

    public RingKernelManager(
        IKernelExecutor executor,
        ILogger<RingKernelManager> logger)
    {
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Launches the ring kernel (starts infinite dispatch loop).
    /// </summary>
    public Task StartAsync(
        int actorCount,
        int messageQueueSize = 4096,
        CancellationToken ct = default)
    {
        if (_ringKernelTask != null)
        {
            throw new InvalidOperationException("Ring kernel already started.");
        }

        _logger.LogInformation("Launching ring kernel for {ActorCount} actors...", actorCount);

        // Allocate GPU memory for ring buffer
        var messageQueue = new ActorMessage[messageQueueSize];
        var actorStates = new ActorState[actorCount];
        var timestamps = new long[actorCount];
        var hlcPhysical = new long[actorCount];
        var hlcLogical = new long[actorCount];
        var queueHead = new[] { 0 };
        var queueTail = new int[actorCount];
        var stopSignal = new[] { false };

        // Launch ring kernel (infinite loop on GPU)
        _ringKernelTask = _executor.ExecuteAsync(
            "ActorMessageProcessorRing",
            new object[]
            {
                timestamps,
                messageQueue,
                queueHead,
                queueTail,
                actorStates,
                hlcPhysical,
                hlcLogical,
                stopSignal
            },
            new LaunchConfiguration { GlobalSize = actorCount },
            ct);

        _logger.LogInformation("Ring kernel launched successfully.");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Gracefully stops the ring kernel.
    /// </summary>
    public async Task StopAsync(CancellationToken ct = default)
    {
        if (_ringKernelTask == null)
        {
            return;
        }

        _logger.LogInformation("Stopping ring kernel...");

        // Set stop signal (kernel will exit loop)
        _stopSignal.Cancel();

        // Wait for kernel to finish
        await _ringKernelTask;

        _logger.LogInformation("Ring kernel stopped.");
    }

    public void Dispose()
    {
        _stopSignal.Dispose();
    }
}
```

---

## Barrier Synchronization

Device-wide barriers enable multi-actor coordination on GPU.

### Example: Temporal Pattern Detection with Barriers

**File**: `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/PatternDetectionKernels.cs`

```csharp
using DotCompute.Abstractions;
using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Temporal;

public static class PatternDetectionKernels
{
    /// <summary>
    /// Detects temporal patterns across multiple actors with device-wide synchronization.
    /// </summary>
    [Kernel(
        EnableBarriers = true,
        BarrierScope = BarrierScope.Device,
        EnableTimestamps = true)]
    public static void DetectTemporalPattern(
        Span<long> timestamps,
        Span<TemporalEvent> events,
        Span<bool> patternDetected,
        long windowSizeNanos)
    {
        int tid = GetGlobalId(0);

        // Step 1: Each thread checks local events
        bool localMatch = CheckLocalPattern(events[tid], windowSizeNanos);

        // BARRIER: Wait for all threads to complete local checks
        DeviceBarrier();

        // Step 2: Global pattern analysis (requires all local results)
        if (tid == 0)
        {
            // Aggregate results from all threads
            bool globalPattern = AnalyzeGlobalPattern(events, patternDetected);
            patternDetected[0] = globalPattern;
        }

        // BARRIER: Wait for global analysis
        DeviceBarrier();

        // Step 3: All threads can now read the result
        if (patternDetected[0])
        {
            HandlePatternDetected(ref events[tid]);
        }
    }
}
```

---

## Memory Ordering for Causal Correctness

Acquire-release semantics ensure causal ordering of actor messages.

### Example: Causal Message Send/Receive

```csharp
[Kernel(MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void SendMessage(
    Span<ActorMessage> messageBuffer,
    Span<long> messageTimestamps,
    int messageId,
    long timestamp)
{
    // Write message data
    messageBuffer[messageId] = new ActorMessage { Payload = data };

    // RELEASE fence: Ensure message write completes before timestamp
    MemoryFence(FenceType.Release);

    // Write timestamp (signals message is ready)
    messageTimestamps[messageId] = timestamp;
}

[Kernel(MemoryOrdering = MemoryOrderingMode.ReleaseAcquire)]
public static void ReceiveMessage(
    Span<ActorMessage> messageBuffer,
    Span<long> messageTimestamps,
    int messageId)
{
    // ACQUIRE: Read timestamp first
    long timestamp = messageTimestamps[messageId];

    // ACQUIRE fence: Ensure timestamp read completes before message read
    MemoryFence(FenceType.Acquire);

    // Now safe to read message (causal ordering guaranteed)
    var message = messageBuffer[messageId];
    ProcessMessage(message, timestamp);
}
```

---

## Complete Integration Example

Full end-to-end example combining all Phase 5 features.

**File**: `examples/Temporal/GpuResidentActorExample.cs`

```csharp
using Orleans.GpuBridge.Backends.DotCompute.Temporal;
using Orleans.GpuBridge.Runtime.Temporal;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var host = Host.CreateDefaultBuilder(args)
    .ConfigureServices((context, services) =>
    {
        services.AddGpuBridge(options => options.PreferGpu = true)
                .AddDotComputeBackend()
                .AddGpuTiming(options =>
                {
                    options.EnableTimestampInjection = true;
                    options.AutoCalibrate = true;
                })
                .AddTemporalActors();
    })
    .Build();

await host.StartAsync();

// Get services
var calibrator = host.Services.GetRequiredService<GpuClockCalibrator>();
var ringManager = host.Services.GetRequiredService<RingKernelManager>();

// Calibrate GPU clock
var calibration = await calibrator.CalibrateAsync(sampleCount: 1000);
Console.WriteLine($"GPU clock offset: {calibration.OffsetNanos}ns, drift: {calibration.DriftPPM}ppm");

// Launch ring kernel
await ringManager.StartAsync(actorCount: 1000);

// ... actors now process messages on GPU with sub-microsecond latency ...

// Graceful shutdown
await ringManager.StopAsync();
await host.StopAsync();
```

---

## Testing and Validation

### Test 1: GPU Timestamp Accuracy

```csharp
[Fact]
public async Task GpuTimestamp_ShouldHaveNanosecondPrecision()
{
    var timingProvider = GetTimingProvider();

    var t1 = await timingProvider.GetGpuTimestampAsync();
    await Task.Delay(10); // 10ms
    var t2 = await timingProvider.GetGpuTimestampAsync();

    var elapsed = t2 - t1;

    // Should be ~10ms ± 100μs
    elapsed.Should().BeGreaterThan(9_900_000); // 9.9ms
    elapsed.Should().BeLessThan(10_100_000);   // 10.1ms
}
```

### Test 2: Ring Kernel Message Latency

```csharp
[Fact]
public async Task RingKernel_ShouldHaveSubMicrosecondLatency()
{
    var ringManager = GetRingKernelManager();
    await ringManager.StartAsync(actorCount: 1);

    var start = Stopwatch.GetTimestamp();

    // Send 1000 messages
    for (int i = 0; i < 1000; i++)
    {
        await SendMessageToRingKernel(actorId: 0, message);
    }

    var elapsed = Stopwatch.GetElapsedTime(start);
    var avgLatency = elapsed.TotalNanoseconds / 1000;

    // Average latency should be < 1μs (1000ns)
    avgLatency.Should().BeLessThan(1000);
}
```

### Test 3: Causal Ordering with Memory Fences

```csharp
[Fact]
public async Task CausalOrdering_ShouldMaintainHappensBefore()
{
    // Send message A -> B with causal dependency
    var messageA = await SendCausalMessage(actorA, "data-A");
    var messageB = await SendCausalMessage(actorB, "data-B", dependsOn: messageA.Id);

    // Verify HLC ordering: B.timestamp > A.timestamp
    messageB.HLC.CompareTo(messageA.HLC).Should().BeGreaterThan(0);
}
```

---

## Performance Tuning

### Optimization 1: Batch Timestamp Queries

Instead of querying timestamps individually, batch them:

```csharp
// ❌ BAD: Individual queries
for (int i = 0; i < 1000; i++)
{
    timestamps[i] = await timingProvider.GetGpuTimestampAsync();
}

// ✅ GOOD: Batch query
var timestamps = await timingProvider.GetGpuTimestampsBatchAsync(count: 1000);
```

### Optimization 2: Clock Calibration Frequency

Calibrate less frequently for stable clocks:

```csharp
// Calibrate every 5 minutes instead of every message
if (DateTime.UtcNow - lastCalibration > TimeSpan.FromMinutes(5))
{
    await calibrator.CalibrateAsync();
}
```

### Optimization 3: Ring Buffer Sizing

Larger ring buffers reduce contention but use more GPU memory:

```csharp
// High-frequency: Larger buffer to prevent overflow
await ringManager.StartAsync(actorCount: 1000, messageQueueSize: 8192);

// Low-frequency: Smaller buffer to save memory
await ringManager.StartAsync(actorCount: 1000, messageQueueSize: 1024);
```

---

## Next Steps

1. ✅ Update DotCompute to 0.4.2-rc2
2. ⏳ Implement timing service integration
3. ⏳ Create ring kernel for resident actors
4. ⏳ Add barrier support for pattern detection
5. ⏳ Test GPU timestamp accuracy
6. ⏳ Benchmark ring kernel latency
7. ⏳ Validate causal ordering with memory fences

Proceed to Phase 6 (Physical Time Precision) after Phase 5 validation is complete.

---

*Guide Version: 1.0*
*Last Updated: 2025-01-11*
*DotCompute Version: 0.4.2-rc2*
