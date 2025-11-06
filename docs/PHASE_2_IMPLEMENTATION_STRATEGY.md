# Phase 2: Orleans Integration - Implementation Strategy

**Date**: 2025-01-06
**Building On**: Phase 1 GPU Acceleration Foundation (✅ Complete & Validated)
**GPU Hardware**: NVIDIA RTX 2000 Ada Generation (8GB, Compute 8.9, CUDA 13.0.48, WSL2)

---

## 🎯 Phase 2 Objectives

Integrate Orleans.GpuBridge GPU acceleration with Orleans distributed framework, enabling:
- **GPU-aware grain placement** - Intelligently place grains on GPU-capable silos
- **Resource-based scheduling** - Select silos based on GPU memory and queue depth
- **Stream processing** - Real-time GPU-accelerated stream processing
- **Stateful GPU grains** - Keep data resident in GPU memory
- **Production patterns** - Best practices for GPU grain development

---

## 📊 Current State Analysis

### ✅ What We Have (Phase 1 Complete)
- DotCompute backend with real GPU execution
- Kernel compilation (CUDA, OpenCL, etc.)
- Memory allocation and data transfer
- Zero-copy execution pathways
- 100% GPU validation on RTX hardware

### 🏗️ What Exists (Skeleton Implementations)
```
src/Orleans.GpuBridge.Grains/
  ├── Batch/GpuBatchGrain.cs          ✓ Uses IGpuBridge abstraction
  ├── Stream/GpuStreamGrain.cs        ✓ Stream processing skeleton
  └── Implementation/GpuResidentGrain.cs  ✓ Stateful grain skeleton

src/Orleans.GpuBridge.Runtime/
  ├── GpuPlacementStrategy.cs         ⚠️ Basic (needs GPU awareness)
  └── GpuPlacementDirector.cs         ⚠️ Simple (needs capacity tracking)
```

### 🔨 What Needs to Be Built
1. **GPU Capacity Tracking System**
   - Track GPU memory per silo
   - Monitor queue depth and utilization
   - Expose capacity to placement director

2. **Enhanced Placement Director**
   - Select best silo based on GPU resources
   - Fallback to CPU silos when GPU unavailable
   - Local placement optimization

3. **GPU-Accelerated Grain Implementations**
   - Wire existing grains to DotCompute backend
   - Add GPU memory management
   - Implement result streaming

4. **Integration Testing**
   - Multi-silo cluster tests
   - GPU/CPU fallback scenarios
   - Performance validation

---

## 🏗️ Implementation Plan

### Week 1: GPU Placement & Capacity Tracking

#### Day 1-2: GPU Capacity Grain
**Location**: `src/Orleans.GpuBridge.Grains/Capacity/`

**Files to Create**:
```csharp
// IGpuCapacityGrain.cs - Interface for capacity tracking
public interface IGpuCapacityGrain : IGrainWithIntegerKey
{
    Task RegisterSiloAsync(SiloAddress silo, GpuCapacity capacity);
    Task UnregisterSiloAsync(SiloAddress silo);
    Task UpdateCapacityAsync(SiloAddress silo, GpuCapacity capacity);
    Task<List<SiloGpuCapacity>> GetGpuCapableSilosAsync();
    Task<GpuCapacity?> GetSiloCapacityAsync(SiloAddress silo);
}

// GpuCapacityGrain.cs - Capacity tracking implementation
[Reentrant]
public sealed class GpuCapacityGrain : Grain, IGpuCapacityGrain
{
    private readonly Dictionary<SiloAddress, GpuCapacity> _capacities = new();
    // Track GPU memory, queue depth, device count per silo
}

// GpuCapacity.cs - Model for GPU resources
public sealed record GpuCapacity(
    int DeviceCount,
    long TotalMemoryMB,
    long AvailableMemoryMB,
    int QueueDepth,
    DateTime LastUpdated);

// SiloGpuCapacity.cs - Silo + capacity combined
public sealed record SiloGpuCapacity(
    SiloAddress SiloAddress,
    GpuCapacity Capacity)
{
    public long AvailableMemoryMB => Capacity.AvailableMemoryMB;
    public int QueueDepth => Capacity.QueueDepth;
}
```

**Integration Points**:
- Silos register on startup with DeviceBroker stats
- Periodic updates every 30 seconds
- Unregister on graceful shutdown

#### Day 3-4: Enhanced Placement Director
**Location**: `src/Orleans.GpuBridge.Runtime/Placement/`

**File to Enhance**: `GpuPlacementDirector.cs`

**Key Changes**:
```csharp
public sealed class GpuPlacementDirector : IPlacementDirector
{
    private readonly ILocalSiloDetails _localSilo;
    private readonly IGrainFactory _grainFactory;
    private readonly ILogger<GpuPlacementDirector> _logger;

    public async Task<SiloAddress> OnAddActivation(
        PlacementStrategy strategy,
        PlacementTarget target,
        IPlacementContext context)
    {
        // 1. Get GPU-capable silos from capacity grain
        // 2. Filter by minimum memory requirements
        // 3. Prefer local silo if PreferLocalPlacement = true
        // 4. Select silo with most available memory
        // 5. Fallback to any compatible silo if no GPU available
    }
}
```

**Test Scenarios**:
- Placement with GPU available
- Placement with GPU memory constrained
- Placement with no GPU (CPU fallback)
- Local placement preference

#### Day 5: Silo Lifecycle Integration
**Location**: `src/Orleans.GpuBridge.Runtime/Infrastructure/`

**File to Create**: `GpuSiloLifecycleParticipant.cs`

```csharp
public sealed class GpuSiloLifecycleParticipant : ILifecycleParticipant<ISiloLifecycle>
{
    public void Participate(ISiloLifecycle lifecycle)
    {
        // Register GPU capacity on silo startup
        lifecycle.Subscribe(
            nameof(GpuSiloLifecycleParticipant),
            ServiceLifecycleStage.ApplicationServices,
            OnStart,
            OnStop);
    }

    private async Task OnStart(CancellationToken ct)
    {
        // Get GPU capacity from DeviceBroker
        // Register with GpuCapacityGrain
    }

    private async Task OnStop(CancellationToken ct)
    {
        // Unregister from GpuCapacityGrain
    }
}
```

---

### Week 2: Grain Implementations & Testing

#### Day 6-7: Enhanced GpuBatchGrain
**Location**: `src/Orleans.GpuBridge.Grains/Batch/GpuBatchGrain.cs`

**Current State**: Uses IGpuBridge abstraction ✅

**Enhancements Needed**:
1. Add DotCompute memory pre-allocation for large batches
2. Implement batch partitioning for memory constraints
3. Add performance telemetry
4. GPU memory pressure handling

**Test Coverage**:
```csharp
// tests/Orleans.GpuBridge.Tests/Integration/GpuBatchGrainTests.cs
[Fact]
public async Task BatchGrain_WithGpu_Should_Execute_Successfully()
{
    // Test with actual GPU execution via DotCompute
}

[Fact]
public async Task BatchGrain_LargeBatch_Should_Partition()
{
    // Test automatic partitioning for memory-constrained batches
}

[Fact]
public async Task BatchGrain_NoGpu_Should_Fallback_ToCpu()
{
    // Test CPU fallback behavior
}
```

#### Day 8: GpuStreamGrain Enhancement
**Location**: `src/Orleans.GpuBridge.Grains/Stream/GpuStreamGrain.cs`

**Current State**: Stream processing skeleton ✅

**Enhancements Needed**:
1. Wire to DotCompute kernel execution
2. Add batch accumulation (collect items before GPU submission)
3. Implement backpressure handling
4. Add stream metrics

**Key Pattern**:
```csharp
private async Task ProcessStreamAsync(CancellationToken ct)
{
    const int batchSize = 128; // Configurable
    var batch = new List<TIn>(batchSize);

    while (!ct.IsCancellationRequested)
    {
        // Accumulate items from input stream
        while (batch.Count < batchSize && TryReadFromChannel(out var item))
        {
            batch.Add(item);
        }

        // Process batch on GPU when threshold reached
        if (batch.Count >= batchSize)
        {
            await ProcessBatchOnGpuAsync(batch, ct);
            batch.Clear();
        }

        // Timeout-based processing for partial batches
        await PeriodicTimerTick(ct);
    }
}
```

#### Day 9: GpuResidentGrain Enhancement
**Location**: `src/Orleans.GpuBridge.Grains/Implementation/GpuResidentGrain.cs`

**Current State**: Stateful grain skeleton ✅

**Enhancements Needed**:
1. Load state into GPU memory on activation
2. Keep state GPU-resident during active period
3. Evict after idle timeout (10 minutes default)
4. Serialize/deserialize state efficiently

**GPU Memory Pattern**:
```csharp
public override async Task OnActivateAsync(CancellationToken ct)
{
    await base.OnActivateAsync(ct);

    // Load persistent state
    await _state.ReadStateAsync();

    // Try to load into GPU memory
    await LoadToGpuMemoryAsync();

    // Register eviction timer
    RegisterTimer(CheckEvictionAsync, null,
        TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));
}

private async Task LoadToGpuMemoryAsync()
{
    var allocator = ServiceProvider
        .GetRequiredService<IMemoryAllocator>();

    var stateBytes = SerializeState(_state.State);
    _gpuMemory = await allocator.AllocateAsync<byte>(
        stateBytes.Length,
        new MemoryAllocationOptions { AllocationFlags = MemoryFlags.HostVisible },
        CancellationToken.None);

    await _gpuMemory.CopyFromHostAsync(stateBytes, 0, 0, stateBytes.Length);
}
```

#### Day 10: Integration Testing with Orleans TestingHost
**Location**: `tests/Orleans.GpuBridge.Tests/Integration/`

**Test Suite**:
```csharp
// GpuOrleansIntegrationTests.cs
public class GpuOrleansIntegrationTests : IAsyncLifetime
{
    private TestCluster _cluster = default!;

    public async Task InitializeAsync()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<GpuSiloConfigurator>();

        _cluster = builder.Build();
        await _cluster.DeployAsync();
    }

    [Fact]
    public async Task MultiSilo_BatchGrain_Should_Route_ToGpuSilo()
    {
        // Test GPU-aware placement across multiple silos
    }

    [Fact]
    public async Task StreamGrain_Should_Process_ContinuousStream()
    {
        // Test continuous stream processing
    }

    [Fact]
    public async Task ResidentGrain_Should_KeepState_InGpuMemory()
    {
        // Test GPU-resident state management
    }
}

private class GpuSiloConfigurator : ISiloConfigurator
{
    public void Configure(ISiloBuilder siloBuilder)
    {
        siloBuilder.Services.AddGpuBridge(options =>
        {
            options.PreferGpu = true;
            options.EnableProfiling = true;
        });

        siloBuilder.Services.AddDotComputeBackend(options =>
        {
            // DotCompute configuration
        });

        siloBuilder.AddMemoryGrainStorageAsDefault();
        siloBuilder.AddMemoryStreams("Default");
    }
}
```

---

## 📁 File Organization

```
src/Orleans.GpuBridge.Grains/
  ├── Capacity/
  │   ├── IGpuCapacityGrain.cs          [NEW]
  │   ├── GpuCapacityGrain.cs           [NEW]
  │   ├── GpuCapacity.cs                [NEW]
  │   └── SiloGpuCapacity.cs            [NEW]
  │
  ├── Batch/
  │   ├── IGpuBatchGrain.cs             [EXISTS]
  │   └── GpuBatchGrain.cs              [ENHANCE]
  │
  ├── Stream/
  │   ├── IGpuStreamGrain.cs            [EXISTS]
  │   └── GpuStreamGrain.cs             [ENHANCE]
  │
  └── Implementation/
      ├── IGpuResidentGrain.cs          [EXISTS]
      └── GpuResidentGrain.cs           [ENHANCE]

src/Orleans.GpuBridge.Runtime/
  ├── Placement/
  │   ├── GpuPlacementStrategy.cs       [ENHANCE]
  │   └── GpuPlacementDirector.cs       [ENHANCE]
  │
  └── Infrastructure/
      ├── GpuSiloLifecycleParticipant.cs [NEW]
      └── ServiceCollectionExtensions.cs [UPDATE]

tests/Orleans.GpuBridge.Tests/
  └── Integration/
      ├── GpuBatchGrainTests.cs         [NEW]
      ├── GpuStreamGrainTests.cs        [NEW]
      ├── GpuResidentGrainTests.cs      [NEW]
      ├── GpuPlacementTests.cs          [NEW]
      └── GpuOrleansIntegrationTests.cs [NEW]
```

---

## 🎯 Success Criteria

### Functional Requirements ✅
- [ ] Grains activate on GPU-capable silos
- [ ] Automatic fallback to CPU silos when GPU unavailable
- [ ] GPU capacity accurately tracked and reported
- [ ] Stream processing maintains order
- [ ] Resident state persists correctly

### Performance Targets 🚀
- [ ] Grain activation < 100ms
- [ ] Stream latency < 10ms per item
- [ ] Batch processing scales linearly
- [ ] Memory usage stable under load
- [ ] GPU utilization > 80% during processing

### Integration Quality 📐
- [ ] All Orleans patterns properly implemented
- [ ] Clean separation of concerns
- [ ] Proper lifecycle management
- [ ] No memory leaks
- [ ] Comprehensive error handling

### Test Coverage 🧪
- [ ] Unit tests for placement logic
- [ ] Integration tests with Orleans TestingHost
- [ ] Multi-silo cluster scenarios
- [ ] GPU/CPU fallback validation
- [ ] Performance benchmarks

---

## 🔗 Dependencies on Phase 1

**Required Phase 1 Components** (All ✅ Complete):
- ✅ `DotComputeKernelCompiler` - Kernel compilation
- ✅ `DotComputeKernelExecutor` - Kernel execution
- ✅ `DotComputeMemoryAllocator` - Memory management
- ✅ `DotComputeDeviceManager` - Device enumeration
- ✅ `DotComputeBackendProvider` - Backend registration

**Integration Pattern**:
```csharp
// Grains use abstractions (IGpuBridge, IGpuKernel)
// Runtime wires to DotCompute implementations
services.AddGpuBridge(options => options.PreferGpu = true)
    .AddDotComputeBackend(options => {
        options.EnableProfiling = true;
        options.MaxMemoryPoolSizeMB = 2048;
    });
```

---

## 🚀 Next Steps After Phase 2

Phase 3 will focus on:
- **Advanced GPU Runtime Features**
  - Persistent kernel hosts with ring buffers
  - Multi-kernel coordination
  - GPUDirect Storage integration
  - CUDA Graph optimization

- **Performance Optimization**
  - Memory pooling strategies
  - Batch size auto-tuning
  - Kernel fusion techniques
  - Zero-copy streaming

---

## 📊 Estimated Timeline

| Week | Days | Component | Status |
|------|------|-----------|--------|
| **Week 1** | Day 1-2 | GPU Capacity Grain | 🔜 Pending |
| | Day 3-4 | Enhanced Placement Director | 🔜 Pending |
| | Day 5 | Silo Lifecycle Integration | 🔜 Pending |
| **Week 2** | Day 6-7 | Enhanced GpuBatchGrain | 🔜 Pending |
| | Day 8 | GpuStreamGrain Enhancement | 🔜 Pending |
| | Day 9 | GpuResidentGrain Enhancement | 🔜 Pending |
| | Day 10 | Integration Testing | 🔜 Pending |

---

## 🎉 Expected Outcome

After Phase 2 completion, Orleans.GpuBridge will provide:

1. **Production-Ready GPU Grains**
   - Batch processing with automatic GPU acceleration
   - Stream processing with GPU compute
   - Stateful grains with GPU-resident data

2. **Intelligent Resource Management**
   - GPU-aware grain placement
   - Automatic CPU fallback
   - Capacity-based load balancing

3. **Orleans Integration**
   - Full Orleans lifecycle support
   - Orleans Streams compatibility
   - Multi-silo cluster support

4. **Developer Experience**
   - Simple grain-based API
   - Automatic GPU utilization
   - Clear performance metrics

---

**Ready to proceed?** Let me know and we'll start with GPU Capacity Tracking! 🚀

*Report Generated: 2025-01-06*
*GPU: NVIDIA RTX 2000 Ada Generation (8GB, SM 8.9)*
*Framework: Orleans.GpuBridge.Core + DotCompute v0.4.1-rc2*
*Status: PHASE 2 PLANNING COMPLETE ✅*
