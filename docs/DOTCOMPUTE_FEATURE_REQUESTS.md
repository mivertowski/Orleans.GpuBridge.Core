# DotCompute Feature Requests for Orleans.GpuBridge.Core Integration

**Date**: 2025-01-06
**DotCompute Version**: v0.3.0-rc1
**Source**: Orleans.GpuBridge.Core Production Integration Requirements
**Purpose**: Requested features and enhancements for production GPU acceleration in distributed systems

---

## Overview

This document outlines feature requests from Orleans.GpuBridge.Core integration experience. These features would enhance DotCompute's capabilities for production distributed computing scenarios.

**Context**: Orleans.GpuBridge.Core provides GPU acceleration for Microsoft Orleans grains (distributed actors). Our requirements focus on:
- Production monitoring and observability
- Multi-grain coordination
- Fault tolerance and recovery
- Performance optimization
- Distributed GPU workloads

---

## 🔴 High Priority Features

### 1. Device Health Monitoring APIs

**Problem**: Production systems need continuous health monitoring to detect thermal throttling, hardware errors, and performance degradation.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface IAccelerator
{
    /// <summary>
    /// Query device sensor data (temperature, power, fan speed, etc.)
    /// </summary>
    Task<SensorData?> GetSensorDataAsync(
        SensorType sensorType,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get comprehensive device metrics in a single call
    /// </summary>
    Task<DeviceMetrics> GetMetricsAsync(
        CancellationToken cancellationToken = default);
}

public enum SensorType
{
    Temperature,      // GPU temperature in Celsius
    PowerConsumption, // Power draw in watts
    FanSpeed,         // Fan speed percentage
    ClockFrequency,   // Current clock in MHz
    Utilization,      // GPU utilization percentage
    MemoryUtilization // Memory utilization percentage
}

public readonly struct SensorData
{
    public SensorType Type { get; }
    public double Value { get; }
    public string Unit { get; }
    public DateTime Timestamp { get; }
    public bool IsValid { get; }
}

public readonly struct DeviceMetrics
{
    public float ComputeUtilizationPercent { get; }
    public float MemoryUtilizationPercent { get; }
    public float TemperatureCelsius { get; }
    public float PowerWatts { get; }
    public int FanSpeedPercent { get; }
    public int ClockFrequencyMHz { get; }
    public long KernelsExecuted { get; }
    public long BytesTransferred { get; }
    public TimeSpan Uptime { get; }
    public DateTime Timestamp { get; }
}
```

**Use Cases**:
- **Thermal Throttling Detection**: Detect when GPU is throttling due to temperature
- **Load Balancing**: Distribute work based on current device utilization
- **Predictive Maintenance**: Identify devices with degrading performance
- **SLA Monitoring**: Track metrics for service-level agreements
- **Adaptive Scheduling**: Route work to underutilized devices

**Implementation Notes**:
- Could query NVML (NVIDIA), ROCm SMI (AMD), or backend-specific APIs
- Some sensors may not be available on all devices (return null)
- Consider caching sensor reads with TTL (sensors update ~1Hz)
- Async to avoid blocking on sensor query latency (~10-50ms)

**Priority**: 🔴 **HIGH** - Essential for production monitoring

---

### 2. Device Reset API

**Problem**: When GPU kernels hang or encounter errors, need ability to reset device to clean state without restarting entire application.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface IAccelerator
{
    /// <summary>
    /// Reset device to clean state, clearing all memory and stopping kernels
    /// </summary>
    /// <param name="options">Reset options (soft vs hard reset)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task ResetAsync(
        DeviceResetOptions? options = null,
        CancellationToken cancellationToken = default);
}

public class DeviceResetOptions
{
    /// <summary>
    /// Soft reset: Stop kernels and clear memory
    /// Hard reset: Full device reset (may require driver reinitialization)
    /// </summary>
    public ResetLevel Level { get; set; } = ResetLevel.Soft;

    /// <summary>
    /// Timeout for reset operation
    /// </summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);
}

public enum ResetLevel
{
    Soft,  // Stop kernels, clear memory, fast (~100-500ms)
    Hard   // Full device reset, slow (~1-3 seconds)
}
```

**Use Cases**:
- **Error Recovery**: Recover from hung kernels without restarting app
- **Grain Deactivation**: Clean up GPU state when Orleans grain deactivates
- **Testing**: Reset to clean state between test runs
- **Resource Cleanup**: Ensure no leaked resources after errors

**Implementation Notes**:
- CUDA: `cudaDeviceReset()`
- OpenCL: Re-create context
- May require draining command queues first
- Should invalidate all compiled kernels
- Should clear all allocations

**Priority**: 🔴 **HIGH** - Critical for fault tolerance

---

### 3. MaxWorkItemDimensions Property

**Problem**: Need to validate 3D workgroup dimensions before kernel launch. Currently may be in Capabilities dictionary but should be first-class property.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public readonly struct AcceleratorInfo
{
    // Existing properties...
    public string Id { get; }
    public string Name { get; }
    public string Architecture { get; }
    public int WarpSize { get; }

    /// <summary>
    /// Maximum dimensions for work items (typically 3 for GPU, 1 for CPU)
    /// </summary>
    public int MaxWorkItemDimensions { get; }  // NEW

    /// <summary>
    /// Maximum work items per dimension
    /// Example: [1024, 1024, 64] means max 1024 threads in X and Y, 64 in Z
    /// </summary>
    public IReadOnlyList<int> MaxWorkItemSizes { get; }  // NEW
}
```

**Use Cases**:
- **Dimension Validation**: Ensure kernel launch parameters are valid
- **Automatic Sizing**: Calculate optimal grid/block sizes
- **Cross-Platform**: Handle different limits (GPU vs CPU)

**Implementation Notes**:
- CUDA: `cudaDeviceProp.maxGridSize` and `maxThreadsDim`
- OpenCL: `CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS` and `CL_DEVICE_MAX_WORK_ITEM_SIZES`
- Typically 3 for GPUs, 1 for CPUs

**Priority**: 🔴 **HIGH** - Needed for kernel launch validation

---

## 🟡 Medium Priority Features

### 4. Async Memory Statistics

**Problem**: Current `Statistics` property is synchronous. For accurate memory statistics, prefer async API that can query device.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface IUnifiedMemoryManager
{
    // Keep existing synchronous property for quick access
    MemoryStatistics Statistics { get; }

    /// <summary>
    /// Get accurate memory statistics with device query
    /// </summary>
    Task<MemoryStatistics> GetStatisticsAsync(
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get statistics for specific memory pool
    /// </summary>
    Task<MemoryPoolStatistics> GetPoolStatisticsAsync(
        string poolName,
        CancellationToken cancellationToken = default);
}

public readonly struct MemoryPoolStatistics
{
    public string PoolName { get; }
    public long TotalBytes { get; }
    public long AllocatedBytes { get; }
    public long FreeBytes { get; }
    public int AllocationCount { get; }
    public int PoolHitCount { get; }
    public int PoolMissCount { get; }
    public double HitRate { get; }
}
```

**Use Cases**:
- **Accurate Monitoring**: Get real-time device memory state
- **Pool Analysis**: Understand pool effectiveness
- **Memory Optimization**: Identify fragmentation issues

**Implementation Notes**:
- CUDA: `cudaMemGetInfo()`
- OpenCL: `clGetMemAllocInfoINTEL()`
- Query may take 1-10ms

**Priority**: 🟡 **MEDIUM** - Nice to have, existing Statistics property sufficient

---

### 5. Event Notification System

**Problem**: Need reactive notifications for device state changes, errors, and thermal events without polling.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface IAccelerator
{
    /// <summary>
    /// Device state changed (available, busy, error)
    /// </summary>
    event EventHandler<DeviceStateChangedEventArgs> StateChanged;

    /// <summary>
    /// Device error occurred
    /// </summary>
    event EventHandler<DeviceErrorEventArgs> ErrorOccurred;

    /// <summary>
    /// Thermal event (temperature threshold exceeded)
    /// </summary>
    event EventHandler<ThermalEventArgs> ThermalEvent;

    /// <summary>
    /// Memory event (low memory, allocation failure)
    /// </summary>
    event EventHandler<MemoryEventArgs> MemoryEvent;
}

public class DeviceStateChangedEventArgs : EventArgs
{
    public DeviceState PreviousState { get; }
    public DeviceState CurrentState { get; }
    public DateTime Timestamp { get; }
}

public class DeviceErrorEventArgs : EventArgs
{
    public string ErrorCode { get; }
    public string ErrorMessage { get; }
    public Exception? Exception { get; }
    public bool IsFatal { get; }
    public DateTime Timestamp { get; }
}

public class ThermalEventArgs : EventArgs
{
    public float CurrentTemperature { get; }
    public float ThresholdTemperature { get; }
    public ThermalEventType EventType { get; }
    public DateTime Timestamp { get; }
}

public enum ThermalEventType
{
    Warning,    // Approaching threshold
    Critical,   // Exceeded threshold
    Throttling  // Performance throttling active
}

public class MemoryEventArgs : EventArgs
{
    public long AvailableBytes { get; }
    public long RequestedBytes { get; }
    public MemoryEventType EventType { get; }
    public DateTime Timestamp { get; }
}

public enum MemoryEventType
{
    LowMemory,          // Available memory below threshold
    AllocationFailed,   // Allocation attempt failed
    FragmentationHigh   // High fragmentation detected
}
```

**Use Cases**:
- **Proactive Monitoring**: React to device issues immediately
- **Load Shedding**: Stop sending work to failing devices
- **Thermal Management**: Reduce load when temperature high
- **Memory Pressure**: Trigger cleanup when memory low

**Implementation Notes**:
- Background thread polls sensors and raises events
- Consider event throttling to prevent spam
- May require platform-specific notification APIs
- Could use `IObservable<T>` pattern instead of events

**Priority**: 🟡 **MEDIUM** - Nice for production, can poll as workaround

---

### 6. Multi-GPU Synchronization and Peer Access

**Problem**: Multi-GPU workloads need coordination and efficient peer-to-peer memory transfer.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface IAcceleratorManager
{
    /// <summary>
    /// Synchronize all devices (wait for all operations to complete)
    /// </summary>
    Task SynchronizeAllAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if peer-to-peer memory access is supported
    /// </summary>
    bool CanAccessPeer(IAccelerator source, IAccelerator target);

    /// <summary>
    /// Enable peer-to-peer memory access between devices
    /// </summary>
    Task EnablePeerAccessAsync(
        IAccelerator source,
        IAccelerator target,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Disable peer access
    /// </summary>
    Task DisablePeerAccessAsync(
        IAccelerator source,
        IAccelerator target,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get peer-to-peer topology information
    /// </summary>
    Task<PeerTopology> GetPeerTopologyAsync(
        CancellationToken cancellationToken = default);
}

public readonly struct PeerTopology
{
    /// <summary>
    /// Devices that can communicate peer-to-peer
    /// </summary>
    public IReadOnlyList<PeerConnection> Connections { get; }
}

public readonly struct PeerConnection
{
    public IAccelerator Source { get; }
    public IAccelerator Target { get; }
    public bool IsSupported { get; }
    public bool IsEnabled { get; }
    public long BandwidthBytesPerSecond { get; }
    public ConnectionType Type { get; }
}

public enum ConnectionType
{
    None,         // No peer access
    NVLink,       // NVIDIA NVLink
    InfinityFabric, // AMD Infinity Fabric
    PCIe,         // PCIe peer access
    SystemMemory  // Through system memory
}
```

**Use Cases**:
- **Data Parallelism**: Split work across multiple GPUs
- **Model Parallelism**: Large models split across devices
- **Efficient Transfer**: P2P faster than host-device-host
- **Multi-GPU Pipelines**: Stream processing across GPUs

**Implementation Notes**:
- CUDA: `cudaDeviceCanAccessPeer()`, `cudaDeviceEnablePeerAccess()`
- Requires compatible devices (same architecture, same node)
- NVLink provides best performance (~300 GB/s vs ~16 GB/s PCIe)
- May require elevated privileges

**Priority**: 🟡 **MEDIUM** - Important for multi-GPU, can workaround with host transfers

---

## 🟢 Low Priority Features

### 7. Vulkan and Metal Backend Support

**Problem**: Additional backend support for cross-platform and game engine integration.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public enum AcceleratorType
{
    CPU,
    GPU,      // Generic GPU
    CUDA,     // NVIDIA CUDA
    OpenCL,   // OpenCL device
    Vulkan,   // NEW: Vulkan compute
    Metal,    // NEW: Apple Metal
    DirectX   // NEW: DirectX compute
}
```

**Use Cases**:
- **Cross-Platform**: Vulkan works on Windows, Linux, Android
- **Apple Silicon**: Metal for M1/M2/M3 Macs
- **Gaming Integration**: Shared compute with graphics
- **Mobile**: Vulkan/Metal for mobile GPU compute

**Implementation Notes**:
- Vulkan compute shaders via SPIR-V
- Metal compute via MSL (Metal Shading Language)
- DirectCompute via HLSL
- May require backend-specific kernel language support

**Priority**: 🟢 **LOW** - CUDA/OpenCL cover most scenarios

---

### 8. Kernel Caching and Persistence

**Problem**: Kernel compilation is slow (100-1000ms). Caching compiled kernels speeds up startup.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface IAccelerator
{
    /// <summary>
    /// Compile kernel with optional persistent caching
    /// </summary>
    Task<ICompiledKernel> CompileKernelAsync(
        KernelDefinition definition,
        CompilationOptions options,
        string? cacheKey = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if kernel is cached
    /// </summary>
    Task<bool> IsCachedAsync(string cacheKey);

    /// <summary>
    /// Clear kernel cache
    /// </summary>
    Task ClearCacheAsync();

    /// <summary>
    /// Export compiled kernel to file
    /// </summary>
    Task ExportKernelAsync(
        ICompiledKernel kernel,
        string path,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Import compiled kernel from file
    /// </summary>
    Task<ICompiledKernel> ImportKernelAsync(
        string path,
        CancellationToken cancellationToken = default);
}

public class CompilationOptions
{
    // Existing...
    public OptimizationLevel OptimizationLevel { get; set; }

    // NEW: Caching options
    public bool EnablePersistentCache { get; set; } = true;
    public string? CacheDirectory { get; set; }
    public TimeSpan? CacheExpiration { get; set; }
}
```

**Use Cases**:
- **Faster Startup**: Skip compilation on subsequent runs
- **Deployment**: Pre-compile kernels for deployment
- **Version Control**: Store compiled kernels with source
- **CI/CD**: Compile once, deploy everywhere

**Implementation Notes**:
- CUDA: PTX or CUBIN serialization
- OpenCL: Serialize program binaries
- Cache key from source hash + compiler version + device
- Store in user's temp directory or specified location

**Priority**: 🟢 **LOW** - Optimization, not essential

---

### 9. Profiling and Debugging APIs

**Problem**: Performance optimization requires detailed kernel profiling information.

**Requested API**:
```csharp
namespace DotCompute.Abstractions;

public interface ICompiledKernel
{
    /// <summary>
    /// Get profiling information for this kernel
    /// </summary>
    Task<KernelProfilingInfo> GetProfilingInfoAsync(
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Number of registers used per thread
    /// </summary>
    int RegisterCount { get; }

    /// <summary>
    /// Shared memory used in bytes
    /// </summary>
    int SharedMemoryBytes { get; }

    /// <summary>
    /// Estimated occupancy (0.0 - 1.0)
    /// </summary>
    double EstimatedOccupancy { get; }

    /// <summary>
    /// Maximum threads per block for optimal occupancy
    /// </summary>
    int OptimalThreadsPerBlock { get; }
}

public readonly struct KernelProfilingInfo
{
    public int RegistersPerThread { get; }
    public int SharedMemoryPerBlock { get; }
    public int ConstantMemoryBytes { get; }
    public int LocalMemoryPerThread { get; }
    public double TheoreticalOccupancy { get; }
    public int WarpCount { get; }
    public int MaxActiveBlocks { get; }
    public TimeSpan CompilationTime { get; }
    public string IntermediateRepresentation { get; }
}

public class KernelExecutionOptions
{
    // NEW: Profiling
    public bool EnableProfiling { get; set; }
    public bool CollectMemoryStats { get; set; }
    public bool CollectTimingStats { get; set; }
}

public readonly struct KernelExecutionStats
{
    public TimeSpan ExecutionTime { get; }
    public long BytesRead { get; }
    public long BytesWritten { get; }
    public double EffectiveBandwidthGBps { get; }
    public double ComputeUtilization { get; }
    public double MemoryUtilization { get; }
}
```

**Use Cases**:
- **Performance Tuning**: Identify bottlenecks
- **Occupancy Analysis**: Optimize register usage
- **Memory Analysis**: Understand memory patterns
- **Debugging**: Diagnose performance issues

**Implementation Notes**:
- CUDA: `cudaFuncGetAttributes()`, CUDA profiling API
- OpenCL: `CL_KERNEL_WORK_GROUP_SIZE`, profiling events
- May require debug compilation for some info
- Consider overhead of profiling

**Priority**: 🟢 **LOW** - Development tool, not production feature

---

## 📊 Feature Priority Summary

| Priority | Features | Total |
|----------|----------|-------|
| 🔴 **HIGH** | Device Health Monitoring, Device Reset, MaxWorkItemDimensions | 3 |
| 🟡 **MEDIUM** | Async Memory Stats, Event Notifications, Multi-GPU Sync | 3 |
| 🟢 **LOW** | Vulkan/Metal Support, Kernel Caching, Profiling APIs | 3 |
| **TOTAL** | | **9** |

---

## 🎯 Implementation Suggestions

### Phase 1: Core Production Features (High Priority)
**Timeline**: 1-2 months
1. Device Health Monitoring APIs
2. Device Reset API
3. MaxWorkItemDimensions property

**Impact**: Enables production deployment with monitoring and fault tolerance

---

### Phase 2: Advanced Production Features (Medium Priority)
**Timeline**: 2-3 months
1. Async Memory Statistics
2. Event Notification System
3. Multi-GPU Synchronization

**Impact**: Enhanced production capabilities, better observability

---

### Phase 3: Optimization Features (Low Priority)
**Timeline**: 3-6 months
1. Additional Backend Support (Vulkan, Metal)
2. Kernel Caching System
3. Profiling and Debugging APIs

**Impact**: Performance optimization, developer experience improvements

---

## 💡 Alternative Approaches

For each feature, consider:

### 1. Backend-Specific Extensions
Instead of first-class APIs, could provide backend-specific extensions:

```csharp
// Get CUDA-specific interface
if (accelerator is ICudaAccelerator cudaAccelerator)
{
    var temperature = cudaAccelerator.GetTemperature();
    cudaAccelerator.Reset();
}
```

**Pros**: Faster to implement, full backend capabilities
**Cons**: Not portable, requires casting, harder to use

---

### 2. Capabilities Dictionary
Store advanced features in `Capabilities` dictionary:

```csharp
if (accelerator.Capabilities.TryGetValue("SupportsReset", out var value))
{
    await accelerator.ResetAsync();
}
```

**Pros**: Flexible, backward compatible
**Cons**: Not discoverable, no type safety, harder to use

---

### 3. Plugin System
Allow users to add custom functionality:

```csharp
accelerator.RegisterExtension(new HealthMonitoringExtension());
var monitor = accelerator.GetExtension<IHealthMonitoring>();
```

**Pros**: Extensible, community contributions
**Cons**: Complex, may fragment ecosystem

---

## 📝 Implementation Notes

### For DotCompute Team

**Testing Recommendations**:
- Test on NVIDIA, AMD, and Intel devices
- Test with multiple concurrent callers
- Test with devices in error states
- Load test event notification system
- Benchmark async operations overhead

**Documentation Recommendations**:
- Provide examples for each feature
- Document platform-specific limitations
- Show common patterns (monitoring, multi-GPU)
- Include troubleshooting guide

**Backward Compatibility**:
- All features should be optional
- Provide sensible defaults
- Gracefully handle unsupported features
- Version APIs appropriately

---

## ✅ Conclusion

These feature requests stem from real production integration requirements for distributed GPU computing with Orleans. Most features fall into three categories:

1. **Monitoring & Observability** - Essential for production operations
2. **Fault Tolerance** - Critical for reliable distributed systems
3. **Performance Optimization** - Nice-to-have for advanced scenarios

**Recommendation**: Focus on **High Priority** features first (monitoring, reset, dimensions). These unblock production deployment scenarios.

---

## 📞 Contact

**Project**: Orleans.GpuBridge.Core
**Repository**: https://github.com/mivertowski/Orleans.GpuBridge.Core (if applicable)
**Maintainer**: Michael Ivertowski

For questions or clarifications on any feature request, please reach out through DotCompute's preferred communication channel.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-06
**DotCompute Version**: v0.3.0-rc1
**Status**: Active Feature Requests
