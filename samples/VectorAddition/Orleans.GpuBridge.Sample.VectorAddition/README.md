# Orleans.GpuBridge Ring Kernel API Sample

This sample application demonstrates the **Ring Kernel API** for Orleans.GpuBridge.Core, showcasing GPU-resident memory management and Direct Memory Access (DMA) operations.

## Overview

The Ring Kernel API (`IGpuResidentGrain<T>`) provides persistent GPU memory allocations that survive across grain activations. This sample demonstrates:

- **GPU Memory Allocation** - Allocating device memory with different types
- **DMA Transfers** - Host-to-device and device-to-host data transfers
- **Memory Management** - Querying memory statistics and releasing allocations
- **Kernel Execution** - Using resident memory buffers for GPU computation
- **Production Patterns** - Error handling, logging, and resource cleanup

## Architecture

### Ring Kernel API Components

```
┌─────────────────────────────────────────────────────────┐
│              IGpuResidentGrain<T>                       │
│  (Orleans Grain Interface for GPU Memory Management)   │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  AllocateAsync│  │  WriteAsync  │  │  ReadAsync   │
│              │  │              │  │              │
│ GPU Memory   │  │ Host→Device  │  │ Device→Host  │
│ Allocation   │  │ DMA Transfer │  │ DMA Transfer │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Key Interfaces

#### IGpuResidentGrain<T>
The main grain interface for managing GPU-resident memory:

```csharp
public interface IGpuResidentGrain<T> : IGrainWithStringKey where T : unmanaged
{
    Task<GpuMemoryHandle> AllocateAsync(long sizeBytes, GpuMemoryType memoryType);
    Task WriteAsync<TData>(GpuMemoryHandle handle, TData[] data, int offset = 0);
    Task<TData[]> ReadAsync<TData>(GpuMemoryHandle handle, int count, int offset = 0);
    Task<GpuComputeResult> ComputeAsync(KernelId kernelId, GpuMemoryHandle input,
                                        GpuMemoryHandle output, GpuComputeParams? parameters);
    Task ReleaseAsync(GpuMemoryHandle handle);
    Task<GpuMemoryInfo> GetMemoryInfoAsync();
    Task ClearAsync();
}
```

#### Memory Types

```csharp
public enum GpuMemoryType
{
    Default,    // Standard GPU memory (optimal for kernels)
    Pinned,     // Page-locked host memory (faster transfers)
    Shared,     // Unified memory (CPU + GPU accessible)
    Texture,    // Optimized for spatial access patterns
    Constant    // Read-only cached memory
}
```

## Sample Workflow

### 1. Initialize Orleans Cluster
```csharp
var builder = new TestClusterBuilder();
builder.AddSiloBuilderConfigurator<SiloConfigurator>();
var cluster = builder.Build();
await cluster.DeployAsync();
```

### 2. Configure GPU Bridge
```csharp
services.AddGpuBridge(options =>
{
    options.PreferGpu = true;
    options.EnableFallback = true;
});
```

### 3. Obtain Resident Grain
```csharp
var grain = grainFactory.GetGrain<IGpuResidentGrain<float>>("my-grain");
```

### 4. Allocate GPU Memory
```csharp
var handle = await grain.AllocateAsync(
    sizeBytes: 1024 * sizeof(float),
    memoryType: GpuMemoryType.Default);

// Result: GpuMemoryHandle
// - Id: Unique identifier for this allocation
// - SizeBytes: Total allocated size
// - Type: Memory allocation type
// - DeviceIndex: GPU device number
// - AllocatedAt: Timestamp of allocation
```

### 5. Write Data to GPU (DMA Transfer)
```csharp
var data = new float[1024];
await grain.WriteAsync(handle, data, offset: 0);

// Host-to-Device transfer via DMA
// - Transfers data from CPU memory to GPU memory
// - Supports partial writes via offset parameter
// - Validates bounds automatically
```

### 6. Read Data from GPU (DMA Transfer)
```csharp
var result = await grain.ReadAsync<float>(handle, count: 1024, offset: 0);

// Device-to-Host transfer via DMA
// - Transfers data from GPU memory to CPU memory
// - Returns strongly-typed array
// - Efficient for result retrieval
```

### 7. Query Memory Statistics
```csharp
var info = await grain.GetMemoryInfoAsync();

// Returns GpuMemoryInfo with:
// - TotalMemoryBytes: Total GPU VRAM
// - AllocatedMemoryBytes: Currently allocated
// - FreeMemoryBytes: Available memory
// - UtilizationPercentage: Memory usage %
// - FragmentationPercentage: Memory fragmentation %
// - Device information and timestamps
```

### 8. Execute GPU Kernel (Optional)
```csharp
var result = await grain.ComputeAsync(
    kernelId: KernelId.Parse("vector-add"),
    input: inputHandle,
    output: outputHandle,
    parameters: new GpuComputeParams(WorkGroupSize: 256));

// Result: GpuComputeResult
// - Success: Execution status
// - ExecutionTime: Kernel execution time
// - Error: Optional error message
```

### 9. Release Memory
```csharp
await grain.ReleaseAsync(handle);
await grain.ClearAsync(); // Releases all allocations
```

## Running the Sample

### Prerequisites
- .NET 9.0 SDK or later
- Orleans.GpuBridge.Core packages
- GPU device (optional - CPU fallback available)

### Build and Run
```bash
cd samples/VectorAddition/Orleans.GpuBridge.Sample.VectorAddition
dotnet restore
dotnet build
dotnet run
```

### Expected Output

```
═══════════════════════════════════════════════════════════════
  Orleans.GpuBridge.Core - Ring Kernel API Sample
  Demonstrating GPU Resident Memory & DMA Operations
═══════════════════════════════════════════════════════════════

▶ Step 1: Initializing Orleans cluster with GPU Bridge...
✓ Orleans cluster initialized successfully

▶ Step 2: Obtaining GPU Resident Grain...
✓ Acquired grain with key: vector-resident-grain

▶ Step 3: Allocating GPU memory...
  ✓ Allocated Input Buffer:
    - Handle ID: a1b2c3d4e5f67890
    - Size: 4.00 KB
    - Type: Default
    - Device: GPU 0
    - Timestamp: 2025-01-07 16:30:45.123 UTC
  ✓ Allocated Output Buffer:
    - Handle ID: f9e8d7c6b5a43210
    - Size: 4.00 KB
    - Type: Default
    - Device: GPU 0
    - Timestamp: 2025-01-07 16:30:45.234 UTC

▶ Step 4: Writing test data to GPU via DMA...
  ✓ Generated 1024 test values
    - Range: 0.0 to 100.0
    - Sample values: [42.15, 87.93, 23.45, ...]
  ✓ DMA write completed:
    - Elements transferred: 1,024
    - Data size: 4.00 KB
    - Transfer time: 0.125 ms
    - Bandwidth: 31.25 GB/s

▶ Step 5: Reading data back from GPU via DMA...
  ✓ DMA read completed:
    - Elements transferred: 1,024
    - Data size: 4.00 KB
    - Transfer time: 0.089 ms
    - Bandwidth: 43.82 GB/s
    - Sample values: [42.15, 87.93, 23.45, ...]
  ✓ Data validation successful:
    - All 1,024 elements match
    - Tolerance: ±1.00E-06

▶ Step 6: Retrieving GPU memory metrics...
  ✓ GPU Memory Statistics:
    ┌─────────────────────────────────────────────────
    │ Device: NVIDIA RTX 4090 (GPU 0)
    │ Timestamp: 2025-01-07 16:30:45.567 UTC
    ├─────────────────────────────────────────────────
    │ Total Memory:        24.00 GB
    │ Allocated:            8.00 KB
    │ Free:                23.99 GB
    │ Reserved:               0 B
    ├─────────────────────────────────────────────────
    │ Buffer Memory:        8.00 KB
    │ Kernel Memory:           0 B
    │ Texture Memory:          0 B
    ├─────────────────────────────────────────────────
    │ Utilization:           0.00%
    │ Fragmentation:         0.00%
    └─────────────────────────────────────────────────

▶ Step 7: Demonstrating kernel execution...
  ℹ Kernel execution demonstration:
    - Kernel ID: vector-add-kernel
    - Input Handle: a1b2c3d4e5f67890
    - Output Handle: f9e8d7c6b5a43210
    - Work Group Size: 256
    - Work Groups: Auto
  ℹ Note: Actual kernel execution requires registered kernels
    See Orleans.GpuBridge documentation for kernel registration

▶ Step 8: Releasing GPU memory allocations...
  ✓ Released Input Buffer (Handle: a1b2c3d4e5f67890)
  ✓ Released Output Buffer (Handle: f9e8d7c6b5a43210)

▶ Step 9: Performing final cleanup...
✓ All resident memory cleared

═══════════════════════════════════════════════════════════════
  Sample completed successfully!
═══════════════════════════════════════════════════════════════
```

## Performance Considerations

### Memory Allocation
- **Persistent allocations** survive grain deactivation
- **Memory pooling** reuses allocations when possible
- **Type safety** enforced through generics (`IGpuResidentGrain<T>`)

### DMA Transfers
- **Asynchronous operations** don't block Orleans scheduler
- **Optimal bandwidth** achieved with large transfers (>1 MB)
- **Pinned memory** provides ~2x faster transfer rates

### Memory Types Performance

| Type | Use Case | Transfer Speed | Kernel Access |
|------|----------|----------------|---------------|
| Default | General compute | Standard | Optimal |
| Pinned | Frequent transfers | Fast (~2x) | Standard |
| Shared | Unified access | Variable | Good |
| Texture | Spatial data | Standard | Cached |
| Constant | Read-only broadcast | Standard | Very Fast |

## Error Handling

The sample demonstrates production-grade error handling:

```csharp
try
{
    var handle = await grain.AllocateAsync(sizeBytes, memoryType);
}
catch (OutOfMemoryException)
{
    // GPU memory exhausted - consider smaller allocation or cleanup
}
catch (InvalidOperationException)
{
    // No GPU available - CPU fallback active
}
```

### Common Exceptions

- `OutOfMemoryException` - Insufficient GPU memory
- `ArgumentException` - Invalid memory handle
- `ArgumentOutOfRangeException` - Out-of-bounds access
- `InvalidOperationException` - GPU unavailable or allocation failed

## Best Practices

### 1. Resource Management
```csharp
// Always release memory when done
try
{
    var handle = await grain.AllocateAsync(size);
    // Use handle...
}
finally
{
    await grain.ReleaseAsync(handle);
}
```

### 2. Batch Operations
```csharp
// Prefer larger transfers over many small ones
var largeBuffer = new float[1024 * 1024]; // 4 MB
await grain.WriteAsync(handle, largeBuffer); // One large transfer
```

### 3. Memory Type Selection
```csharp
// Choose appropriate memory type for workload
GpuMemoryType.Default   // Most kernels
GpuMemoryType.Pinned    // Frequent CPU↔GPU transfers
GpuMemoryType.Shared    // Occasional CPU access during compute
```

### 4. Monitoring
```csharp
// Regularly check memory usage
var info = await grain.GetMemoryInfoAsync();
if (info.UtilizationPercentage > 90.0)
{
    // Consider cleanup or smaller allocations
}
```

## Integration with Orleans

### Grain Lifecycle
```csharp
// Memory persists across activations
var grain = grainFactory.GetGrain<IGpuResidentGrain<float>>("key");
await grain.AllocateAsync(size); // Activation 1

// Grain deactivates...

var grain2 = grainFactory.GetGrain<IGpuResidentGrain<float>>("key");
var info = await grain2.GetMemoryInfoAsync(); // Activation 2
// Memory still allocated!
```

### Placement Strategies
```csharp
// GPU-aware placement (future feature)
[GpuAwarePlacement(MinMemoryMB = 1024)]
public class MyGpuGrain : Grain, IGpuResidentGrain<float>
{
    // Placed on silos with sufficient GPU memory
}
```

## Further Reading

- [Orleans Documentation](https://learn.microsoft.com/en-us/dotnet/orleans/)
- [GPU Memory Management Best Practices](../../docs/MEMORY_MANAGEMENT.md)
- [Kernel Registration Guide](../../docs/KERNEL_REGISTRATION.md)
- [Performance Tuning](../../docs/PERFORMANCE.md)

## Support

For issues or questions:
- GitHub Issues: [Orleans.GpuBridge.Core/issues](https://github.com/your-org/Orleans.GpuBridge.Core/issues)
- Documentation: [docs/](../../docs/)

## License

This sample is part of Orleans.GpuBridge.Core and follows the project's license terms.

---

**Orleans.GpuBridge.Core** - Bringing GPU acceleration to distributed computing.
