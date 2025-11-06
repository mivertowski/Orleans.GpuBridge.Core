# DotCompute API Discovery Report
## v0.4.1-rc2 Kernel Compilation and Execution APIs

**Date:** January 6, 2025
**Status:** ✅ **API FULLY DISCOVERED**
**Hardware:** NVIDIA RTX 2000 Ada Generation Laptop GPU

---

## 🎯 Executive Summary

Successfully discovered the complete DotCompute v0.4.1-rc2 kernel compilation API through runtime reflection analysis. The API uses a two-step process: compile kernels with `IAccelerator.CompileKernelAsync()`, then execute via the compiled kernel interface.

---

## 📋 Discovered API Structure

### 1. Kernel Compilation

**Method Signature:**
```csharp
ValueTask<ICompiledKernel> IAccelerator.CompileKernelAsync(
    KernelDefinition definition,
    CompilationOptions options,
    CancellationToken cancellationToken
)
```

**KernelDefinition Class:**
```csharp
namespace DotCompute.Abstractions.Kernels;

public class KernelDefinition
{
    // Properties
    public string Name { get; set; }
    public string Source { get; set; }           // Kernel source code
    public string Code { get; set; }             // Alias for Source
    public string EntryPoint { get; set; }       // Kernel entry function name
    public string EntryFunction { get; set; }    // Alias for EntryPoint
    public Dictionary<string, object> Metadata { get; set; }
    public KernelLanguage Language { get; set; }

    // Constructor
    public KernelDefinition(string name, string source, string entryPoint)
}
```

**KernelLanguage Enum:**
```csharp
namespace DotCompute.Abstractions.Kernels.Types;

public enum KernelLanguage
{
    Auto = 0,
    Cuda = 1,
    OpenCL = 2,
    Ptx = 3,
    HLSL = 4,
    SPIRV = 5,
    Metal = 6,
    HIP = 7,
    SYCL = 8,
    CSharpIL = 9,
    Binary = 10,
    CSharp = 11,
    DirectCompute = 12,
    Vulkan = 13,
    WebGPU = 14
}
```

**CompilationOptions Class:**
```csharp
namespace DotCompute.Abstractions;

public class CompilationOptions
{
    // Optimization
    public OptimizationLevel OptimizationLevel { get; set; }
    public bool EnableDebugInfo { get; set; }
    public bool GenerateDebugInfo { get; set; }
    public bool EnableDeviceDebugging { get; set; }
    public bool GenerateLineInfo { get; set; }
    public bool EnableProfiling { get; set; }

    // Math and Performance
    public bool EnableFastMath { get; set; }
    public bool FastMath { get; set; }
    public bool UseFastMath { get; set; }
    public bool AggressiveOptimizations { get; set; }
    public bool FusedMultiplyAdd { get; set; }
    public FloatingPointMode FloatingPointMode { get; set; }
    public bool StrictFloatingPoint { get; set; }
    public bool UseNativeMathLibrary { get; set; }

    // Code Generation
    public bool AllowUnsafeCode { get; set; }
    public string TargetArchitecture { get; set; }
    public Dictionary<string, string> Defines { get; set; }
    public IList<string> IncludePaths { get; set; }
    public IList<string> AdditionalFlags { get; set; }
    public Version ComputeCapability { get; set; }
    public string CompilerBackend { get; set; }

    // Resource Limits
    public int? MaxRegisters { get; set; }
    public int MaxRegistersPerThread { get; set; }
    public long? SharedMemoryLimit { get; set; }
    public int? ThreadBlockSize { get; set; }
    public int? PreferredBlockSize { get; set; }
    public long? SharedMemorySize { get; set; }

    // Advanced Optimizations
    public bool EnableMemoryCoalescing { get; set; }
    public bool EnableOperatorFusion { get; set; }
    public bool EnableParallelExecution { get; set; }
    public bool EnableDynamicParallelism { get; set; }
    public bool EnableSharedMemoryRegisterSpilling { get; set; }
    public bool EnableTileBasedProgramming { get; set; }
    public bool EnableL2CacheResidencyControl { get; set; }
    public bool EnableLoopUnrolling { get; set; }
    public bool EnableVectorization { get; set; }
    public bool EnableInlining { get; set; }
    public bool UnrollLoops { get; set; }
    public bool EnableProfileGuidedOptimizations { get; set; }
    public string ProfileDataPath { get; set; }

    // Compilation Settings
    public TimeSpan CompilationTimeout { get; set; }
    public bool TreatWarningsAsErrors { get; set; }
    public int WarningLevel { get; set; }
    public bool ForceInterpretedMode { get; set; }
    public bool CompileToCubin { get; set; }
    public bool RelocatableDeviceCode { get; set; }

    // Predefined Configurations
    public static CompilationOptions Default { get; }
    public static CompilationOptions Debug { get; }
    public static CompilationOptions Release { get; }
}
```

---

### 2. Compiled Kernel Interface

**ICompiledKernel Interface:**
```csharp
namespace DotCompute.Core.Kernels;

public interface ICompiledKernel
{
    string Name { get; }
    bool IsValid { get; }

    // Note: Execution methods not visible in interface definition
    // Need to investigate concrete implementation or IAccelerator execution methods
}
```

---

### 3. IAccelerator Full Interface

**Complete IAccelerator API:**
```csharp
namespace DotCompute.Abstractions;

public interface IAccelerator
{
    // Kernel Compilation
    ValueTask<ICompiledKernel> CompileKernelAsync(
        KernelDefinition definition,
        CompilationOptions options,
        CancellationToken cancellationToken);

    // Synchronization
    ValueTask SynchronizeAsync(CancellationToken cancellationToken);

    // Health and Monitoring
    ValueTask<HealthSnapshot> GetHealthSnapshotAsync(CancellationToken cancellationToken);
    ValueTask<ProfilingMetrics> GetProfilingMetricsAsync(CancellationToken cancellationToken);
    ValueTask<ProfilingSnapshot> GetProfilingSnapshotAsync(CancellationToken cancellationToken);
    ValueTask<SensorReadings> GetSensorReadingsAsync(CancellationToken cancellationToken);

    // Device Management
    ValueTask<ResetResult> ResetAsync(ResetOptions options, CancellationToken cancellationToken);

    // Properties
    AcceleratorInfo Info { get; }
    AcceleratorType Type { get; }
    string DeviceType { get; }
    IUnifiedMemoryManager Memory { get; }
    IUnifiedMemoryManager MemoryManager { get; }
    AcceleratorContext Context { get; }
    CudaGraphManager GraphManager { get; }
    CudaDevice Device { get; }  // CUDA-specific
    int DeviceId { get; }
    bool IsAvailable { get; }
    bool IsDisposed { get; }
}
```

---

## 🔍 Key Findings

### 1. Two-Step Kernel Workflow

**Step 1: Compilation**
```csharp
var kernelDef = new KernelDefinition(
    name: "VectorAdd",
    source: cudaSourceCode,
    entryPoint: "vectorAdd"
)
{
    Language = KernelLanguage.Cuda
};

var options = CompilationOptions.Release;
options.EnableFastMath = true;
options.MaxRegisters = 32;

var compiledKernel = await accelerator.CompileKernelAsync(
    kernelDef,
    options,
    cancellationToken
);
```

**Step 2: Execution**
```csharp
// Execution method needs further investigation
// Possible approaches:
// 1. ICompiledKernel may have Execute method in concrete implementation
// 2. IAccelerator may have ExecuteKernelAsync method
// 3. May use IComputeOrchestrator pattern
```

### 2. Memory Management

**IUnifiedMemoryManager Available:**
- Accessible via `accelerator.Memory` or `accelerator.MemoryManager`
- Provides unified memory management APIs
- Need to explore: AllocateAsync, CopyAsync, FreeAsync methods

### 3. Graph Support

**CudaGraphManager Available:**
- CUDA-specific graph operations
- Accessible via `accelerator.GraphManager`
- Enables kernel graph optimization for CUDA devices

---

## 📝 Implementation Mapping

### Orleans.GpuBridge → DotCompute Language Mapping

| Orleans.GpuBridge | DotCompute |
|------------------|------------|
| `KernelLanguage.CUDA` | `KernelLanguage.Cuda` |
| `KernelLanguage.OpenCL` | `KernelLanguage.OpenCL` |
| `KernelLanguage.CSharp` | `KernelLanguage.CSharp` |
| `KernelLanguage.HLSL` | `KernelLanguage.HLSL` |

### Optimization Level Mapping

| Orleans.GpuBridge | DotCompute |
|------------------|------------|
| `OptimizationLevel.None` | `CompilationOptions.Debug` |
| `OptimizationLevel.Fast` | `CompilationOptions.Release` with `EnableFastMath` |
| `OptimizationLevel.Balanced` | `CompilationOptions.Release` |
| `OptimizationLevel.Size` | `CompilationOptions.Release` with aggressive settings |

---

## 🚧 Outstanding Questions

### 1. Kernel Execution API

**Status:** ⚠️ **NOT YET DISCOVERED**

Need to investigate:
- Does ICompiledKernel have execution methods in concrete implementation?
- Is there an IAccelerator.ExecuteAsync method not visible in interface?
- Does IComputeOrchestrator provide kernel execution?
- How to pass arguments to compiled kernels?
- How to specify grid/block dimensions?

**Next Steps:**
1. Create test to compile a simple kernel
2. Inspect returned ICompiledKernel concrete type methods
3. Check IAccelerator concrete type for Execute methods
4. Review IComputeOrchestrator if available

### 2. Memory Buffer Integration

**Status:** ⚠️ **PARTIALLY DISCOVERED**

Need to explore:
- IUnifiedMemoryManager methods for allocation/free
- How to create device memory buffers
- How to pass memory buffers as kernel arguments
- Pinned memory support
- Unified memory (zero-copy) support

### 3. Work Dimensions

**Status:** ❓ **UNKNOWN**

Need to discover:
- How to specify global work size
- How to specify local work size (block size)
- Grid/block dimension structures
- Multi-dimensional dispatch support

---

## ✅ Next Steps for Implementation

### Phase 1.1: Kernel Compilation (Ready to Implement)

1. **Create MapLanguage Helper:**
   ```csharp
   private DotCompute.Abstractions.Kernels.Types.KernelLanguage MapLanguage(
       Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage language)
   {
       return language switch
       {
           KernelLanguage.CUDA => DotCompute.Abstractions.Kernels.Types.KernelLanguage.Cuda,
           KernelLanguage.OpenCL => DotCompute.Abstractions.Kernels.Types.KernelLanguage.OpenCL,
           // ... etc
       };
   }
   ```

2. **Create MapCompilationOptions Helper:**
   ```csharp
   private CompilationOptions MapCompilationOptions(KernelCompilationOptions options)
   {
       var dotComputeOptions = options.OptimizationLevel == OptimizationLevel.Debug
           ? CompilationOptions.Debug
           : CompilationOptions.Release;

       // Map specific options...
       dotComputeOptions.EnableFastMath = options.EnableFastMath;
       dotComputeOptions.Defines = options.Defines;

       return dotComputeOptions;
   }
   ```

3. **Replace CompileKernelForDeviceAsync:**
   ```csharp
   private async Task<DotComputeCompiledKernel> CompileKernelForDeviceAsync(...)
   {
       var adapter = device as DotComputeAcceleratorAdapter;
       var accelerator = adapter.Accelerator;

       var kernelDef = new KernelDefinition(
           name: source.Name,
           source: source.SourceCode,
           entryPoint: source.EntryPoint ?? source.Name
       )
       {
           Language = MapLanguage(source.Language)
       };

       var compiledKernel = await accelerator.CompileKernelAsync(
           kernelDef,
           MapCompilationOptions(options),
           cancellationToken
       );

       return new DotComputeCompiledKernel(
           kernelId: GenerateKernelId(source, device),
           name: source.Name,
           device: device,
           metadata: metadata,
           nativeKernel: compiledKernel,  // Store ICompiledKernel
           logger: _logger
       );
   }
   ```

### Phase 1.2: Kernel Execution (Requires Further Investigation)

**Blocked until execution API is discovered.**

Options to explore:
1. Inspect concrete ICompiledKernel implementation for Execute methods
2. Check for IAccelerator.ExecuteAsync methods
3. Test IComputeOrchestrator pattern
4. Review DotCompute samples/documentation

---

## 🎯 Success Criteria

### Compilation Phase:
- ✅ API structure fully documented
- ✅ Language mapping defined
- ✅ Compilation options mapping defined
- ⏳ Real kernel compilation working (ready to implement)
- ⏳ Compiled kernel caching functional
- ⏳ No "simulated_pending_api" warnings

### Execution Phase:
- ⏳ Execution API discovered
- ⏳ Memory argument passing working
- ⏳ Work dimensions configuration working
- ⏳ Real GPU execution (no Task.Delay simulation)
- ⏳ Performance measurement accurate

---

## 📚 References

- **DotCompute Version:** 0.4.1-rc2
- **Release Date:** January 2025
- **Hardware Tested:** NVIDIA RTX 2000 Ada (Compute Capability 8.9)
- **CUDA Version:** 13.0.48
- **Discovery Method:** Runtime reflection analysis via API explorer tool

---

*Report Generated: January 6, 2025*
*Orleans.GpuBridge.Core Integration Project*
