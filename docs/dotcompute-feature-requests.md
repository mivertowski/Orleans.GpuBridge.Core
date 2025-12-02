# DotCompute Feature Requests

## ✅ SOLVED: Runtime C#-to-CUDA Translation

**Status**: **RESOLVED** - Using `RingKernelHandlerTranslator` from `DotCompute.Backends.CUDA`
**Solution**: Pass `RingKernelHandlerTranslator` to `CudaRingKernelCompiler` constructor

### How It Was Resolved

The compiler has **two translation strategies**:
1. **Strategy 1** (Legacy): Uses `DotCompute.Generators.MemoryPack.HandlerTranslationService` - requires DLL
2. **Strategy 2** (Unified API): Uses `RingKernelHandlerTranslator` from `DotCompute.Backends.CUDA` - ✅ **Works!**

**Working Code:**
```csharp
var handlerTranslator = new RingKernelHandlerTranslator(logger);
var compiler = new CudaRingKernelCompiler(logger, discovery, stubGen, serializerGen, handlerTranslator);
```

**Test Results:**
- ✅ C#-to-CUDA translation: **SUCCESS** (32KB CUDA code generated)
- ✅ PTX compilation: **SUCCESS** (108KB PTX)
- ✅ Device runtime linking: **SUCCESS** (263KB CUBIN)
- ✅ Queue initialization: **SUCCESS**
- ⚠️ Kernel launch/activation: **HANGS** (separate issue)

## 1. ~~Runtime Access to DotCompute.Generators Assembly~~ (NO LONGER NEEDED)

**Priority**: ~~High~~ **RESOLVED**
**Component**: DotCompute.Backends.CUDA, DotCompute.Generators
**Status**: ~~Blocking GPU-native actor testing~~ **FIXED - Use Strategy 2**

### Problem

The `CudaRingKernelCompiler` requires runtime access to `DotCompute.Generators.dll` to translate C# handler code to CUDA, but the generators package is configured as a build-time analyzer only:

```xml
<PackageReference Include="DotCompute.Generators"
                  Version="0.5.1"
                  OutputItemType="Analyzer"
                  ReferenceOutputAssembly="false"
                  PrivateAssets="all" />
```

### Error

```
System.IO.FileNotFoundException: Could not load file or assembly 'DotCompute.Generators, Version=0.5.1.0, Culture=neutral, PublicKeyToken=null'.
The system cannot find the file specified.

at DotCompute.Backends.CUDA.RingKernels.CudaRingKernelCompiler.TryTranslateCSharpHandler(String handlerName, DiscoveredRingKernel kernel)
```

### Current Workarounds

None available - GPU testing is blocked.

### Requested Solution

One of:
1. **Separate runtime translation package**: Create `DotCompute.Generators.Runtime` with translation APIs
2. **Include translators in main package**: Move C#-to-CUDA translation into `DotCompute.Backends.CUDA`
3. **Dual-mode generators**: Support both compile-time source generation and runtime translation

### Impact

- **Blocks**: GPU-native actor testing on real hardware
- **Affects**: All projects using unified ring kernel API with inline handlers
- **Workaround effort**: Requires manually writing CUDA `.cu` files instead of using inline C# handlers

### Test Environment

- **Hardware**: NVIDIA RTX 2000 Ada (Compute Capability 8.9)
- **OS**: Native Linux (not WSL2)
- **CUDA**: 13.0 compatible (Driver 580.95.05)
- **DotCompute**: 0.5.1

### Additional Context

The unified ring kernel API (`[RingKernel]` attribute with inline handler methods) is the recommended DotCompute pattern. Making this work requires runtime C#-to-CUDA translation, which is currently not available.

---

## 2. CPU Backend Assembly Registration Support

**Priority**: Medium
**Component**: DotCompute.Backends.CPU
**Status**: Blocks CPU message passing tests

### Problem

The `CpuRingKernelRuntime` doesn't provide a `RegisterAssembly` method for kernel discovery, unlike `CudaRingKernelRuntime`. This makes it impossible to test unified ring kernels consistently across backends.

### Error

```csharp
var cpuRuntime = new CpuRingKernelRuntime(logger);
cpuRuntime.RegisterAssembly(typeof(VectorAddRingKernel).Assembly);
// Error CS1061: 'CpuRingKernelRuntime' does not contain a definition for 'RegisterAssembly'
```

### Current Behavior

- CPU backend launches kernels but detects wrong message types:
  ```
  Detected message types for kernel 'vectoradd_processor': Input=Byte, Output=Byte
  ```
- Should detect: `Input=VectorAddProcessorRingRequest, Output=VectorAddProcessorRingResponse`
- Named queues cannot be found, causing message passing to fail

### Requested Solution

Add `RegisterAssembly` method to `CpuRingKernelRuntime` to match `CudaRingKernelRuntime` API:

```csharp
public void RegisterAssembly(Assembly assembly)
{
    // Discover ring kernels in assembly and register message types
}
```

### Impact

- **Blocks**: CPU backend message passing tests
- **Workaround**: None - kernels can be launched but can't process messages
- **Consistency**: CPU and CUDA backends have different discovery mechanisms

---

*Generated: 2025-12-01*
*Reporter: Orleans.GpuBridge.Core validation tests*
