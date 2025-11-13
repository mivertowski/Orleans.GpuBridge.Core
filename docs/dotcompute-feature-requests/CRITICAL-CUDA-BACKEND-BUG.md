# CRITICAL BUG: CUDA Backend LibraryImport Missing EntryPoint

**Date**: November 13, 2025
**Severity**: 🔴 **BLOCKER** - Prevents all CUDA backend usage
**Component**: DotCompute.Backends.CUDA
**File**: `src/Backends/DotCompute.Backends.CUDA/Types/Native/CudaApi.cs`

---

## Executive Summary

The CUDA backend cannot execute **any** ring kernels due to incorrect LibraryImport declarations. The methods are named with `_Internal` suffix but don't specify the actual CUDA Driver API function names via `EntryPoint` parameter.

## Error Details

### Runtime Error

```text
System.EntryPointNotFoundException: Unable to find an entry point named 'cuMemAlloc_Internal' in shared library 'cuda'.
```

### Test Environment
- **GPU**: NVIDIA RTX 2000 Ada Generation Laptop GPU
- **Driver**: 581.15
- **CUDA**: 13.0
- **Compute Capability**: 8.9
- **Platform**: Linux (WSL2)
- **libcuda.so**: Available at `/usr/lib/wsl/lib/libcuda.so.1`

### Test Status
- ✅ GPU detection: **PASSED**
- ✅ Runtime creation: **PASSED**
- ✅ Wrapper instantiation: **PASSED**
- ❌ Kernel launch: **FAILED** (cuMemAlloc_Internal not found)

## Root Cause

### Current Code (BROKEN)
```csharp
// File: DotCompute.Backends.CUDA/Types/Native/CudaApi.cs

#if WINDOWS
    private const string CUDA_DRIVER_LIBRARY = "nvcuda";
#else
    private const string CUDA_DRIVER_LIBRARY = "cuda";  // ✅ Library name is correct
#endif

[LibraryImport(CUDA_DRIVER_LIBRARY)]
[DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
private static partial int cuMemAlloc_Internal(ref IntPtr dptr, nuint bytesize);  // ❌ Missing EntryPoint!

public static CudaError cuMemAlloc(ref IntPtr dptr, nuint bytesize)
    => (CudaError)cuMemAlloc_Internal(ref dptr, bytesize);
```

**Problem**: The `LibraryImport` generator looks for a native function called `cuMemAlloc_Internal`, but the actual CUDA Driver API function is `cuMemAlloc`.

### The Fix (REQUIRED)
```csharp
[LibraryImport(CUDA_DRIVER_LIBRARY, EntryPoint = "cuMemAlloc")]  // ✅ Maps C# method to native function
[DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
private static partial int cuMemAlloc_Internal(ref IntPtr dptr, nuint bytesize);

public static CudaError cuMemAlloc(ref IntPtr dptr, nuint bytesize)
    => (CudaError)cuMemAlloc_Internal(ref dptr, bytesize);
```

## Affected Functions

All CUDA Driver API functions in `CudaApi.cs` are affected:

### Memory Management
- `cuMemAlloc_Internal` → needs `EntryPoint = "cuMemAlloc"`
- `cuMemFree_Internal` → needs `EntryPoint = "cuMemFree"`
- `cuMemcpyHtoD_Internal` → needs `EntryPoint = "cuMemcpyHtoD"`
- `cuMemcpyDtoH_Internal` → needs `EntryPoint = "cuMemcpyDtoH"`
- `cuMemcpyDtoD_Internal` → needs `EntryPoint = "cuMemcpyDtoD"`
- `cuMemsetD8_Internal` → needs `EntryPoint = "cuMemsetD8"`
- `cuMemsetD16_Internal` → needs `EntryPoint = "cuMemsetD16"`
- `cuMemsetD32_Internal` → needs `EntryPoint = "cuMemsetD32"`

### Module Management
- `cuModuleLoad_Internal` → needs `EntryPoint = "cuModuleLoad"`
- `cuModuleUnload_Internal` → needs `EntryPoint = "cuModuleUnload"`
- `cuModuleGetFunction_Internal` → needs `EntryPoint = "cuModuleGetFunction"`

### Kernel Execution
- `cuLaunchKernel_Internal` → needs `EntryPoint = "cuLaunchKernel"`

### Context Management
- `cuCtxCreate_Internal` → needs `EntryPoint = "cuCtxCreate"`
- `cuCtxDestroy_Internal` → needs `EntryPoint = "cuCtxDestroy"`
- `cuCtxSetCurrent_Internal` → needs `EntryPoint = "cuCtxSetCurrent"`
- `cuCtxGetCurrent_Internal` → needs `EntryPoint = "cuCtxGetCurrent"`

### Device Management
- `cuDeviceGet_Internal` → needs `EntryPoint = "cuDeviceGet"`
- `cuDeviceGetCount_Internal` → needs `EntryPoint = "cuDeviceGetCount"`
- `cuDeviceGetName_Internal` → needs `EntryPoint = "cuDeviceGetName"`

### Stream Management
- `cuStreamCreate_Internal` → needs `EntryPoint = "cuStreamCreate"`
- `cuStreamDestroy_Internal` → needs `EntryPoint = "cuStreamDestroy"`
- `cuStreamSynchronize_Internal` → needs `EntryPoint = "cuStreamSynchronize"`

### Event Management
- `cuEventCreate_Internal` → needs `EntryPoint = "cuEventCreate"`
- `cuEventRecord_Internal` → needs `EntryPoint = "cuEventRecord"`
- `cuEventSynchronize_Internal` → needs `EntryPoint = "cuEventSynchronize"`
- `cuEventElapsedTime_Internal` → needs `EntryPoint = "cuEventElapsedTime"`

## Impact

### Functionality Impact
- ❌ **ALL CUDA backend operations are blocked**
- ❌ Ring kernel GPU execution impossible
- ❌ GPU memory allocation fails immediately
- ❌ GPU-native actors cannot run on actual GPUs
- ✅ CPU backend works fine (unaffected)

### Workaround
**None**. The CPU backend can be used for development/testing, but the entire purpose of Orleans.GpuBridge.Core (GPU-native actors) cannot be validated.

### Timeline Impact
- **Phase 5 GPU validation**: BLOCKED until fixed
- **Production deployment**: BLOCKED until fixed
- **Performance benchmarks**: Cannot compare CPU vs GPU

## Required Action

### Immediate (Priority 1)
1. Add `EntryPoint` parameter to **all** `LibraryImport` declarations in `CudaApi.cs`
2. Rebuild DotCompute.Backends.CUDA package
3. Publish fixed version to NuGet (suggest v0.4.3)

### Example Fix Pattern
```csharp
// Before (BROKEN):
[LibraryImport(CUDA_DRIVER_LIBRARY)]
private static partial int cuMemAlloc_Internal(ref IntPtr dptr, nuint bytesize);

// After (FIXED):
[LibraryImport(CUDA_DRIVER_LIBRARY, EntryPoint = "cuMemAlloc")]
private static partial int cuMemAlloc_Internal(ref IntPtr dptr, nuint bytesize);
```

### Testing Requirements
After fix:
1. ✅ Verify `nvidia-smi` detects GPU
2. ✅ Verify runtime creation succeeds
3. ✅ Verify kernel launch succeeds (cuMemAlloc works)
4. ✅ Verify message queue initialization
5. ✅ Verify infinite dispatch loop activates
6. ✅ Measure GPU performance vs CPU baseline

## Why This Wasn't Caught Earlier

1. **Source generator creates correct code**: The LibraryImportGenerator creates correct P/Invoke stubs
2. **Windows testing might work differently**: If DotCompute was primarily tested on Windows, the nvcuda.dll exports might have matched
3. **CPU backend masked the issue**: Development/testing on CPU backend wouldn't reveal this
4. **Linux/WSL specific**: This is the first test on Linux with actual CUDA hardware

## Related Issues

This is separate from the other API version mismatches discovered:
- ✅ `OpenCLDeviceManager` missing (documented in PHASE5-WEEK15-SDK-UPGRADE-REQUIREMENT.md)
- ✅ `Metal.RingKernels` missing (documented in PHASE5-WEEK15-SDK-UPGRADE-REQUIREMENT.md)
- 🔴 **This is a runtime bug**, not a packaging issue

## References

- **CUDA Driver API**: <https://docs.nvidia.com/cuda/cuda-driver-api/>
- **LibraryImportAttribute**: <https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.libraryimportattribute>
- **EntryPoint Parameter**: Required when C# method name differs from native function name

---

**URGENT**: This blocks all GPU-native actor validation. Please prioritize this fix for v0.4.3 release.

**Test Command**: `dotnet run --project tests/RingKernelValidation/RingKernelValidation.csproj -- cuda`

**Expected**: Ring kernel launches on GPU and processes messages at <200ns latency.

**Actual**: Crash at cuMemAlloc with EntryPointNotFoundException.
