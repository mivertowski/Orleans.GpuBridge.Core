# DotCompute Device Enumeration Research Report

**Date:** 2025-11-05
**Project:** Orleans.GpuBridge.Core
**Component:** Orleans.GpuBridge.Backends.DotCompute
**Research Focus:** IComputeOrchestrator API and Device Discovery Patterns

---

## Executive Summary

This research was conducted to understand DotCompute's device enumeration API for implementing GPU device discovery in `DotComputeDeviceManager.cs`. Based on codebase analysis and technical documentation review, **DotCompute appears to be a proprietary/internal framework** developed specifically for this project rather than a publicly available NuGet package.

**Key Finding:** The DotCompute framework (v0.2.0-alpha) is referenced in the project but lacks public documentation. Implementation must be inferred from existing abstractions and similar GPU compute frameworks.

---

## Research Methodology

### 1. Web Search Results
- **Publicly Available GPU Compute Libraries for .NET:**
  - **ILGPU** - Modern GPU compiler for .NET (CUDA, OpenCL, CPU)
  - **ComputeSharp** - DirectX 12 compute shaders in C#
  - **ManagedCUDA** - .NET wrapper for NVIDIA CUDA
  - **Hybridizer** - C# to GPU code compiler

- **DotCompute Findings:**
  - No public NuGet packages found
  - No GitHub repository found
  - No Stack Overflow discussions
  - No official documentation available online

### 2. Codebase Analysis
- **Project Reference:** `Orleans.GpuBridge.Backends.DotCompute.csproj`
- **NuGet Dependencies:**
  ```xml
  <PackageReference Include="DotCompute.Core" Version="0.2.0-alpha" />
  <PackageReference Include="DotCompute.Runtime" Version="0.2.0-alpha" />
  <PackageReference Include="DotCompute.Backends.CUDA" Version="0.2.0-alpha" />
  <PackageReference Include="DotCompute.Backends.OpenCL" Version="0.2.0-alpha" />
  <PackageReference Include="DotCompute.Backends.Metal" Version="0.2.0-alpha" />
  ```

- **Technical Requirements (from TECHNICAL_REQUIREMENTS.md):**
  - Repository: https://github.com/mivertowski/DotCompute (not found publicly)
  - Build from source mentioned but repository not accessible
  - Supports CUDA, OpenCL, DirectCompute, and Metal backends

---

## Inferred Architecture from Existing Code

### Device Manager Pattern

Based on the existing stub implementation in `DotComputeDeviceManager.cs`, the expected pattern is:

```csharp
// Current CPU Fallback Implementation
public async Task InitializeAsync(CancellationToken cancellationToken = default)
{
    // TODO: Implement device discovery using DotCompute IComputeOrchestrator
    _devices.Add(CreateCpuFallbackDevice());
}
```

### Expected DotCompute API Pattern

Based on similar frameworks (ILGPU, ComputeSharp, OpenCL.NET), the likely API pattern is:

```csharp
// Hypothetical IComputeOrchestrator API
public interface IComputeOrchestrator : IDisposable
{
    // Device enumeration
    IReadOnlyList<IComputeDevice> EnumerateDevices();
    IReadOnlyList<IComputeDevice> EnumerateDevices(DeviceType type);

    // Device filtering
    IComputeDevice GetDefaultDevice();
    IComputeDevice GetDevice(int index);
    IComputeDevice GetDevice(string deviceId);

    // Context creation
    IComputeContext CreateContext(IComputeDevice device);
    IComputeContext CreateContext(IComputeDevice device, ContextOptions options);
}

// Device information model (inferred)
public interface IDotComputeDevice
{
    // Basic properties
    string DeviceId { get; }
    int Index { get; }
    string Name { get; }
    DeviceBackend Backend { get; } // CUDA, OpenCL, Metal, CPU

    // Hardware properties
    string Vendor { get; }
    string Architecture { get; }
    Version ComputeCapability { get; }

    // Memory information
    long TotalMemoryBytes { get; }
    long AvailableMemoryBytes { get; }

    // Compute capabilities
    int ComputeUnits { get; }
    int MaxWorkGroupSize { get; }
    int[] MaxWorkGroupDimensions { get; }
    int MaxClockFrequencyMHz { get; }
    int WarpSize { get; }

    // Status and capabilities
    DeviceStatus Status { get; }
    IReadOnlyDictionary<string, object> Properties { get; }
    bool SupportsFeature(string feature);
}

// Likely backend enum
public enum DeviceBackend
{
    CPU,
    CUDA,
    OpenCL,
    Metal,
    DirectCompute
}
```

---

## Property Mapping Guide

### DotCompute Device → IComputeDevice Interface

| IComputeDevice Property | DotCompute Source | Type Mapping | Notes |
|------------------------|-------------------|--------------|-------|
| `DeviceId` | `IDotComputeDevice.DeviceId` | string → string | Direct mapping |
| `Index` | `IDotComputeDevice.Index` | int → int | Direct mapping |
| `Name` | `IDotComputeDevice.Name` | string → string | Direct mapping |
| `Type` | `IDotComputeDevice.Backend` | DeviceBackend → DeviceType | Enum conversion needed |
| `Vendor` | `IDotComputeDevice.Vendor` | string → string | Direct mapping |
| `Architecture` | `IDotComputeDevice.Architecture` | string → string | Direct mapping |
| `ComputeCapability` | `IDotComputeDevice.ComputeCapability` | Version → Version | Direct mapping |
| `TotalMemoryBytes` | `IDotComputeDevice.TotalMemoryBytes` | long → long | Direct mapping |
| `AvailableMemoryBytes` | `IDotComputeDevice.AvailableMemoryBytes` | long → long | Direct mapping |
| `ComputeUnits` | `IDotComputeDevice.ComputeUnits` | int → int | Direct mapping |
| `MaxClockFrequencyMHz` | `IDotComputeDevice.MaxClockFrequencyMHz` | int → int | Direct mapping |
| `MaxThreadsPerBlock` | `IDotComputeDevice.MaxWorkGroupSize` | int → int | Property name differs |
| `MaxWorkGroupDimensions` | `IDotComputeDevice.MaxWorkGroupDimensions` | int[] → int[] | Direct mapping |
| `WarpSize` | `IDotComputeDevice.WarpSize` | int → int | Direct mapping |
| `Properties` | `IDotComputeDevice.Properties` | Dictionary → Dictionary | Direct mapping |

### DeviceBackend → DeviceType Enum Conversion

```csharp
private static DeviceType MapBackendToDeviceType(DeviceBackend backend)
{
    return backend switch
    {
        DeviceBackend.CPU => DeviceType.CPU,
        DeviceBackend.CUDA => DeviceType.CUDA,
        DeviceBackend.OpenCL => DeviceType.OpenCL,
        DeviceBackend.Metal => DeviceType.Metal,
        DeviceBackend.DirectCompute => DeviceType.GPU, // Generic GPU
        _ => throw new ArgumentException($"Unknown backend: {backend}")
    };
}
```

---

## Recommended Implementation Approach

### Phase 1: Stub with CPU Fallback (CURRENT STATE ✅)

The current implementation provides a CPU fallback device and serves as a foundation:

```csharp
private static IComputeDevice CreateCpuFallbackDevice()
{
    return new CpuFallbackDevice
    {
        DeviceId = "cpu-0",
        Name = "CPU (Fallback)",
        Type = DeviceType.CPU,
        // ... other properties
    };
}
```

### Phase 2: DotCompute Integration (RECOMMENDED NEXT STEP)

**Prerequisites:**
1. Verify DotCompute NuGet packages are restorable
2. Check if DotCompute source code is available
3. Review actual DotCompute API via IntelliSense

**Implementation Pattern:**

```csharp
using DotCompute.Core; // Hypothetical namespace
using DotCompute.Runtime; // Hypothetical namespace

public async Task InitializeAsync(CancellationToken cancellationToken = default)
{
    if (_isInitialized) return;

    _logger.LogInformation("Discovering devices via DotCompute");

    try
    {
        // Step 1: Create compute orchestrator
        using var orchestrator = new ComputeOrchestrator(_options);

        // Step 2: Enumerate all devices
        var dotComputeDevices = orchestrator.EnumerateDevices();

        // Step 3: Map to IComputeDevice interface
        foreach (var dcDevice in dotComputeDevices)
        {
            var device = MapDotComputeDevice(dcDevice);
            _devices.Add(device);

            _logger.LogInformation(
                "Discovered {Backend} device: {Name} (Index: {Index}, Memory: {Memory:N0} GB)",
                device.Type, device.Name, device.Index, device.TotalMemoryBytes / 1_073_741_824.0);
        }

        // Step 4: Fallback to CPU if no GPU devices found
        if (_devices.Count == 0)
        {
            _logger.LogWarning("No GPU devices found, adding CPU fallback");
            _devices.Add(CreateCpuFallbackDevice());
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Failed to enumerate DotCompute devices, using CPU fallback");
        _devices.Add(CreateCpuFallbackDevice());
    }

    _isInitialized = true;
    _logger.LogInformation("Device discovery complete: {DeviceCount} device(s) found", _devices.Count);
}

private IComputeDevice MapDotComputeDevice(IDotComputeDevice dcDevice)
{
    return new DotComputeDeviceAdapter
    {
        DeviceId = dcDevice.DeviceId,
        Index = dcDevice.Index,
        Name = dcDevice.Name,
        Type = MapBackendToDeviceType(dcDevice.Backend),
        Vendor = dcDevice.Vendor,
        Architecture = dcDevice.Architecture,
        ComputeCapability = new Version(dcDevice.ComputeCapability.Major, dcDevice.ComputeCapability.Minor),
        TotalMemoryBytes = dcDevice.TotalMemoryBytes,
        AvailableMemoryBytes = dcDevice.AvailableMemoryBytes,
        ComputeUnits = dcDevice.ComputeUnits,
        MaxClockFrequencyMHz = dcDevice.MaxClockFrequencyMHz,
        // ... additional properties
    };
}
```

### Phase 3: Advanced Features (FUTURE)

- Device filtering by capability
- Dynamic device hotplug detection
- Device health monitoring
- Multi-GPU topology discovery

---

## Alternative Implementation Strategies

### Option A: ILGPU Backend (Proven Technology)

Since DotCompute documentation is unavailable, consider using **ILGPU** which is:
- ✅ Well-documented
- ✅ Production-ready
- ✅ Supports CUDA and OpenCL
- ✅ Active community support

**Device Enumeration with ILGPU:**

```csharp
using ILGPU;
using ILGPU.Runtime;

public async Task InitializeAsync(CancellationToken cancellationToken = default)
{
    using var context = Context.Create(builder => builder.Default().EnableAlgorithms());

    // Enumerate CUDA devices
    foreach (var device in context.GetCudaDevices())
    {
        _devices.Add(MapILGPUDevice(device, DeviceType.CUDA));
    }

    // Enumerate OpenCL devices
    foreach (var device in context.GetCLDevices())
    {
        _devices.Add(MapILGPUDevice(device, DeviceType.OpenCL));
    }

    // CPU fallback
    var cpuDevice = context.GetCPUDevice(0);
    _devices.Add(MapILGPUDevice(cpuDevice, DeviceType.CPU));
}
```

### Option B: Hybrid Approach

- Use ILGPU for device enumeration
- Map to Orleans.GpuBridge abstractions
- Maintain DotCompute compatibility for future migration

---

## Code Examples from Similar Frameworks

### ILGPU Device Properties

```csharp
var accelerator = context.CreateCudaAccelerator(0);
var properties = new
{
    Name = accelerator.Name,
    MemorySize = accelerator.MemorySize,
    MaxGridSize = accelerator.MaxGridSize,
    MaxGroupSize = accelerator.MaxGroupSize,
    WarpSize = accelerator.WarpSize,
    NumMultiprocessors = accelerator.NumMultiprocessors
};
```

### ComputeSharp Device Query

```csharp
using ComputeSharp;

var device = GraphicsDevice.GetDefault();
var properties = new
{
    Name = device.Name,
    DedicatedMemorySize = device.DedicatedMemorySize,
    SharedMemorySize = device.SharedSystemMemory,
    Architecture = device.Description
};
```

### OpenCL Device Enumeration

```csharp
using OpenCL.Net;

foreach (var platform in Cl.GetPlatformIDs())
{
    var devices = Cl.GetDeviceIDs(platform, DeviceType.All);
    foreach (var device in devices)
    {
        var name = Cl.GetDeviceInfo(device, DeviceInfo.Name).ToString();
        var memory = (long)Cl.GetDeviceInfo(device, DeviceInfo.GlobalMemSize);
        var computeUnits = (int)Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits);
    }
}
```

---

## Critical Questions to Resolve

### 1. DotCompute Package Availability
**Question:** Can the DotCompute NuGet packages be restored?

**Investigation Steps:**
```bash
cd src/Orleans.GpuBridge.Backends.DotCompute
dotnet restore
dotnet build
```

**Expected Outcomes:**
- ✅ Success → DotCompute is available, proceed with implementation
- ❌ Failure → Consider migrating to ILGPU backend

### 2. DotCompute API Discovery
**Question:** What is the actual API surface of DotCompute?

**Investigation Steps:**
```csharp
// Use IntelliSense to discover API
using DotCompute.Core;
using DotCompute.Runtime;

// List available types
var orchestratorType = typeof(IComputeOrchestrator);
var methods = orchestratorType.GetMethods();
```

### 3. Backend Selection Strategy
**Question:** Should we prioritize DotCompute or switch to ILGPU?

**Decision Matrix:**

| Criterion | DotCompute | ILGPU |
|-----------|------------|-------|
| Documentation | ❌ None | ✅ Excellent |
| Community Support | ❌ Unknown | ✅ Active |
| Production Readiness | ⚠️ Alpha | ✅ Stable |
| Feature Completeness | ⚠️ Unknown | ✅ Comprehensive |
| Orleans Integration | ✅ Designed for it | ⚠️ Needs adapter |

**Recommendation:** If DotCompute packages cannot be restored, migrate to ILGPU.

---

## Implementation Checklist

### Immediate Actions (Priority 1)
- [ ] Verify DotCompute package restoration
- [ ] Document actual DotCompute API via IntelliSense
- [ ] Create device adapter class mapping DotCompute → IComputeDevice
- [ ] Implement device enumeration logic
- [ ] Add comprehensive logging

### Testing Strategy (Priority 2)
- [ ] Unit tests for device mapping
- [ ] Integration tests with real GPU hardware
- [ ] Fallback testing (no GPU scenarios)
- [ ] Multi-GPU enumeration tests

### Documentation (Priority 3)
- [ ] API documentation for device manager
- [ ] Usage examples
- [ ] Troubleshooting guide
- [ ] Performance tuning recommendations

---

## Conclusion

**Current State:** The DotCompute backend has a solid architectural foundation but requires actual device enumeration implementation.

**Blocker:** Lack of DotCompute documentation and uncertain package availability.

**Recommended Path Forward:**

1. **Attempt DotCompute integration:**
   - Verify package restoration
   - Use IntelliSense to discover API
   - Implement based on similar framework patterns

2. **Fallback to ILGPU if needed:**
   - Well-documented alternative
   - Production-proven
   - Easy migration path

3. **Maintain abstraction layer:**
   - Keep IDeviceManager interface unchanged
   - Backend implementation can be swapped
   - Orleans integration remains consistent

---

## References

### Similar Framework Documentation
- **ILGPU Documentation:** https://ilgpu.net/
- **ComputeSharp GitHub:** https://github.com/Sergio0694/ComputeSharp
- **OpenCL.NET:** https://github.com/Thraka/OpenCL.Net
- **ManagedCUDA:** https://github.com/kunzmi/managedCuda

### Orleans.GpuBridge Codebase
- **DotComputeDeviceManager.cs:** `/src/Orleans.GpuBridge.Backends.DotCompute/Adapters/DotComputeDeviceManager.cs`
- **IDeviceManager Interface:** `/src/Orleans.GpuBridge.Abstractions/Providers/IDeviceManager.cs`
- **IComputeDevice Interface:** `/src/Orleans.GpuBridge.Abstractions/Providers/IComputeDevice.cs`
- **Technical Requirements:** `/docs/planning/TECHNICAL_REQUIREMENTS.md`

---

**Research Status:** COMPLETE ✅
**Next Action:** Verify DotCompute package availability and document actual API
**Estimated Implementation Time:** 4-8 hours (with DotCompute), 8-16 hours (ILGPU migration)
