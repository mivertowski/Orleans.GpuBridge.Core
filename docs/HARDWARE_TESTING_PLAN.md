# Hardware Testing Plan - Device Discovery

**Date**: 2025-01-06
**Phase**: 1 - Device Discovery Testing
**Status**: Ready for Execution
**Prerequisites**: ✅ Clean build, ✅ Real API integration complete

---

## Test Environment

### Hardware Available
- **GPU**: NVIDIA RTX (CUDA-capable)
- **CUDA Version**: 13.x
- **Drivers**: Latest NVIDIA drivers installed
- **CPU**: Multi-core CPU for fallback testing

### Software Stack
- **.NET**: 9.0
- **DotCompute**: v0.3.0-rc1
- **Backends**: CUDA, OpenCL, CPU
- **Orleans**: v9.2.1

---

## Test Scenarios

### Scenario 1: GPU Discovery (Primary Path)
**Objective**: Verify real GPU device is discovered and properties are correct

**Test Steps**:
1. Ensure NVIDIA GPU is available and drivers loaded
2. Run application with device discovery
3. Verify DotCompute discovers GPU via CUDA backend
4. Check device properties logged correctly

**Expected Output**:
```
[INF] Initializing DotCompute device manager with real API integration (v0.3.0-rc1)
[DBG] DotCompute AcceleratorManager created successfully
[INF] Starting DotCompute device discovery using real API
[INF] Discovered DotCompute device: dotcompute-gpu-0 - NVIDIA GeForce RTX XXXX (GPU, sm_XX)
[DBG] Device details: ComputeUnits=XX, Memory=XX.XXGB, WarpSize=32
[INF] Device discovery complete. Found 1 real device(s)
```

**Validation Checks**:
- ✅ Device ID format: `dotcompute-gpu-0`
- ✅ Device Type: `GPU`
- ✅ Architecture: `sm_XX` (CUDA compute capability)
- ✅ WarpSize: `32` (NVIDIA standard)
- ✅ ComputeUnits: > 0
- ✅ Memory: > 0 GB
- ✅ Device name contains "RTX" or GPU model

---

### Scenario 2: CPU Fallback Discovery
**Objective**: Verify CPU backend discovered when GPU unavailable

**Test Steps**:
1. Disable GPU or run on GPU-less system
2. Ensure DotCompute.Backends.CPU package loaded
3. Run device discovery
4. Verify CPU backend discovered as fallback

**Expected Output**:
```
[INF] Discovered DotCompute device: dotcompute-cpu-0 - CPU (CPU, x86_64)
[DBG] Device details: ComputeUnits=8, Memory=XX.XXGB, WarpSize=1
[INF] Device discovery complete. Found 1 real device(s)
```

**Validation Checks**:
- ✅ Device ID format: `dotcompute-cpu-0`
- ✅ Device Type: `CPU`
- ✅ Architecture: `x86_64` or `ARM64`
- ✅ WarpSize: `1` (CPU has no warp concept)
- ✅ ComputeUnits: Matches CPU core count
- ✅ Memory: System RAM size

---

### Scenario 3: Multiple Devices
**Objective**: Test discovery when multiple accelerators available

**Test Steps**:
1. Run on system with multiple GPUs or GPU + CPU backend
2. Verify all devices discovered
3. Check device indexing is sequential (0, 1, 2, ...)

**Expected Output**:
```
[INF] Discovered DotCompute device: dotcompute-gpu-0 - NVIDIA GeForce RTX XXXX (GPU, sm_XX)
[INF] Discovered DotCompute device: dotcompute-cpu-0 - CPU (CPU, x86_64)
[INF] Device discovery complete. Found 2 real device(s)
```

**Validation Checks**:
- ✅ Each device has unique ID
- ✅ Device indices sequential (0, 1, 2...)
- ✅ All devices logged with full details
- ✅ No duplicate device IDs

---

### Scenario 4: Device Health Monitoring
**Objective**: Test GetDeviceHealthAsync with real device

**Test Steps**:
1. Call `GetDeviceHealthAsync()` for discovered device
2. Verify memory usage reported
3. Check device status is `Available`

**Expected Behavior**:
```csharp
var healthInfo = await deviceManager.GetDeviceHealthAsync("dotcompute-gpu-0");

// Validation:
Assert.NotNull(healthInfo);
Assert.Equal(DeviceStatus.Available, healthInfo.Status);
Assert.InRange(healthInfo.MemoryUtilizationPercent, 0, 100);
Assert.Equal("dotcompute-gpu-0", healthInfo.DeviceId);
```

**Memory Info Validation**:
- Total memory > 0
- Allocated memory >= 0
- Utilization percentage: 0-100%
- Statistics property accessible

---

### Scenario 5: Device Properties Accuracy
**Objective**: Verify adapter correctly maps DotCompute properties

**Test Steps**:
1. Get discovered device via `GetDeviceAsync()`
2. Check all IComputeDevice properties
3. Validate property values make sense

**Properties to Verify**:
```csharp
var device = await deviceManager.GetDeviceAsync("dotcompute-gpu-0");

// Identity
✅ device.Id - Format: "dotcompute-{type}-{index}"
✅ device.Name - Contains GPU model name
✅ device.DeviceId - Matches Id
✅ device.Type - GPU or CPU enum
✅ device.Index - Sequential integer

// Capabilities
✅ device.Architecture - CUDA: "sm_XX", CPU: "x86_64"
✅ device.WarpSize - GPU: 32, CPU: 1
✅ device.ComputeUnits - > 0
✅ device.MaxWorkGroupSize - > 0

// Memory
✅ device.TotalMemoryBytes - > 0
✅ device.AvailableMemoryBytes - > 0, <= Total
✅ device.MemoryBandwidth - > 0

// Versions
✅ device.MajorVersion - >= 0
✅ device.MinorVersion - >= 0

// Extensions
✅ device.Extensions - Non-null collection
✅ device.SupportsExtension("...") - Returns bool
```

---

## Test Execution

### Manual Testing
```bash
# Run test application
cd /home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core
dotnet run --project tests/Orleans.GpuBridge.Tests

# Or create simple test program
dotnet new console -n DeviceDiscoveryTest
# Add references and test code
dotnet run
```

### Test Code Template
```csharp
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Microsoft.Extensions.Logging;

var loggerFactory = LoggerFactory.Create(builder =>
    builder.AddConsole().SetMinimumLevel(LogLevel.Debug));

var logger = loggerFactory.CreateLogger<DotComputeDeviceManager>();
var deviceManager = new DotComputeDeviceManager(logger, null);

// Initialize and discover
await deviceManager.InitializeAsync();

// Get all devices
var devices = await deviceManager.GetAllDevicesAsync();
Console.WriteLine($"Discovered {devices.Count} device(s)");

foreach (var device in devices)
{
    Console.WriteLine($"\nDevice: {device.Id}");
    Console.WriteLine($"  Name: {device.Name}");
    Console.WriteLine($"  Type: {device.Type}");
    Console.WriteLine($"  Architecture: {device.Architecture}");
    Console.WriteLine($"  Compute Units: {device.ComputeUnits}");
    Console.WriteLine($"  Memory: {device.TotalMemoryBytes / (1024.0 * 1024.0 * 1024.0):F2} GB");
    Console.WriteLine($"  Warp Size: {device.WarpSize}");

    // Test health monitoring
    var health = await deviceManager.GetDeviceHealthAsync(device.Id);
    Console.WriteLine($"  Status: {health.Status}");
    Console.WriteLine($"  Memory Usage: {health.MemoryUtilizationPercent:F1}%");
    Console.WriteLine($"  Temperature: {health.TemperatureCelsius}°C");
}

// Cleanup
await deviceManager.DisposeAsync();
```

---

## Success Criteria

### Must Pass
- ✅ At least one device discovered (GPU or CPU)
- ✅ Device properties match hardware capabilities
- ✅ Memory info accessible and within valid ranges
- ✅ Device status reports as Available
- ✅ No exceptions during discovery or health checks
- ✅ Logging output clear and informative

### Should Pass
- ✅ GPU discovered on systems with NVIDIA GPU + CUDA
- ✅ CPU fallback works on GPU-less systems
- ✅ Multiple devices discovered if multiple accelerators present
- ✅ Temperature reading (if sensors available)

---

## Known Limitations

### Simulated Values
⚠️ **Temperature**: Currently simulated (45°C GPU, 0°C CPU)
**Reason**: DotCompute v0.3.0-rc1 sensor APIs not yet available
**Impact**: Non-blocking, sensor support planned for future

⚠️ **Context Creation**: Not yet implemented
**Reason**: Context creation pattern unclear in v0.3.0-rc1
**Impact**: Non-blocking for device discovery phase

---

## Error Scenarios to Test

### No Devices Available
**Setup**: Run with no GPU and no CPU backend
**Expected**: Log warning, empty device list or exception

### Driver Not Loaded
**Setup**: GPU present but CUDA drivers not loaded
**Expected**: CPU fallback used, warning logged

### Invalid Device ID
**Setup**: Request health for non-existent device ID
**Expected**: ArgumentException thrown

---

## Performance Expectations

- **Discovery Time**: < 1 second for 1-2 devices
- **Health Check**: < 100ms per device
- **Memory Overhead**: < 50MB for device manager
- **CPU Impact**: Minimal (< 5% during discovery)

---

## Post-Test Actions

### On Success
1. ✅ Document actual device properties discovered
2. ✅ Save test output logs for reference
3. ✅ Move to kernel compilation API investigation
4. ✅ Begin unit test development

### On Failure
1. ❌ Capture full error logs and stack traces
2. ❌ Document hardware/software environment
3. ❌ Check DotCompute package versions
4. ❌ Verify CUDA drivers and toolkit installed
5. ❌ Test with simple DotCompute sample code (isolate issue)

---

## Next Phase Dependencies

**Kernel Compilation Testing** depends on:
- ✅ Device discovery working
- ✅ Devices accessible via adapter
- ⏳ Kernel compilation API signature confirmed
- ⏳ Test kernels compiled for discovered devices

---

**Status**: Ready for hardware testing execution
**Estimated Time**: 30-45 minutes
**Risk Level**: Low (CPU fallback ensures basic functionality)

---

*Test plan created for DotCompute v0.3.0-rc1 device discovery validation*
