# Phase 6 Week 12: Software PTP & Integration - Usage Guide

## Overview

Week 12 completes Phase 6 (Physical Time Precision) by adding:
- **Software PTP**: IEEE 1588 implementation without hardware requirements
- **Integration Tests**: Verify Phase 5 + Phase 6 work together
- **Performance Benchmarks**: Measure clock source overhead
- **Hardware Setup**: Automated PTP device configuration

---

## 1. Software PTP Clock Source

### When to Use

Use `SoftwarePtpClockSource` when:
- ✅ PTP hardware unavailable (no `/dev/ptp*` devices)
- ✅ Running in cloud environments (Azure, AWS, GCP)
- ✅ Need better precision than NTP (±10ms) but don't have hardware PTP
- ✅ Acceptable precision: ±1-10μs (vs ±50ns-1μs for hardware PTP)

### Basic Usage

```csharp
using Orleans.GpuBridge.Runtime.Temporal.Clock;
using Microsoft.Extensions.Logging;

// Initialize Software PTP with NTP server
var softwarePtp = new SoftwarePtpClockSource(
    logger,
    masterAddress: "time.nist.gov",
    syncInterval: TimeSpan.FromMinutes(1));

bool initialized = await softwarePtp.InitializeAsync();

if (initialized)
{
    // Get time with ±1-10μs precision
    long timeNanos = softwarePtp.GetCurrentTimeNanos();
    long errorBound = softwarePtp.GetErrorBound(); // ±1-10μs

    Console.WriteLine($"Time: {timeNanos}ns, Error: ±{errorBound / 1_000}μs");
}

softwarePtp.Dispose();
```

### Automatic Fallback Chain

The `ClockSourceSelector` automatically tries sources in preference order:

```csharp
var selector = new ClockSourceSelector(logger);
await selector.InitializeAsync();

// Fallback chain: GPS → PTP Hardware → PTP Software → NTP → System Clock
var activeSource = selector.ActiveSource;

Console.WriteLine($"Using: {activeSource.GetType().Name}");
Console.WriteLine($"Error Bound: ±{activeSource.GetErrorBound()}ns");
```

**Fallback Logic**:
1. **GPS Clock**: ±50ns (if GPS receiver available) ❌ Not implemented
2. **PTP Hardware**: ±50ns-1μs (if `/dev/ptp*` accessible) ✅
3. **PTP Software**: ±1-10μs (if NTP servers reachable) ✅ **NEW**
4. **NTP Client**: ±10ms (placeholder, not implemented) ❌
5. **System Clock**: ±100ms (always available) ✅

---

## 2. Integration with Phase 5 (Clock Calibration)

### GPU Clock Calibration with PTP

```csharp
using Orleans.GpuBridge.Runtime.Temporal;
using Orleans.GpuBridge.Runtime.Temporal.Clock;

// Initialize clock source (PTP if available)
var clockSelector = new ClockSourceSelector(logger);
await clockSelector.InitializeAsync();

// Create GPU clock calibrator
var calibrator = new GpuClockCalibrator(logger);

// Get timestamps from both CPU and GPU clocks
long cpuTimeNanos = clockSelector.ActiveSource.GetCurrentTimeNanos();
long gpuTimeNanos = calibrator.GetGpuTimeNanos();

// Calculate clock skew
long skew = gpuTimeNanos - cpuTimeNanos;
Console.WriteLine($"GPU-CPU Skew: {skew / 1_000}μs");
```

### Network Latency Compensation

```csharp
using Orleans.GpuBridge.Runtime.Temporal.Network;
using System.Net;

// Initialize clock and compensator
var clockSelector = new ClockSourceSelector(logger);
await clockSelector.InitializeAsync();

var compensator = new NetworkLatencyCompensator(logger);

// Measure latency to remote node
var remoteEndpoint = new IPEndPoint(IPAddress.Parse("192.168.1.100"), 12345);
var rtt = await compensator.MeasureLatencyAsync(remoteEndpoint);

Console.WriteLine($"RTT to {remoteEndpoint}: {rtt.TotalMilliseconds:F3}ms");

// Compensate remote timestamp
long remoteTimestamp = receiveTimestampFromNetwork(); // Your network code
long compensatedTimestamp = compensator.CompensateTimestamp(
    remoteTimestamp,
    remoteEndpoint);

// Now use compensated timestamp for distributed ordering
```

---

## 3. Performance Benchmarks

### Running Benchmarks

```bash
cd tests/Orleans.GpuBridge.Temporal.Tests

# Run all benchmarks
dotnet test --filter "FullyQualifiedName~Benchmarks"

# Run specific benchmark
dotnet test --filter "Benchmark_PtpClockSource_TimeReadLatency"
```

### Expected Performance

| Clock Source | Time Read Latency | Error Bound | Throughput |
|--------------|-------------------|-------------|------------|
| PTP Hardware | 50-100ns | ±50ns-1μs | 10-20M reads/sec |
| Software PTP | 100-500ns | ±1-10μs | 2-10M reads/sec |
| System Clock | 50-200ns | ±100ms | 5-20M reads/sec |

### Benchmark Example Output

```
=== PTP Clock Source Benchmark ===
Iterations: 10,000
Total Time: 5ms
Avg Latency: 78.23ns per read
Throughput: 12,789,456 reads/second
Error Bound: ±500ns

=== Software PTP Clock Source Benchmark ===
Iterations: 10,000
Total Time: 8ms
Avg Latency: 142.67ns per read
Throughput: 7,012,345 reads/second
Error Bound: ±5μs
```

---

## 4. Hardware Setup (PTP Permissions)

### Automated Setup

Run the provided script to configure PTP device access:

```bash
cd scripts
chmod +x setup-ptp-permissions.sh
sudo ./setup-ptp-permissions.sh
```

**Script Actions**:
1. Creates `ptp` group
2. Adds current user to `ptp` group
3. Creates udev rules for `/dev/ptp*` devices
4. Sets permissions: `group=ptp, mode=0660`
5. Reloads udev rules

### Manual Setup (if script fails)

```bash
# Create ptp group
sudo groupadd ptp

# Add your user to ptp group
sudo usermod -aG ptp $USER

# Create udev rule
echo 'SUBSYSTEM=="ptp", GROUP="ptp", MODE="0660"' | \
  sudo tee /etc/udev/rules.d/99-ptp.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Log out and back in for group changes
```

### Verify PTP Access

```bash
# Check PTP devices
ls -l /dev/ptp*

# Expected output:
# crw-rw---- 1 root ptp 248, 0 Jan 11 10:30 /dev/ptp0

# Check group membership
groups | grep ptp

# Test PTP clock
cd tests/Orleans.GpuBridge.Temporal.Tests
dotnet test --filter "PtpClockSource"
```

---

## 5. Configuration Examples

### ASP.NET Core Service Configuration

```csharp
// Startup.cs or Program.cs
public void ConfigureServices(IServiceCollection services)
{
    // Register clock source selector
    services.AddSingleton<IPhysicalClockSource>(sp =>
    {
        var logger = sp.GetRequiredService<ILogger<ClockSourceSelector>>();
        var selector = new ClockSourceSelector(logger);
        selector.InitializeAsync().Wait();
        return selector.ActiveSource;
    });

    // Register network latency compensator
    services.AddSingleton<NetworkLatencyCompensator>();

    // Register GPU clock calibrator
    services.AddSingleton<GpuClockCalibrator>();

    // ... other services
}
```

### Orleans Silo Configuration

```csharp
// Configure Orleans with temporal clock sources
var siloBuilder = new SiloBuilder()
    .ConfigureServices(services =>
    {
        // Add temporal clock sources
        services.AddSingleton(sp =>
        {
            var logger = sp.GetRequiredService<ILogger<ClockSourceSelector>>();
            var selector = new ClockSourceSelector(logger);
            selector.InitializeAsync().Wait();
            return selector;
        });

        services.AddSingleton(sp =>
            sp.GetRequiredService<ClockSourceSelector>().ActiveSource);

        services.AddSingleton<NetworkLatencyCompensator>();
    });
```

---

## 6. Troubleshooting

### PTP Hardware Not Found

**Symptom**: `/dev/ptp*` devices don't exist

**Solutions**:
1. Check for PTP-capable NIC:
   ```bash
   ethtool -T eth0 | grep PTP
   ```

2. Load PTP kernel module:
   ```bash
   sudo modprobe ptp_pch  # For Intel NICs
   sudo modprobe ptp_kvm  # For KVM guests
   ```

3. For Hyper-V VMs:
   - Ensure Enhanced Session Mode enabled
   - Check for `hv_utils` module: `lsmod | grep hv_utils`
   - Hyper-V synthetic NIC provides `/dev/ptp0`

4. Use Software PTP fallback if hardware unavailable

### Software PTP Initialization Fails

**Symptom**: `SoftwarePtpClockSource.InitializeAsync()` returns `false`

**Solutions**:
1. Check network connectivity:
   ```bash
   ping time.nist.gov
   ```

2. Try alternative NTP servers:
   ```csharp
   var softwarePtp = new SoftwarePtpClockSource(
       logger,
       masterAddress: "pool.ntp.org");  // Alternative server
   ```

3. Check firewall rules:
   ```bash
   sudo ufw allow 123/udp  # NTP port
   ```

### Permission Denied on /dev/ptp0

**Symptom**: `Permission denied` when accessing PTP clock

**Solutions**:
1. Run setup script:
   ```bash
   sudo ./scripts/setup-ptp-permissions.sh
   ```

2. Verify group membership:
   ```bash
   groups | grep ptp
   ```

3. Log out and back in (for group changes)

4. Temporary workaround (testing only):
   ```bash
   sudo chmod 666 /dev/ptp0
   ```

---

## 7. Integration Test Examples

### Test Clock Calibration with PTP

```bash
# Run Phase 5+6 integration tests
dotnet test --filter "ClockCalibrationIntegrationTests"

# Example tests:
# - GpuClockCalibrator uses PTP source when available
# - Clock skew detection with PTP precision
# - Drift compensation over time
# - Fallback to system clock when PTP unavailable
```

### Test Network Compensation

```bash
# Run network compensation tests
dotnet test --filter "NetworkCompensationIntegrationTests"

# Example tests:
# - Combined clock calibration + network compensation
# - Distributed timestamp ordering with causal relationships
# - HLC integration with physical clocks
# - Cross-node timestamp alignment
```

---

## 8. API Reference

### SoftwarePtpClockSource

```csharp
public sealed class SoftwarePtpClockSource : IPhysicalClockSource
{
    // Constructor
    public SoftwarePtpClockSource(
        ILogger<SoftwarePtpClockSource> logger,
        string masterAddress = "time.nist.gov",
        int ptpPort = 319,
        TimeSpan? syncInterval = null);

    // Properties
    public bool IsSynchronized { get; }
    public string MasterAddress { get; }
    public long CurrentOffsetNanos { get; }

    // Methods
    public Task<bool> InitializeAsync(CancellationToken ct = default);
    public long GetCurrentTimeNanos();
    public long GetErrorBound();
    public double GetClockDrift();
    public void Dispose();
}
```

### PtpClientProtocol

```csharp
public sealed class PtpClientProtocol
{
    public Task<PtpSyncResult?> ExchangeAsync(
        string masterAddress,
        int port,
        CancellationToken ct = default);
}

public sealed class PtpSyncResult
{
    public long OffsetNanos { get; init; }
    public long RoundTripDelayNanos { get; init; }
    public long MasterTimestampNanos { get; init; }
    public long LocalTimestampNanos { get; init; }
    public double? DriftRatePpm { get; init; }
}
```

---

## 9. Best Practices

### Clock Source Selection

1. **Production Environments**:
   - Use PTP hardware if available (best precision)
   - Fallback to Software PTP for cloud deployments
   - Always use `ClockSourceSelector` for automatic fallback

2. **Development Environments**:
   - System clock sufficient for most testing
   - Use Software PTP to test precision-dependent features

3. **High-Precision Requirements** (GPU-native actors):
   - Require PTP hardware (±50ns-1μs)
   - Verify error bound: `errorBound < 10_000` (< 10μs)

### Performance Optimization

1. **Minimize Time Reads**:
   - Cache timestamps when possible
   - Batch time reads in tight loops

2. **Sync Interval Tuning**:
   - Software PTP default: 1 minute
   - Increase for stable networks: 5-10 minutes
   - Decrease for unstable networks: 30 seconds

3. **Network Compensation**:
   - Measure RTT periodically (not per-message)
   - Use sliding window statistics (100 samples)
   - Remove outliers (top 20%)

### Error Handling

1. **Initialization Failures**:
   - Always check `InitializeAsync()` return value
   - Log warnings but don't fail application startup
   - Fallback to system clock if all sources fail

2. **Runtime Errors**:
   - Handle `InvalidOperationException` on unsynchronized clocks
   - Retry synchronization after temporary failures
   - Monitor error bounds for quality degradation

---

## 10. Next Steps (Phase 7+)

After Week 12, proceed with:

1. **Week 13**: HLC with PTP integration (Phase 7)
2. **Week 14**: Vector Clocks with network compensation (Phase 7)
3. **Week 15**: Temporal graph patterns (Phase 9 preview)

**Related Documentation**:
- [PHASE6-IMPLEMENTATION-GUIDE.md](./PHASE6-IMPLEMENTATION-GUIDE.md)
- [PHASE6-READINESS-SUMMARY.md](./PHASE6-READINESS-SUMMARY.md)
- [IMPLEMENTATION-ROADMAP.md](./IMPLEMENTATION-ROADMAP.md)

---

**Phase 6 Week 12**: ✅ **COMPLETE**
**Last Updated**: January 11, 2025
