# Phase 6 Implementation Guide: Physical Time Precision
## Orleans.GpuBridge.Core - Weeks 11-12

## Overview

Phase 6 implements **sub-microsecond physical time synchronization** across distributed GPU-resident actors using **Precision Time Protocol (PTP)** and advanced network latency compensation.

### Why This Matters

**GPU-native actors require nanosecond-precision timing:**
- Ring kernels process messages in 100-500ns
- Cross-node communication adds microseconds of latency
- Clock drift accumulates rapidly at this scale

**Without sub-μs synchronization:**
- Causal ordering breaks down across nodes
- Temporal patterns miss events separated by <1μs
- Physics simulations drift from reality

**With PTP synchronization:**
- ±100ns clock accuracy across the cluster
- Accurate causal ordering at GPU timescales
- Correct temporal analytics on distributed events

### Key Innovations

1. **PTP Hardware Integration**: Direct access to NIC hardware clocks (when available)
2. **Software PTP Fallback**: High-precision software sync when hardware unavailable
3. **Network Latency Compensation**: Automatic RTT measurement and timestamp adjustment
4. **GPU Clock Alignment**: Synchronize GPU `%%globaltimer` with PTP time
5. **Multi-Tier Clock Hierarchy**: GPS → PTP → NTP → System → GPU

## Goals

- ✅ PTP clock synchronization (±100ns accuracy)
- ✅ Network latency measurement and compensation
- ✅ Sub-microsecond cross-node timing
- ✅ Automatic clock source selection (PTP → NTP → System)
- 🔵 Optional: GPS receiver integration for grand master clock

## Architecture

### Clock Source Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    GPS Time Source (Optional)                 │
│                  Accuracy: ±50ns absolute                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PTP Grand Master Clock (Hardware)               │
│                  Accuracy: ±100ns cluster-wide               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                PTP Slave Clocks (Each Node)                  │
│              /sys/class/ptp/ptp0 (Linux)                    │
│              IOCTL_PTP_GET_TIME (Windows)                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Network Latency Compensator (Software)             │
│              RTT measurement + timestamp adjustment          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              GPU Clock Calibrator (Phase 5)                  │
│              GPU %%globaltimer ↔ PTP time                   │
└─────────────────────────────────────────────────────────────┘
```

### Fallback Chain

If PTP hardware unavailable, automatically fall back:

```
PTP Hardware → PTP Software → Chrony/NTPd → NTP Client → System Clock
(±100ns)       (±1μs)          (±500μs)       (±10ms)      (±100ms)
```

## Implementation Plan

### Week 11: PTP Foundation

**Day 1-2: PTP Clock Source (Linux)**
- File: `src/Orleans.GpuBridge.Runtime/Temporal/PtpClockSource.cs`
- Implement Linux PTP via `/dev/ptp0` device file
- Use `ptp_clock_gettime()` system call
- Detect available PTP hardware clocks

**Day 3-4: PTP Clock Source (Windows)**
- Windows PTP via `IOCTL_PTP_GET_TIME` IOCTL
- Detect PTP-capable NICs (Intel i210, Mellanox ConnectX)
- Handle driver differences (Intel vs Mellanox)

**Day 5: Software PTP Implementation**
- File: `src/Orleans.GpuBridge.Runtime/Temporal/SoftwarePtpClient.cs`
- Implement software PTP synchronization
- PTP message exchange (Sync, Follow_Up, Delay_Req, Delay_Resp)
- Offset calculation and drift correction

### Week 12: Network Compensation & Integration

**Day 1-2: Network Latency Measurement**
- File: `src/Orleans.GpuBridge.Runtime/Temporal/NetworkLatencyCompensator.cs`
- Implement RTT measurement (ICMP ping or TCP roundtrip)
- Statistical RTT analysis (min, median, p99)
- Detect and handle RTT outliers

**Day 3: Timestamp Compensation**
- Adjust remote timestamps based on measured latency
- Apply compensation to incoming messages
- Handle asymmetric network paths

**Day 4: Clock Source Selection**
- File: `src/Orleans.GpuBridge.Runtime/Temporal/ClockSourceSelector.cs`
- Auto-detect best available clock source
- Implement fallback chain
- Runtime clock source switching

**Day 5: Integration & Testing**
- Integrate PTP with existing `HybridTimestamp`
- Update `GpuClockCalibrator` to use PTP
- Comprehensive testing suite

## File Structure

```
src/Orleans.GpuBridge.Runtime/Temporal/
├── Clock/
│   ├── PtpClockSource.cs           # PTP hardware clock access
│   ├── SoftwarePtpClient.cs        # Software PTP implementation
│   ├── GpsClockSource.cs           # GPS receiver integration (optional)
│   ├── ClockSourceSelector.cs      # Auto clock source selection
│   └── IPhysicalClockSource.cs     # Base interface (Phase 1)
├── Network/
│   ├── NetworkLatencyCompensator.cs   # RTT measurement & compensation
│   ├── PtpMessageExchange.cs          # PTP protocol implementation
│   └── LatencyStatistics.cs           # RTT statistical analysis
└── Integration/
    └── HighPrecisionClockProvider.cs  # Unified clock access API

tests/Orleans.GpuBridge.Temporal.Tests/
├── Unit/
│   ├── PtpClockSourceTests.cs
│   ├── NetworkLatencyTests.cs
│   └── ClockSourceSelectionTests.cs
└── Integration/
    ├── PtpSynchronizationTests.cs
    ├── CrossNodeTimingTests.cs
    └── ClockDriftTests.cs
```

## Detailed Implementation

### 1. PTP Clock Source (Linux)

**File**: `src/Orleans.GpuBridge.Runtime/Temporal/PtpClockSource.cs`

```csharp
using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// PTP hardware clock source for sub-microsecond timing.
/// Accesses /dev/ptp* devices on Linux or uses IOCTL on Windows.
/// </summary>
public sealed class PtpClockSource : IPhysicalClockSource, IDisposable
{
    private readonly ILogger<PtpClockSource> _logger;
    private SafeFileHandle? _ptpDevice;
    private int _ptpClockId;
    private bool _isHardwarePtp;

    public bool IsSynchronized => _ptpDevice != null;
    public string ClockPath { get; }

    public PtpClockSource(ILogger<PtpClockSource> logger, string ptpDevicePath = "/dev/ptp0")
    {
        _logger = logger;
        ClockPath = ptpDevicePath;
    }

    /// <summary>
    /// Initializes PTP clock access.
    /// </summary>
    public async Task<bool> InitializeAsync(CancellationToken ct = default)
    {
        if (OperatingSystem.IsLinux())
        {
            return InitializeLinuxPtp();
        }
        else if (OperatingSystem.IsWindows())
        {
            return InitializeWindowsPtp();
        }
        else
        {
            _logger.LogWarning("PTP hardware not supported on this OS. Falling back to software PTP.");
            return false;
        }
    }

    /// <summary>
    /// Gets current PTP time in nanoseconds since Unix epoch.
    /// </summary>
    public long GetCurrentTimeNanos()
    {
        if (!IsSynchronized)
            throw new InvalidOperationException("PTP clock not initialized");

        if (OperatingSystem.IsLinux())
        {
            return GetLinuxPtpTime();
        }
        else if (OperatingSystem.IsWindows())
        {
            return GetWindowsPtpTime();
        }

        throw new PlatformNotSupportedException();
    }

    /// <summary>
    /// Gets PTP clock uncertainty (error bound) in nanoseconds.
    /// </summary>
    public long GetErrorBound()
    {
        // PTP hardware clocks typically have ±100ns accuracy
        // Software PTP has ±1μs accuracy
        return _isHardwarePtp ? 100 : 1_000;
    }

    private bool InitializeLinuxPtp()
    {
        try
        {
            // Open PTP device file
            _ptpDevice = File.OpenHandle(ClockPath, FileMode.Open, FileAccess.Read);

            // Get PTP clock ID via ioctl
            const int PTP_CLOCK_GETCAPS = 0x80086d01;
            var caps = new PtpClockCaps();

            if (Ioctl(_ptpDevice, PTP_CLOCK_GETCAPS, ref caps) == 0)
            {
                _ptpClockId = caps.ClockId;
                _isHardwarePtp = true;

                _logger.LogInformation(
                    "PTP hardware clock initialized: {Device} (ID={ClockId}, MaxAdj={MaxAdj} ppb)",
                    ClockPath,
                    _ptpClockId,
                    caps.MaxAdj);

                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to initialize PTP hardware clock at {Path}", ClockPath);
        }

        return false;
    }

    private bool InitializeWindowsPtp()
    {
        try
        {
            // Windows PTP via IOCTL_PTP_GET_TIME
            // Implementation depends on NIC driver (Intel i210, Mellanox ConnectX)

            // TODO: Implement Windows PTP when PTP-capable hardware available
            _logger.LogWarning("Windows PTP not yet implemented");
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to initialize Windows PTP");
            return false;
        }
    }

    private long GetLinuxPtpTime()
    {
        // Use clock_gettime with CLOCK_REALTIME or custom PTP clock ID
        var timespec = new Timespec();

        if (clock_gettime(_ptpClockId, ref timespec) == 0)
        {
            return timespec.Seconds * 1_000_000_000L + timespec.Nanoseconds;
        }

        throw new InvalidOperationException("Failed to read PTP time");
    }

    private long GetWindowsPtpTime()
    {
        // Windows implementation via IOCTL
        throw new NotImplementedException("Windows PTP not yet implemented");
    }

    public void Dispose()
    {
        _ptpDevice?.Dispose();
    }

    // P/Invoke declarations
    [DllImport("libc", SetLastError = true)]
    private static extern int clock_gettime(int clockId, ref Timespec tp);

    [DllImport("libc", SetLastError = true)]
    private static extern int ioctl(SafeFileHandle fd, uint request, ref PtpClockCaps caps);

    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Seconds;
        public long Nanoseconds;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PtpClockCaps
    {
        public int ClockId;
        public int MaxAdj;
        public int NAdjust;
        public int PpsMode;
        public int NPerOut;
        public int NExtTimestamp;
        public int Reserved1;
        public int Reserved2;
    }
}
```

### 2. Network Latency Compensator

**File**: `src/Orleans.GpuBridge.Runtime/Temporal/NetworkLatencyCompensator.cs`

```csharp
namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Measures and compensates for network latency in distributed timestamps.
/// </summary>
public sealed class NetworkLatencyCompensator
{
    private readonly ILogger<NetworkLatencyCompensator> _logger;
    private readonly ConcurrentDictionary<IPEndPoint, LatencyStatistics> _latencyCache = new();
    private readonly TimeSpan _measurementInterval = TimeSpan.FromMinutes(1);

    public NetworkLatencyCompensator(ILogger<NetworkLatencyCompensator> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Measures round-trip time (RTT) to remote endpoint.
    /// Returns median RTT from 10 measurements.
    /// </summary>
    public async Task<TimeSpan> MeasureLatencyAsync(
        IPEndPoint remote,
        CancellationToken ct = default)
    {
        const int sampleCount = 10;
        var samples = new long[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            var stopwatch = Stopwatch.StartNew();

            // TCP connection roundtrip (more reliable than ICMP ping)
            using var client = new TcpClient();
            await client.ConnectAsync(remote.Address, remote.Port, ct);

            stopwatch.Stop();
            samples[i] = stopwatch.ElapsedTicks;

            // Small delay between measurements
            if (i < sampleCount - 1)
                await Task.Delay(10, ct);
        }

        // Calculate median RTT (more robust than mean)
        Array.Sort(samples);
        long medianTicks = samples[sampleCount / 2];
        var rtt = TimeSpan.FromTicks(medianTicks);

        // Cache result
        var stats = _latencyCache.GetOrAdd(remote, _ => new LatencyStatistics(remote));
        stats.AddSample(rtt);

        _logger.LogDebug(
            "Measured RTT to {Remote}: {RTT:F3}ms (median of {Count} samples)",
            remote,
            rtt.TotalMilliseconds,
            sampleCount);

        return rtt;
    }

    /// <summary>
    /// Compensates remote timestamp for network latency.
    /// Adjusts timestamp assuming symmetric network path (RTT/2).
    /// </summary>
    public long CompensateTimestamp(long remoteTimestampNanos, IPEndPoint sourceEndpoint)
    {
        if (!_latencyCache.TryGetValue(sourceEndpoint, out var stats))
        {
            // No latency data available - return uncompensated
            _logger.LogWarning(
                "No latency data for {Source} - returning uncompensated timestamp",
                sourceEndpoint);
            return remoteTimestampNanos;
        }

        // Assume symmetric path: one-way latency ≈ RTT / 2
        long oneWayLatencyNanos = stats.MedianRtt.Ticks * 100 / 2; // Convert ticks to nanos

        // Compensate by adding one-way latency
        long compensatedTimestamp = remoteTimestampNanos + oneWayLatencyNanos;

        _logger.LogTrace(
            "Compensated timestamp from {Source}: {Original}ns → {Compensated}ns (Δ={Delta}μs)",
            sourceEndpoint,
            remoteTimestampNanos,
            compensatedTimestamp,
            oneWayLatencyNanos / 1_000);

        return compensatedTimestamp;
    }

    /// <summary>
    /// Starts background latency measurement for known endpoints.
    /// </summary>
    public void StartPeriodicMeasurements(IEnumerable<IPEndPoint> endpoints)
    {
        foreach (var endpoint in endpoints)
        {
            _ = Task.Run(async () =>
            {
                while (true)
                {
                    try
                    {
                        await MeasureLatencyAsync(endpoint);
                        await Task.Delay(_measurementInterval);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Failed to measure latency to {Endpoint}", endpoint);
                    }
                }
            });
        }
    }
}

/// <summary>
/// Statistical analysis of network latency measurements.
/// </summary>
internal sealed class LatencyStatistics
{
    private const int MaxSamples = 100;
    private readonly IPEndPoint _endpoint;
    private readonly List<TimeSpan> _samples = new();
    private readonly object _lock = new();

    public TimeSpan MedianRtt { get; private set; }
    public TimeSpan MinRtt { get; private set; } = TimeSpan.MaxValue;
    public TimeSpan MaxRtt { get; private set; }
    public TimeSpan P99Rtt { get; private set; }

    public LatencyStatistics(IPEndPoint endpoint)
    {
        _endpoint = endpoint;
    }

    public void AddSample(TimeSpan rtt)
    {
        lock (_lock)
        {
            _samples.Add(rtt);

            // Sliding window - keep last 100 samples
            if (_samples.Count > MaxSamples)
                _samples.RemoveAt(0);

            // Update statistics
            var sorted = _samples.OrderBy(s => s).ToArray();
            MedianRtt = sorted[sorted.Length / 2];
            MinRtt = sorted[0];
            MaxRtt = sorted[^1];
            P99Rtt = sorted[(int)(sorted.Length * 0.99)];
        }
    }
}
```

### 3. Clock Source Selector

**File**: `src/Orleans.GpuBridge.Runtime/Temporal/ClockSourceSelector.cs`

```csharp
namespace Orleans.GpuBridge.Runtime.Temporal;

/// <summary>
/// Automatically selects best available clock source with fallback.
/// Preference: GPS → PTP Hardware → PTP Software → Chrony → NTP → System.
/// </summary>
public sealed class ClockSourceSelector
{
    private readonly ILogger<ClockSourceSelector> _logger;
    private readonly List<IPhysicalClockSource> _clockSources = new();
    private IPhysicalClockSource? _activeSource;

    public IPhysicalClockSource ActiveSource => _activeSource ??
        throw new InvalidOperationException("No clock source available");

    public ClockSourceSelector(ILogger<ClockSourceSelector> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes and selects best available clock source.
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Detecting available clock sources...");

        // Try clock sources in order of preference
        await TryInitializeGpsClock(ct);
        await TryInitializePtpHardware(ct);
        await TryInitializePtpSoftware(ct);
        await TryInitializeNtpClient(ct);
        await TryInitializeSystemClock(ct);

        if (_activeSource == null)
        {
            throw new InvalidOperationException(
                "No clock source available - at least system clock should work");
        }

        _logger.LogInformation(
            "Clock source selected: {Source} (Accuracy: ±{ErrorBound}ns)",
            _activeSource.GetType().Name,
            _activeSource.GetErrorBound());
    }

    private async Task TryInitializeGpsClock(CancellationToken ct)
    {
        try
        {
            var gpsClock = new GpsClockSource(
                _logger.CreateLogger<GpsClockSource>());

            if (await gpsClock.InitializeAsync(ct))
            {
                _clockSources.Add(gpsClock);
                _activeSource = gpsClock;
                _logger.LogInformation("GPS clock source available (±50ns accuracy)");
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "GPS clock not available");
        }
    }

    private async Task TryInitializePtpHardware(CancellationToken ct)
    {
        if (_activeSource != null) return; // Already have better source

        try
        {
            var ptpClock = new PtpClockSource(
                _logger.CreateLogger<PtpClockSource>());

            if (await ptpClock.InitializeAsync(ct))
            {
                _clockSources.Add(ptpClock);
                _activeSource = ptpClock;
                _logger.LogInformation("PTP hardware clock available (±100ns accuracy)");
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "PTP hardware clock not available");
        }
    }

    private async Task TryInitializePtpSoftware(CancellationToken ct)
    {
        if (_activeSource != null) return;

        try
        {
            var softwarePtp = new SoftwarePtpClient(
                _logger.CreateLogger<SoftwarePtpClient>());

            if (await softwarePtp.InitializeAsync(ct))
            {
                _clockSources.Add(softwarePtp);
                _activeSource = softwarePtp;
                _logger.LogInformation("Software PTP available (±1μs accuracy)");
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Software PTP not available");
        }
    }

    private async Task TryInitializeNtpClient(CancellationToken ct)
    {
        if (_activeSource != null) return;

        try
        {
            // Use existing NtpClockSource from Phase 1
            var ntpClock = new NtpClockSource(
                _logger.CreateLogger<NtpClockSource>());

            if (await ntpClock.InitializeAsync(ct))
            {
                _clockSources.Add(ntpClock);
                _activeSource = ntpClock;
                _logger.LogInformation("NTP client available (±10ms accuracy)");
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "NTP client not available");
        }
    }

    private Task TryInitializeSystemClock(CancellationToken ct)
    {
        // System clock always available as last resort
        var systemClock = new SystemClockSource(
            _logger.CreateLogger<SystemClockSource>());

        _clockSources.Add(systemClock);
        _activeSource ??= systemClock;

        _logger.LogInformation("System clock available (±100ms accuracy)");
        return Task.CompletedTask;
    }
}

/// <summary>
/// System clock fallback (always available but least accurate).
/// </summary>
internal sealed class SystemClockSource : IPhysicalClockSource
{
    private readonly ILogger<SystemClockSource> _logger;

    public bool IsSynchronized => true; // Always available

    public SystemClockSource(ILogger<SystemClockSource> logger)
    {
        _logger = logger;
    }

    public long GetCurrentTimeNanos()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeNanoseconds();
    }

    public long GetErrorBound()
    {
        return 100_000_000; // ±100ms
    }
}
```

## Testing Strategy

### Unit Tests

**File**: `tests/Orleans.GpuBridge.Temporal.Tests/Unit/PtpClockSourceTests.cs`

```csharp
public sealed class PtpClockSourceTests
{
    [Fact]
    public async Task PtpClockSource_InitializesSuccessfully()
    {
        // Arrange
        var logger = NullLogger<PtpClockSource>.Instance;
        var ptpClock = new PtpClockSource(logger);

        // Act
        bool initialized = await ptpClock.InitializeAsync();

        // Assert
        if (initialized)
        {
            ptpClock.IsSynchronized.Should().BeTrue();
            var time = ptpClock.GetCurrentTimeNanos();
            time.Should().BeGreaterThan(0);
        }
        else
        {
            // PTP hardware not available - not a test failure
            logger.LogInformation("PTP hardware not available on this system");
        }
    }

    [Fact]
    public async Task PtpClockSource_ReturnsNanosecondPrecision()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);
        await ptpClock.InitializeAsync();

        if (!ptpClock.IsSynchronized)
        {
            // Skip test if PTP unavailable
            return;
        }

        // Act
        long time1 = ptpClock.GetCurrentTimeNanos();
        await Task.Delay(1); // 1ms delay
        long time2 = ptpClock.GetCurrentTimeNanos();

        // Assert
        long deltaNanos = time2 - time1;
        deltaNanos.Should().BeGreaterThan(1_000_000); // > 1ms
        deltaNanos.Should().BeLessThan(10_000_000);   // < 10ms
    }

    [Fact]
    public void PtpClockSource_ErrorBoundIsReasonable()
    {
        // Arrange
        var ptpClock = new PtpClockSource(NullLogger<PtpClockSource>.Instance);

        // Act
        long errorBound = ptpClock.GetErrorBound();

        // Assert
        errorBound.Should().BeInRange(100, 10_000); // Between 100ns and 10μs
    }
}
```

### Integration Tests

**File**: `tests/Orleans.GpuBridge.Temporal.Tests/Integration/PtpSynchronizationTests.cs`

```csharp
public sealed class PtpSynchronizationTests
{
    [Fact(Skip = "Requires PTP hardware")]
    public async Task PtpClock_SynchronizesAcrossNodes()
    {
        // Test cross-node clock synchronization accuracy
        // Requires multi-node setup with PTP grand master
    }

    [Fact]
    public async Task ClockSourceSelector_SelectsBestAvailable()
    {
        // Arrange
        var selector = new ClockSourceSelector(
            NullLogger<ClockSourceSelector>.Instance);

        // Act
        await selector.InitializeAsync();

        // Assert
        selector.ActiveSource.Should().NotBeNull();
        selector.ActiveSource.IsSynchronized.Should().BeTrue();
    }

    [Fact]
    public async Task NetworkLatencyCompensator_MeasuresRtt()
    {
        // Arrange
        var compensator = new NetworkLatencyCompensator(
            NullLogger<NetworkLatencyCompensator>.Instance);
        var localhost = new IPEndPoint(IPAddress.Loopback, 80);

        // Act
        var rtt = await compensator.MeasureLatencyAsync(localhost);

        // Assert
        rtt.Should().BeGreaterThan(TimeSpan.Zero);
        rtt.Should().BeLessThan(TimeSpan.FromMilliseconds(10));
    }
}
```

## Performance Targets

### Clock Accuracy

| Clock Source | Target Accuracy | Typical Accuracy | Use Case |
|--------------|----------------|------------------|----------|
| GPS          | ±50ns          | ±30ns            | Grand master |
| PTP Hardware | ±100ns         | ±80ns            | Distributed cluster |
| PTP Software | ±1μs           | ±500ns           | No PTP hardware |
| Chrony/NTPd  | ±500μs         | ±200μs           | Traditional DC |
| NTP Client   | ±10ms          | ±5ms             | Internet sync |
| System Clock | ±100ms         | ±50ms            | Fallback only |

### Network Compensation

- RTT measurement: < 1ms per endpoint
- Compensation calculation: < 10ns overhead
- Measurement frequency: Every 1 minute per endpoint
- Cache size: 100 samples per endpoint

## Integration with Phase 5

Phase 6 extends Phase 5's GPU clock calibration:

**Before Phase 6:**
```csharp
var gpuCalibrator = new GpuClockCalibrator(logger);
var calibration = await gpuCalibrator.CalibrateAsync(sampleCount: 1000);
// Calibrates GPU clock against CPU system clock (±10ms accuracy)
```

**After Phase 6:**
```csharp
// Select best clock source
var clockSelector = new ClockSourceSelector(logger);
await clockSelector.InitializeAsync();

// Calibrate GPU against PTP (±100ns accuracy)
var gpuCalibrator = new GpuClockCalibrator(logger, clockSelector.ActiveSource);
var calibration = await gpuCalibrator.CalibrateAsync(sampleCount: 1000);

// GPU now synchronized with sub-microsecond precision
```

## Hardware Requirements

### PTP-Capable Network Cards

**Supported NICs:**
- Intel i210/i211 (most common, $30-50)
- Intel i350/X540/X550 (enterprise)
- Mellanox ConnectX-3/4/5 (high-end)
- Broadcom NetXtreme (server)

**How to Check:**
```bash
# Linux
ethtool -T eth0

# Should show:
Time stamping parameters for eth0:
Capabilities:
        hardware-transmit     (SOF_TIMESTAMPING_TX_HARDWARE)
        hardware-receive      (SOF_TIMESTAMPING_RX_HARDWARE)
        hardware-raw-clock    (SOF_TIMESTAMPING_RAW_HARDWARE)
```

### Software Requirements

**Linux:**
- `linuxptp` package (provides `ptp4l` daemon)
- Kernel PTP support (CONFIG_PTP_1588_CLOCK=y)
- `/dev/ptp0` device node

**Windows:**
- Driver with PTP support (Intel or Mellanox)
- Windows 10/Server 2019+ (IOCTL_PTP_GET_TIME)

## Production Deployment

### 1. Verify PTP Hardware
```bash
# Linux
ls /sys/class/ptp/
ethtool -T eth0

# Start PTP daemon
sudo ptp4l -i eth0 -m
```

### 2. Select Grand Master
- Designate one node as PTP grand master
- Ideally connected to GPS receiver
- All other nodes synchronize to it

### 3. Configure Orleans.GpuBridge
```csharp
services.AddGpuBridge(options =>
{
    options.EnableHighPrecisionTiming = true;
    options.PtpDevice = "/dev/ptp0";
    options.NetworkLatencyCompensation = true;
});
```

### 4. Monitor Clock Drift
```csharp
var metrics = temporalMetrics.GetClockDrift();
if (metrics.DriftNanos > 1_000_000) // > 1ms drift
{
    logger.LogWarning("Clock drift exceeds threshold: {Drift}ns", metrics.DriftNanos);
    await recalibrateClocks();
}
```

## Success Criteria

### Functional Requirements
- ✅ PTP clock source functional on Linux
- ✅ PTP clock source functional on Windows
- ✅ Software PTP fallback works without hardware
- ✅ Network latency compensation applied
- ✅ Automatic clock source selection
- ✅ Runtime clock source switching

### Performance Requirements
- ✅ Clock accuracy: ±100ns (PTP hardware)
- ✅ Clock accuracy: ±1μs (PTP software)
- ✅ RTT measurement: < 1ms
- ✅ Timestamp compensation: < 10ns overhead
- ✅ Clock source detection: < 100ms

### Quality Requirements
- ✅ >90% code coverage
- ✅ All unit tests passing
- ✅ Integration tests passing (hardware-dependent tests skippable)
- ✅ Graceful degradation without PTP hardware

## Known Limitations

1. **Hardware Dependency**: PTP hardware not available on all systems
2. **Driver Support**: Windows PTP support varies by NIC vendor
3. **Network Topology**: PTP works best on switched networks (not routed)
4. **Asymmetric Paths**: Network latency compensation assumes symmetric RTT
5. **Clock Drift**: Hardware clocks drift over time, requiring periodic recalibration

## Next Phase Preview

**Phase 7: Integration & Optimization (Weeks 13-14)**
- End-to-end performance optimization
- Production monitoring and observability
- Fault tolerance and recovery
- Complete documentation

---

*Phase 6 Implementation Guide*
*Version: 1.0*
*Last Updated: 2025-11-11*
*Author: Claude Code Assistant*
