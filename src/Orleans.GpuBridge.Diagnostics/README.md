# Orleans.GpuBridge.Diagnostics

GPU telemetry, metrics collection, and observability for Orleans GPU Bridge.

## Overview

Orleans.GpuBridge.Diagnostics provides comprehensive monitoring and observability capabilities for GPU-accelerated Orleans applications. It enables real-time metrics collection, telemetry export, and performance tracking across GPU devices.

## Key Features

- **GPU Device Metrics**: Memory usage, utilization, temperature, and queue depth
- **System Metrics**: CPU, memory, and overall system health monitoring
- **OpenTelemetry Integration**: Export metrics to Prometheus, Jaeger, and other backends
- **Transfer Direction Tracking**: Monitor host-to-device and device-to-host data movement
- **Custom Metric Providers**: Extensible architecture for domain-specific metrics

## Installation

```bash
dotnet add package Orleans.GpuBridge.Diagnostics
```

## Quick Start

### Basic Configuration

```csharp
using Orleans.GpuBridge.Diagnostics;
using Microsoft.Extensions.DependencyInjection;

services.AddGpuBridge()
    .AddGpuTelemetry(options =>
    {
        options.CollectionInterval = TimeSpan.FromSeconds(5);
        options.EnableGpuMetrics = true;
        options.EnableSystemMetrics = true;
    });
```

### Accessing Telemetry

```csharp
public class MonitoringService
{
    private readonly IGpuTelemetry _telemetry;

    public MonitoringService(IGpuTelemetry telemetry)
    {
        _telemetry = telemetry;
    }

    public async Task<GpuDeviceMetrics> GetDeviceMetricsAsync(int deviceIndex)
    {
        return await _telemetry.GetDeviceMetricsAsync(deviceIndex);
    }

    public async Task<SystemMetrics> GetSystemMetricsAsync()
    {
        return await _telemetry.GetSystemMetricsAsync();
    }
}
```

### OpenTelemetry Integration

```csharp
services.AddOpenTelemetry()
    .WithMetrics(builder => builder
        .AddGpuBridgeInstrumentation()
        .AddPrometheusExporter());
```

## Configuration Options

```csharp
public class GpuMetricsOptions
{
    // Collection frequency
    public TimeSpan CollectionInterval { get; set; } = TimeSpan.FromSeconds(10);

    // Enable/disable metric categories
    public bool EnableGpuMetrics { get; set; } = true;
    public bool EnableSystemMetrics { get; set; } = true;
    public bool EnableTransferMetrics { get; set; } = true;

    // Retention settings
    public int MetricHistorySize { get; set; } = 1000;
}
```

## Available Metrics

### GPU Device Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `gpu.memory.used` | Used GPU memory | bytes |
| `gpu.memory.total` | Total GPU memory | bytes |
| `gpu.utilization` | GPU compute utilization | percent |
| `gpu.temperature` | GPU temperature | celsius |
| `gpu.queue.depth` | Current queue depth | count |
| `gpu.power.draw` | Current power consumption | watts |

### System Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `system.cpu.utilization` | CPU utilization | percent |
| `system.memory.used` | Used system memory | bytes |
| `system.memory.available` | Available system memory | bytes |

### Transfer Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `transfer.host_to_device.bytes` | Data transferred to GPU | bytes |
| `transfer.device_to_host.bytes` | Data transferred from GPU | bytes |
| `transfer.duration` | Transfer operation duration | milliseconds |

## API Reference

### IGpuTelemetry Interface

```csharp
public interface IGpuTelemetry
{
    Task<GpuDeviceMetrics> GetDeviceMetricsAsync(int deviceIndex);
    Task<IReadOnlyList<GpuDeviceMetrics>> GetAllDeviceMetricsAsync();
    Task<SystemMetrics> GetSystemMetricsAsync();
    void RecordTransfer(TransferDirection direction, long bytes, TimeSpan duration);
}
```

### GpuMetricsCollector

The `GpuMetricsCollector` is a background service that continuously collects metrics:

```csharp
// Automatically started when telemetry is enabled
services.AddHostedService<GpuMetricsCollector>();
```

## Dependencies

- **Microsoft.Extensions.Diagnostics** (>= 9.0.0)
- **OpenTelemetry.Api** (>= 1.9.0)
- **System.Diagnostics.DiagnosticSource** (>= 9.0.0)

## License

MIT License - Copyright (c) 2025 Michael Ivertowski

---

For more information, see the [Orleans.GpuBridge.Core Documentation](https://github.com/mivertowski/Orleans.GpuBridge.Core).
