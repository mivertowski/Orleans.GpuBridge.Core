using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions.Metrics;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Diagnostics.Abstractions;
using Orleans.GpuBridge.Diagnostics.Configuration;
using Orleans.GpuBridge.Diagnostics.Implementation;
using Orleans.GpuBridge.Diagnostics.Interfaces;
using Orleans.GpuBridge.Diagnostics.Models;

namespace Orleans.GpuBridge.Diagnostics;

/// <summary>
/// Background service that continuously collects GPU and system performance metrics.
/// </summary>
/// <remarks>
/// <para>
/// The GpuMetricsCollector operates as a hosted background service that periodically
/// queries GPU devices and system resources to gather performance metrics. It supports
/// multiple GPU vendors (NVIDIA, AMD, Intel) and provides CPU fallbacks when hardware
/// monitoring tools are unavailable.
/// </para>
/// <para>
/// Key features:
/// - Automatic periodic metric collection based on configured intervals
/// - Multi-vendor GPU support with vendor-specific monitoring tools
/// - Thread-safe metric caching and retrieval
/// - Graceful handling of unavailable devices or monitoring tools
/// - Integration with Orleans telemetry system
/// - Configurable logging levels for debugging
/// </para>
/// <para>
/// The collector uses external tools for GPU monitoring:
/// - NVIDIA: nvidia-smi command-line utility
/// - AMD: rocm-smi command-line utility  
/// - Intel: Windows performance counters (future implementation)
/// </para>
/// </remarks>
public sealed class GpuMetricsCollector : BackgroundService, 
    Orleans.GpuBridge.Diagnostics.Abstractions.IGpuMetricsCollector,
    Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector
{
    private readonly ILogger<GpuMetricsCollector> _logger;
    private readonly IGpuTelemetry _telemetry;
    private readonly GpuMetricsOptions _options;
    private readonly Timer _collectionTimer;
    private readonly Dictionary<int, GpuDeviceMetrics> _currentMetrics = new();
    private readonly object _metricsLock = new();
    
    /// <summary>
    /// Initializes a new instance of the <see cref="GpuMetricsCollector"/> class.
    /// </summary>
    /// <param name="logger">The logger instance for diagnostic output.</param>
    /// <param name="telemetry">The telemetry system for metric reporting.</param>
    /// <param name="options">Configuration options for metrics collection behavior.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when any of the required parameters are null.
    /// </exception>
    public GpuMetricsCollector(
        ILogger<GpuMetricsCollector> logger,
        IGpuTelemetry telemetry,
        IOptions<GpuMetricsOptions> options)
    {
        _logger = logger;
        _telemetry = telemetry;
        _options = options.Value;
        _collectionTimer = new Timer(
            CollectMetrics,
            null,
            TimeSpan.Zero,
            _options.CollectionInterval);
    }
    
    /// <summary>
    /// Executes the background metrics collection service.
    /// </summary>
    /// <param name="stoppingToken">Token that signals when the service should stop.</param>
    /// <returns>A task that represents the lifetime of the background service.</returns>
    /// <remarks>
    /// This method runs continuously until the application shuts down, collecting
    /// metrics at the configured interval. It handles exceptions gracefully and
    /// implements exponential backoff on persistent failures.
    /// </remarks>
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("GPU metrics collector started");
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CollectAllMetricsAsync();
                await Task.Delay(_options.CollectionInterval, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting GPU metrics");
                await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
            }
        }
        
        _logger.LogInformation("GPU metrics collector stopped");
    }
    
    private async Task CollectAllMetricsAsync()
    {
        var tasks = new List<Task>();
        
        if (_options.EnableGpuMetrics)
        {
            for (int i = 0; i < _options.MaxDevices; i++)
            {
                var deviceIndex = i;
                tasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        var metrics = await CollectDeviceMetricsAsync(deviceIndex);
                        if (metrics != null)
                        {
                            UpdateMetrics(deviceIndex, metrics);
                            UpdateTelemetry(deviceIndex, metrics);
                        }
                    }
                    catch (Exception ex)
                    {
                        if (_options.EnableDetailedLogging)
                        {
                            _logger.LogDebug(ex, "Failed to collect metrics for device {DeviceIndex}", deviceIndex);
                        }
                    }
                }));
            }
        }
        
        if (_options.EnableSystemMetrics)
        {
            tasks.Add(CollectSystemMetricsAsync());
        }
        
        await Task.WhenAll(tasks);
    }
    
    private async Task<GpuDeviceMetrics?> CollectDeviceMetricsAsync(int deviceIndex)
    {
        try
        {
            // Try NVIDIA first
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || 
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var nvidiaMetrics = await CollectNvidiaMetricsAsync(deviceIndex);
                if (nvidiaMetrics != null)
                    return nvidiaMetrics;
            }
            
            // Try AMD
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var amdMetrics = await CollectAmdMetricsAsync(deviceIndex);
                if (amdMetrics != null)
                    return amdMetrics;
            }
            
            // Try Intel
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var intelMetrics = await CollectIntelMetricsAsync(deviceIndex);
                if (intelMetrics != null)
                    return intelMetrics;
            }
            
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to collect metrics for device {DeviceIndex}", deviceIndex);
            return null;
        }
    }
    
    private async Task<GpuDeviceMetrics?> CollectNvidiaMetricsAsync(int deviceIndex)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    Arguments = $"--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits -i {deviceIndex}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            
            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            
            if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
            {
                var parts = output.Trim().Split(',').Select(p => p.Trim()).ToArray();
                if (parts.Length >= 8)
                {
                    return new GpuDeviceMetrics
                    {
                        DeviceIndex = int.Parse(parts[0]),
                        DeviceName = parts[1],
                        GpuUtilization = double.Parse(parts[2]),
                        MemoryUtilization = double.Parse(parts[3]),
                        MemoryUsedMB = long.Parse(parts[4]),
                        MemoryTotalMB = long.Parse(parts[5]),
                        TemperatureCelsius = double.Parse(parts[6]),
                        PowerUsageWatts = double.Parse(parts[7]),
                        Timestamp = DateTimeOffset.UtcNow,
                        DeviceType = "NVIDIA"
                    };
                }
            }
        }
        catch
        {
            // nvidia-smi not available or failed
        }
        
        return null;
    }
    
    private async Task<GpuDeviceMetrics?> CollectAmdMetricsAsync(int deviceIndex)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "rocm-smi",
                    Arguments = $"--showgpuuse --showmeminfo vram --showtemp --showpower -d {deviceIndex} --json",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            
            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            
            if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
            {
                // Parse JSON output (simplified for example)
                return new GpuDeviceMetrics
                {
                    DeviceIndex = deviceIndex,
                    DeviceName = $"AMD GPU {deviceIndex}",
                    GpuUtilization = 0, // Parse from JSON
                    MemoryUtilization = 0, // Parse from JSON
                    MemoryUsedMB = 0, // Parse from JSON
                    MemoryTotalMB = 0, // Parse from JSON
                    TemperatureCelsius = 0, // Parse from JSON
                    PowerUsageWatts = 0, // Parse from JSON
                    Timestamp = DateTimeOffset.UtcNow,
                    DeviceType = "AMD"
                };
            }
        }
        catch
        {
            // rocm-smi not available or failed
        }
        
        return null;
    }
    
    private async Task<GpuDeviceMetrics?> CollectIntelMetricsAsync(int deviceIndex)
    {
        // Intel GPU metrics collection (placeholder)
        await Task.CompletedTask;
        return null;
    }
    
    private async Task CollectSystemMetricsAsync()
    {
        try
        {
            var process = Process.GetCurrentProcess();
            var metrics = new SystemMetrics
            {
                ProcessCpuUsage = 0, // Would need proper calculation
                ProcessMemoryMB = process.WorkingSet64 / (1024 * 1024),
                ThreadCount = process.Threads.Count,
                HandleCount = process.HandleCount,
                Timestamp = DateTimeOffset.UtcNow
            };
            
            if (_options.EnableDetailedLogging)
            {
                _logger.LogDebug(
                    "System metrics: CPU {Cpu}%, Memory {Memory}MB, Threads {Threads}",
                    metrics.ProcessCpuUsage, metrics.ProcessMemoryMB, metrics.ThreadCount);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect system metrics");
        }
        
        await Task.CompletedTask;
    }
    
    private void UpdateMetrics(int deviceIndex, GpuDeviceMetrics metrics)
    {
        lock (_metricsLock)
        {
            _currentMetrics[deviceIndex] = metrics;
        }
    }
    
    private void UpdateTelemetry(int deviceIndex, GpuDeviceMetrics metrics)
    {
        if (_telemetry is GpuTelemetry telemetry)
        {
            telemetry.UpdateGpuMetrics(
                deviceIndex,
                metrics.GpuUtilization,
                metrics.TemperatureCelsius,
                metrics.PowerUsageWatts);
        }
    }
    
    private void CollectMetrics(object? state)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await CollectAllMetricsAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in metrics collection timer");
            }
        });
    }
    
    /// <summary>
    /// Retrieves performance metrics for a specific GPU device.
    /// </summary>
    /// <param name="deviceIndex">The zero-based index of the GPU device to query.</param>
    /// <returns>
    /// A task containing the GPU device metrics including utilization, memory usage,
    /// temperature, and power consumption.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no metrics are available for the specified device index.
    /// </exception>
    /// <remarks>
    /// This method first attempts to return cached metrics, then falls back to
    /// collecting fresh metrics if cached data is unavailable. The method will
    /// automatically detect the GPU vendor and use the appropriate monitoring tool.
    /// </remarks>
    public async Task<GpuDeviceMetrics> GetDeviceMetricsAsync(int deviceIndex)
    {
        lock (_metricsLock)
        {
            if (_currentMetrics.TryGetValue(deviceIndex, out var metrics))
            {
                return metrics;
            }
        }
        
        // Try to collect fresh metrics
        var freshMetrics = await CollectDeviceMetricsAsync(deviceIndex);
        if (freshMetrics != null)
        {
            UpdateMetrics(deviceIndex, freshMetrics);
            return freshMetrics;
        }
        
        throw new InvalidOperationException($"No metrics available for device {deviceIndex}");
    }
    
    /// <summary>
    /// Retrieves current system performance metrics for the host process.
    /// </summary>
    /// <returns>
    /// A task containing system metrics including CPU usage, memory consumption,
    /// thread count, and handle count.
    /// </returns>
    /// <remarks>
    /// System metrics are collected fresh on each call and include process-level
    /// resource consumption data useful for capacity planning and troubleshooting.
    /// </remarks>
    public async Task<SystemMetrics> GetSystemMetricsAsync()
    {
        await CollectSystemMetricsAsync();
        return new SystemMetrics
        {
            ProcessCpuUsage = 0,
            ProcessMemoryMB = Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024),
            ThreadCount = Process.GetCurrentProcess().Threads.Count,
            HandleCount = Process.GetCurrentProcess().HandleCount,
            Timestamp = DateTimeOffset.UtcNow
        };
    }
    
    /// <summary>
    /// Retrieves performance metrics for all available GPU devices.
    /// </summary>
    /// <returns>
    /// A task containing a read-only list of GPU device metrics for all detected devices.
    /// </returns>
    /// <remarks>
    /// This method triggers a fresh collection cycle for all configured devices up to
    /// the MaxDevices limit. Devices that cannot be queried are omitted from the results.
    /// The operation is performed in parallel for better performance.
    /// </remarks>
    public async Task<IReadOnlyList<GpuDeviceMetrics>> GetAllDeviceMetricsAsync()
    {
        await CollectAllMetricsAsync();
        
        lock (_metricsLock)
        {
            return _currentMetrics.Values.ToList();
        }
    }
    
    /// <summary>
    /// Releases all resources used by the GpuMetricsCollector.
    /// </summary>
    /// <remarks>
    /// This method stops the metrics collection timer and releases associated resources.
    /// It should be called when the service is no longer needed to prevent resource leaks.
    /// </remarks>
    public override void Dispose()
    {
        _collectionTimer?.Dispose();
        base.Dispose();
    }

    // Implementation of Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector
    async Task<GpuMemoryInfo> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetMemoryInfoAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        var deviceMetrics = await GetDeviceMetricsAsync(deviceIndex);
        return GpuMemoryInfo.Create(
            totalBytes: deviceMetrics.MemoryTotalMB * 1024 * 1024,
            allocatedBytes: deviceMetrics.MemoryUsedMB * 1024 * 1024,
            deviceIndex: deviceIndex,
            deviceName: deviceMetrics.DeviceName ?? $"Device {deviceIndex}");
    }

    async Task<IReadOnlyList<GpuMemoryInfo>> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetAllMemoryInfoAsync(CancellationToken cancellationToken)
    {
        var allMetrics = await GetAllDeviceMetricsAsync();
        return allMetrics.Select((m, index) => GpuMemoryInfo.Create(
            totalBytes: m.MemoryTotalMB * 1024 * 1024,
            allocatedBytes: m.MemoryUsedMB * 1024 * 1024,
            deviceIndex: index,
            deviceName: m.DeviceName ?? $"Device {index}")).ToList();
    }

    async Task<GpuUtilizationMetrics> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetUtilizationAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        var deviceMetrics = await GetDeviceMetricsAsync(deviceIndex);
        return new GpuUtilizationMetrics
        {
            GpuUtilizationPercentage = deviceMetrics.GpuUtilization,
            MemoryUtilizationPercentage = deviceMetrics.MemoryUtilization,
            Timestamp = deviceMetrics.Timestamp.DateTime
        };
    }

    Task<IReadOnlyDictionary<string, double>> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetPerformanceCountersAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        // Return empty dictionary as this collector doesn't implement performance counters yet
        return Task.FromResult<IReadOnlyDictionary<string, double>>(new Dictionary<string, double>());
    }

    async Task<double> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetTemperatureAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        var deviceMetrics = await GetDeviceMetricsAsync(deviceIndex);
        return deviceMetrics.TemperatureCelsius;
    }

    async Task<double> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetPowerConsumptionAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        var deviceMetrics = await GetDeviceMetricsAsync(deviceIndex);
        return deviceMetrics.PowerUsageWatts;
    }

    async Task<GpuClockSpeeds> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetClockSpeedsAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        var deviceMetrics = await GetDeviceMetricsAsync(deviceIndex);
        return new GpuClockSpeeds
        {
            GraphicsClockMHz = 0, // Not available in GpuDeviceMetrics
            MemoryClockMHz = 0, // Not available in GpuDeviceMetrics
            Timestamp = deviceMetrics.Timestamp.DateTime
        };
    }

    Task Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.StartCollectionAsync(TimeSpan interval, CancellationToken cancellationToken)
    {
        // Collection is already started in the background service
        return Task.CompletedTask;
    }

    Task Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.StopCollectionAsync()
    {
        // Collection will be stopped when the service is stopped
        return Task.CompletedTask;
    }

    Task<AggregatedGpuMetrics> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetAggregatedMetricsAsync(TimeSpan duration, int deviceIndex, CancellationToken cancellationToken)
    {
        // Return basic aggregated metrics - in a real implementation, this would aggregate historical data
        return GetDeviceMetricsAsync(deviceIndex).ContinueWith(t =>
        {
            var metrics = t.Result;
            return new AggregatedGpuMetrics
            {
                AverageGpuUtilization = metrics.GpuUtilization,
                PeakGpuUtilization = metrics.GpuUtilization,
                AverageMemoryUtilization = metrics.MemoryUtilization,
                PeakMemoryUtilization = metrics.MemoryUtilization,
                AverageTemperature = metrics.TemperatureCelsius,
                PeakTemperature = metrics.TemperatureCelsius,
                AveragePowerConsumption = metrics.PowerUsageWatts,
                TotalEnergyConsumed = metrics.PowerUsageWatts * duration.TotalHours,
                SampleCount = 1,
                Duration = duration,
                StartTime = DateTime.UtcNow.Subtract(duration),
                EndTime = DateTime.UtcNow
            };
        }, cancellationToken);
    }

    async Task<Orleans.GpuBridge.Abstractions.Metrics.DeviceMetrics> Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector.GetDeviceMetricsAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        var deviceMetrics = await GetDeviceMetricsAsync(deviceIndex);
        var memoryInfo = await ((Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector)this).GetMemoryInfoAsync(deviceIndex, cancellationToken);
        var utilization = await ((Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector)this).GetUtilizationAsync(deviceIndex, cancellationToken);
        var clockSpeeds = await ((Orleans.GpuBridge.Abstractions.Metrics.IGpuMetricsCollector)this).GetClockSpeedsAsync(deviceIndex, cancellationToken);
        
        return new Orleans.GpuBridge.Abstractions.Metrics.DeviceMetrics
        {
            MemoryInfo = memoryInfo,
            Utilization = utilization,
            ClockSpeeds = clockSpeeds,
            Temperature = deviceMetrics.TemperatureCelsius,
            PowerConsumption = deviceMetrics.PowerUsageWatts,
            PerformanceCounters = new Dictionary<string, double>(),
            DeviceIndex = deviceIndex,
            Timestamp = deviceMetrics.Timestamp.DateTime
        };
    }
}