using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Orleans.GpuBridge.Diagnostics;

public class GpuMetricsOptions
{
    public TimeSpan CollectionInterval { get; set; } = TimeSpan.FromSeconds(10);
    public bool EnableSystemMetrics { get; set; } = true;
    public bool EnableGpuMetrics { get; set; } = true;
    public bool EnableDetailedLogging { get; set; } = false;
    public int MaxDevices { get; set; } = 8;
}

public interface IGpuMetricsCollector
{
    Task<GpuDeviceMetrics> GetDeviceMetricsAsync(int deviceIndex);
    Task<SystemMetrics> GetSystemMetricsAsync();
    Task<IReadOnlyList<GpuDeviceMetrics>> GetAllDeviceMetricsAsync();
}

public sealed class GpuMetricsCollector : BackgroundService, IGpuMetricsCollector
{
    private readonly ILogger<GpuMetricsCollector> _logger;
    private readonly IGpuTelemetry _telemetry;
    private readonly GpuMetricsOptions _options;
    private readonly Timer _collectionTimer;
    private readonly Dictionary<int, GpuDeviceMetrics> _currentMetrics = new();
    private readonly object _metricsLock = new();
    
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
    
    public async Task<IReadOnlyList<GpuDeviceMetrics>> GetAllDeviceMetricsAsync()
    {
        await CollectAllMetricsAsync();
        
        lock (_metricsLock)
        {
            return _currentMetrics.Values.ToList();
        }
    }
    
    public override void Dispose()
    {
        _collectionTimer?.Dispose();
        base.Dispose();
    }
}

public class GpuDeviceMetrics
{
    public int DeviceIndex { get; set; }
    public string DeviceName { get; set; } = string.Empty;
    public string DeviceType { get; set; } = string.Empty;
    public double GpuUtilization { get; set; }
    public double MemoryUtilization { get; set; }
    public long MemoryUsedMB { get; set; }
    public long MemoryTotalMB { get; set; }
    public double TemperatureCelsius { get; set; }
    public double PowerUsageWatts { get; set; }
    public DateTimeOffset Timestamp { get; set; }
    
    public long MemoryAvailableMB => MemoryTotalMB - MemoryUsedMB;
    public double MemoryUsagePercent => MemoryTotalMB > 0 ? (MemoryUsedMB * 100.0 / MemoryTotalMB) : 0;
}

public class SystemMetrics
{
    public double ProcessCpuUsage { get; set; }
    public long ProcessMemoryMB { get; set; }
    public int ThreadCount { get; set; }
    public int HandleCount { get; set; }
    public DateTimeOffset Timestamp { get; set; }
}