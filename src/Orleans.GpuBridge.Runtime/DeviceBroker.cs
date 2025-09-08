using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Management;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Manages GPU devices and work distribution
/// </summary>
public sealed class DeviceBroker : IDisposable
{
    private readonly ILogger<DeviceBroker> _logger;
    private readonly GpuBridgeOptions _options;
    private readonly List<GpuDevice> _devices;
    private readonly ConcurrentDictionary<int, DeviceWorkQueue> _workQueues;
    private readonly SemaphoreSlim _initLock;
    private readonly Timer _monitoringTimer;
    private bool _initialized;
    private bool _disposed;
    
    public int DeviceCount => _devices.Count;
    public long TotalMemoryBytes => _devices.Sum(d => d.TotalMemoryBytes);
    public int CurrentQueueDepth => _workQueues.Values.Sum(q => q.QueuedItems);
    
    public DeviceBroker(
        ILogger<DeviceBroker> logger,
        IOptions<GpuBridgeOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        _devices = new List<GpuDevice>();
        _workQueues = new ConcurrentDictionary<int, DeviceWorkQueue>();
        _initLock = new SemaphoreSlim(1, 1);
        
        // Start monitoring timer for device health
        _monitoringTimer = new Timer(
            MonitorDeviceHealth,
            null,
            TimeSpan.FromSeconds(30),
            TimeSpan.FromSeconds(30));
    }
    
    public async Task InitializeAsync(CancellationToken ct)
    {
        await _initLock.WaitAsync(ct);
        try
        {
            if (_initialized) return;
            
            _logger.LogInformation("Initializing device broker with GPU detection");
            
            // Detect physical GPUs
            await DetectGpuDevicesAsync(ct);
            
            // Always add CPU fallback device
            AddCpuDevice();
            
            // Initialize work queues for each device
            foreach (var device in _devices)
            {
                _workQueues[device.Index] = new DeviceWorkQueue(device);
            }
            
            _initialized = true;
            _logger.LogInformation(
                "Device broker initialized with {Count} devices, {Memory:N0} bytes total memory",
                _devices.Count, TotalMemoryBytes);
        }
        finally
        {
            _initLock.Release();
        }
    }
    
    private async Task DetectGpuDevicesAsync(CancellationToken ct)
    {
        var detectedDevices = new List<GpuDevice>();
        
        // Try to detect NVIDIA GPUs
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || 
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            detectedDevices.AddRange(await DetectNvidiaGpusAsync(ct));
        }
        
        // Try to detect AMD GPUs
        detectedDevices.AddRange(await DetectAmdGpusAsync(ct));
        
        // Try to detect Intel GPUs
        detectedDevices.AddRange(await DetectIntelGpusAsync(ct));
        
        // On macOS, detect Metal devices
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            detectedDevices.AddRange(await DetectMetalDevicesAsync(ct));
        }
        
        _devices.AddRange(detectedDevices);
        
        if (detectedDevices.Count > 0)
        {
            _logger.LogInformation(
                "Detected {Count} GPU devices",
                detectedDevices.Count);
        }
        else if (_options.PreferGpu)
        {
            _logger.LogWarning(
                "GPU requested but no physical GPUs detected, will use CPU fallback");
        }
    }
    
    private async Task<List<GpuDevice>> DetectNvidiaGpusAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            // Try nvidia-smi for detailed GPU information
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    Arguments = "--query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            
            if (process.Start())
            {
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync(ct);
                
                if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                {
                    var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        var parts = line.Split(',').Select(p => p.Trim()).ToArray();
                        if (parts.Length >= 5)
                        {
                            var device = new GpuDevice(
                                Index: int.Parse(parts[0]),
                                Name: parts[1],
                                Type: DeviceType.CUDA,
                                TotalMemoryBytes: long.Parse(parts[2]) * 1024 * 1024,
                                AvailableMemoryBytes: long.Parse(parts[3]) * 1024 * 1024,
                                ComputeUnits: GetCudaCoreCount(parts[1]),
                                Capabilities: new[]
                                {
                                    "CUDA",
                                    $"Compute {parts[4]}",
                                    "Tensor Cores",
                                    "NVENC",
                                    "NVDEC"
                                });
                            
                            devices.Add(device);
                            _logger.LogInformation(
                                "Detected NVIDIA GPU: {Name} with {Memory:N0} MB memory",
                                device.Name, device.TotalMemoryBytes / (1024 * 1024));
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "nvidia-smi not available or failed");
        }
        
        return devices;
    }
    
    private async Task<List<GpuDevice>> DetectAmdGpusAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            // Try rocm-smi for AMD GPUs
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "rocm-smi",
                    Arguments = "--showid --showname --showmeminfo vram",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            
            if (process.Start())
            {
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync(ct);
                
                if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
                {
                    // Parse AMD GPU information
                    _logger.LogDebug("AMD GPU detection output: {Output}", output);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "rocm-smi not available");
        }
        
        return devices;
    }
    
    private async Task<List<GpuDevice>> DetectIntelGpusAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        // Intel GPU detection through Level Zero or OpenCL
        await Task.CompletedTask; // Placeholder for actual implementation
        
        return devices;
    }
    
    private async Task<List<GpuDevice>> DetectMetalDevicesAsync(CancellationToken ct)
    {
        var devices = new List<GpuDevice>();
        
        try
        {
            // Use system_profiler on macOS
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "system_profiler",
                    Arguments = "SPDisplaysDataType -json",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                }
            };
            
            if (process.Start())
            {
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync(ct);
                
                if (process.ExitCode == 0)
                {
                    // Parse JSON output for GPU information
                    _logger.LogDebug("Metal device detection output received");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "system_profiler not available");
        }
        
        return devices;
    }
    
    private void AddCpuDevice()
    {
        var cpuInfo = GetCpuInfo();
        var cpuDevice = new GpuDevice(
            Index: _devices.Count,
            Name: cpuInfo.Name,
            Type: DeviceType.CPU,
            TotalMemoryBytes: cpuInfo.TotalMemory,
            AvailableMemoryBytes: cpuInfo.AvailableMemory,
            ComputeUnits: Environment.ProcessorCount,
            Capabilities: cpuInfo.Capabilities);
        
        _devices.Add(cpuDevice);
    }
    
    private CpuInfo GetCpuInfo()
    {
        var name = "Unknown CPU";
        var capabilities = new List<string> { "CPU", "Multi-threaded" };
        
        try
        {
            // Get CPU name and features
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                using var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_Processor");
                foreach (var obj in searcher.Get())
                {
                    name = obj["Name"]?.ToString() ?? name;
                    break;
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var cpuinfo = System.IO.File.ReadAllText("/proc/cpuinfo");
                var modelLine = cpuinfo.Split('\n')
                    .FirstOrDefault(l => l.StartsWith("model name"));
                if (modelLine != null)
                {
                    name = modelLine.Split(':')[1].Trim();
                }
            }
            
            // Detect SIMD capabilities
            if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
                capabilities.Add("AVX512");
            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                capabilities.Add("AVX2");
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported)
                capabilities.Add("AVX");
            if (System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported)
                capabilities.Add("NEON");
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to get detailed CPU information");
        }
        
        var totalMemory = GC.GetTotalMemory(false) + Environment.WorkingSet;
        var availableMemory = totalMemory - GC.GetTotalMemory(false);
        
        return new CpuInfo
        {
            Name = name,
            TotalMemory = totalMemory,
            AvailableMemory = availableMemory,
            Capabilities = capabilities.ToArray()
        };
    }
    
    private int GetCudaCoreCount(string gpuName)
    {
        // Estimate CUDA core count based on GPU model
        return gpuName switch
        {
            var n when n.Contains("4090") => 16384,
            var n when n.Contains("4080") => 9728,
            var n when n.Contains("4070") => 5888,
            var n when n.Contains("3090") => 10496,
            var n when n.Contains("3080") => 8704,
            var n when n.Contains("3070") => 5888,
            var n when n.Contains("A100") => 6912,
            var n when n.Contains("V100") => 5120,
            _ => 1024 // Default estimate
        };
    }
    
    private void MonitorDeviceHealth(object? state)
    {
        if (_disposed) return;
        
        try
        {
            foreach (var queue in _workQueues.Values)
            {
                var metrics = queue.GetMetrics();
                if (metrics.ErrorRate > 0.1) // More than 10% errors
                {
                    _logger.LogWarning(
                        "Device {Index} has high error rate: {ErrorRate:P}",
                        queue.Device.Index, metrics.ErrorRate);
                }
                
                if (metrics.QueuedItems > 1000)
                {
                    _logger.LogWarning(
                        "Device {Index} has large queue: {Items} items",
                        queue.Device.Index, metrics.QueuedItems);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error monitoring device health");
        }
    }
    
    public async Task ShutdownAsync(CancellationToken ct)
    {
        _logger.LogInformation("Shutting down device broker");
        
        // Stop all work queues
        var shutdownTasks = _workQueues.Values
            .Select(q => q.ShutdownAsync(ct))
            .ToArray();
        
        await Task.WhenAll(shutdownTasks);
        
        _workQueues.Clear();
        _devices.Clear();
        _initialized = false;
    }
    
    public IReadOnlyList<GpuDevice> GetDevices()
    {
        EnsureInitialized();
        return _devices.AsReadOnly();
    }
    
    public GpuDevice? GetDevice(int index)
    {
        EnsureInitialized();
        return _devices.FirstOrDefault(d => d.Index == index);
    }
    
    public GpuDevice? GetBestDevice()
    {
        EnsureInitialized();
        
        // Score devices based on availability and performance
        return _devices
            .Select(d => new
            {
                Device = d,
                Score = CalculateDeviceScore(d)
            })
            .OrderByDescending(x => x.Score)
            .Select(x => x.Device)
            .FirstOrDefault();
    }
    
    private double CalculateDeviceScore(GpuDevice device)
    {
        var queue = _workQueues.GetValueOrDefault(device.Index);
        if (queue == null) return 0;
        
        var metrics = queue.GetMetrics();
        
        // Calculate score based on multiple factors
        double score = 0;
        
        // Memory availability (40% weight)
        score += (device.AvailableMemoryBytes / (double)device.TotalMemoryBytes) * 40;
        
        // Queue depth (30% weight) - lower is better
        score += Math.Max(0, 30 - (metrics.QueuedItems / 10.0));
        
        // Success rate (20% weight)
        score += (1 - metrics.ErrorRate) * 20;
        
        // Device type preference (10% weight)
        score += device.Type switch
        {
            DeviceType.CUDA => 10,
            DeviceType.OpenCL => 8,
            DeviceType.Metal => 7,
            DeviceType.DirectCompute => 6,
            DeviceType.CPU => 5,
            _ => 0
        };
        
        return score;
    }
    
    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Device broker not initialized");
        }
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        _disposed = true;
        _monitoringTimer?.Dispose();
        _initLock?.Dispose();
        
        // Shutdown all queues
        var shutdownTask = ShutdownAsync(CancellationToken.None);
        shutdownTask.GetAwaiter().GetResult();
    }
    
    private sealed class CpuInfo
    {
        public string Name { get; set; } = default!;
        public long TotalMemory { get; set; }
        public long AvailableMemory { get; set; }
        public string[] Capabilities { get; set; } = default!;
    }
}

/// <summary>
/// Work queue for a specific device
/// </summary>
internal sealed class DeviceWorkQueue
{
    private readonly GpuDevice _device;
    private readonly ConcurrentQueue<WorkItem> _queue;
    private readonly SemaphoreSlim _semaphore;
    private readonly CancellationTokenSource _cts;
    private readonly Task _processingTask;
    private long _processedItems;
    private long _failedItems;
    private bool _shutdown;
    
    public GpuDevice Device => _device;
    public int QueuedItems => _queue.Count;
    
    public DeviceWorkQueue(GpuDevice device)
    {
        _device = device;
        _queue = new ConcurrentQueue<WorkItem>();
        _semaphore = new SemaphoreSlim(0);
        _cts = new CancellationTokenSource();
        _processingTask = ProcessQueueAsync(_cts.Token);
    }
    
    public Task<WorkHandle> EnqueueAsync(
        Func<CancellationToken, Task> work,
        CancellationToken ct = default)
    {
        if (_shutdown)
        {
            throw new InvalidOperationException("Queue is shutting down");
        }
        
        var item = new WorkItem
        {
            Id = Guid.NewGuid().ToString(),
            Work = work,
            CompletionSource = new TaskCompletionSource(),
            EnqueuedAt = DateTime.UtcNow
        };
        
        _queue.Enqueue(item);
        _semaphore.Release();
        
        return Task.FromResult(new WorkHandle(item.Id, item.CompletionSource.Task));
    }
    
    private async Task ProcessQueueAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await _semaphore.WaitAsync(ct);
                
                if (_queue.TryDequeue(out var item))
                {
                    try
                    {
                        await item.Work(ct);
                        item.CompletionSource.SetResult();
                        Interlocked.Increment(ref _processedItems);
                    }
                    catch (Exception ex)
                    {
                        item.CompletionSource.SetException(ex);
                        Interlocked.Increment(ref _failedItems);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }
    
    public DeviceMetrics GetMetrics()
    {
        var processed = Interlocked.Read(ref _processedItems);
        var failed = Interlocked.Read(ref _failedItems);
        var total = processed + failed;
        
        return new DeviceMetrics
        {
            QueuedItems = _queue.Count,
            ProcessedItems = processed,
            FailedItems = failed,
            ErrorRate = total > 0 ? failed / (double)total : 0
        };
    }
    
    public async Task ShutdownAsync(CancellationToken ct)
    {
        _shutdown = true;
        _cts.Cancel();
        
        try
        {
            await _processingTask.WaitAsync(ct);
        }
        catch (OperationCanceledException)
        {
            // Expected
        }
        
        // Complete remaining items with cancellation
        while (_queue.TryDequeue(out var item))
        {
            item.CompletionSource.SetCanceled();
        }
    }
    
    private sealed class WorkItem
    {
        public string Id { get; init; } = default!;
        public Func<CancellationToken, Task> Work { get; init; } = default!;
        public TaskCompletionSource CompletionSource { get; init; } = default!;
        public DateTime EnqueuedAt { get; init; }
    }
}

internal sealed record WorkHandle(string Id, Task CompletionTask);

internal sealed class DeviceMetrics
{
    public int QueuedItems { get; init; }
    public long ProcessedItems { get; init; }
    public long FailedItems { get; init; }
    public double ErrorRate { get; init; }
}
