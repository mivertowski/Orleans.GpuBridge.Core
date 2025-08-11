using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.DotCompute;

/// <summary>
/// Manages DotCompute devices and backend selection
/// </summary>
public sealed class DotComputeDeviceManager : IDisposable
{
    private readonly ILogger<DotComputeDeviceManager> _logger;
    private readonly DotComputeOptions _options;
    private readonly List<IComputeDevice> _devices;
    private readonly SemaphoreSlim _initLock;
    private bool _initialized;
    private IBackendProvider? _activeBackend;
    
    public DotComputeDeviceManager(
        ILogger<DotComputeDeviceManager> logger,
        IOptions<DotComputeOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        _devices = new List<IComputeDevice>();
        _initLock = new SemaphoreSlim(1, 1);
    }
    
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await _initLock.WaitAsync(ct);
        try
        {
            if (_initialized) return;
            
            _logger.LogInformation("Initializing DotCompute device manager");
            
            // Select backend based on platform and availability
            _activeBackend = SelectBackend();
            
            if (_activeBackend == null)
            {
                throw new InvalidOperationException("No compatible compute backend found");
            }
            
            _logger.LogInformation(
                "Selected backend: {Backend}",
                _activeBackend.Name);
            
            // Initialize backend and enumerate devices
            await _activeBackend.InitializeAsync(ct);
            
            var devices = await _activeBackend.EnumerateDevicesAsync(ct);
            _devices.AddRange(devices);
            
            // Always add CPU fallback device
            if (!_devices.Any(d => d.Type == DeviceType.Cpu))
            {
                _devices.Add(new CpuComputeDevice(_logger));
            }
            
            _initialized = true;
            
            _logger.LogInformation(
                "Initialized {Count} compute devices: {Devices}",
                _devices.Count,
                string.Join(", ", _devices.Select(d => $"{d.Name} ({d.Type})")));
        }
        finally
        {
            _initLock.Release();
        }
    }
    
    public IReadOnlyList<IComputeDevice> GetDevices()
    {
        EnsureInitialized();
        return _devices.AsReadOnly();
    }
    
    public IComputeDevice? GetDevice(int index)
    {
        EnsureInitialized();
        return _devices.FirstOrDefault(d => d.Index == index);
    }
    
    public IComputeDevice GetBestDevice(ComputeRequirements? requirements = null)
    {
        EnsureInitialized();
        
        requirements ??= new ComputeRequirements();
        
        // Score each device based on requirements
        var scoredDevices = _devices
            .Where(d => d.IsAvailable)
            .Select(d => new
            {
                Device = d,
                Score = CalculateDeviceScore(d, requirements)
            })
            .OrderByDescending(x => x.Score)
            .ToList();
        
        var selected = scoredDevices.FirstOrDefault()?.Device 
            ?? _devices.First(d => d.Type == DeviceType.Cpu);
        
        _logger.LogDebug(
            "Selected device {Device} for requirements {Requirements}",
            selected.Name, requirements);
        
        return selected;
    }
    
    private IBackendProvider? SelectBackend()
    {
        var backends = new List<IBackendProvider>();
        
        // Check CUDA availability
        if (_options.EnableCuda && RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || 
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            var cuda = new CudaBackendProvider(_logger);
            if (cuda.IsAvailable())
            {
                backends.Add(cuda);
            }
        }
        
        // Check OpenCL availability
        if (_options.EnableOpenCl)
        {
            var opencl = new OpenClBackendProvider(_logger);
            if (opencl.IsAvailable())
            {
                backends.Add(opencl);
            }
        }
        
        // Check DirectCompute availability (Windows only)
        if (_options.EnableDirectCompute && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var directCompute = new DirectComputeBackendProvider(_logger);
            if (directCompute.IsAvailable())
            {
                backends.Add(directCompute);
            }
        }
        
        // Check Metal availability (macOS only)
        if (_options.EnableMetal && RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            var metal = new MetalBackendProvider(_logger);
            if (metal.IsAvailable())
            {
                backends.Add(metal);
            }
        }
        
        // Select backend based on priority
        return _options.PreferredBackend switch
        {
            ComputeBackend.Cuda => backends.FirstOrDefault(b => b is CudaBackendProvider),
            ComputeBackend.OpenCl => backends.FirstOrDefault(b => b is OpenClBackendProvider),
            ComputeBackend.DirectCompute => backends.FirstOrDefault(b => b is DirectComputeBackendProvider),
            ComputeBackend.Metal => backends.FirstOrDefault(b => b is MetalBackendProvider),
            _ => backends.FirstOrDefault()
        } ?? backends.FirstOrDefault();
    }
    
    private double CalculateDeviceScore(IComputeDevice device, ComputeRequirements requirements)
    {
        double score = 0;
        
        // Device type preference
        score += device.Type switch
        {
            DeviceType.Cuda => requirements.PreferGpu ? 1000 : 100,
            DeviceType.OpenCl => requirements.PreferGpu ? 800 : 80,
            DeviceType.DirectCompute => requirements.PreferGpu ? 600 : 60,
            DeviceType.Metal => requirements.PreferGpu ? 700 : 70,
            DeviceType.Cpu => requirements.PreferGpu ? 0 : 500,
            _ => 0
        };
        
        // Memory requirements
        if (device.AvailableMemory >= requirements.MinMemoryBytes)
        {
            score += 100;
            score += Math.Min(100, device.AvailableMemory / (double)requirements.MinMemoryBytes * 10);
        }
        
        // Compute units
        score += Math.Min(100, device.ComputeUnits * 5);
        
        // Availability
        if (device.IsAvailable)
        {
            score += 50;
        }
        
        return score;
    }
    
    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Device manager not initialized");
        }
    }
    
    public void Dispose()
    {
        _activeBackend?.Dispose();
        _devices.Clear();
        _initLock?.Dispose();
    }
}

/// <summary>
/// DotCompute configuration options
/// </summary>
public sealed class DotComputeOptions
{
    public bool EnableCuda { get; set; } = true;
    public bool EnableOpenCl { get; set; } = true;
    public bool EnableDirectCompute { get; set; } = true;
    public bool EnableMetal { get; set; } = true;
    public ComputeBackend PreferredBackend { get; set; } = ComputeBackend.Auto;
    public bool EnableKernelCaching { get; set; } = true;
    public int MaxCachedKernels { get; set; } = 100;
    public bool EnableMemoryPooling { get; set; } = true;
    public long MaxPooledMemoryBytes { get; set; } = 1024L * 1024 * 1024; // 1GB
}

/// <summary>
/// Compute backend type
/// </summary>
public enum ComputeBackend
{
    Auto,
    Cuda,
    OpenCl,
    DirectCompute,
    Metal,
    Cpu
}

/// <summary>
/// Requirements for device selection
/// </summary>
public sealed class ComputeRequirements
{
    public bool PreferGpu { get; set; } = true;
    public long MinMemoryBytes { get; set; } = 256 * 1024 * 1024; // 256MB
    public int MinComputeUnits { get; set; } = 1;
    public DeviceType? RequiredType { get; set; }
}

/// <summary>
/// Backend provider interface
/// </summary>
public interface IBackendProvider : IDisposable
{
    string Name { get; }
    bool IsAvailable();
    Task InitializeAsync(CancellationToken ct = default);
    Task<IReadOnlyList<IComputeDevice>> EnumerateDevicesAsync(CancellationToken ct = default);
}

/// <summary>
/// CPU compute device for fallback
/// </summary>
internal sealed class CpuComputeDevice : IComputeDevice
{
    private readonly ILogger _logger;
    
    public string Name => "CPU (Multi-threaded SIMD)";
    public DeviceType Type => DeviceType.Cpu;
    public int Index => -1;
    public long TotalMemory => Environment.WorkingSet;
    public long AvailableMemory => Environment.WorkingSet / 2;
    public int ComputeUnits => Environment.ProcessorCount;
    public bool IsAvailable => true;
    
    public CpuComputeDevice(ILogger logger)
    {
        _logger = logger;
    }
    
    public Task<IUnifiedBuffer<T>> AllocateBufferAsync<T>(
        int size,
        BufferFlags flags = BufferFlags.ReadWrite,
        CancellationToken ct = default) where T : unmanaged
    {
        return Task.FromResult<IUnifiedBuffer<T>>(
            new CpuUnifiedBuffer<T>(size, flags));
    }
    
    public Task<ICompiledKernel> CompileKernelAsync(
        string code,
        string entryPoint,
        CompilationOptions? options = null,
        CancellationToken ct = default)
    {
        // CPU kernels don't need compilation
        return Task.FromResult<ICompiledKernel>(
            new CpuCompiledKernel(entryPoint, this));
    }
    
    public Task<IKernelExecution> LaunchKernelAsync(
        ICompiledKernel kernel,
        KernelLaunchParams launchParams,
        CancellationToken ct = default)
    {
        return Task.FromResult<IKernelExecution>(
            new CpuKernelExecution(launchParams));
    }
}

/// <summary>
/// CPU unified buffer implementation
/// </summary>
internal sealed class CpuUnifiedBuffer<T> : IUnifiedBuffer<T> where T : unmanaged
{
    private readonly T[] _data;
    private readonly BufferFlags _flags;
    private bool _disposed;
    
    public int Length => _disposed ? throw new ObjectDisposedException(nameof(CpuUnifiedBuffer<T>)) : _data.Length;
    public Memory<T> Memory => _disposed ? throw new ObjectDisposedException(nameof(CpuUnifiedBuffer<T>)) : _data.AsMemory();
    public bool IsResident => !_disposed;
    
    public CpuUnifiedBuffer(int size, BufferFlags flags)
    {
        _data = new T[size];
        _flags = flags;
    }
    
    public Task CopyToDeviceAsync(CancellationToken ct = default)
    {
        // No-op for CPU
        return Task.CompletedTask;
    }
    
    public Task CopyFromDeviceAsync(CancellationToken ct = default)
    {
        // No-op for CPU
        return Task.CompletedTask;
    }
    
    public Task<IUnifiedBuffer<T>> CloneAsync(CancellationToken ct = default)
    {
        var clone = new CpuUnifiedBuffer<T>(_data.Length, _flags);
        _data.CopyTo(clone._data, 0);
        return Task.FromResult<IUnifiedBuffer<T>>(clone);
    }
    
    public void Dispose()
    {
        _disposed = true;
    }
}

/// <summary>
/// CPU compiled kernel (no actual compilation needed)
/// </summary>
internal sealed class CpuCompiledKernel : ICompiledKernel
{
    private readonly Dictionary<int, IUnifiedBuffer<byte>> _buffers = new();
    private readonly Dictionary<string, object> _constants = new();
    
    public string Name { get; }
    public IComputeDevice Device { get; }
    
    public CpuCompiledKernel(string name, IComputeDevice device)
    {
        Name = name;
        Device = device;
    }
    
    public void SetBuffer(int index, IUnifiedBuffer<byte> buffer)
    {
        _buffers[index] = buffer;
    }
    
    public void SetConstant<T>(string name, T value) where T : unmanaged
    {
        _constants[name] = value;
    }
    
    public void Dispose()
    {
        _buffers.Clear();
        _constants.Clear();
    }
}

/// <summary>
/// CPU kernel execution
/// </summary>
internal sealed class CpuKernelExecution : IKernelExecution
{
    private readonly DateTime _startTime;
    private readonly DateTime _endTime;
    
    public bool IsComplete => true;
    
    public CpuKernelExecution(KernelLaunchParams launchParams)
    {
        _startTime = DateTime.UtcNow;
        // Simulate execution
        Thread.Sleep(1);
        _endTime = DateTime.UtcNow;
    }
    
    public Task WaitForCompletionAsync(CancellationToken ct = default)
    {
        return Task.CompletedTask;
    }
    
    public TimeSpan GetExecutionTime()
    {
        return _endTime - _startTime;
    }
}

/// <summary>
/// Placeholder backend providers (to be implemented when DotCompute is available)
/// </summary>
internal abstract class BackendProviderBase : IBackendProvider
{
    protected readonly ILogger _logger;
    
    public abstract string Name { get; }
    
    protected BackendProviderBase(ILogger logger)
    {
        _logger = logger;
    }
    
    public virtual bool IsAvailable()
    {
        // Check if runtime libraries are available
        return false;
    }
    
    public virtual Task InitializeAsync(CancellationToken ct = default)
    {
        _logger.LogInformation("Initializing {Backend} backend", Name);
        return Task.CompletedTask;
    }
    
    public virtual Task<IReadOnlyList<IComputeDevice>> EnumerateDevicesAsync(CancellationToken ct = default)
    {
        return Task.FromResult<IReadOnlyList<IComputeDevice>>(Array.Empty<IComputeDevice>());
    }
    
    public virtual void Dispose()
    {
        // Cleanup
    }
}

internal sealed class CudaBackendProvider : BackendProviderBase
{
    public override string Name => "CUDA";
    public CudaBackendProvider(ILogger logger) : base(logger) { }
}

internal sealed class OpenClBackendProvider : BackendProviderBase
{
    public override string Name => "OpenCL";
    public OpenClBackendProvider(ILogger logger) : base(logger) { }
}

internal sealed class DirectComputeBackendProvider : BackendProviderBase
{
    public override string Name => "DirectCompute";
    public DirectComputeBackendProvider(ILogger logger) : base(logger) { }
}

internal sealed class MetalBackendProvider : BackendProviderBase
{
    public override string Name => "Metal";
    public MetalBackendProvider(ILogger logger) : base(logger) { }
}