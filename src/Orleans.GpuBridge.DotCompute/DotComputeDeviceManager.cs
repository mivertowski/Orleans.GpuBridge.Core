using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.DotCompute.Configuration;
using Orleans.GpuBridge.DotCompute.Devices;
using Orleans.GpuBridge.DotCompute.Models;

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
            if (!_devices.Any(d => d.Type == DeviceType.CPU))
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
            ?? _devices.First(d => d.Type == DeviceType.CPU);
        
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
        if (_options.EnableOpenCL)
        {
            var opencl = new OpenCLBackendProvider(_logger);
            if (opencl.IsAvailable())
            {
                backends.Add(opencl);
            }
        }
        
        // Check DirectCompute availability (Windows only)
        if (_options.EnableDirectCompute && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var directCompute = new DirectGpuBackendProvider(_logger);
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
            GpuBackend.CUDA => backends.FirstOrDefault(b => b is CudaBackendProvider),
            GpuBackend.OpenCL => backends.FirstOrDefault(b => b is OpenCLBackendProvider),
            GpuBackend.DirectCompute => backends.FirstOrDefault(b => b is DirectGpuBackendProvider),
            GpuBackend.Metal => backends.FirstOrDefault(b => b is MetalBackendProvider),
            _ => backends.FirstOrDefault()
        } ?? backends.FirstOrDefault();
    }
    
    private double CalculateDeviceScore(IComputeDevice device, ComputeRequirements requirements)
    {
        double score = 0;
        
        // Device type preference
        score += device.Type switch
        {
            DeviceType.CUDA => requirements.PreferGpu ? 1000 : 100,
            DeviceType.OpenCL => requirements.PreferGpu ? 800 : 80,
            DeviceType.DirectCompute => requirements.PreferGpu ? 600 : 60,
            DeviceType.Metal => requirements.PreferGpu ? 700 : 70,
            DeviceType.CPU => requirements.PreferGpu ? 0 : 500,
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

