using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Builders;
using Orleans.GpuBridge.DotCompute.Devices;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.DotCompute.Execution;
using Orleans.GpuBridge.DotCompute.Kernels;
using Orleans.GpuBridge.DotCompute.Memory;

namespace Orleans.GpuBridge.DotCompute;

/// <summary>
/// Main bridge implementation using DotCompute for GPU acceleration
/// </summary>
public sealed class DotComputeBridge : IGpuBridge, IDisposable
{
    private readonly ILogger<DotComputeBridge> _logger;
    private readonly DotComputeDeviceManager _deviceManager;
    private readonly KernelCompiler _kernelCompiler;
    private readonly KernelRegistry _kernelRegistry;
    private readonly KernelCatalog _kernelCatalog;
    private readonly IServiceProvider _serviceProvider;
    private readonly SemaphoreSlim _initLock;
    private bool _initialized;
    
    public DotComputeBridge(
        ILogger<DotComputeBridge> logger,
        DotComputeDeviceManager deviceManager,
        KernelCompiler kernelCompiler,
        KernelRegistry kernelRegistry,
        KernelCatalog kernelCatalog,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _deviceManager = deviceManager;
        _kernelCompiler = kernelCompiler;
        _kernelRegistry = kernelRegistry;
        _kernelCatalog = kernelCatalog;
        _serviceProvider = serviceProvider;
        _initLock = new SemaphoreSlim(1, 1);
    }
    
    public async ValueTask InitializeAsync(CancellationToken ct = default)
    {
        await _initLock.WaitAsync(ct);
        try
        {
            if (_initialized) return;
            
            _logger.LogInformation("Initializing DotCompute bridge");
            
            // Initialize device manager
            await _deviceManager.InitializeAsync(ct);
            
            // Scan for kernel methods in loaded assemblies
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                try
                {
                    _kernelRegistry.ScanAssembly(assembly);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, 
                        "Failed to scan assembly {Assembly} for kernels",
                        assembly.FullName);
                }
            }
            
            _initialized = true;
            
            _logger.LogInformation(
                "DotCompute bridge initialized with {DeviceCount} devices and {KernelCount} kernels",
                _deviceManager.GetDevices().Count,
                _kernelRegistry.GetAllRegistrations().Count());
        }
        finally
        {
            _initLock.Release();
        }
    }
    
    public async ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default)
    {
        await EnsureInitializedAsync(ct);
        
        var devices = _deviceManager.GetDevices();
        var totalMemory = devices.Sum(d => d.TotalMemory);
        var hasGpu = devices.Any(d => d.Type != DeviceType.CPU);
        
        var metadata = new Dictionary<string, object>
        {
            ["DeviceList"] = devices.Select(d => d.Name).ToList(),
            ["SupportedKernels"] = _kernelRegistry.GetAllRegistrations()
                .Select(r => r.Id.Value)
                .ToList()
        };
        
        return new GpuBridgeInfo(
            Version: "1.0.0",
            DeviceCount: devices.Count,
            TotalMemoryBytes: totalMemory,
            Backend: GpuBackend.OpenCL, // Default to OpenCL for now
            IsGpuAvailable: hasGpu,
            Metadata: metadata);
    }
    
    public async ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull
    {
        await EnsureInitializedAsync(ct);
        
        // First check if kernel is registered
        var registration = _kernelRegistry.GetRegistration(kernelId);
        if (registration != null)
        {
            // Select best device for this kernel
            var device = _deviceManager.GetBestDevice(new ComputeRequirements
            {
                PreferGpu = true,
                MinMemoryBytes = 256 * 1024 * 1024
            });
            
            // Compile kernel for device
            CompiledKernelInfo compiledInfo;
            
            if (registration.Method != null)
            {
                compiledInfo = await _kernelCompiler.CompileFromMethodAsync<TIn, TOut>(
                    registration.Method,
                    device,
                    ct);
            }
            else if (registration.SourceCode != null && registration.EntryPoint != null)
            {
                compiledInfo = await _kernelCompiler.CompileFromSourceAsync(
                    registration.SourceCode,
                    registration.EntryPoint,
                    device,
                    typeof(TIn),
                    typeof(TOut),
                    ct);
            }
            else
            {
                throw new InvalidOperationException(
                    $"Kernel {kernelId} has no implementation");
            }
            
            // Create kernel wrapper
            var kernelLogger = _serviceProvider.GetRequiredService<ILogger<DotComputeKernel<TIn, TOut>>>();
            return new CompiledDotComputeKernel<TIn, TOut>(
                kernelId,
                compiledInfo,
                device,
                kernelLogger);
        }
        
        // Fall back to kernel catalog (includes CPU fallback)
        return await _kernelCatalog.ResolveAsync<TIn, TOut>(kernelId, _serviceProvider);
    }
    
    public async ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default)
    {
        await EnsureInitializedAsync(ct);
        
        var devices = _deviceManager.GetDevices();
        return devices.Select(d => new GpuDevice(
            Index: d.Index,
            Name: d.Name,
            Type: d.Type,
            TotalMemoryBytes: d.TotalMemory,
            AvailableMemoryBytes: d.AvailableMemory,
            ComputeUnits: d.ComputeUnits,
            Capabilities: GetDeviceCapabilities(d))).ToList();
    }
    
    private string[] GetDeviceCapabilities(IComputeDevice device)
    {
        var capabilities = new List<string>();
        
        capabilities.Add(device.Type.ToString());
        
        if (device.Type == DeviceType.CUDA)
        {
            capabilities.Add("CUDA");
            capabilities.Add("PTX");
            capabilities.Add("Tensor Cores");
        }
        else if (device.Type == DeviceType.OpenCL)
        {
            capabilities.Add("OpenCL 2.0");
            capabilities.Add("SPIR-V");
        }
        else if (device.Type == DeviceType.DirectCompute)
        {
            capabilities.Add("DirectX 11");
            capabilities.Add("HLSL");
        }
        else if (device.Type == DeviceType.Metal)
        {
            capabilities.Add("Metal 2");
            capabilities.Add("MSL");
        }
        else if (device.Type == DeviceType.CPU)
        {
            capabilities.Add("AVX512");
            capabilities.Add("AVX2");
            capabilities.Add("Multi-threaded");
        }
        
        return capabilities.ToArray();
    }
    
    private async ValueTask EnsureInitializedAsync(CancellationToken ct)
    {
        if (!_initialized)
        {
            await InitializeAsync(ct);
        }
    }
    
    public void Dispose()
    {
        _kernelCompiler.ClearCache();
        _deviceManager.Dispose();
        _initLock?.Dispose();
    }
}

/// <summary>
/// Compiled DotCompute kernel wrapper
/// </summary>
internal sealed class CompiledDotComputeKernel<TIn, TOut> : DotComputeKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly CompiledKernelInfo _compiledInfo;
    
    public CompiledDotComputeKernel(
        KernelId kernelId,
        CompiledKernelInfo compiledInfo,
        IComputeDevice device,
        ILogger<DotComputeKernel<TIn, TOut>> logger)
        : base(kernelId, device, string.Empty, logger)
    {
        _compiledInfo = compiledInfo;
    }
    
    protected override async Task ExecuteKernelAsync(
        KernelExecutionContext context,
        CancellationToken ct)
    {
        try
        {
            var inputCount = context.InputData.Count;
            
            // Allocate buffers
            var inputBuffer = await AllocateInputBufferAsync(context.InputData, ct);
            var outputBuffer = await AllocateOutputBufferAsync(inputCount, ct);
            
            // Set kernel parameters
            _compiledInfo.CompiledKernel.SetBuffer(0, inputBuffer);
            _compiledInfo.CompiledKernel.SetBuffer(1, outputBuffer);
            _compiledInfo.CompiledKernel.SetConstant("size", inputCount);
            
            // Calculate launch parameters
            var workGroupSize = context.Hints?.MaxMicroBatch ?? 256;
            var globalWorkSize = ((inputCount + workGroupSize - 1) / workGroupSize) * workGroupSize;
            
            var launchParams = new KernelLaunchParams
            {
                GlobalWorkSize = globalWorkSize,
                LocalWorkSize = workGroupSize,
                Buffers = new Dictionary<int, IUnifiedBuffer<byte>>
                {
                    [0] = inputBuffer,
                    [1] = outputBuffer
                },
                Constants = new Dictionary<string, object>
                {
                    ["size"] = inputCount
                }
            };
            
            // Launch kernel
            var execution = await _device.LaunchKernelAsync(
                _compiledInfo.CompiledKernel,
                launchParams,
                ct);
            
            // Wait for completion
            await execution.WaitForCompletionAsync(ct);
            
            // Copy results back
            await outputBuffer.CopyFromDeviceAsync(ct);
            
            // Read results
            var results = await ReadResultsFromBufferAsync<TOut>(outputBuffer, inputCount, ct);
            context.Results.AddRange(results);
            
            _logger.LogDebug(
                "Kernel execution completed in {Time}ms for {Count} items",
                execution.GetExecutionTime().TotalMilliseconds,
                inputCount);
        }
        catch (Exception ex)
        {
            context.Error = ex;
            _logger.LogError(ex, "Kernel execution failed");
        }
        finally
        {
            context.IsComplete = true;
        }
    }
    
    private async Task<IUnifiedBuffer<byte>> AllocateInputBufferAsync(
        IReadOnlyList<TIn> data,
        CancellationToken ct)
    {
        // Serialize input data to byte buffer
        var serializedData = SerializeData(data);
        var buffer = await _device.AllocateBufferAsync<byte>(
            serializedData.Length,
            BufferFlags.ReadOnly | BufferFlags.HostVisible,
            ct);
        
        serializedData.CopyTo(buffer.Memory.Span);
        await buffer.CopyToDeviceAsync(ct);
        
        return buffer;
    }
    
    private async Task<IUnifiedBuffer<byte>> AllocateOutputBufferAsync(
        int count,
        CancellationToken ct)
    {
        // Estimate output buffer size
        var estimatedSize = count * EstimateItemSize<TOut>();
        var buffer = await _device.AllocateBufferAsync<byte>(
            estimatedSize,
            BufferFlags.WriteOnly | BufferFlags.HostVisible,
            ct);
        
        return buffer;
    }
    
    private byte[] SerializeData<T>(IReadOnlyList<T> data)
    {
        // Simple serialization for POD types
        // In production, use proper serialization
        if (typeof(T).IsPrimitive)
        {
            var size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            var bytes = new byte[data.Count * size];
            Buffer.BlockCopy(data.ToArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }
        
        // For complex types, use JSON or other serialization
        var json = System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(data);
        return json;
    }
    
    private Task<List<T>> ReadResultsFromBufferAsync<T>(
        IUnifiedBuffer<byte> buffer,
        int count,
        CancellationToken ct)
    {
        var results = new List<T>(count);
        
        // Deserialize results
        if (typeof(T).IsPrimitive)
        {
            var size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            var array = new T[count];
            Buffer.BlockCopy(buffer.Memory.ToArray(), 0, array, 0, count * size);
            results.AddRange(array);
        }
        else
        {
            // For complex types, use JSON or other deserialization
            var deserialized = System.Text.Json.JsonSerializer.Deserialize<List<T>>(
                buffer.Memory.Span);
            if (deserialized != null)
            {
                results.AddRange(deserialized);
            }
        }
        
        return Task.FromResult(results);
    }
    
    private int EstimateItemSize<T>()
    {
        if (typeof(T).IsPrimitive)
        {
            return System.Runtime.InteropServices.Marshal.SizeOf<T>();
        }
        
        // Estimate for complex types
        return 256;
    }
}

/// <summary>
/// Service collection extensions for DotCompute
/// </summary>
public static class DotComputeServiceExtensions
{
    public static IGpuBridgeBuilder AddDotCompute(
        this IGpuBridgeBuilder builder,
        Action<DotComputeOptions>? configure = null)
    {
        var services = builder.Services;
        
        // Configure options
        if (configure != null)
        {
            services.Configure(configure);
        }
        
        // Register DotCompute services
        services.AddSingleton<DotComputeDeviceManager>();
        services.AddSingleton<KernelCompiler>();
        services.AddSingleton<KernelRegistry>();
        services.AddSingleton<IGpuBridge, DotComputeBridge>();
        
        // Register as hosted service for initialization
        services.AddHostedService<DotComputeInitializerService>();
        
        return builder;
    }
}

/// <summary>
/// Background service to initialize DotCompute on startup
/// </summary>
internal sealed class DotComputeInitializerService : Microsoft.Extensions.Hosting.BackgroundService
{
    private readonly DotComputeBridge _bridge;
    private readonly ILogger<DotComputeInitializerService> _logger;
    
    public DotComputeInitializerService(
        IGpuBridge bridge,
        ILogger<DotComputeInitializerService> logger)
    {
        _bridge = (DotComputeBridge)bridge;
        _logger = logger;
    }
    
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            _logger.LogInformation("Starting DotCompute initialization");
            await _bridge.InitializeAsync(stoppingToken);
            _logger.LogInformation("DotCompute initialization completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize DotCompute");
            throw;
        }
    }
}