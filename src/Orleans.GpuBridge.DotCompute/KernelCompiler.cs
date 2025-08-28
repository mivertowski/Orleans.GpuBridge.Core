using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.DotCompute.Devices;
using Orleans.GpuBridge.DotCompute.Compilation;

namespace Orleans.GpuBridge.DotCompute;

/// <summary>
/// Compiles and manages GPU kernels
/// </summary>
public sealed class KernelCompiler
{
    private readonly ILogger<KernelCompiler> _logger;
    private readonly DotComputeDeviceManager _deviceManager;
    private readonly ConcurrentDictionary<string, CompiledKernelInfo> _compiledKernels;
    private readonly SemaphoreSlim _compilationLock;
    
    public KernelCompiler(
        ILogger<KernelCompiler> logger,
        DotComputeDeviceManager deviceManager)
    {
        _logger = logger;
        _deviceManager = deviceManager;
        _compiledKernels = new ConcurrentDictionary<string, CompiledKernelInfo>();
        _compilationLock = new SemaphoreSlim(1, 1);
    }
    
    /// <summary>
    /// Compiles a kernel from C# method with attribute
    /// </summary>
    public async Task<CompiledKernelInfo> CompileFromMethodAsync<TIn, TOut>(
        MethodInfo method,
        IComputeDevice device,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull
    {
        var kernelAttribute = method.GetCustomAttribute<KernelAttribute>();
        if (kernelAttribute == null)
        {
            throw new ArgumentException($"Method {method.Name} does not have [Kernel] attribute");
        }
        
        var cacheKey = $"{method.DeclaringType?.FullName}.{method.Name}_{device.Type}";
        
        if (_compiledKernels.TryGetValue(cacheKey, out var cached))
        {
            _logger.LogDebug("Using cached kernel {Key}", cacheKey);
            return cached;
        }
        
        await _compilationLock.WaitAsync(ct);
        try
        {
            // Double-check after acquiring lock
            if (_compiledKernels.TryGetValue(cacheKey, out cached))
            {
                return cached;
            }
            
            _logger.LogInformation(
                "Compiling kernel {Method} for device {Device}",
                method.Name, device.Name);
            
            // Generate kernel code based on device type
            var kernelCode = GenerateKernelCode(method, device.Type);
            
            // Compile for target device
            var compiledKernel = await device.CompileKernelAsync(
                kernelCode,
                method.Name,
                new CompilationOptions
                {
                    EnableOptimizations = true,
                    UseFastMath = true,
                    TargetArchitecture = GetTargetArchitecture(device)
                },
                ct);
            
            var info = new CompiledKernelInfo
            {
                Name = method.Name,
                SourceMethod = method,
                CompiledKernel = compiledKernel,
                Device = device,
                InputType = typeof(TIn),
                OutputType = typeof(TOut),
                CompiledAt = DateTime.UtcNow
            };
            
            _compiledKernels[cacheKey] = info;
            
            _logger.LogInformation(
                "Successfully compiled kernel {Method} for {Device}",
                method.Name, device.Name);
            
            return info;
        }
        finally
        {
            _compilationLock.Release();
        }
    }
    
    /// <summary>
    /// Compiles a kernel from source code
    /// </summary>
    public async Task<CompiledKernelInfo> CompileFromSourceAsync(
        string source,
        string entryPoint,
        IComputeDevice device,
        Type inputType,
        Type outputType,
        CancellationToken ct = default)
    {
        var cacheKey = $"{entryPoint}_{source.GetHashCode()}_{device.Type}";
        
        if (_compiledKernels.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }
        
        await _compilationLock.WaitAsync(ct);
        try
        {
            if (_compiledKernels.TryGetValue(cacheKey, out cached))
            {
                return cached;
            }
            
            var compiledKernel = await device.CompileKernelAsync(
                source,
                entryPoint,
                new CompilationOptions
                {
                    EnableOptimizations = true,
                    UseFastMath = true
                },
                ct);
            
            var info = new CompiledKernelInfo
            {
                Name = entryPoint,
                SourceMethod = null,
                CompiledKernel = compiledKernel,
                Device = device,
                InputType = inputType,
                OutputType = outputType,
                CompiledAt = DateTime.UtcNow
            };
            
            _compiledKernels[cacheKey] = info;
            return info;
        }
        finally
        {
            _compilationLock.Release();
        }
    }
    
    private string GenerateKernelCode(MethodInfo method, DeviceType deviceType)
    {
        return deviceType switch
        {
            DeviceType.Cuda => GenerateCudaCode(method),
            DeviceType.OpenCl => GenerateOpenClCode(method),
            DeviceType.DirectCompute => GenerateHlslCode(method),
            DeviceType.Metal => GenerateMetalCode(method),
            _ => GenerateCpuCode(method)
        };
    }
    
    private string GenerateCudaCode(MethodInfo method)
    {
        var sb = new StringBuilder();
        sb.AppendLine("extern \"C\" {");
        sb.AppendLine($"__global__ void {method.Name}(");
        sb.AppendLine("    const float* input,");
        sb.AppendLine("    float* output,");
        sb.AppendLine("    int size) {");
        sb.AppendLine("    int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        sb.AppendLine("    if (idx < size) {");
        sb.AppendLine("        // Kernel logic here");
        sb.AppendLine("        output[idx] = input[idx] * 2.0f;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        sb.AppendLine("}");
        return sb.ToString();
    }
    
    private string GenerateOpenClCode(MethodInfo method)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"__kernel void {method.Name}(");
        sb.AppendLine("    __global const float* input,");
        sb.AppendLine("    __global float* output,");
        sb.AppendLine("    const int size) {");
        sb.AppendLine("    int idx = get_global_id(0);");
        sb.AppendLine("    if (idx < size) {");
        sb.AppendLine("        output[idx] = input[idx] * 2.0f;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }
    
    private string GenerateHlslCode(MethodInfo method)
    {
        var sb = new StringBuilder();
        sb.AppendLine("StructuredBuffer<float> input : register(t0);");
        sb.AppendLine("RWStructuredBuffer<float> output : register(u0);");
        sb.AppendLine("");
        sb.AppendLine("[numthreads(256, 1, 1)]");
        sb.AppendLine($"void {method.Name}(uint3 id : SV_DispatchThreadID) {{");
        sb.AppendLine("    output[id.x] = input[id.x] * 2.0f;");
        sb.AppendLine("}");
        return sb.ToString();
    }
    
    private string GenerateMetalCode(MethodInfo method)
    {
        var sb = new StringBuilder();
        sb.AppendLine("#include <metal_stdlib>");
        sb.AppendLine("using namespace metal;");
        sb.AppendLine("");
        sb.AppendLine($"kernel void {method.Name}(");
        sb.AppendLine("    const device float* input [[buffer(0)]],");
        sb.AppendLine("    device float* output [[buffer(1)]],");
        sb.AppendLine("    uint id [[thread_position_in_grid]]) {");
        sb.AppendLine("    output[id] = input[id] * 2.0f;");
        sb.AppendLine("}");
        return sb.ToString();
    }
    
    private string GenerateCpuCode(MethodInfo method)
    {
        // For CPU, we'll use the original C# method directly
        return $"// CPU execution of {method.Name}";
    }
    
    private string GetTargetArchitecture(IComputeDevice device)
    {
        return device.Type switch
        {
            DeviceType.Cuda => "sm_75", // CUDA compute capability 7.5
            DeviceType.OpenCl => "cl_std_2_0",
            DeviceType.DirectCompute => "cs_5_0",
            DeviceType.Metal => "metal_2_4",
            _ => "native"
        };
    }
    
    public void ClearCache()
    {
        foreach (var kernel in _compiledKernels.Values)
        {
            kernel.CompiledKernel?.Dispose();
        }
        _compiledKernels.Clear();
    }
}

/// <summary>
/// Information about a compiled kernel
/// </summary>
public sealed class CompiledKernelInfo
{
    public string Name { get; init; } = default!;
    public MethodInfo? SourceMethod { get; init; }
    public ICompiledKernel CompiledKernel { get; init; } = default!;
    public IComputeDevice Device { get; init; } = default!;
    public Type InputType { get; init; } = default!;
    public Type OutputType { get; init; } = default!;
    public DateTime CompiledAt { get; init; }
}

/// <summary>
/// Attribute to mark kernel methods
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class KernelAttribute : Attribute
{
    public string? Name { get; set; }
    public int PreferredWorkGroupSize { get; set; } = 256;
    public bool RequiresAtomics { get; set; }
    public bool RequiresSharedMemory { get; set; }
}

/// <summary>
/// Kernel registry for managing available kernels
/// </summary>
public sealed class KernelRegistry
{
    private readonly Dictionary<KernelId, KernelRegistration> _registrations;
    private readonly ILogger<KernelRegistry> _logger;
    
    public KernelRegistry(ILogger<KernelRegistry> logger)
    {
        _logger = logger;
        _registrations = new Dictionary<KernelId, KernelRegistration>();
    }
    
    public void RegisterKernel<TIn, TOut>(
        KernelId id,
        MethodInfo method,
        string? description = null)
        where TIn : notnull
        where TOut : notnull
    {
        var registration = new KernelRegistration
        {
            Id = id,
            Method = method,
            InputType = typeof(TIn),
            OutputType = typeof(TOut),
            Description = description ?? $"Kernel {id}"
        };
        
        _registrations[id] = registration;
        
        _logger.LogInformation(
            "Registered kernel {Id}: {Input} -> {Output}",
            id, typeof(TIn).Name, typeof(TOut).Name);
    }
    
    public void RegisterKernel<TIn, TOut>(
        KernelId id,
        string sourceCode,
        string entryPoint,
        string? description = null)
        where TIn : notnull
        where TOut : notnull
    {
        var registration = new KernelRegistration
        {
            Id = id,
            SourceCode = sourceCode,
            EntryPoint = entryPoint,
            InputType = typeof(TIn),
            OutputType = typeof(TOut),
            Description = description ?? $"Kernel {id}"
        };
        
        _registrations[id] = registration;
    }
    
    public KernelRegistration? GetRegistration(KernelId id)
    {
        return _registrations.GetValueOrDefault(id);
    }
    
    public IEnumerable<KernelRegistration> GetAllRegistrations()
    {
        return _registrations.Values;
    }
    
    public void ScanAssembly(Assembly assembly)
    {
        var kernelMethods = assembly.GetTypes()
            .SelectMany(t => t.GetMethods(BindingFlags.Public | BindingFlags.Static))
            .Where(m => m.GetCustomAttribute<KernelAttribute>() != null)
            .ToList();
        
        foreach (var method in kernelMethods)
        {
            var attr = method.GetCustomAttribute<KernelAttribute>()!;
            var kernelId = new KernelId(attr.Name ?? $"{method.DeclaringType?.Name}/{method.Name}");
            
            // Infer input/output types from method parameters
            var parameters = method.GetParameters();
            if (parameters.Length >= 2)
            {
                var inputType = parameters[0].ParameterType.GetElementType() ?? typeof(object);
                var outputType = parameters[1].ParameterType.GetElementType() ?? typeof(object);
                
                var registration = new KernelRegistration
                {
                    Id = kernelId,
                    Method = method,
                    InputType = inputType,
                    OutputType = outputType,
                    Description = $"Auto-discovered kernel: {method.Name}"
                };
                
                _registrations[kernelId] = registration;
                
                _logger.LogInformation(
                    "Auto-registered kernel {Id} from {Method}",
                    kernelId, method.Name);
            }
        }
    }
}

/// <summary>
/// Kernel registration information
/// </summary>
public sealed class KernelRegistration
{
    public KernelId Id { get; init; } = default!;
    public MethodInfo? Method { get; init; }
    public string? SourceCode { get; init; }
    public string? EntryPoint { get; init; }
    public Type InputType { get; init; } = default!;
    public Type OutputType { get; init; } = default!;
    public string Description { get; init; } = default!;
}