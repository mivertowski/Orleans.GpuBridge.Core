using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// ILGPU kernel compiler implementation
/// </summary>
internal sealed class ILGPUKernelCompiler : IKernelCompiler
{
    private readonly ILogger<ILGPUKernelCompiler> _logger;
    private readonly Context _context;
    private readonly ILGPUDeviceManager _deviceManager;
    private readonly ConcurrentDictionary<string, CompiledKernel> _compilationCache;
    private readonly ConcurrentDictionary<string, Action<AcceleratorStream, KernelConfig, object[]>> _ilgpuKernelCache;
    private bool _disposed;

    public ILGPUKernelCompiler(
        ILogger<ILGPUKernelCompiler> logger,
        Context context,
        ILGPUDeviceManager deviceManager)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _compilationCache = new ConcurrentDictionary<string, CompiledKernel>();
        _ilgpuKernelCache = new ConcurrentDictionary<string, Action<AcceleratorStream, KernelConfig, object[]>>();
    }

    public async Task<CompiledKernel> CompileFromMethodAsync(
        MethodInfo method,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));

        if (options == null)
            options = new KernelCompilationOptions();

        var cacheKey = $"method:{method.DeclaringType?.FullName}.{method.Name}:{options.OptimizationLevel}";

        // Check cache first
        if (_compilationCache.TryGetValue(cacheKey, out var cachedKernel))
        {
            _logger.LogDebug("Using cached kernel compilation: {CacheKey}", cacheKey);
            return cachedKernel;
        }

        _logger.LogInformation("Compiling ILGPU kernel from method: {MethodName}", method.Name);

        try
        {
            // Validate the method
            var validationResult = await ValidateMethodAsync(method, cancellationToken);
            if (!validationResult.IsValid)
            {
                throw new InvalidOperationException($"Method validation failed: {validationResult.ErrorMessage}");
            }

            // Get a suitable device for compilation
            var device = _deviceManager.GetDefaultDevice() as ILGPUComputeDevice;
            if (device == null)
            {
                throw new InvalidOperationException("No suitable ILGPU device available for compilation");
            }

            var accelerator = device.Accelerator;

            // Compile the kernel using ILGPU
            var startTime = DateTime.UtcNow;
            
            // Load the kernel method using ILGPU's LoadAutoGroupedStreamKernel
            var loadedKernel = accelerator.LoadAutoGroupedStreamKernel(method);
            
            var compilationTime = DateTime.UtcNow - startTime;

            // Create metadata
            var metadata = new KernelMetadata(
                RequiredSharedMemory: 0, // ILGPU manages this automatically
                RequiredRegisters: 0,
                MaxThreadsPerBlock: device.MaxThreadsPerBlock,
                PreferredBlockSize: CalculatePreferredBlockSize(accelerator),
                UsesAtomics: AnalyzeMethodForAtomics(method),
                UsesSharedMemory: AnalyzeMethodForSharedMemory(method),
                UsesDynamicParallelism: false, // ILGPU doesn't support dynamic parallelism
                ExtendedMetadata: new Dictionary<string, object>
                {
                    ["compilation_time_ms"] = compilationTime.TotalMilliseconds,
                    ["accelerator_type"] = accelerator.AcceleratorType.ToString(),
                    ["optimization_level"] = options.OptimizationLevel.ToString()
                });

            // Create the compiled kernel wrapper
            var compiledKernel = new CompiledKernel
            {
                KernelId = cacheKey,
                Name = method.Name,
                CompiledCode = new byte[0], // ILGPU handles binary internally
                Metadata = metadata,
                NativeHandle = IntPtr.Zero // Not applicable for ILGPU
            };

            // Cache the ILGPU kernel delegate
            _ilgpuKernelCache[cacheKey] = loadedKernel;

            // Cache the compiled kernel
            _compilationCache[cacheKey] = compiledKernel;

            _logger.LogInformation(
                "Successfully compiled ILGPU kernel: {KernelName} in {CompilationTime}ms",
                method.Name, compilationTime.TotalMilliseconds);

            return compiledKernel;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile ILGPU kernel from method: {MethodName}", method.Name);
            throw;
        }
    }

    public async Task<CompiledKernel> CompileFromSourceAsync(
        string sourceCode,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(sourceCode))
            throw new ArgumentException("Source code cannot be null or empty", nameof(sourceCode));

        if (string.IsNullOrEmpty(entryPoint))
            throw new ArgumentException("Entry point cannot be null or empty", nameof(entryPoint));

        // ILGPU primarily works with compiled C# methods, not source code compilation
        // For source code compilation, we would need to:
        // 1. Use Roslyn to compile the source code to a method
        // 2. Load the method via reflection
        // 3. Compile using CompileFromMethodAsync

        _logger.LogWarning(
            "Source code compilation not fully implemented for ILGPU. " +
            "Consider using CompileFromMethodAsync instead.");

        // For now, create a placeholder kernel that represents source compilation failure TODO
        var compiledKernel = new CompiledKernel
        {
            KernelId = $"source:{entryPoint}",
            Name = entryPoint,
            CompiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode),
            Metadata = new KernelMetadata(
                ExtendedMetadata: new Dictionary<string, object>
                {
                    ["source_language"] = language.ToString(),
                    ["compilation_method"] = "source_placeholder"
                })
        };

        return await Task.FromResult(compiledKernel);
    }

    public async Task<CompiledKernel> CompileFromAssemblyAsync(
        Assembly assembly,
        string typeName,
        string methodName,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (assembly == null)
            throw new ArgumentNullException(nameof(assembly));

        if (string.IsNullOrEmpty(typeName))
            throw new ArgumentException("Type name cannot be null or empty", nameof(typeName));

        if (string.IsNullOrEmpty(methodName))
            throw new ArgumentException("Method name cannot be null or empty", nameof(methodName));

        _logger.LogInformation("Compiling ILGPU kernel from assembly: {TypeName}.{MethodName}", typeName, methodName);

        try
        {
            // Find the type in the assembly
            var type = assembly.GetType(typeName);
            if (type == null)
            {
                throw new ArgumentException($"Type '{typeName}' not found in assembly '{assembly.FullName}'");
            }

            // Find the method
            var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
            if (method == null)
            {
                throw new ArgumentException($"Method '{methodName}' not found in type '{typeName}'");
            }

            // Delegate to method compilation
            return await CompileFromMethodAsync(method, options ?? new KernelCompilationOptions(), cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile ILGPU kernel from assembly: {TypeName}.{MethodName}", typeName, methodName);
            throw;
        }
    }

    public Task<KernelValidationResult> ValidateMethodAsync(
        MethodInfo method,
        CancellationToken cancellationToken = default)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));

        var errors = new List<string>();
        var warnings = new List<string>();
        var unsupportedFeatures = new List<string>();

        try
        {
            // Check if method is static
            if (!method.IsStatic)
            {
                errors.Add("Kernel methods must be static");
            }

            // Check return type
            if (method.ReturnType != typeof(void))
            {
                errors.Add("Kernel methods must return void");
            }

            // Check parameters
            var parameters = method.GetParameters();
            foreach (var param in parameters)
            {
                if (!IsValidParameterType(param.ParameterType))
                {
                    errors.Add($"Parameter '{param.Name}' has unsupported type: {param.ParameterType}");
                }
            }

            // Check for unsupported constructs (basic analysis)
            if (ContainsUnsupportedFeatures(method))
            {
                warnings.Add("Method may contain constructs not supported by ILGPU");
            }

            // Check method body size (heuristic)
            var methodBody = method.GetMethodBody();
            if (methodBody != null && methodBody.GetILAsByteArray()?.Length > 10000)
            {
                warnings.Add("Large method body may cause compilation issues");
            }

            var isValid = errors.Count == 0;
            
            var result = new KernelValidationResult(
                IsValid: isValid,
                ErrorMessage: isValid ? null : string.Join("; ", errors),
                Warnings: warnings.Count > 0 ? warnings : null,
                UnsupportedFeatures: unsupportedFeatures.Count > 0 ? unsupportedFeatures : null);

            _logger.LogDebug(
                "Kernel validation result for {MethodName}: Valid={IsValid}, Errors={ErrorCount}, Warnings={WarningCount}",
                method.Name, isValid, errors.Count, warnings.Count);

            return Task.FromResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating kernel method: {MethodName}", method.Name);
            
            return Task.FromResult(new KernelValidationResult(
                IsValid: false,
                ErrorMessage: $"Validation error: {ex.Message}"));
        }
    }

    public Task<CompilationDiagnostics> GetDiagnosticsAsync(
        CompiledKernel kernel,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        try
        {
            var diagnostics = new CompilationDiagnostics(
                IntermediateCode: "ILGPU IL intermediate code not exposed",
                AssemblyCode: "ILGPU assembly code not exposed",
                OptimizationReport: "ILGPU optimization details not available",
                CompilationTime: TimeSpan.FromMilliseconds(
                    kernel.Metadata.ExtendedMetadata?.TryGetValue("compilation_time_ms", out var timeObj) == true
                        ? Convert.ToDouble(timeObj) 
                        : 0),
                CompiledCodeSize: kernel.CompiledCode?.Length ?? 0,
                AdditionalInfo: kernel.Metadata.ExtendedMetadata);

            return Task.FromResult(diagnostics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get diagnostics for kernel: {KernelId}", kernel.KernelId);
            throw;
        }
    }

    public void ClearCache()
    {
        _logger.LogInformation("Clearing ILGPU kernel compilation cache");

        try
        {
            // Dispose cached kernels
            foreach (var kernel in _compilationCache.Values)
            {
                try
                {
                    kernel.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disposing cached kernel: {KernelId}", kernel.KernelId);
                }
            }

            _compilationCache.Clear();
            _ilgpuKernelCache.Clear();

            _logger.LogInformation("ILGPU kernel compilation cache cleared");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing ILGPU kernel compilation cache");
            throw;
        }
    }

    internal Action<AcceleratorStream, KernelConfig, object[]>? GetCachedILGPUKernel(string kernelId)
    {
        return _ilgpuKernelCache.TryGetValue(kernelId, out var kernel) ? kernel : null;
    }

    private bool IsValidParameterType(Type parameterType)
    {
        // ILGPU supports various parameter types
        if (parameterType.IsPrimitive)
            return true;

        if (parameterType == typeof(string))
            return false; // Strings are not supported

        // Check for array types
        if (parameterType.IsArray)
        {
            var elementType = parameterType.GetElementType();
            return elementType != null && elementType.IsPrimitive;
        }

        // Check for ILGPU specific types (ArrayView, etc.)
        if (parameterType.Namespace?.StartsWith("ILGPU") == true)
            return true;

        // Check for struct types
        if (parameterType.IsValueType && !parameterType.IsEnum)
        {
            // Simple structs with primitive fields are usually supported
            return true;
        }

        return false;
    }

    private bool ContainsUnsupportedFeatures(MethodInfo method)
    {
        // This is a simplified check - a full implementation would analyze IL code
        // For now, just check method attributes and basic characteristics TODO
        
        var methodBody = method.GetMethodBody();
        if (methodBody == null)
            return false;

        // Check for recursive calls (not supported by ILGPU)
        try
        {
            var ilBytes = methodBody.GetILAsByteArray();
            if (ilBytes != null && ilBytes.Length > 0)
            {
                // Simple heuristic: very large methods might contain unsupported features
                return ilBytes.Length > 5000;
            }
        }
        catch
        {
            // If we can't analyze, assume it might contain unsupported features
            return true;
        }

        return false;
    }

    private bool AnalyzeMethodForAtomics(MethodInfo method)
    {
        // Check if method name or declaring type suggests atomic operations
        var methodName = method.Name.ToLowerInvariant();
        return methodName.Contains("atomic") || 
               methodName.Contains("interlocked") ||
               method.DeclaringType?.Name.ToLowerInvariant().Contains("atomic") == true;
    }

    private bool AnalyzeMethodForSharedMemory(MethodInfo method)
    {
        // Check if method parameters or return type suggest shared memory usage
        var parameters = method.GetParameters();
        return parameters.Any(p => 
            p.ParameterType.Name.Contains("SharedMemory") ||
            p.ParameterType.Namespace?.Contains("ILGPU") == true);
    }

    private int CalculatePreferredBlockSize(Accelerator accelerator)
    {
        // Calculate optimal block size based on device characteristics
        return accelerator.AcceleratorType switch
        {
            AcceleratorType.Cuda => Math.Min(512, accelerator.MaxNumThreadsPerGroup),
            AcceleratorType.OpenCL => Math.Min(256, accelerator.MaxNumThreadsPerGroup),
            AcceleratorType.CPU => Environment.ProcessorCount,
            _ => 256
        };
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU kernel compiler");

        try
        {
            ClearCache();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU kernel compiler");
        }

        _disposed = true;
    }
}