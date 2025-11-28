using System.Diagnostics.CodeAnalysis;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using DotCompute.Abstractions;
// Type aliases to avoid ambiguity with Orleans.GpuBridge types
using DotComputeKernelDef = DotCompute.Abstractions.Kernels.KernelDefinition;
using DotComputeCompilationOptions = DotCompute.Abstractions.CompilationOptions;
using DotComputeKernelLanguage = DotCompute.Abstractions.Kernels.Types.KernelLanguage;

namespace Orleans.GpuBridge.Backends.DotCompute.Kernels;

/// <summary>
/// DotCompute kernel compiler implementation
/// </summary>
internal sealed class DotComputeKernelCompiler : IKernelCompiler
{
    private readonly IDeviceManager _deviceManager;
    private readonly ILogger<DotComputeKernelCompiler> _logger;
    private readonly ConcurrentDictionary<string, DotComputeCompiledKernel> _compiledKernels;
    private readonly ConcurrentDictionary<string, object> _nativeKernels;
    private bool _disposed;

    public DotComputeKernelCompiler(
        IDeviceManager deviceManager,
        ILogger<DotComputeKernelCompiler> logger)
    {
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _compiledKernels = new ConcurrentDictionary<string, DotComputeCompiledKernel>();
        _nativeKernels = new ConcurrentDictionary<string, object>();
    }

    /// <summary>
    /// Compiles kernel source to GPU-executable code using DotCompute v0.5.1 NVRTC/CUDA Driver API.
    /// </summary>
    /// <remarks>
    /// <para>Uses real GPU compilation via <see cref="IAccelerator.CompileKernelAsync"/>.</para>
    /// <para>Compiled kernels are cached for performance.</para>
    /// </remarks>
    public async Task<CompiledKernel> CompileAsync(
        KernelSource source,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        if (options == null)
            options = new KernelCompilationOptions();

        try
        {
            _logger.LogDebug("Compiling DotCompute kernel: {KernelName}", source.Name);

            // Check if already compiled (caching pattern will remain with real APIs)
            var cacheKey = GenerateCacheKey(source, options);
            if (_compiledKernels.TryGetValue(cacheKey, out var cached))
            {
                _logger.LogDebug("Using cached compiled kernel: {KernelName}", source.Name);
                return cached.BaseKernel;
            }

            // Select target device
            var device = SelectTargetDevice(options.TargetDevice as IComputeDevice);

            // Compile kernel for DotCompute
            var compiledKernel = await CompileKernelForDeviceAsync(source, device, options, cancellationToken);

            // Cache the compiled kernel
            _compiledKernels[cacheKey] = compiledKernel;

            _logger.LogDebug("Successfully compiled DotCompute kernel: {KernelName}", source.Name);
            return compiledKernel.BaseKernel;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile DotCompute kernel: {KernelName}", source.Name);
            throw new InvalidOperationException($"DotCompute kernel compilation failed: {ex.Message}", ex);
        }
    }

    public async Task<CompiledKernel> CompileFromFileAsync(
        string filePath,
        string kernelName,
        KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        if (string.IsNullOrEmpty(kernelName))
            throw new ArgumentException("Kernel name cannot be null or empty", nameof(kernelName));

        try
        {
            _logger.LogDebug("Compiling DotCompute kernel from file: {FilePath}", filePath);

            // Read kernel source from file
            var sourceCode = await System.IO.File.ReadAllTextAsync(filePath, cancellationToken);
            var source = new KernelSource(
                Name: kernelName,
                SourceCode: sourceCode,
                Language: DetectLanguageFromFile(filePath),
                EntryPoint: kernelName);

            return await CompileAsync(source, options, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile DotCompute kernel from file: {FilePath}", filePath);
            throw;
        }
    }

    [RequiresUnreferencedCode("Uses method validation which may not work with trimming.")]
    public async Task<CompiledKernel> CompileFromMethodAsync(
        [NotNull] MethodInfo method,
        [NotNull] KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));

        if (options == null)
            options = new KernelCompilationOptions();

        try
        {
            _logger.LogDebug("Compiling DotCompute kernel from method: {MethodName}", method.Name);

            // Convert C# method to kernel source
            var sourceCode = GenerateKernelSourceFromMethod(method);
            var source = new KernelSource(
                Name: method.Name,
                SourceCode: sourceCode,
                Language: KernelLanguage.CSharp,
                EntryPoint: method.Name);

            return await CompileAsync(source, options, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile DotCompute kernel from method: {MethodName}", method.Name);
            throw;
        }
    }

    public async Task<CompiledKernel> CompileFromSourceAsync(
        [NotNull] string sourceCode,
        [NotNull] string entryPoint,
        KernelLanguage language,
        [NotNull] KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(sourceCode))
            throw new ArgumentException("Source code cannot be null or empty", nameof(sourceCode));

        if (string.IsNullOrEmpty(entryPoint))
            throw new ArgumentException("Entry point cannot be null or empty", nameof(entryPoint));

        if (options == null)
            options = new KernelCompilationOptions();

        try
        {
            _logger.LogDebug("Compiling DotCompute kernel from source: {EntryPoint}", entryPoint);

            var source = new KernelSource(
                Name: entryPoint,
                SourceCode: sourceCode,
                Language: language,
                EntryPoint: entryPoint);

            return await CompileAsync(source, options, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile DotCompute kernel from source: {EntryPoint}", entryPoint);
            throw;
        }
    }

    [RequiresUnreferencedCode("Uses reflection to find types and methods which may not work with trimming.")]
    public async Task<CompiledKernel> CompileFromAssemblyAsync(
        [NotNull] Assembly assembly,
        [NotNull] string typeName,
        [NotNull] string methodName,
        [NotNull] KernelCompilationOptions options,
        CancellationToken cancellationToken = default)
    {
        if (assembly == null)
            throw new ArgumentNullException(nameof(assembly));

        if (string.IsNullOrEmpty(typeName))
            throw new ArgumentException("Type name cannot be null or empty", nameof(typeName));

        if (string.IsNullOrEmpty(methodName))
            throw new ArgumentException("Method name cannot be null or empty", nameof(methodName));

        if (options == null)
            options = new KernelCompilationOptions();

        try
        {
            _logger.LogDebug("Compiling DotCompute kernel from assembly: {TypeName}.{MethodName}", typeName, methodName);

            // Find the method using reflection
            var type = assembly.GetType(typeName);
            if (type == null)
                throw new InvalidOperationException($"Type not found: {typeName}");

            var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static);
            if (method == null)
                throw new InvalidOperationException($"Method not found: {typeName}.{methodName}");

            return await CompileFromMethodAsync(method, options, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile DotCompute kernel from assembly: {TypeName}.{MethodName}", typeName, methodName);
            throw;
        }
    }

    [RequiresUnreferencedCode("Uses method validation which may not work with trimming.")]
    public async Task<KernelValidationResult> ValidateMethodAsync(
        [NotNull] MethodInfo method,
        CancellationToken cancellationToken = default)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));

        try
        {
            _logger.LogDebug("Validating method for DotCompute compilation: {MethodName}", method.Name);

            var errors = new List<string>();
            var warnings = new List<string>();

            // Basic validation checks
            if (!method.IsStatic)
            {
                errors.Add("Kernel methods must be static");
            }

            if (method.IsGenericMethodDefinition || method.DeclaringType?.IsGenericType == true)
            {
                errors.Add("Generic methods and types are not supported in kernels");
            }

            // Check return type
            if (method.ReturnType != typeof(void))
            {
                warnings.Add("Kernel methods typically should return void");
            }

            // Check parameters (simplified validation)
            foreach (var param in method.GetParameters())
            {
                if (!IsValidKernelParameterType(param.ParameterType))
                {
                    errors.Add($"Parameter type '{param.ParameterType.Name}' is not supported in kernels");
                }
            }

            var isValid = errors.Count == 0;
            
            return await Task.FromResult(new KernelValidationResult(
                IsValid: isValid,
                ErrorMessage: errors.Count > 0 ? string.Join("; ", errors) : null,
                Warnings: warnings.AsReadOnly()));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to validate method: {MethodName}", method.Name);
            throw;
        }
    }

    public async Task<CompilationDiagnostics> GetDiagnosticsAsync(
        [NotNull] CompiledKernel kernel,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        try
        {
            _logger.LogDebug("Getting diagnostics for kernel: {KernelId}", kernel.KernelId);

            // Simulate diagnostics generation
            var diagnostics = new CompilationDiagnostics(
                IntermediateCode: $"DotCompute IR for {kernel.Name}",
                OptimizationReport: "DotCompute optimizations applied",
                CompilationTime: TimeSpan.FromMilliseconds(150),
                AdditionalInfo: new Dictionary<string, object>
                {
                    ["registers_used"] = 32,
                    ["shared_memory_used"] = 1024,
                    ["estimated_occupancy"] = 0.75,
                    ["warnings"] = new[] { "No specific warnings" }
                });

            return await Task.FromResult(diagnostics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to get diagnostics for kernel: {KernelId}", kernel.KernelId);
            throw;
        }
    }

    public bool IsKernelCached(string kernelId)
    {
        return _compiledKernels.ContainsKey(kernelId);
    }

    public CompiledKernel? GetCachedKernel(string kernelId)
    {
        _compiledKernels.TryGetValue(kernelId, out var kernel);
        return kernel?.BaseKernel;
    }

    public void ClearCache()
    {
        _logger.LogInformation("Clearing DotCompute kernel compilation cache");

        foreach (var kernel in _compiledKernels.Values)
        {
            try
            {
                kernel.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing cached kernel: {KernelId}", kernel.KernelId);
            }
        }

        _compiledKernels.Clear();
        _nativeKernels.Clear();
    }

    public DotComputeCompiledKernel? GetCachedDotComputeKernel(string kernelId)
    {
        _compiledKernels.TryGetValue(kernelId, out var kernel);
        return kernel;
    }

    /// <summary>
    /// Maps Orleans.GpuBridge kernel language to DotCompute kernel language
    /// </summary>
    private static DotComputeKernelLanguage MapLanguage(Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage language)
    {
        return language switch
        {
            Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.CUDA => DotComputeKernelLanguage.Cuda,
            Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.OpenCL => DotComputeKernelLanguage.OpenCL,
            Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.CSharp => DotComputeKernelLanguage.CSharp,
            Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.HLSL => DotComputeKernelLanguage.HLSL,
            Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.PTX => DotComputeKernelLanguage.Ptx,
            Orleans.GpuBridge.Abstractions.Enums.Compilation.KernelLanguage.SPIRV => DotComputeKernelLanguage.SPIRV,
            _ => DotComputeKernelLanguage.Auto
        };
    }

    /// <summary>
    /// Maps Orleans.GpuBridge compilation options to DotCompute compilation options
    /// </summary>
    private static DotComputeCompilationOptions MapCompilationOptions(KernelCompilationOptions options)
    {
        // Start with appropriate base options based on optimization level
        var dotComputeOptions = options.OptimizationLevel == OptimizationLevel.O0
            ? DotComputeCompilationOptions.Debug
            : DotComputeCompilationOptions.Release;

        // Map optimization-specific settings
        if (options.EnableFastMath)
        {
            dotComputeOptions.EnableFastMath = true;
            dotComputeOptions.FastMath = true;
            dotComputeOptions.UseFastMath = true;
        }

        // Apply aggressive optimizations for O3
        if (options.OptimizationLevel == OptimizationLevel.O3)
        {
            dotComputeOptions.AggressiveOptimizations = true;
            dotComputeOptions.EnableLoopUnrolling = true;
            dotComputeOptions.UnrollLoops = true;
            dotComputeOptions.EnableVectorization = true;
            dotComputeOptions.EnableInlining = true;
        }

        // Map defines (CompilationOptions.Defines is read-only, need to add to existing dictionary)
        if (options.Defines?.Any() == true)
        {
            foreach (var define in options.Defines)
            {
                dotComputeOptions.Defines.Add(define.Key, define.Value);
            }
        }

        // Map register count settings
        if (options.MaxRegisterCount > 0)
        {
            dotComputeOptions.MaxRegisters = options.MaxRegisterCount;
            dotComputeOptions.MaxRegistersPerThread = options.MaxRegisterCount;
        }

        // Map debug settings
        if (options.EnableDebugInfo)
        {
            dotComputeOptions.EnableDebugInfo = true;
            dotComputeOptions.GenerateDebugInfo = true;
            dotComputeOptions.EnableDeviceDebugging = true;
            dotComputeOptions.GenerateLineInfo = true;
        }

        // Map profiling settings
        if (options.EnableProfiling)
        {
            dotComputeOptions.EnableProfiling = true;
        }

        return dotComputeOptions;
    }

    /// <summary>
    /// Compiles kernel for DotCompute device using real GPU compilation
    /// </summary>
    private async Task<DotComputeCompiledKernel> CompileKernelForDeviceAsync(
        KernelSource source,
        IComputeDevice device,
        KernelCompilationOptions options,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Compiling DotCompute kernel: {KernelName} for device: {DeviceId}",
            source.Name, device.DeviceId);

        // Extract DotCompute accelerator from device adapter
        var adapter = device as DotComputeAcceleratorAdapter
            ?? throw new InvalidOperationException($"Device {device.DeviceId} is not a DotCompute device");

        var accelerator = adapter.Accelerator;

        // Create kernel definition with source and entry point
        var kernelDef = new DotComputeKernelDef(
            name: source.Name,
            source: source.SourceCode,
            entryPoint: source.EntryPoint ?? source.Name)
        {
            Language = MapLanguage(source.Language)
        };

        // Map compilation options from Orleans to DotCompute
        var compilationOptions = MapCompilationOptions(options);

        try
        {
            // ✅ REAL API: Compile kernel using DotCompute NVRTC/CUDA Driver API
            var nativeKernel = await accelerator.CompileKernelAsync(
                kernelDef,
                compilationOptions,
                cancellationToken);

            _logger.LogInformation(
                "Successfully compiled DotCompute kernel: {KernelName} (ID: {KernelId})",
                nativeKernel.Name,
                nativeKernel.Id);

            // Generate unique kernel ID
            var kernelId = $"{source.Name}_{device.DeviceId}_{nativeKernel.Id}";

            // Store native kernel for execution
            _nativeKernels[kernelId] = nativeKernel;

            // Build metadata
            var metadata = new Dictionary<string, object>
            {
                ["compiled_for_device"] = device.DeviceId,
                ["compilation_time"] = DateTime.UtcNow,
                ["language"] = source.Language.ToString(),
                ["entry_point"] = source.EntryPoint ?? source.Name,
                ["optimization_level"] = options.OptimizationLevel.ToString(),
                ["native_kernel_id"] = nativeKernel.Id.ToString(),
                ["status"] = "compiled_real_gpu_kernel"  // ✅ No longer simulated!
            };

            if (options.Defines?.Any() == true)
            {
                metadata["defines"] = string.Join(";", options.Defines.Select(kv => $"{kv.Key}={kv.Value}"));
            }

            // Create Orleans kernel wrapper
            return new DotComputeCompiledKernel(
                kernelId: kernelId,
                name: source.Name,
                device: device,
                metadata: metadata,
                nativeKernel: nativeKernel,  // Store real CudaCompiledKernel
                logger: _logger);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile kernel {KernelName} with DotCompute", source.Name);
            throw new InvalidOperationException($"Kernel compilation failed: {ex.Message}", ex);
        }
    }

    private IComputeDevice SelectTargetDevice(IComputeDevice? preferredDevice)
    {
        if (preferredDevice != null)
        {
            return preferredDevice;
        }

        // Use default device selection
        return _deviceManager.GetDefaultDevice();
    }

    private string GenerateCacheKey(KernelSource source, KernelCompilationOptions options)
    {
        var keyParts = new[]
        {
            source.Name,
            source.GetHashCode().ToString(),
            options.OptimizationLevel.ToString(),
            options.Defines?.Any() == true ? string.Join(";", options.Defines.Select(kv => $"{kv.Key}={kv.Value}")) : string.Empty
        };

        return string.Join("_", keyParts.Where(p => !string.IsNullOrEmpty(p)));
    }

    private KernelLanguage DetectLanguageFromFile(string filePath)
    {
        var extension = System.IO.Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".cl" => KernelLanguage.OpenCL,
            ".cu" => KernelLanguage.CUDA,
            ".hlsl" => KernelLanguage.HLSL,
            ".cs" => KernelLanguage.CSharp,
            _ => KernelLanguage.OpenCL
        };
    }

    private string GenerateKernelSourceFromMethod(MethodInfo method)
    {
        // Simplified C# method to kernel source conversion
        // In a real implementation, this would perform IL analysis and code generation
        return $@"// Generated kernel from method: {method.Name}
__kernel void {method.Name}(/* parameters would be analyzed and converted */)
{{
    // Method body would be analyzed and converted to GPU code
    // This is a placeholder implementation
}}";
    }

    private bool IsValidKernelParameterType(Type parameterType)
    {
        // Simplified parameter type validation
        // In a real implementation, this would check against supported GPU types
        return parameterType.IsPrimitive || 
               parameterType.IsArray || 
               parameterType == typeof(string) ||
               parameterType.IsValueType;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute kernel compiler");

        try
        {
            ClearCache();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute kernel compiler");
        }

        _disposed = true;
    }
}

/// <summary>
/// DotCompute compiled kernel wrapper
/// </summary>
internal sealed class DotComputeCompiledKernel : IDisposable
{
    private readonly CompiledKernel _baseKernel;
    private readonly object _nativeKernel;
    private readonly ILogger _logger;
    private bool _disposed;

    public string KernelId => _baseKernel.KernelId;
    public string Name => _baseKernel.Name;
    public CompiledKernel BaseKernel => _baseKernel;

    public DotComputeCompiledKernel(
        string kernelId,
        string name,
        IComputeDevice device,
        IReadOnlyDictionary<string, object> metadata,
        object nativeKernel,
        ILogger logger)
    {
        _nativeKernel = nativeKernel ?? throw new ArgumentNullException(nameof(nativeKernel));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        
        // Create CompiledKernel with simulated binary code
        var simulatedBinaryCode = System.Text.Encoding.UTF8.GetBytes($"DotComputeKernel_{kernelId}");
        var kernelMetadata = new KernelMetadata(
            ExtendedMetadata: metadata
        );

        _baseKernel = new CompiledKernel
        {
            KernelId = kernelId,
            Name = name,
            CompiledCode = simulatedBinaryCode,
            Metadata = kernelMetadata,
            NativeHandle = IntPtr.Zero // Will be set by actual DotCompute implementation
        };
    }

    public object GetNativeKernel() => _nativeKernel;

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogTrace("Disposing DotCompute compiled kernel: {KernelId}", KernelId);

        // Production implementation would dispose DotCompute kernel resources properly
        // This ensures proper cleanup of GPU resources and prevents memory leaks
        if (_nativeKernel is IDisposable disposableKernel)
        {
            disposableKernel.Dispose();
        }

        _baseKernel?.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Native DotCompute kernel wrapper (placeholder implementation)
/// </summary>
internal sealed class DotComputeNativeKernel : IDisposable
{
    public string Name { get; }
    public string SourceCode { get; }

    public DotComputeNativeKernel(string name, string sourceCode)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        SourceCode = sourceCode ?? throw new ArgumentNullException(nameof(sourceCode));
    }

    public void Dispose()
    {
        // Cleanup native DotCompute kernel resources
    }
}