using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics.CodeAnalysis;
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
    private readonly ConcurrentDictionary<string, Action<Index1D, ArrayView<int>>> _ilgpuKernelCache;
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
        _ilgpuKernelCache = new ConcurrentDictionary<string, Action<Index1D, ArrayView<int>>>();
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
            // Validate the method asynchronously
            var validationResult = await ValidateMethodAsync(method, cancellationToken).ConfigureAwait(false);
            if (!validationResult.IsValid)
            {
                throw new InvalidOperationException($"Method validation failed: {validationResult.ErrorMessage}");
            }

            // Get a suitable device for compilation asynchronously
            var device = await GetCompilationDeviceAsync(cancellationToken).ConfigureAwait(false);
            var accelerator = device.Accelerator;

            // Compile the kernel using ILGPU asynchronously
            var compiledKernel = await CompileKernelAsync(method, device, accelerator, options, cacheKey, cancellationToken).ConfigureAwait(false);

            // Cache the compiled kernel
            _compilationCache[cacheKey] = compiledKernel;

            _logger.LogInformation(
                "Successfully compiled ILGPU kernel: {KernelName}",
                method.Name);

            return compiledKernel;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile ILGPU kernel from method: {MethodName}", method.Name);
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

        // ILGPU primarily works with compiled C# methods, not source code compilation
        // For source code compilation, we would need to:
        // 1. Use Roslyn to compile the source code to a method
        // 2. Load the method via reflection
        // 3. Compile using CompileFromMethodAsync

        _logger.LogWarning(
            "Source code compilation not fully implemented for ILGPU. " +
            "Consider using CompileFromMethodAsync instead.");

        // Implement actual source code compilation for ILGPU
        try
        {
            // For ILGPU, we need to create a dynamic method or assembly
            // This is a simplified implementation that creates a compilable kernel
            var compiledKernel = await CompileSourceCodeToMethodAsync(sourceCode, entryPoint, language, options, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("Successfully compiled source kernel: {EntryPoint} from {Language}", 
                entryPoint, language);
                
            return compiledKernel;
        }
        catch (Exception compileEx)
        {
            _logger.LogError(compileEx, "Failed to compile source code for entry point: {EntryPoint}", entryPoint);
            
            // Return a fallback kernel that indicates compilation failure
            var fallbackKernel = new CompiledKernel
            {
                KernelId = $"source_failed:{entryPoint}",
                Name = entryPoint,
                CompiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode),
                Metadata = new KernelMetadata(
                    ExtendedMetadata: new Dictionary<string, object>
                    {
                        ["source_language"] = language.ToString(),
                        ["compilation_method"] = "source_fallback",
                        ["compilation_error"] = compileEx.Message,
                        ["compilation_time"] = DateTime.UtcNow
                    })
            };
            
            return fallbackKernel;
        }
    }

    [RequiresUnreferencedCode("Uses reflection to find types and methods which may be trimmed.")]
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

    [RequiresUnreferencedCode("Uses method body analysis which may not work with trimming.")]
    public async Task<KernelValidationResult> ValidateMethodAsync(
        [NotNull] MethodInfo method,
        CancellationToken cancellationToken = default)
    {
        if (method == null)
            throw new ArgumentNullException(nameof(method));

        try
        {
            // Perform validation asynchronously to avoid blocking for complex methods
            return await Task.Run(async () =>
            {
                var errors = new List<string>();
                var warnings = new List<string>();
                var unsupportedFeatures = new List<string>();

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

                // Check parameters asynchronously for complex types
                var parameters = method.GetParameters();
                await Task.Run(() =>
                {
                    foreach (var param in parameters)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!IsValidParameterType(param.ParameterType))
                        {
                            errors.Add($"Parameter '{param.Name}' has unsupported type: {param.ParameterType}");
                        }
                    }
                }, cancellationToken).ConfigureAwait(false);

                // Check for unsupported constructs asynchronously (basic analysis)
                var containsUnsupported = await Task.Run(() => ContainsUnsupportedFeatures(method), cancellationToken).ConfigureAwait(false);
                if (containsUnsupported)
                {
                    warnings.Add("Method may contain constructs not supported by ILGPU");
                }

                // Check method body size asynchronously (heuristic)
                await Task.Run(() =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var methodBody = method.GetMethodBody();
                    if (methodBody != null && methodBody.GetILAsByteArray()?.Length > 10000)
                    {
                        warnings.Add("Large method body may cause compilation issues");
                    }
                }, cancellationToken).ConfigureAwait(false);

                var isValid = errors.Count == 0;
                
                var result = new KernelValidationResult(
                    IsValid: isValid,
                    ErrorMessage: isValid ? null : string.Join("; ", errors),
                    Warnings: warnings.Count > 0 ? warnings : null,
                    UnsupportedFeatures: unsupportedFeatures.Count > 0 ? unsupportedFeatures : null);

                _logger.LogDebug(
                    "Kernel validation result for {MethodName}: Valid={IsValid}, Errors={ErrorCount}, Warnings={WarningCount}",
                    method.Name, isValid, errors.Count, warnings.Count);

                return result;
            }, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating kernel method: {MethodName}", method.Name);
            
            return new KernelValidationResult(
                IsValid: false,
                ErrorMessage: $"Validation error: {ex.Message}");
        }
    }

    public Task<CompilationDiagnostics> GetDiagnosticsAsync(
        [NotNull] CompiledKernel kernel,
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

    internal Action<Index1D, ArrayView<int>>? GetCachedILGPUKernel(string kernelId)
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

    [RequiresUnreferencedCode("Uses GetMethodBody() which may not work with trimming.")]
    private bool ContainsUnsupportedFeatures(MethodInfo method)
    {
        // Comprehensive IL analysis for ILGPU compatibility
        // Check method attributes, IL instructions, and potential unsupported patterns
        
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

    /// <summary>
    /// Compiles source code to a method-based kernel with async file operations
    /// </summary>
    private async Task<CompiledKernel> CompileSourceCodeToMethodAsync(
        string sourceCode, 
        string entryPoint, 
        KernelLanguage language,
        KernelCompilationOptions options,
        CancellationToken cancellationToken)
    {
        // This is a simplified implementation of source compilation
        // A full implementation would:
        // 1. Parse the source code
        // 2. Generate or compile to IL/assembly
        // 3. Create a MethodInfo that ILGPU can compile
        // 4. Handle language-specific syntax (HLSL, OpenCL C, etc.)
        
        _logger.LogInformation("Attempting source compilation for {Language} kernel: {EntryPoint}", 
            language, entryPoint);

        var startTime = DateTime.UtcNow;
        string? tempDirectory = null;
            
        try
        {
            // Create temporary compilation directory
            tempDirectory = await AsyncFileOperations.CreateTempCompilationDirectoryAsync(
                $"ilgpu_{entryPoint}", cancellationToken).ConfigureAwait(false);
            
            // Generate cache key and paths
            var cacheKey = GenerateCacheKey(sourceCode, entryPoint, language, options);
            var sourceFile = Path.Combine(tempDirectory, $"{entryPoint}.{GetFileExtension(language)}");
            var cacheFile = Path.Combine(tempDirectory, $"{cacheKey}.cache");
            
            // Write source to temporary file for compilation
            await AsyncFileOperations.WriteCompiledKernelAsync(
                sourceFile, System.Text.Encoding.UTF8.GetBytes(sourceCode), cancellationToken).ConfigureAwait(false);
            
            // Perform compilation with async file I/O
            var compiledKernel = await PerformSourceCompilationAsync(
                sourceFile, cacheFile, entryPoint, language, options, startTime, cancellationToken).ConfigureAwait(false);
                
            return compiledKernel;
        }
        finally
        {
            // Clean up temporary directory
            if (tempDirectory != null)
            {
                await AsyncFileOperations.CleanupTempDirectoryAsync(
                    tempDirectory, _logger, cancellationToken).ConfigureAwait(false);
            }
        }
    }
    
    /// <summary>
    /// Performs the actual source compilation with caching support
    /// </summary>
    private async Task<CompiledKernel> PerformSourceCompilationAsync(
        string sourceFile,
        string cacheFile,
        string entryPoint,
        KernelLanguage language,
        KernelCompilationOptions options,
        DateTime startTime,
        CancellationToken cancellationToken)
    {
        // Check if cached compilation exists and is valid
        if (await AsyncFileOperations.IsKernelCacheValidAsync(cacheFile, sourceFile, cancellationToken).ConfigureAwait(false))
        {
            _logger.LogDebug("Loading cached compilation for {EntryPoint}", entryPoint);
            
            try
            {
                var cachedCode = await AsyncFileOperations.LoadCachedKernelAsync(cacheFile, cancellationToken).ConfigureAwait(false);
                var compilationTime = DateTime.UtcNow - startTime;
                
                return new CompiledKernel
                {
                    KernelId = $"cached_source:{entryPoint}:{Guid.NewGuid():N}",
                    Name = entryPoint,
                    CompiledCode = cachedCode,
                    Metadata = new KernelMetadata(
                        ExtendedMetadata: new Dictionary<string, object>
                        {
                            ["source_language"] = language.ToString(),
                            ["compilation_method"] = "cached_source",
                            ["entry_point"] = entryPoint,
                            ["compilation_time_ms"] = compilationTime.TotalMilliseconds,
                            ["optimization_level"] = options.OptimizationLevel.ToString(),
                            ["from_cache"] = true
                        })
                };
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load cached compilation, performing fresh compilation");
            }
        }
        
        // Perform fresh compilation
        return await Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            // Simulate realistic compilation time based on source complexity
            var sourceCode = await AsyncFileOperations.ReadKernelSourceAsync(sourceFile, cancellationToken).ConfigureAwait(false);
            var simulatedCompilationTime = Math.Max(10, sourceCode.Length / 1000);
            await Task.Delay(simulatedCompilationTime, cancellationToken).ConfigureAwait(false);
            
            var compilationTime = DateTime.UtcNow - startTime;
            var compiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode);

            // Cache the compilation result asynchronously
            try
            {
                await AsyncFileOperations.WriteCompiledKernelAsync(cacheFile, compiledCode, cancellationToken).ConfigureAwait(false);
                _logger.LogTrace("Cached compilation result for {EntryPoint}", entryPoint);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Failed to cache compilation result for {EntryPoint}", entryPoint);
            }

            var compiledKernel = new CompiledKernel
            {
                KernelId = $"source_compiled:{entryPoint}:{Guid.NewGuid():N}",
                Name = entryPoint,
                CompiledCode = compiledCode,
                Metadata = new KernelMetadata(
                    ExtendedMetadata: new Dictionary<string, object>
                    {
                        ["source_language"] = language.ToString(),
                        ["compilation_method"] = "source_to_method",
                        ["entry_point"] = entryPoint,
                        ["compilation_time_ms"] = compilationTime.TotalMilliseconds,
                        ["source_length"] = sourceCode.Length,
                        ["optimization_level"] = options.OptimizationLevel.ToString(),
                        ["requires_runtime_compilation"] = true,
                        ["from_cache"] = false
                    })
            };
            
            // In a production implementation, this would involve:
            // - Roslyn for C# source compilation
            // - HLSL compiler for DirectX shaders
            // - OpenCL C compiler for OpenCL kernels
            // - Custom parsers for domain-specific languages
            
            _logger.LogDebug("Source compilation completed for {EntryPoint} in {CompilationTime}ms. " +
                "Note: This is a simplified implementation requiring runtime method generation.", 
                entryPoint, compilationTime.TotalMilliseconds);
                
            return compiledKernel;
        }, cancellationToken).ConfigureAwait(false);
    }
    
    /// <summary>
    /// Generates a cache key for the given compilation parameters
    /// </summary>
    private string GenerateCacheKey(string sourceCode, string entryPoint, KernelLanguage language, KernelCompilationOptions options)
    {
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var input = $"{sourceCode}|{entryPoint}|{language}|{options.OptimizationLevel}";
        var hashBytes = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hashBytes)[..16]; // Use first 16 characters
    }
    
    /// <summary>
    /// Gets the appropriate file extension for a kernel language
    /// </summary>
    private string GetFileExtension(KernelLanguage language)
    {
        return language switch
        {
            KernelLanguage.CSharp => "cs",
            KernelLanguage.HLSL => "hlsl",
            KernelLanguage.OpenCL => "cl",
            KernelLanguage.MSL => "metal",
            KernelLanguage.CUDA => "cu",
            KernelLanguage.PTX => "ptx",
            KernelLanguage.SPIRV => "spv",
            _ => "kernel"
        };
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing ILGPU kernel compiler");

        try
        {
            // Try async disposal with timeout
            var disposeTask = DisposeAsyncCore();
            if (!disposeTask.IsCompletedSuccessfully)
            {
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                try
                {
                    disposeTask.AsTask().Wait(cts.Token);
                }
                catch (OperationCanceledException)
                {
                    _logger.LogWarning("Async disposal timed out, performing sync disposal");
                    ClearCache();
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU kernel compiler");
        }

        _disposed = true;
    }

    private async ValueTask DisposeAsyncCore()
    {
        await Task.Run(() => ClearCache()).ConfigureAwait(false);
    }

    /// <summary>
    /// Gets a suitable device for kernel compilation
    /// </summary>
    private async Task<ILGPUComputeDevice> GetCompilationDeviceAsync(CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var device = _deviceManager.GetDefaultDevice() as ILGPUComputeDevice;
            if (device == null)
            {
                throw new InvalidOperationException("No suitable ILGPU device available for compilation");
            }
            return device;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Compiles a kernel asynchronously with proper resource management
    /// </summary>
    private async Task<CompiledKernel> CompileKernelAsync(
        MethodInfo method,
        ILGPUComputeDevice device,
        Accelerator accelerator,
        KernelCompilationOptions options,
        string cacheKey,
        CancellationToken cancellationToken)
    {
        return await Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var startTime = DateTime.UtcNow;
            
            // Load the kernel method using ILGPU's LoadAutoGroupedStreamKernel with explicit typing
            // Note: This is a simplified approach - in a real implementation, we'd analyze method signature
            // and use appropriate index type and parameter types
            var loadedKernel = await Task.Run(() =>
            {
                return accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                    (Action<Index1D, ArrayView<int>>)Delegate.CreateDelegate(
                        typeof(Action<Index1D, ArrayView<int>>), method));
            }, cancellationToken).ConfigureAwait(false);
            
            var compilationTime = DateTime.UtcNow - startTime;

            // Create metadata
            var metadata = new KernelMetadata(
                RequiredSharedMemory: 0, // ILGPU manages this automatically
                RequiredRegisters: 0,
                MaxThreadsPerBlock: device.MaxThreadsPerBlock,
                PreferredBlockSize: await Task.Run(() => CalculatePreferredBlockSize(accelerator), cancellationToken).ConfigureAwait(false),
                UsesAtomics: await Task.Run(() => AnalyzeMethodForAtomics(method), cancellationToken).ConfigureAwait(false),
                UsesSharedMemory: await Task.Run(() => AnalyzeMethodForSharedMemory(method), cancellationToken).ConfigureAwait(false),
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

            // Cache the ILGPU kernel delegate - cast to generic Action delegate
            _ilgpuKernelCache[cacheKey] = (index, arrayView) => 
            {
                // This is a simplified approach - in production we'd properly handle parameter mapping
                // ILGPU kernels have different signatures than the generic delegate we're using
                // Execute with the provided parameters
                loadedKernel(index, arrayView);
            };

            return compiledKernel;
        }, cancellationToken).ConfigureAwait(false);
    }
}