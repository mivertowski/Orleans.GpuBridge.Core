using System;
using Microsoft.Extensions.Logging;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// ILGPU kernel compiler - Main compilation methods
/// </summary>
internal sealed partial class ILGPUKernelCompiler
{
    #region Public Compilation Methods

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

        _logger.LogWarning(
            "Source code compilation not fully implemented for ILGPU. " +
            "Consider using CompileFromMethodAsync instead.");

        try
        {
            var compiledKernel = await CompileSourceCodeToMethodAsync(sourceCode, entryPoint, language, options, cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("Successfully compiled source kernel: {EntryPoint} from {Language}",
                entryPoint, language);

            return compiledKernel;
        }
        catch (Exception compileEx)
        {
            _logger.LogError(compileEx, "Failed to compile source code for entry point: {EntryPoint}", entryPoint);

            return CreateFallbackKernel(sourceCode, entryPoint, language, compileEx);
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

        // Ensure options is not null before proceeding
        options ??= new KernelCompilationOptions();

        _logger.LogInformation("Compiling ILGPU kernel from assembly: {TypeName}.{MethodName}", typeName, methodName);

        try
        {
            var type = assembly.GetType(typeName);
            if (type == null)
            {
                throw new ArgumentException($"Type '{typeName}' not found in assembly '{assembly.FullName}'");
            }

            var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
            if (method == null)
            {
                throw new ArgumentException($"Method '{methodName}' not found in type '{typeName}'");
            }

            return await CompileFromMethodAsync(method, options, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile ILGPU kernel from assembly: {TypeName}.{MethodName}", typeName, methodName);
            throw;
        }
    }

    #endregion

    #region Kernel Compilation Helpers

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

            var loadedKernel = await Task.Run(() =>
            {
                return accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(
                    (Action<Index1D, ArrayView<int>>)Delegate.CreateDelegate(
                        typeof(Action<Index1D, ArrayView<int>>), method));
            }, cancellationToken).ConfigureAwait(false);

            var compilationTime = DateTime.UtcNow - startTime;

            var metadata = await CreateKernelMetadataAsync(method, device, accelerator, options, compilationTime, cancellationToken).ConfigureAwait(false);

            var compiledKernel = new CompiledKernel
            {
                KernelId = cacheKey,
                Name = method.Name,
                CompiledCode = [],
                Metadata = metadata,
                NativeHandle = IntPtr.Zero
            };

            _ilgpuKernelCache[cacheKey] = (index, arrayView) =>
            {
                loadedKernel(index, arrayView);
            };

            return compiledKernel;
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Creates kernel metadata asynchronously
    /// </summary>
    private async Task<KernelMetadata> CreateKernelMetadataAsync(
        MethodInfo method,
        ILGPUComputeDevice device,
        Accelerator accelerator,
        KernelCompilationOptions options,
        TimeSpan compilationTime,
        CancellationToken cancellationToken)
    {
        return await Task.Run(async () =>
        {
            var preferredBlockSize = await Task.Run(() => CalculatePreferredBlockSize(accelerator), cancellationToken).ConfigureAwait(false);
            var usesAtomics = await Task.Run(() => AnalyzeMethodForAtomics(method), cancellationToken).ConfigureAwait(false);
            var usesSharedMemory = await Task.Run(() => AnalyzeMethodForSharedMemory(method), cancellationToken).ConfigureAwait(false);

            return new KernelMetadata(
                RequiredSharedMemory: 0,
                RequiredRegisters: 0,
                MaxThreadsPerBlock: device.MaxThreadsPerBlock,
                PreferredBlockSize: preferredBlockSize,
                UsesAtomics: usesAtomics,
                UsesSharedMemory: usesSharedMemory,
                UsesDynamicParallelism: false,
                ExtendedMetadata: new System.Collections.Generic.Dictionary<string, object>
                {
                    ["compilation_time_ms"] = compilationTime.TotalMilliseconds,
                    ["accelerator_type"] = accelerator.AcceleratorType.ToString(),
                    ["optimization_level"] = options.OptimizationLevel.ToString()
                });
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Creates a fallback kernel for failed source compilation
    /// </summary>
    private static CompiledKernel CreateFallbackKernel(string sourceCode, string entryPoint, KernelLanguage language, Exception compileEx)
    {
        return new CompiledKernel
        {
            KernelId = $"source_failed:{entryPoint}",
            Name = entryPoint,
            CompiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode),
            Metadata = new KernelMetadata(
                ExtendedMetadata: new System.Collections.Generic.Dictionary<string, object>
                {
                    ["source_language"] = language.ToString(),
                    ["compilation_method"] = "source_fallback",
                    ["compilation_error"] = compileEx.Message,
                    ["compilation_time"] = DateTime.UtcNow
                })
        };
    }

    #endregion
}
