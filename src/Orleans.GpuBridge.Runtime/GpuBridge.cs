using System;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Main implementation of the GPU bridge
/// </summary>
public sealed class GpuBridge : IGpuBridge
{
    private readonly ILogger<GpuBridge> _logger;
    private readonly KernelCatalog _kernelCatalog;
    private readonly DeviceBroker _deviceBroker;
    private readonly GpuBridgeOptions _options;
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Cache for dynamic kernel execution delegates to avoid repeated reflection.
    /// Key: (inputType, outputType) tuple.
    /// </summary>
    private readonly ConcurrentDictionary<(Type, Type), Func<KernelId, object, CancellationToken, Task<object>>> _dynamicExecutorCache = new();

    /// <summary>
    /// MethodInfo for the generic ResolveAsync method, cached for performance.
    /// </summary>
    private static readonly MethodInfo ResolveAsyncMethod = typeof(KernelCatalog)
        .GetMethod(nameof(KernelCatalog.ResolveAsync))!;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuBridge"/> class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="kernelCatalog">Kernel catalog for kernel management</param>
    /// <param name="deviceBroker">Device broker for GPU management</param>
    /// <param name="options">GPU bridge configuration options</param>
    /// <param name="serviceProvider">Service provider for dependency injection</param>
    public GpuBridge(
        ILogger<GpuBridge> logger,
        KernelCatalog kernelCatalog,
        DeviceBroker deviceBroker,
        IOptions<GpuBridgeOptions> options,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _kernelCatalog = kernelCatalog;
        _deviceBroker = deviceBroker;
        _options = options.Value;
        _serviceProvider = serviceProvider;
    }

    /// <inheritdoc/>
    public ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var version = assembly.GetName().Version?.ToString() ?? "1.0.0";

        var info = new GpuBridgeInfo(
            Version: version,
            DeviceCount: _deviceBroker.DeviceCount,
            TotalMemoryBytes: _deviceBroker.TotalMemoryBytes,
            Backend: _options.PreferGpu ? GpuBackend.CUDA : GpuBackend.CPU,
            IsGpuAvailable: _deviceBroker.DeviceCount > 0,
            Metadata: new Dictionary<string, object>
            {
                ["MaxConcurrentKernels"] = _options.MaxConcurrentKernels,
                ["MemoryPoolSizeMB"] = _options.MemoryPoolSizeMB,
                ["EnableProfiling"] = _options.EnableProfiling
            });

        return new ValueTask<GpuBridgeInfo>(info);
    }

    /// <inheritdoc/>
    public async ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull
    {
        _logger.LogDebug("Getting kernel {KernelId}", kernelId);

        var kernel = await _kernelCatalog.ResolveAsync<TIn, TOut>(kernelId, _serviceProvider);

        if (kernel == null)
        {
            _logger.LogWarning("Kernel {KernelId} not found, using CPU passthrough", kernelId);
            kernel = new CpuPassthroughKernel<TIn, TOut>();
        }

        return kernel;
    }

    /// <inheritdoc/>
    public ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default)
    {
        var devices = _deviceBroker.GetDevices();
        return new ValueTask<IReadOnlyList<GpuDevice>>(devices);
    }

    /// <summary>
    /// Executes a kernel with dynamic type resolution (non-generic overload)
    /// </summary>
    /// <param name="kernelId">Kernel identifier</param>
    /// <param name="input">Input data (type will be inferred)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Kernel execution result</returns>
    [RequiresDynamicCode("Dynamic kernel execution uses runtime reflection to create generic method calls.")]
    [RequiresUnreferencedCode("Dynamic kernel execution uses reflection which may not work with trimming.")]
    public async ValueTask<object> ExecuteKernelAsync(string kernelId, object input, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);

        _logger.LogDebug("Executing kernel {KernelId} with dynamic input type {InputType}", kernelId, input.GetType().Name);

        // Get input type from the runtime object
        var inputType = input.GetType();

        // For dynamic execution without compile-time output type knowledge,
        // we use the same type for output as input (common pattern for transforms)
        // or use 'object' as the output type for maximum flexibility
        var outputType = inputType;

        // Get or create the dynamic executor for this type combination
        var executor = _dynamicExecutorCache.GetOrAdd((inputType, outputType), types =>
            CreateDynamicExecutor(types.Item1, types.Item2));

        try
        {
            var result = await executor(new KernelId(kernelId), input, ct);
            _logger.LogDebug("Kernel {KernelId} executed successfully", kernelId);
            return result;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Dynamic kernel execution failed for {KernelId}", kernelId);
            throw new InvalidOperationException(
                $"Dynamic kernel execution failed for kernel '{kernelId}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Executes a kernel dynamically with specified input and output types.
    /// </summary>
    /// <typeparam name="TRequest">The request type.</typeparam>
    /// <typeparam name="TResponse">The response type.</typeparam>
    /// <param name="kernelId">The kernel identifier.</param>
    /// <param name="input">The input object.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The execution result.</returns>
    public async ValueTask<TResponse> ExecuteKernelAsync<TRequest, TResponse>(
        string kernelId,
        TRequest input,
        CancellationToken ct = default)
        where TRequest : notnull
        where TResponse : notnull
    {
        _logger.LogDebug(
            "Executing kernel {KernelId} with typed request {RequestType} -> {ResponseType}",
            kernelId, typeof(TRequest).Name, typeof(TResponse).Name);

        var kernel = await _kernelCatalog.ResolveAsync<TRequest, TResponse>(
            new KernelId(kernelId),
            _serviceProvider,
            ct);

        return await kernel.ExecuteAsync(input, ct);
    }

    /// <summary>
    /// Creates a dynamic executor delegate for the specified type combination.
    /// Uses reflection to invoke the generic ResolveAsync method.
    /// </summary>
    [RequiresDynamicCode("Calls System.Reflection.MethodInfo.MakeGenericMethod(params Type[])")]
    [RequiresUnreferencedCode("Uses reflection to access properties and methods dynamically.")]
    private Func<KernelId, object, CancellationToken, Task<object>> CreateDynamicExecutor(Type inputType, Type outputType)
    {
        _logger.LogDebug(
            "Creating dynamic executor for types {InputType} -> {OutputType}",
            inputType.Name, outputType.Name);

        // Create the generic method for the specific types
        var genericMethod = ResolveAsyncMethod.MakeGenericMethod(inputType, outputType);

        // Return a delegate that invokes the method dynamically
        return async (kernelId, input, ct) =>
        {
            // Invoke ResolveAsync<TIn, TOut>(kernelId, serviceProvider, ct)
            var task = (Task)genericMethod.Invoke(
                _kernelCatalog,
                new object[] { kernelId, _serviceProvider, ct })!;

            await task;

            // Get the Result property from the generic Task<T>
            var resultProperty = task.GetType().GetProperty("Result")!;
            var kernel = resultProperty.GetValue(task)!;

            // Get the ExecuteAsync method from the kernel
            var executeMethod = kernel.GetType().GetMethod("ExecuteAsync", new[] { inputType, typeof(CancellationToken) });
            if (executeMethod == null)
            {
                throw new InvalidOperationException(
                    $"Kernel does not have ExecuteAsync method for input type {inputType.Name}");
            }

            // Invoke ExecuteAsync(input, ct)
            var executeTask = (Task)executeMethod.Invoke(kernel, new[] { input, ct })!;
            await executeTask;

            // Get the result from the execute task
            var executeResultProperty = executeTask.GetType().GetProperty("Result")!;
            return executeResultProperty.GetValue(executeTask)!;
        };
    }
}