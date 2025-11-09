using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Execution;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Backends.DotCompute.Kernels;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
// Type aliases to avoid ambiguity with Orleans.GpuBridge types
using OrleansCompiledKernel = Orleans.GpuBridge.Abstractions.Models.CompiledKernel;
using DotComputeKernelArguments = DotCompute.Abstractions.Kernels.KernelArguments;
using DotComputeCompiledKernel = DotCompute.Abstractions.ICompiledKernel;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// DotCompute kernel executor implementation
/// </summary>
/// <remarks>
/// This partial class contains the core execution and coordination logic.
/// Additional functionality is split across partial class files for maintainability.
/// </remarks>
internal sealed partial class DotComputeKernelExecutor : IKernelExecutor
{
    private readonly ILogger<DotComputeKernelExecutor> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private readonly IDeviceManager _deviceManager;
    private readonly IMemoryAllocator _memoryAllocator;
    private readonly IKernelCompiler _kernelCompiler;
    private readonly ConcurrentDictionary<string, DotComputeKernelExecution> _activeExecutions;
    private readonly ExecutionStatistics _statistics;
    private readonly SemaphoreSlim _executionSemaphore;
    private bool _disposed;

    public DotComputeKernelExecutor(
        ILogger<DotComputeKernelExecutor> logger,
        ILoggerFactory loggerFactory,
        IDeviceManager deviceManager,
        IMemoryAllocator memoryAllocator,
        IKernelCompiler kernelCompiler)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        _deviceManager = deviceManager ?? throw new ArgumentNullException(nameof(deviceManager));
        _memoryAllocator = memoryAllocator ?? throw new ArgumentNullException(nameof(memoryAllocator));
        _kernelCompiler = kernelCompiler ?? throw new ArgumentNullException(nameof(kernelCompiler));
        _activeExecutions = new ConcurrentDictionary<string, DotComputeKernelExecution>();
        _statistics = new ExecutionStatistics(0, 0, 0, TimeSpan.Zero, TimeSpan.Zero, 0, 0, new System.Collections.Generic.Dictionary<string, long>());
        _executionSemaphore = new SemaphoreSlim(50, 50); // Limit concurrent executions
    }

    /// <summary>
    /// Executes a kernel asynchronously with GPU acceleration
    /// </summary>
    public async Task<KernelExecutionResult> ExecuteAsync(
        OrleansCompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var stopwatch = Stopwatch.StartNew();

        try
        {
            _logger.LogDebug("Executing DotCompute kernel: {KernelName}", kernel.Name);

            await _executionSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);

            try
            {
                // Select device for execution
                var device = SelectExecutionDevice(parameters);

                // Get the compiled DotCompute kernel
                var dotComputeKernel = (_kernelCompiler as DotComputeKernelCompiler)?.GetCachedDotComputeKernel(kernel.KernelId);
                if (dotComputeKernel == null)
                {
                    throw new InvalidOperationException($"DotCompute kernel not found: {kernel.KernelId}");
                }

                var nativeKernel = dotComputeKernel.GetNativeKernel();

                // Prepare kernel arguments
                var kernelArgs = await PrepareKernelArgumentsAsync(parameters, device, cancellationToken).ConfigureAwait(false);

                // Calculate work dimensions
                var workDimensions = CalculateWorkDimensions(parameters);

                // Execute the kernel
                var executionStart = Stopwatch.StartNew();

                try
                {
                    await ExecuteDotComputeKernelAsync(
                        nativeKernel,
                        kernelArgs,
                        workDimensions,
                        device,
                        cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "DotCompute kernel execution failed: {KernelName}", kernel.Name);
                    throw new InvalidOperationException($"DotCompute kernel execution failed: {ex.Message}", ex);
                }

                executionStart.Stop();

                // Update statistics
                UpdateExecutionStatistics(kernel.Name, executionStart.Elapsed, true);

                var timing = new KernelTiming(
                    QueueTime: TimeSpan.FromMilliseconds(stopwatch.ElapsedMilliseconds - executionStart.ElapsedMilliseconds),
                    KernelTime: executionStart.Elapsed,
                    TotalTime: stopwatch.Elapsed);

                _logger.LogDebug(
                    "DotCompute kernel execution completed: {KernelName} in {ExecutionTime}ms",
                    kernel.Name, executionStart.ElapsedMilliseconds);

                return new KernelExecutionResult(
                    Success: true,
                    Timing: timing,
                    Metadata: new System.Collections.Generic.Dictionary<string, object>
                    {
                        ["device_name"] = device.Name,
                        ["device_type"] = device.Type.ToString(),
                        ["global_work_size"] = parameters.GlobalWorkSize,
                        ["local_work_size"] = parameters.LocalWorkSize ?? Array.Empty<int>()
                    });
            }
            finally
            {
                _executionSemaphore.Release();
            }
        }
        catch (Exception ex)
        {
            UpdateExecutionStatistics(kernel.Name, stopwatch.Elapsed, false);

            _logger.LogError(ex, "Failed to execute DotCompute kernel: {KernelName}", kernel.Name);

            return new KernelExecutionResult(
                Success: false,
                ErrorMessage: ex.Message,
                Timing: new KernelTiming(
                    QueueTime: TimeSpan.Zero,
                    KernelTime: TimeSpan.Zero,
                    TotalTime: stopwatch.Elapsed));
        }
    }

    /// <summary>
    /// Executes a kernel asynchronously in non-blocking mode
    /// </summary>
    public Task<IKernelExecution> ExecuteAsyncNonBlocking(
        OrleansCompiledKernel kernel,
        KernelExecutionParameters parameters,
        CancellationToken cancellationToken = default)
    {
        if (kernel == null)
            throw new ArgumentNullException(nameof(kernel));

        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        var executionId = Guid.NewGuid().ToString("N");
        var execution = new DotComputeKernelExecution(
            executionId,
            kernel,
            this,
            _loggerFactory.CreateLogger<DotComputeKernelExecution>());

        _activeExecutions[executionId] = execution;

        // Start execution in background
        _ = Task.Run(async () =>
        {
            try
            {
                var result = await ExecuteAsync(kernel, parameters, cancellationToken).ConfigureAwait(false);
                execution.SetResult(result);
            }
            catch (Exception ex)
            {
                execution.SetError(ex);
            }
            finally
            {
                _activeExecutions.TryRemove(executionId, out _);
            }
        }, cancellationToken);

        return Task.FromResult<IKernelExecution>(execution);
    }

    /// <summary>
    /// Executes a DotCompute kernel on the GPU with real GPU acceleration
    /// </summary>
    /// <remarks>
    /// This method performs actual GPU kernel execution using DotCompute's ExecuteAsync API.
    ///
    /// Key features:
    /// - Real CUDA kernel execution via NVRTC
    /// - Automatic launch configuration (DotCompute handles grid/block dimensions)
    /// - Asynchronous GPU synchronization
    /// - Production-grade error handling
    ///
    /// WorkDimensions are currently informational only - DotCompute v0.4.1-rc2
    /// automatically determines optimal launch configuration based on kernel characteristics.
    /// </remarks>
    private async Task ExecuteDotComputeKernelAsync(
        object kernel,
        DotComputeKernelArguments arguments,
        WorkDimensions workDimensions,
        IComputeDevice device,
        CancellationToken cancellationToken)
    {
        // Validate kernel type - must be DotCompute ICompiledKernel
        if (kernel is not DotComputeCompiledKernel compiledKernel)
        {
            throw new InvalidOperationException(
                $"Kernel is not a DotCompute ICompiledKernel. Got type: {kernel?.GetType().FullName ?? "null"}");
        }

        _logger.LogDebug(
            "Executing DotCompute kernel '{KernelName}' on device '{DeviceId}' with {ArgCount} arguments",
            compiledKernel.Name,
            device.DeviceId,
            arguments.Count);

        try
        {
            // ✅ REAL API: Execute kernel on GPU using DotCompute
            // DotCompute v0.4.1-rc2: ExecuteAsync automatically handles:
            // - Optimal grid/block dimension calculation
            // - GPU memory synchronization
            // - Asynchronous execution with proper await
            await compiledKernel.ExecuteAsync(arguments, cancellationToken).ConfigureAwait(false);

            _logger.LogDebug(
                "Successfully executed DotCompute kernel '{KernelName}' on GPU",
                compiledKernel.Name);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Failed to execute DotCompute kernel '{KernelName}' on device '{DeviceId}'",
                compiledKernel.Name,
                device.DeviceId);

            throw new InvalidOperationException(
                $"DotCompute kernel execution failed: {ex.Message}",
                ex);
        }
    }

    internal void RemoveActiveExecution(string executionId)
    {
        _activeExecutions.TryRemove(executionId, out _);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _logger.LogDebug("Disposing DotCompute kernel executor");

        try
        {
            // Cancel all active executions
            foreach (var execution in _activeExecutions.Values)
            {
                try
                {
                    _ = execution.CancelAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error cancelling active execution: {ExecutionId}", execution.ExecutionId);
                }
            }

            _activeExecutions.Clear();
            _executionSemaphore?.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute kernel executor");
        }

        _disposed = true;
    }
}
