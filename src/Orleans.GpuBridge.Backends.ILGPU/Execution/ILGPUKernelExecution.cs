using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;
using Orleans.GpuBridge.Abstractions.Models;

namespace Orleans.GpuBridge.Backends.ILGPU.Execution;

/// <summary>
/// ILGPU kernel execution tracking
/// </summary>
internal sealed class ILGPUKernelExecution : IKernelExecution
{
    private readonly ILGPUKernelExecutor _executor;
    private readonly ILogger<ILGPUKernelExecution> _logger;
    private readonly TaskCompletionSource<KernelExecutionResult> _completionSource;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private KernelTiming? _timing;
    private volatile KernelExecutionStatus _status;
    private volatile int _progressInt; // Progress as int (0-100)

    public string ExecutionId { get; }
    public CompiledKernel Kernel { get; }
    public KernelExecutionStatus Status => _status;
    public bool IsComplete => _status == KernelExecutionStatus.Completed ||
                             _status == KernelExecutionStatus.Failed ||
                             _status == KernelExecutionStatus.Cancelled ||
                             _status == KernelExecutionStatus.Timeout;
    public double Progress => _progressInt / 100.0;

    public ILGPUKernelExecution(
        string executionId,
        CompiledKernel kernel,
        ILGPUKernelExecutor executor,
        ILogger<ILGPUKernelExecution> logger)
    {
        ExecutionId = executionId ?? throw new ArgumentNullException(nameof(executionId));
        Kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _completionSource = new TaskCompletionSource<KernelExecutionResult>();
        _cancellationTokenSource = new CancellationTokenSource();
        _status = KernelExecutionStatus.Queued;
        _progressInt = 0;

        _logger.LogDebug("Created kernel execution: {ExecutionId} for kernel: {KernelName}",
            ExecutionId, Kernel.Name);
    }

    public async Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogDebug("Waiting for kernel execution completion: {ExecutionId}", ExecutionId);

            using var combinedTokenSource = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, _cancellationTokenSource.Token);

            var result = await _completionSource.Task.ConfigureAwait(false);

            _logger.LogDebug(
                "Kernel execution completed: {ExecutionId}, Success: {Success}",
                ExecutionId, result.Success);

            return result;
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("Kernel execution wait cancelled: {ExecutionId}", ExecutionId);

            if (cancellationToken.IsCancellationRequested)
                throw;

            // Internal cancellation
            return new KernelExecutionResult(false, "Execution was cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error waiting for kernel execution: {ExecutionId}", ExecutionId);
            throw;
        }
    }

    public Task CancelAsync()
    {
        if (IsComplete)
        {
            _logger.LogDebug("Cannot cancel already completed execution: {ExecutionId}", ExecutionId);
            return Task.CompletedTask;
        }

        try
        {
            _logger.LogInformation("Cancelling kernel execution: {ExecutionId}", ExecutionId);

            _status = KernelExecutionStatus.Cancelled;
            _cancellationTokenSource.Cancel();

            var result = new KernelExecutionResult(false, "Execution was cancelled");
            _completionSource.TrySetResult(result);

            // Remove from active executions
            _executor.RemoveActiveExecution(ExecutionId);

            _logger.LogInformation("Kernel execution cancelled: {ExecutionId}", ExecutionId);
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cancelling kernel execution: {ExecutionId}", ExecutionId);
            throw;
        }
    }

    public KernelTiming? GetTiming()
    {
        return _timing;
    }

    internal void SetRunning()
    {
        _status = KernelExecutionStatus.Running;
        _progressInt = 10; // Small progress to indicate start (10%)

        _logger.LogDebug("Kernel execution started: {ExecutionId}", ExecutionId);
    }

    internal void UpdateProgress(double progress)
    {
        if (progress < 0.0 || progress > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(progress), "Progress must be between 0.0 and 1.0");
        }

        _progressInt = (int)(progress * 100);

        _logger.LogTrace("Kernel execution progress: {ExecutionId} - {Progress:P1}", ExecutionId, progress);
    }

    internal void SetResult(KernelExecutionResult result)
    {
        if (result == null)
            throw new ArgumentNullException(nameof(result));

        try
        {
            _status = result.Success ? KernelExecutionStatus.Completed : KernelExecutionStatus.Failed;
            _progressInt = 100; // 100% complete
            _timing = result.Timing;

            _completionSource.TrySetResult(result);

            _logger.LogDebug(
                "Kernel execution result set: {ExecutionId}, Success: {Success}",
                ExecutionId, result.Success);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting kernel execution result: {ExecutionId}", ExecutionId);
            SetError(ex);
        }
    }

    internal void SetError(Exception exception)
    {
        if (exception == null)
            throw new ArgumentNullException(nameof(exception));

        try
        {
            _status = KernelExecutionStatus.Failed;
            _progressInt = 100; // Mark as complete even though failed

            var result = new KernelExecutionResult(false, exception.Message);
            _completionSource.TrySetResult(result);

            _logger.LogError(exception, "Kernel execution failed: {ExecutionId}", ExecutionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting kernel execution error: {ExecutionId}", ExecutionId);

            // Last resort - set exception directly
            _completionSource.TrySetException(ex);
        }
    }

    internal void SetTimeout()
    {
        try
        {
            _status = KernelExecutionStatus.Timeout;
            _progressInt = 100; // Mark as complete due to timeout

            var result = new KernelExecutionResult(false, "Kernel execution timed out");
            _completionSource.TrySetResult(result);

            _logger.LogWarning("Kernel execution timed out: {ExecutionId}", ExecutionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting kernel execution timeout: {ExecutionId}", ExecutionId);
        }
    }

    internal void Cancel()
    {
        if (!IsComplete)
        {
            _ = CancelAsync();
        }
    }
}
