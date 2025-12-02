using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Fallback executor for CPU operations with float arrays
/// </summary>
internal sealed class CpuFallbackExecutor : IFallbackExecutor<float[], float>, IFallbackAware
{
    private readonly ILogger<CpuFallbackExecutor> _logger;

    public FallbackLevel Level => FallbackLevel.Degraded;

    public int Priority => (int)FallbackLevel.Degraded;

    public CpuFallbackExecutor(ILogger<CpuFallbackExecutor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<float> ExecuteAsync(float[] input, string operationName, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Executing CPU fallback for {OperationName} with {InputLength} elements",
            operationName, input.Length);

        await Task.Yield(); // Make it async

        // Simple CPU implementation - sum all elements
        var sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            sum += input[i];
        }

        return sum;
    }

    public bool ShouldFallback(Exception exception)
    {
        // Fallback on GPU-specific exceptions but not on general compute errors
        return exception is Orleans.GpuBridge.Abstractions.Exceptions.GpuDeviceException or
               Orleans.GpuBridge.Abstractions.Exceptions.GpuKernelException or
               Orleans.GpuBridge.Abstractions.Exceptions.GpuMemoryException;
    }
}
