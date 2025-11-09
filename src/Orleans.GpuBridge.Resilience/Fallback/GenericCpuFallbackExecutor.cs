using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Generic CPU fallback executor
/// </summary>
internal sealed class GenericCpuFallbackExecutor : IFallbackExecutor<object, object>, IFallbackAware
{
    private readonly ILogger<GenericCpuFallbackExecutor> _logger;

    public FallbackLevel Level => FallbackLevel.Degraded;

    public GenericCpuFallbackExecutor(ILogger<GenericCpuFallbackExecutor> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task<object> ExecuteAsync(object input, string operationName, CancellationToken cancellationToken)
    {
        _logger.LogDebug("Executing generic CPU fallback for {OperationName}", operationName);

        await Task.Yield(); // Make it async

        // Generic fallback - just return the input (identity operation)
        return input;
    }

    public bool ShouldFallback(Exception exception)
    {
        return exception is Orleans.GpuBridge.Abstractions.Exceptions.GpuBridgeException;
    }
}
