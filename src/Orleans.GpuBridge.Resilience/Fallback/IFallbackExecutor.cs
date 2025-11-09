using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Resilience.Fallback;

/// <summary>
/// Interface for fallback executors
/// </summary>
public interface IFallbackExecutor<in TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    /// <summary>
    /// Priority/level of this executor (lower = higher priority)
    /// </summary>
    FallbackLevel Level { get; }

    /// <summary>
    /// Executes the operation
    /// </summary>
    Task<TOut> ExecuteAsync(TIn input, string operationName, CancellationToken cancellationToken);
}
