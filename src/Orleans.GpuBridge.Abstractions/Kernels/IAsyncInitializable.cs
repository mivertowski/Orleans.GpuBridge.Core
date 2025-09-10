using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Abstractions.Kernels;

/// <summary>
/// Interface for kernels that support asynchronous initialization
/// </summary>
public interface IAsyncInitializable
{
    /// <summary>
    /// Asynchronously initializes the kernel with optional cancellation support
    /// </summary>
    /// <param name="cancellationToken">Cancellation token to observe</param>
    /// <returns>A task representing the initialization operation</returns>
    Task InitializeAsync(CancellationToken cancellationToken = default);
}