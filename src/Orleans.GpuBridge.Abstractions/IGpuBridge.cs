using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Main interface for GPU bridge operations
/// </summary>
public interface IGpuBridge
{
    /// <summary>
    /// Gets information about the GPU bridge and available resources
    /// </summary>
    ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default);

    /// <summary>
    /// Gets a kernel executor for the specified kernel ID
    /// </summary>
    ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull;

    /// <summary>
    /// Gets the list of available GPU devices
    /// </summary>
    ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default);

    /// <summary>
    /// Executes a kernel with dynamic input/output types
    /// </summary>
    /// <param name="kernelId">The kernel identifier</param>
    /// <param name="input">The input data (dynamic typed)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>The kernel execution result</returns>
    /// <remarks>
    /// This method provides dynamic kernel execution for scenarios where
    /// compile-time type information is not available. Use with caution.
    /// TODO: Implement full dynamic kernel execution support
    /// </remarks>
    [RequiresDynamicCode("Dynamic kernel execution uses runtime reflection to create generic method calls.")]
    [RequiresUnreferencedCode("Dynamic kernel execution uses reflection which may not work with trimming.")]
    ValueTask<object> ExecuteKernelAsync(string kernelId, object input, CancellationToken ct = default);
}