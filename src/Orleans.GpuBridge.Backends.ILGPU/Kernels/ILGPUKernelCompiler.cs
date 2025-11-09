using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// ILGPU kernel compiler implementation - Main entry point
/// </summary>
/// <remarks>
/// This class is split across multiple partial files for maintainability:
/// - ILGPUKernelCompiler.cs: Constructor, fields, and disposal
/// - ILGPUKernelCompiler.Compilation.cs: Compilation methods and helpers
/// - ILGPUKernelCompiler.Validation.cs: Validation and analysis methods
/// - ILGPUKernelCompiler.Cache.cs: Cache management and diagnostics
/// </remarks>
internal sealed partial class ILGPUKernelCompiler : IKernelCompiler
{
    #region Fields

    private readonly ILogger<ILGPUKernelCompiler> _logger;
    private readonly Context _context;
    private readonly ILGPUDeviceManager _deviceManager;
    private readonly ConcurrentDictionary<string, CompiledKernel> _compilationCache;
    private readonly ConcurrentDictionary<string, Action<Index1D, ArrayView<int>>> _ilgpuKernelCache;
    private bool _disposed;

    #endregion

    #region Constructor

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

    #endregion

    #region Internal Access

    /// <summary>
    /// Gets a cached ILGPU kernel by ID
    /// </summary>
    internal Action<Index1D, ArrayView<int>>? GetCachedILGPUKernel(string kernelId)
    {
        return _ilgpuKernelCache.TryGetValue(kernelId, out var kernel) ? kernel : null;
    }

    #endregion

    #region Disposal

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

    #endregion
}
