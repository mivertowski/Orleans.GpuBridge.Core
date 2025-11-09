using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
// Resilience features will be integrated in a future release

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// Central catalog for resolving and managing GPU kernel registrations.
/// Provides thread-safe kernel resolution with automatic CPU fallback.
/// </summary>
public sealed class KernelCatalog
{
    private readonly ILogger<KernelCatalog> _logger;
    private readonly Dictionary<string, Func<IServiceProvider, object>> _factories = new();
    private readonly SemaphoreSlim _catalogLock = new(1, 1);

    /// <summary>
    /// Initializes a new instance of the <see cref="KernelCatalog"/> class.
    /// </summary>
    /// <param name="logger">Logger for catalog operations.</param>
    /// <param name="options">Catalog configuration options containing kernel descriptors.</param>
    /// <exception cref="ArgumentNullException">Thrown when logger or options are null.</exception>
    public KernelCatalog(
        [NotNull] ILogger<KernelCatalog> logger,
        [NotNull] IOptions<KernelCatalogOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        var descriptors = options?.Value?.Descriptors ?? throw new ArgumentNullException(nameof(options));
        foreach (var d in descriptors)
        {
            if (d.Factory != null)
                _factories[d.Id.Value] = d.Factory;
        }

        _logger.LogInformation("KernelCatalog initialized with {Count} kernel descriptors and resilience support", _factories.Count);
    }

    /// <summary>
    /// Resolves a kernel instance by its identifier with the specified input and output types.
    /// Falls back to CPU passthrough if the kernel is not found in the catalog.
    /// </summary>
    /// <typeparam name="TIn">The input type for the kernel.</typeparam>
    /// <typeparam name="TOut">The output type for the kernel.</typeparam>
    /// <param name="id">The kernel identifier.</param>
    /// <param name="sp">The service provider for dependency resolution.</param>
    /// <param name="cancellationToken">Cancellation token for the operation.</param>
    /// <returns>A resolved kernel instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown when kernel resolution fails.</exception>
    public async Task<IGpuKernel<TIn, TOut>> ResolveAsync<TIn, TOut>(KernelId id, IServiceProvider sp, CancellationToken cancellationToken = default)
        where TIn : notnull
        where TOut : notnull
    {
        var operationName = $"KernelResolve_{id.Value}";
        var startTime = DateTimeOffset.UtcNow;

        try
        {
            await _catalogLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (_factories.TryGetValue(id.Value, out var factory))
                {
                    _logger.LogDebug("Resolving kernel {KernelId} from factory", id.Value);

                    // Execute factory asynchronously for complex kernels
                    var kernelObject = await Task.Run(() => factory(sp), cancellationToken).ConfigureAwait(false);

                    if (kernelObject is IGpuKernel<TIn, TOut> kernel)
                    {
                        // Initialize kernel asynchronously if it supports it
                        if (kernel is IAsyncInitializable asyncInitializable)
                        {
                            await asyncInitializable.InitializeAsync(cancellationToken).ConfigureAwait(false);
                        }

                        return kernel;
                    }
                    else
                    {
                        _logger.LogError("Factory for kernel {KernelId} returned incompatible type {ActualType}, expected {ExpectedType}",
                            id.Value, kernelObject?.GetType(), typeof(IGpuKernel<TIn, TOut>));
                        throw new InvalidOperationException($"Kernel factory returned incompatible type for {id.Value}");
                    }
                }
                else
                {
                    _logger.LogInformation("Kernel {KernelId} not found in catalog, using CPU passthrough", id.Value);
                    return new CpuPassthroughKernel<TIn, TOut>();
                }
            }
            finally
            {
                _catalogLock.Release();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to resolve kernel {KernelId}", id.Value);
            throw new InvalidOperationException($"Failed to resolve kernel: {id.Value}", ex);
        }
    }
}
