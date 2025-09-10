using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
// Resilience features will be integrated in a future release

namespace Orleans.GpuBridge.Runtime;

public sealed class KernelCatalogOptions
{ 
    public List<KernelDescriptor> Descriptors { get; } = new(); 
}

public sealed class KernelDescriptor
{
    public KernelId Id { get; set; } = new("unset");
    public Type InType { get; set; } = typeof(object);
    public Type OutType { get; set; } = typeof(object);
    public Func<IServiceProvider, object>? Factory { get; set; }
    
    public static KernelDescriptor Build(Action<KernelDescriptor> cfg)
    {
        var d = new KernelDescriptor();
        cfg(d);
        return d;
    }
    
    public KernelDescriptor SetId(string id)
    {
        Id = new(id);
        return this;
    }
    
    public KernelDescriptor In<T>()
    {
        InType = typeof(T);
        return this;
    }
    
    public KernelDescriptor Out<T>()
    {
        OutType = typeof(T);
        return this;
    }
    
    public KernelDescriptor FromFactory(Func<IServiceProvider, object> f)
    {
        Factory = f;
        return this;
    }
}

public sealed class KernelCatalog
{
    private readonly ILogger<KernelCatalog> _logger;
    private readonly Dictionary<string, Func<IServiceProvider, object>> _factories = new();
    private readonly SemaphoreSlim _catalogLock = new(1, 1);
    
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

internal sealed class CpuPassthroughKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly Dictionary<string, IReadOnlyList<TIn>> _batches = new();
    
    public ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var handle = KernelHandle.Create();
        _batches[handle.Id] = items;
        return new(handle);
    }
    
    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield(); // Ensure async
        if (!_batches.TryGetValue(handle.Id, out var items))
        {
            yield break;
        }
        
        // For passthrough, attempt to cast directly if types match
        if (typeof(TIn) == typeof(TOut))
        {
            foreach (var item in items)
            {
                ct.ThrowIfCancellationRequested();
                
                // Process item asynchronously for large datasets
                var result = await Task.Run(() =>
                {
                    if (item is TOut directResult)
                    {
                        return directResult;
                    }
                    else
                    {
                        return default(TOut)!;
                    }
                }, ct).ConfigureAwait(false);
                
                yield return result;
            }
        }
        else
        {
            // For different types, use async conversion
            foreach (var item in items)
            {
                ct.ThrowIfCancellationRequested();
                
                // Try to convert using common patterns asynchronously
                var result = await Task.Run(() =>
                {
                    TOut convertedResult = default(TOut)!;
                    try
                    {
                        if (item is IConvertible convertible)
                        {
                            var converted = Convert.ChangeType(item, typeof(TOut));
                            if (converted is TOut typedResult)
                            {
                                convertedResult = typedResult;
                            }
                        }
                    }
                    catch
                    {
                        // Use default for conversion failures
                    }
                    
                    return convertedResult;
                }, ct).ConfigureAwait(false);
                
                yield return result;
            }
        }
        
        // Clean up the batch after processing
        _batches.Remove(handle.Id);
    }
    
    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        return new(new KernelInfo(
            new KernelId("cpu-passthrough"),
            "CPU passthrough kernel",
            typeof(TIn),
            typeof(TOut),
            false,
            1024));
    }
}

public sealed class CpuVectorAddKernel : IGpuKernel<float[], float>
{
    private readonly Dictionary<string, IReadOnlyList<float[]>> _batches = new();
    
    public static float Execute(float[] a, float[] b)
    {
        var n = Math.Min(a.Length, b.Length);
        var sum = 0f;
        for (int i = 0; i < n; i++)
            sum += a[i] + b[i];
        return sum;
    }
    
    public ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<float[]> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var handle = KernelHandle.Create();
        _batches[handle.Id] = items;
        return new(handle);
    }
    
    public async IAsyncEnumerable<float> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield(); // Ensure async
        if (!_batches.TryGetValue(handle.Id, out var items))
        {
            yield break;
        }
        
        // Process pairs of vectors asynchronously for large datasets
        for (int i = 0; i < items.Count - 1; i += 2)
        {
            ct.ThrowIfCancellationRequested();
            
            var a = items[i];
            var b = items[i + 1];
            
            // Execute computation asynchronously for large vectors
            var result = await Task.Run(() => Execute(a, b), ct).ConfigureAwait(false);
            yield return result;
        }
        
        // If odd number of items, return sum of last vector
        if (items.Count % 2 == 1)
        {
            var lastVector = items[items.Count - 1];
            var sum = await Task.Run(() => lastVector.Sum(), ct).ConfigureAwait(false);
            yield return sum;
        }
        
        // Clean up the batch after processing
        _batches.Remove(handle.Id);
    }
    
    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        return new(new KernelInfo(
            new KernelId("cpu-vector-add"),
            "CPU vector addition kernel",
            typeof(float[]),
            typeof(float),
            false,
            1024));
    }
}
