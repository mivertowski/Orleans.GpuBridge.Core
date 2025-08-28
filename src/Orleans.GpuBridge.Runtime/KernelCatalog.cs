using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

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
    private readonly Dictionary<string, Func<IServiceProvider, object>> _factories = new();
    
    public KernelCatalog(IOptions<KernelCatalogOptions> options)
    {
        foreach (var d in options.Value.Descriptors)
        {
            if (d.Factory != null)
                _factories[d.Id.Value] = d.Factory;
        }
    }
    
    public Task<IGpuKernel<TIn, TOut>> ResolveAsync<TIn, TOut>(KernelId id, IServiceProvider sp)
        where TIn : notnull
        where TOut : notnull
    {
        if (_factories.TryGetValue(id.Value, out var factory))
        {
            return Task.FromResult((IGpuKernel<TIn, TOut>)factory(sp));
        }
        
        return Task.FromResult<IGpuKernel<TIn, TOut>>(new CpuPassthroughKernel<TIn, TOut>());
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
                yield return (TOut)(object)item;
            }
        }
        else
        {
            // For different types, use default conversion or throw
            foreach (var item in items)
            {
                ct.ThrowIfCancellationRequested();
                
                // Try to convert using common patterns
                if (item is IConvertible convertible && typeof(TOut).IsAssignableFrom(typeof(IConvertible)))
                {
                    yield return (TOut)Convert.ChangeType(item, typeof(TOut));
                }
                else
                {
                    // Return default for type mismatch in passthrough mode
                    yield return default(TOut)!;
                }
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
        if (!_batches.TryGetValue(handle.Id, out var items))
        {
            yield break;
        }
        
        // Process pairs of vectors
        for (int i = 0; i < items.Count - 1; i += 2)
        {
            ct.ThrowIfCancellationRequested();
            
            var a = items[i];
            var b = items[i + 1];
            var result = Execute(a, b);
            yield return result;
        }
        
        // If odd number of items, return sum of last vector
        if (items.Count % 2 == 1)
        {
            var lastVector = items[items.Count - 1];
            var sum = lastVector.Sum();
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
