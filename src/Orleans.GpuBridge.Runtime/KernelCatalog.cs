using System; using System.Collections.Generic; using System.Threading.Tasks; using Microsoft.Extensions.Options; using Orleans.GpuBridge.Abstractions;
namespace Orleans.GpuBridge.Runtime;
public sealed class KernelCatalogOptions{ public List<KernelDescriptor> Descriptors{get;}=new(); }
public sealed class KernelDescriptor
{
    public KernelId Id{get;private set;}=new("unset"); public Type InType{get;private set;}=typeof(object); public Type OutType{get;private set;}=typeof(object);
    public Func<IServiceProvider,object>? Factory{get;private set;}
    public static KernelDescriptor Build(Action<KernelDescriptor> cfg){var d=new KernelDescriptor(); cfg(d); return d;}
    public KernelDescriptor Id(string id){Id=new(id); return this;} public KernelDescriptor In<T>(){InType=typeof(T); return this;} public KernelDescriptor Out<T>(){OutType=typeof(T); return this;}
    public KernelDescriptor FromFactory(Func<IServiceProvider,object> f){Factory=f; return this;}
}
public sealed class KernelCatalog
{
    private readonly Dictionary<string,Func<IServiceProvider,object>> _factories=new();
    public KernelCatalog(IOptions<KernelCatalogOptions> o){foreach(var d in o.Value.Descriptors) if(d.Factory!=null) _factories[d.Id.Value]=d.Factory;}
    public Task<IGpuKernel<TIn,TOut>> ResolveAsync<TIn,TOut>(KernelId id, IServiceProvider sp)
        => Task.FromResult((IGpuKernel<TIn,TOut>)(_factories.TryGetValue(id.Value,out var f)? f(sp) : new CpuPassthroughKernel<TIn,TOut>()));
}
internal sealed class CpuPassthroughKernel<TIn,TOut>:IGpuKernel<TIn,TOut>
{
    public ValueTask<KernelHandle> SubmitBatchAsync(System.Collections.Generic.IReadOnlyList<TIn> items, GpuExecutionHints? hints=null, System.Threading.CancellationToken ct=default)
        => new(new KernelHandle(Guid.NewGuid().ToString("n")));
    public async System.Collections.Generic.IAsyncEnumerable<TOut> ReadResultsAsync(KernelHandle h, System.Threading.CancellationToken ct=default){ yield break; }
}
public sealed class CpuVectorAddKernel:IGpuKernel<float[],float>
{
    public static float[] Execute(float[] a,float[] b){ var n=Math.Min(a.Length,b.Length); var r=new float[n]; for(int i=0;i<n;i++) r[i]=a[i]+b[i]; return r; }
    public ValueTask<KernelHandle> SubmitBatchAsync(System.Collections.Generic.IReadOnlyList<float[]> items, GpuExecutionHints? hints=null, System.Threading.CancellationToken ct=default)
        => new(new KernelHandle(Guid.NewGuid().ToString("n")));
    public async System.Collections.Generic.IAsyncEnumerable<float> ReadResultsAsync(KernelHandle h, System.Threading.CancellationToken ct=default){ yield break; }
}
