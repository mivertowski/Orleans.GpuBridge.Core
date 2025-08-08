using System.Collections.Generic; using System.Threading.Tasks; using Microsoft.Extensions.DependencyInjection; using Orleans;
using Orleans.GpuBridge.Abstractions; using Orleans.GpuBridge.Runtime;
namespace Orleans.GpuBridge.Grains;
public interface IGpuBatchGrain<TIn,TOut>:IGrainWithStringKey{ Task<IReadOnlyList<TOut>> ExecuteAsync(IReadOnlyList<TIn> batch, GpuExecutionHints? hints=null); }
public sealed class GpuBatchGrain<TIn,TOut>:Grain, IGpuBatchGrain<TIn,TOut>
{
    private KernelCatalog _catalog=default!; private IGpuKernel<TIn,TOut> _kernel=default!; private KernelId _kernelId;
    public override async Task OnActivateAsync(System.Threading.CancellationToken ct){ _kernelId=KernelId.Parse(this.GetPrimaryKeyString()); _catalog=ServiceProvider.GetRequiredService<KernelCatalog>(); _kernel=await _catalog.ResolveAsync<TIn,TOut>(_kernelId, ServiceProvider); }
    public async Task<IReadOnlyList<TOut>> ExecuteAsync(IReadOnlyList<TIn> batch, GpuExecutionHints? hints=null)
    { var h=await _kernel.SubmitBatchAsync(batch,hints); var res=new List<TOut>(); await foreach(var x in _kernel.ReadResultsAsync(h)) res.Add(x); return res; }
}
