using System.Collections.Generic; using System.Linq; using System.Threading.Tasks; using Orleans; using Orleans.GpuBridge.Abstractions; using Orleans.GpuBridge.Grains;
namespace Orleans.GpuBridge.BridgeFX;
public sealed class GpuPipeline<TIn,TOut>
{
    private readonly IGrainFactory _grains; private readonly string _kernelId; private int _batchSize=8192; private GpuExecutionHints _hints=new();
    private GpuPipeline(IGrainFactory grains,string kernelId){_grains=grains;_kernelId=kernelId;}
    public static GpuPipeline<TIn,TOut> For(IGrainFactory grains,string kernelId)=>new(grains,kernelId);
    public GpuPipeline<TIn,TOut> WithBatchSize(int size){_batchSize=size; return this;} public GpuPipeline<TIn,TOut> WithHints(GpuExecutionHints h){_hints=h; return this;}
    public async Task<List<TOut>> ExecuteAsync(IEnumerable<TIn> inputs)
    {
        var list=inputs.ToList(); var grain=_grains.GetGrain<IGpuBatchGrain<TIn,TOut>>(_kernelId); var results=new List<TOut>(list.Count);
        for(int i=0;i<list.Count;i+=_batchSize){ var chunk=list.GetRange(i,System.Math.Min(_batchSize,list.Count-i)); var r=await grain.ExecuteAsync(chunk,_hints); results.AddRange(r); }
        return results;
    }
}
