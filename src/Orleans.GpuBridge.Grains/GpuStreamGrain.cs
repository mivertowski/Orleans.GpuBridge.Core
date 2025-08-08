using System.Threading.Tasks; using Orleans;
namespace Orleans.GpuBridge.Grains; public interface IGpuStreamGrain<TIn,TOut>:IGrainWithStringKey{ Task PushAsync(TIn item); }
public sealed class GpuStreamGrain<TIn,TOut>:Grain,IGpuStreamGrain<TIn,TOut>{ public Task PushAsync(TIn item)=>Task.CompletedTask; }
