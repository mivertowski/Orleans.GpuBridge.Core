using System.Threading.Tasks; using Orleans;
namespace Orleans.GpuBridge.Grains; public interface IGpuResidentGrain:IGrainWithStringKey{ Task PinAsync(string uri); Task UnpinAsync(); }
public sealed class GpuResidentGrain:Grain,IGpuResidentGrain{ public Task PinAsync(string uri)=>Task.CompletedTask; public Task UnpinAsync()=>Task.CompletedTask; }
