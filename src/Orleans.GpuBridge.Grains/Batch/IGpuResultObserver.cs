using System;
using System.Threading.Tasks;
using Orleans;

namespace Orleans.GpuBridge.Grains.Batch;

/// <summary>
/// Observer for GPU results
/// </summary>
public interface IGpuResultObserver<T> : IGrainObserver
{
    Task OnNextAsync(T item);
    Task OnErrorAsync(Exception error);
    Task OnCompletedAsync();
}