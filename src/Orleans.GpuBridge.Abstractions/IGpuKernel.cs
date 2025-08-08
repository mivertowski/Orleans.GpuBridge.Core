using System.Collections.Generic; using System.Threading; using System.Threading.Tasks;
namespace Orleans.GpuBridge.Abstractions;
public interface IGpuKernel<TIn,TOut>
{
    ValueTask<KernelHandle> SubmitBatchAsync(IReadOnlyList<TIn> items, GpuExecutionHints? hints=null, CancellationToken ct=default);
    IAsyncEnumerable<TOut> ReadResultsAsync(KernelHandle handle, CancellationToken ct=default);
}
public readonly record struct KernelHandle(string JobId);
