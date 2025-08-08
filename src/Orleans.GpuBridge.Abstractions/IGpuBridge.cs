using System.Threading.Tasks;
namespace Orleans.GpuBridge.Abstractions;
public interface IGpuBridge { ValueTask<GpuBridgeInfo> GetInfoAsync(); ValueTask<IGpuKernel<TIn,TOut>> GetKernelAsync<TIn,TOut>(KernelId id); }
public sealed record GpuBridgeInfo(string Runtime, int DeviceCount);
