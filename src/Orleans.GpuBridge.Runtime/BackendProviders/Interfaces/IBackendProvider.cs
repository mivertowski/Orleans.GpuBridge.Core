using Orleans.GpuBridge.Runtime.BackendProviders.Enums;
using Orleans.GpuBridge.Runtime.BackendProviders.Models;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Backend provider interface
/// </summary>
public interface IBackendProvider
{
    string Name { get; }
    BackendType Type { get; }
    bool IsAvailable { get; }
    int DeviceCount { get; }
    
    bool Initialize();
    void Shutdown();
    IComputeContext CreateContext(int deviceIndex = 0);
    IReadOnlyList<DeviceInfo> GetDevices();
}