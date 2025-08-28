using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;

namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Options;

/// <summary>
/// Options for memory allocation
/// </summary>
public sealed record MemoryAllocationOptions(
    MemoryType Type = MemoryType.Device,
    bool ZeroInitialize = false,
    bool EnablePeerAccess = false,
    int Alignment = 256,
    IComputeDevice? PreferredDevice = null);