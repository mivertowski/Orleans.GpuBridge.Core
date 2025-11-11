using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU device memory for fallback provider
/// </summary>
internal sealed class CpuMemory : IDeviceMemory
{
    public long SizeBytes { get; }
    public IntPtr DevicePointer { get; }
    public IComputeDevice Device => new CpuDevice();

    public CpuMemory(long sizeBytes)
    {
        SizeBytes = sizeBytes;
        DevicePointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)sizeBytes);
    }

    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) => new CpuMemory(sizeBytes);

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(DevicePointer);
    }
}

/// <summary>
/// CPU typed device memory for fallback provider
/// </summary>
internal sealed class CpuMemory<T> : IDeviceMemory<T> where T : unmanaged
{
    public int Length { get; }
    public long SizeBytes { get; }
    public IntPtr DevicePointer { get; }
    public IComputeDevice Device { get; }

    public CpuMemory(int elementCount)
    {
        Length = elementCount;
        unsafe { SizeBytes = elementCount * sizeof(T); }
        DevicePointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)SizeBytes);
        Device = new CpuDevice();
    }

    public unsafe Span<T> AsSpan()
    {
        return new Span<T>((void*)DevicePointer, Length);
    }

    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes)
    {
        unsafe { return new CpuMemory<T>((int)(sizeBytes / sizeof(T))); }
    }

    public Task CopyFromHostAsync(T[] source, int sourceOffset, int destinationOffset, int count, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(T[] destination, int sourceOffset, int destinationOffset, int count, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(T value, int offset, int count, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(DevicePointer);
    }
}

/// <summary>
/// CPU pinned memory for fallback provider
/// </summary>
internal sealed class CpuPinnedMemory : IPinnedMemory
{
    public long SizeBytes { get; }
    public IntPtr HostPointer { get; }

    public CpuPinnedMemory(long sizeBytes)
    {
        SizeBytes = sizeBytes;
        HostPointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)sizeBytes);
    }

    public unsafe Span<byte> AsSpan()
    {
        return new Span<byte>((void*)HostPointer, (int)SizeBytes);
    }

    public Task RegisterWithDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task UnregisterFromDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(HostPointer);
    }
}

/// <summary>
/// CPU unified memory for fallback provider
/// </summary>
internal sealed class CpuUnifiedMemory : IUnifiedMemory
{
    public long SizeBytes { get; }
    public IntPtr DevicePointer { get; }
    public IComputeDevice Device { get; }
    public IntPtr HostPointer => DevicePointer;

    public CpuUnifiedMemory(long sizeBytes)
    {
        SizeBytes = sizeBytes;
        DevicePointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)sizeBytes);
        Device = new CpuDevice();
    }

    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) => new CpuUnifiedMemory(sizeBytes);

    public Task PrefetchAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task AdviseAsync(MemoryAdvice advice, IComputeDevice? device = null, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public unsafe Span<byte> AsHostSpan()
    {
        return new Span<byte>((void*)HostPointer, (int)SizeBytes);
    }

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(DevicePointer);
    }
}
