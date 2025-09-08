using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

namespace Orleans.GpuBridge.Backends.ILGPU.Memory;

/// <summary>
/// ILGPU pinned memory implementation
/// </summary>
internal sealed class ILGPUPinnedMemory : IPinnedMemory
{
    private readonly ILogger _logger;
    private readonly byte[] _managedArray;
    private readonly GCHandle _handle;
    private bool _disposed;

    public long SizeBytes { get; }
    public IntPtr HostPointer { get; }

    public ILGPUPinnedMemory(long sizeBytes, ILogger logger)
    {
        if (sizeBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizeBytes), "Size must be greater than zero");

        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        SizeBytes = sizeBytes;

        try
        {
            // Allocate managed array
            _managedArray = new byte[sizeBytes];

            // Pin the array in memory
            _handle = GCHandle.Alloc(_managedArray, GCHandleType.Pinned);
            HostPointer = _handle.AddrOfPinnedObject();

            _logger.LogDebug(
                "Allocated {SizeBytes} bytes of pinned memory at {Pointer:X}",
                sizeBytes, HostPointer.ToInt64());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate pinned memory");
            Dispose();
            throw;
        }
    }

    public unsafe Span<byte> AsSpan()
    {
        EnsureNotDisposed();
        return new Span<byte>((void*)HostPointer, (int)SizeBytes);
    }

    public async Task RegisterWithDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        try
        {
            _logger.LogDebug(
                "Registering pinned memory with device: {DeviceName}",
                device.Name);

            // ILGPU doesn't have explicit memory registration like CUDA
            // Pinned memory is automatically optimal for transfers
            await Task.CompletedTask;

            _logger.LogDebug("Pinned memory registered successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register pinned memory with device");
            throw;
        }
    }

    public async Task UnregisterFromDeviceAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (device == null)
            throw new ArgumentNullException(nameof(device));

        try
        {
            _logger.LogDebug(
                "Unregistering pinned memory from device: {DeviceName}",
                device.Name);

            // ILGPU doesn't require explicit memory unregistration
            await Task.CompletedTask;

            _logger.LogDebug("Pinned memory unregistered successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to unregister pinned memory from device");
            throw;
        }
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ILGPUPinnedMemory));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogDebug("Disposing pinned memory ({SizeBytes} bytes)", SizeBytes);

            if (_handle.IsAllocated)
            {
                _handle.Free();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing pinned memory");
        }

        _disposed = true;
    }
}

/// <summary>
/// ILGPU unified memory implementation
/// </summary>
internal sealed class ILGPUUnifiedMemory : IUnifiedMemory
{
    private readonly UnifiedMemoryOptions _options;

    public IntPtr HostPointer => DevicePointer; // Unified memory has same pointer

    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _memoryBuffer;
    private readonly ILGPUComputeDevice _device;
    private readonly ILogger _logger;
    private bool _disposed;

    public long SizeBytes { get; }
    public IntPtr DevicePointer => _memoryBuffer.NativePtr;
    public IComputeDevice Device => _device;

    public ILGPUUnifiedMemory(
        MemoryBuffer1D<byte, Stride1D.Dense> memoryBuffer,
        ILGPUComputeDevice device,
        long sizeBytes,
        UnifiedMemoryOptions options,
        ILogger logger)
    {
        _memoryBuffer = memoryBuffer ?? throw new ArgumentNullException(nameof(memoryBuffer));
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        SizeBytes = sizeBytes;
    }

    public async Task PrefetchAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        if (device == null)
            throw new ArgumentNullException(nameof(device));

        try
        {
            // ILGPU doesn't expose prefetch directly, but we can simulate
            // by ensuring data is available on the target device
            if (device is ILGPUComputeDevice ilgpuDevice)
            {
                ilgpuDevice.Accelerator.Synchronize();
            }

            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to prefetch unified memory to device {device.Name}", ex);
        }
    }

    public async Task AdviseAsync(
        MemoryAdvice advice,
        IComputeDevice? device = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogDebug(
                "Memory advice {Advice} for unified memory on device {DeviceName}",
                advice, device?.Name ?? "default");

            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to set memory advice {advice}", ex);
        }
    }

    public unsafe Span<byte> AsHostSpan()
    {
        // For true unified memory, this would be accessible
        // For ILGPU, we need to be careful as this might not be directly accessible
        throw new NotSupportedException(
            "ILGPU unified memory may not be directly accessible from host. " +
            "Use CopyToHostAsync to transfer data to host memory first.");
    }

    // IDeviceMemory interface implementations
    public async Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default)
    {
        if (hostPointer == IntPtr.Zero)
            throw new ArgumentException("Host pointer cannot be zero", nameof(hostPointer));

        var accelerator = _device.Accelerator;
        var stream = accelerator.DefaultStream;
        var targetView = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);

        unsafe
        {
            var hostSpan = new ReadOnlySpan<byte>((void*)hostPointer, (int)sizeBytes);
            targetView.CopyFromCPU(stream, hostSpan);
        }
        stream.Synchronize();
    }

    public async Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default)
    {
        if (hostPointer == IntPtr.Zero)
            throw new ArgumentException("Host pointer cannot be zero", nameof(hostPointer));

        var accelerator = _device.Accelerator;
        var stream = accelerator.DefaultStream;
        var sourceView = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);

        unsafe
        {
            var hostSpan = new Span<byte>((void*)hostPointer, (int)sizeBytes);
            sourceView.CopyToCPU(stream, hostSpan);
        }
        stream.Synchronize();
    }

    public async Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
            
        if (sizeBytes <= 0)
            return;
            
        var accelerator = _device.Accelerator;
        var stream = accelerator.DefaultStream;
        
        // Handle different source types
        if (source is ILGPUUnifiedMemory unifiedSource)
        {
            // Unified to Unified copy
            var sourceView = unifiedSource._memoryBuffer.View.SubView((int)sourceOffset, (int)sizeBytes);
            var destView = _memoryBuffer.View.SubView((int)destinationOffset, (int)sizeBytes);
            sourceView.CopyTo(stream, destView);
        }
        else if (source is ILGPUDeviceMemoryWrapper deviceSource)
        {
            // Device to Unified copy
            var sourceView = deviceSource.GetBuffer().View.SubView((int)sourceOffset, (int)sizeBytes);
            var destView = _memoryBuffer.View.SubView((int)destinationOffset, (int)sizeBytes);
            sourceView.CopyTo(stream, destView);
        }
        else
        {
            // Fallback: Copy through host memory
            var tempBuffer = new byte[sizeBytes];
            unsafe
            {
                fixed (byte* ptr = tempBuffer)
                {
                    await source.CopyToHostAsync(new IntPtr(ptr), sourceOffset, sizeBytes, cancellationToken);
                    await CopyFromHostAsync(new IntPtr(ptr), destinationOffset, sizeBytes, cancellationToken);
                }
            }
            return;
        }
        
        stream.Synchronize();
    }

    public async Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default)
    {
        var accelerator = _device.Accelerator;
        var stream = accelerator.DefaultStream;
        var targetView = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);

        if (value == 0)
        {
            targetView.MemSetToZero(stream);
        }
        else
        {
            var fillBuffer = new byte[sizeBytes];
            Array.Fill(fillBuffer, value);
            targetView.CopyFromCPU(stream, fillBuffer.AsSpan());
        }
        stream.Synchronize();
    }

    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes)
    {
        var subBuffer = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);
        return new ILGPUUnifiedMemory(subBuffer, _device, sizeBytes, _options, _logger);
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        try
        {
            if (!_memoryBuffer.IsDisposed)
                _memoryBuffer.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing unified memory");
        }
        _disposed = true;
    }
}

/// <summary>
/// Fallback implementation for unified memory when not available
/// </summary>
internal sealed class ILGPUUnifiedMemoryFallback : IUnifiedMemory
{
    private readonly IDeviceMemory _deviceMemory;
    private readonly ILogger _logger;
    private readonly byte[] _hostBuffer;
    private bool _disposed;

    public long SizeBytes => _deviceMemory.SizeBytes;
    public IntPtr DevicePointer => _deviceMemory.DevicePointer;
    public IntPtr HostPointer { get; }
    public IComputeDevice Device => _deviceMemory.Device;

    public ILGPUUnifiedMemoryFallback(IDeviceMemory deviceMemory, ILogger logger)
    {
        _deviceMemory = deviceMemory ?? throw new ArgumentNullException(nameof(deviceMemory));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Create host buffer for fallback
        _hostBuffer = new byte[deviceMemory.SizeBytes];
        unsafe
        {
            fixed (byte* ptr = _hostBuffer)
            {
                HostPointer = new IntPtr(ptr);
            }
        }

        _logger.LogDebug(
            "Created unified memory fallback with {SizeBytes} bytes",
            deviceMemory.SizeBytes);
    }

    public async Task PrefetchAsync(IComputeDevice device, CancellationToken cancellationToken = default)
    {
        // For fallback, prefetch means copying data to the target location
        if (device.Type == Orleans.GpuBridge.Abstractions.Enums.DeviceType.CPU)
        {
            // Copy to host buffer
            await CopyToHostAsync(HostPointer, 0, SizeBytes, cancellationToken);
        }
        else
        {
            // Data is already on device
            await Task.CompletedTask;
        }
    }

    public Task AdviseAsync(MemoryAdvice advice, IComputeDevice? device = null, CancellationToken cancellationToken = default)
    {
        _logger.LogDebug("Memory advice {Advice} ignored in fallback mode", advice);
        return Task.CompletedTask;
    }

    public unsafe Span<byte> AsHostSpan()
    {
        return new Span<byte>(_hostBuffer);
    }

    // Delegate all other operations to the underlying device memory
    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) =>
        _deviceMemory.CopyFromHostAsync(hostPointer, offsetBytes, sizeBytes, cancellationToken);

    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) =>
        _deviceMemory.CopyToHostAsync(hostPointer, offsetBytes, sizeBytes, cancellationToken);

    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) =>
        _deviceMemory.CopyFromAsync(source, sourceOffset, destinationOffset, sizeBytes, cancellationToken);

    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) =>
        _deviceMemory.FillAsync(value, offsetBytes, sizeBytes, cancellationToken);

    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) =>
        _deviceMemory.CreateView(offsetBytes, sizeBytes);

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _deviceMemory.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing unified memory fallback");
        }

        _disposed = true;
    }
}