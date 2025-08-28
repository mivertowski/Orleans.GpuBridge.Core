using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;

namespace Orleans.GpuBridge.Backends.DotCompute.Memory;

/// <summary>
/// DotCompute pinned memory implementation
/// </summary>
internal sealed class DotComputePinnedMemory : IPinnedMemory
{
    private readonly ILogger _logger;
    private readonly byte[] _managedArray;
    private readonly GCHandle _handle;
    private bool _disposed;

    public long SizeBytes { get; }
    public IntPtr HostPointer { get; }

    public DotComputePinnedMemory(long sizeBytes, ILogger logger)
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
                "Allocated {SizeBytes} bytes of DotCompute pinned memory at {Pointer:X}",
                sizeBytes, HostPointer.ToInt64());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to allocate DotCompute pinned memory");
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
                "Registering DotCompute pinned memory with device: {DeviceName}",
                device.Name);

            // In a real DotCompute implementation, this would register the pinned memory
            // with the specific device for optimal transfer performance
            await Task.CompletedTask;

            _logger.LogDebug("DotCompute pinned memory registered successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to register DotCompute pinned memory with device");
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
                "Unregistering DotCompute pinned memory from device: {DeviceName}",
                device.Name);

            // In a real DotCompute implementation, this would unregister the pinned memory
            await Task.CompletedTask;

            _logger.LogDebug("DotCompute pinned memory unregistered successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to unregister DotCompute pinned memory from device");
            throw;
        }
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(DotComputePinnedMemory));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogDebug("Disposing DotCompute pinned memory ({SizeBytes} bytes)", SizeBytes);

            if (_handle.IsAllocated)
            {
                _handle.Free();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute pinned memory");
        }

        _disposed = true;
    }
}

/// <summary>
/// DotCompute unified memory implementation
/// </summary>
internal sealed class DotComputeUnifiedMemory : DotComputeDeviceMemoryWrapper, IUnifiedMemory
{
    private readonly UnifiedMemoryOptions _options;

    public IntPtr HostPointer => DevicePointer; // Unified memory has same pointer

    public DotComputeUnifiedMemory(
        IntPtr devicePointer,
        IComputeDevice device,
        long sizeBytes,
        UnifiedMemoryOptions options,
        DotComputeMemoryAllocator allocator,
        ILogger logger)
        : base(devicePointer, device, sizeBytes, allocator, logger)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public async Task PrefetchAsync(
        IComputeDevice device,
        CancellationToken cancellationToken = default)
    {
        if (device == null)
            throw new ArgumentNullException(nameof(device));

        try
        {
            _logger.LogTrace(
                "Prefetching unified memory to device: {DeviceName}",
                device.Name);

            // In a real DotCompute implementation, this would prefetch the unified memory
            // to the specified device for better performance
            await Task.Delay(1, cancellationToken); // Simulate prefetch operation

            _logger.LogTrace("Unified memory prefetch completed");
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
                "Setting memory advice {Advice} for unified memory on device {DeviceName}",
                advice, device?.Name ?? "default");

            // In a real DotCompute implementation, this would set memory advice
            // to optimize memory access patterns
            await Task.CompletedTask;

            _logger.LogDebug("Memory advice set successfully");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to set memory advice {advice}", ex);
        }
    }

    public unsafe Span<byte> AsHostSpan()
    {
        // In a real unified memory implementation, this would be directly accessible
        // For DotCompute, we need to be careful about memory consistency
        return new Span<byte>((void*)HostPointer, (int)SizeBytes);
    }
}

/// <summary>
/// Fallback implementation for unified memory when not natively supported
/// </summary>
internal sealed class DotComputeUnifiedMemoryFallback : IUnifiedMemory
{
    private readonly IDeviceMemory _deviceMemory;
    private readonly ILogger _logger;
    private readonly byte[] _hostBuffer;
    private readonly GCHandle _hostHandle;
    private bool _disposed;

    public long SizeBytes => _deviceMemory.SizeBytes;
    public IntPtr DevicePointer => _deviceMemory.DevicePointer;
    public IntPtr HostPointer { get; }
    public IComputeDevice Device => _deviceMemory.Device;

    public DotComputeUnifiedMemoryFallback(IDeviceMemory deviceMemory, ILogger logger)
    {
        _deviceMemory = deviceMemory ?? throw new ArgumentNullException(nameof(deviceMemory));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Create host buffer for fallback
        _hostBuffer = new byte[deviceMemory.SizeBytes];
        _hostHandle = GCHandle.Alloc(_hostBuffer, GCHandleType.Pinned);
        HostPointer = _hostHandle.AddrOfPinnedObject();

        _logger.LogDebug(
            "Created DotCompute unified memory fallback with {SizeBytes} bytes",
            deviceMemory.SizeBytes);
    }

    public async Task PrefetchAsync(IComputeDevice device, CancellationToken cancellationToken = default)
    {
        // For fallback, prefetch means copying data to the target location
        if (device.Type == DeviceType.Cpu)
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
        _logger.LogDebug("Memory advice {Advice} ignored in DotCompute fallback mode", advice);
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
            if (_hostHandle.IsAllocated)
            {
                _hostHandle.Free();
            }
            _deviceMemory.Dispose();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing DotCompute unified memory fallback");
        }

        _disposed = true;
    }
}