using System;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Backends.ILGPU.DeviceManagement;

namespace Orleans.GpuBridge.Backends.ILGPU.Memory;

/// <summary>
/// ILGPU device memory wrapper (untyped)
/// </summary>
internal sealed class ILGPUDeviceMemoryWrapper : IDeviceMemory
{
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _memoryBuffer;
    private readonly ILGPUComputeDevice _device;
    private readonly ILGPUMemoryAllocator _allocator;
    private readonly ILogger _logger;
    private bool _disposed;

    public long SizeBytes { get; }
    public IntPtr DevicePointer => _memoryBuffer.NativePtr;
    public IComputeDevice Device => _device;
    
    /// <summary>
    /// Gets the underlying ILGPU memory buffer for internal operations
    /// </summary>
    internal MemoryBuffer1D<byte, Stride1D.Dense> GetBuffer() => _memoryBuffer;

    public ILGPUDeviceMemoryWrapper(
        MemoryBuffer1D<byte, Stride1D.Dense> memoryBuffer,
        ILGPUComputeDevice device,
        long sizeBytes,
        ILGPUMemoryAllocator allocator,
        ILogger logger)
    {
        _memoryBuffer = memoryBuffer ?? throw new ArgumentNullException(nameof(memoryBuffer));
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        SizeBytes = sizeBytes;
    }

    public async Task CopyFromHostAsync(
        IntPtr hostPointer,
        long offsetBytes,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (hostPointer == IntPtr.Zero)
            throw new ArgumentException("Host pointer cannot be zero", nameof(hostPointer));

        if (offsetBytes < 0 || offsetBytes + sizeBytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(offsetBytes), "Copy range exceeds buffer bounds");

        try
        {
            _logger.LogTrace(
                "Copying {SizeBytes} bytes from host {HostPtr:X} to device offset {OffsetBytes}",
                sizeBytes, hostPointer.ToInt64(), offsetBytes);

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create a view of the target region
            var targetView = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);

            unsafe
            {
                // Copy from host pointer to device buffer
                var hostSpan = new ReadOnlySpan<byte>((void*)hostPointer, (int)sizeBytes);
                targetView.CopyFromCPU(stream, hostSpan);
            }

            // Synchronize to ensure copy completes
            stream.Synchronize();

            _logger.LogTrace("Host to device copy completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to copy from host to device");
            throw;
        }
    }

    public async Task CopyToHostAsync(
        IntPtr hostPointer,
        long offsetBytes,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (hostPointer == IntPtr.Zero)
            throw new ArgumentException("Host pointer cannot be zero", nameof(hostPointer));

        if (offsetBytes < 0 || offsetBytes + sizeBytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(offsetBytes), "Copy range exceeds buffer bounds");

        try
        {
            _logger.LogTrace(
                "Copying {SizeBytes} bytes from device offset {OffsetBytes} to host {HostPtr:X}",
                sizeBytes, offsetBytes, hostPointer.ToInt64());

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create a view of the source region
            var sourceView = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);

            unsafe
            {
                // Copy from device buffer to host pointer
                var hostSpan = new Span<byte>((void*)hostPointer, (int)sizeBytes);
                sourceView.CopyToCPU(stream, hostSpan);
            }

            // Synchronize to ensure copy completes
            stream.Synchronize();

            _logger.LogTrace("Device to host copy completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to copy from device to host");
            throw;
        }
    }

    public async Task CopyFromAsync(
        IDeviceMemory source,
        long sourceOffset,
        long destinationOffset,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (source is not ILGPUDeviceMemoryWrapper ilgpuSource)
        {
            throw new ArgumentException("Source must be ILGPU device memory", nameof(source));
        }

        if (sourceOffset < 0 || sourceOffset + sizeBytes > source.SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(sourceOffset), "Source copy range exceeds buffer bounds");

        if (destinationOffset < 0 || destinationOffset + sizeBytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(destinationOffset), "Destination copy range exceeds buffer bounds");

        try
        {
            _logger.LogTrace(
                "Copying {SizeBytes} bytes from device buffer to device buffer (offset {DestOffset})",
                sizeBytes, destinationOffset);

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create views of source and destination regions
            var sourceView = ilgpuSource._memoryBuffer.View.SubView((int)sourceOffset, (int)sizeBytes);
            var destView = _memoryBuffer.View.SubView((int)destinationOffset, (int)sizeBytes);

            // Copy between device buffers
            destView.CopyFrom(stream, sourceView);

            // Synchronize to ensure copy completes
            stream.Synchronize();

            _logger.LogTrace("Device to device copy completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to copy between device buffers");
            throw;
        }
    }

    public async Task FillAsync(
        byte value,
        long offsetBytes,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (offsetBytes < 0 || offsetBytes + sizeBytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(offsetBytes), "Fill range exceeds buffer bounds");

        try
        {
            _logger.LogTrace(
                "Filling {SizeBytes} bytes with value {Value:X2} at offset {OffsetBytes}",
                sizeBytes, value, offsetBytes);

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create a view of the target region
            var targetView = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);

            // Fill with the specified byte value
            targetView.MemSetToZero(stream); // ILGPU only supports zero fill directly

            if (value != 0)
            {
                // For non-zero values, we need to copy from a host buffer TODO
                var fillBuffer = new byte[sizeBytes];
                Array.Fill(fillBuffer, value);
                
                unsafe
                {
                    fixed (byte* bufferPtr = fillBuffer)
                    {
                        var hostSpan = new ReadOnlySpan<byte>(bufferPtr, (int)sizeBytes);
                        targetView.CopyFromCPU(stream, hostSpan);
                    }
                }
            }

            // Synchronize to ensure fill completes
            stream.Synchronize();

            _logger.LogTrace("Memory fill completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to fill device memory");
            throw;
        }
    }

    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes)
    {
        EnsureNotDisposed();

        if (offsetBytes < 0 || offsetBytes + sizeBytes > SizeBytes)
            throw new ArgumentOutOfRangeException(nameof(offsetBytes), "View range exceeds buffer bounds");

        // Create a sub-view of the memory buffer
        var subBuffer = _memoryBuffer.View.SubView((int)offsetBytes, (int)sizeBytes);
        
        return new ILGPUDeviceMemoryWrapper(
            subBuffer,
            _device,
            sizeBytes,
            _allocator,
            _logger);
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ILGPUDeviceMemoryWrapper));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogTrace("Disposing ILGPU device memory ({SizeBytes} bytes)", SizeBytes);

            if (!_memoryBuffer.IsDisposed)
            {
                _memoryBuffer.Dispose();
            }

            // Notify allocator
            _allocator.OnAllocationDisposed(DevicePointer, SizeBytes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU device memory");
        }

        _disposed = true;
    }
}

/// <summary>
/// ILGPU device memory wrapper (typed)
/// </summary>
internal sealed class ILGPUDeviceMemoryWrapper<T> : IDeviceMemory<T> where T : unmanaged
{
    private readonly MemoryBuffer1D<T, Stride1D.Dense> _memoryBuffer;
    private readonly ILGPUComputeDevice _device;
    private readonly ILGPUMemoryAllocator _allocator;
    private readonly ILogger _logger;
    private bool _disposed;

    public int Length { get; }
    public long SizeBytes => Length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
    public IntPtr DevicePointer => _memoryBuffer.NativePtr;
    public IComputeDevice Device => _device;

    public ILGPUDeviceMemoryWrapper(
        MemoryBuffer1D<T, Stride1D.Dense> memoryBuffer,
        ILGPUComputeDevice device,
        int length,
        ILGPUMemoryAllocator allocator,
        ILogger logger)
    {
        _memoryBuffer = memoryBuffer ?? throw new ArgumentNullException(nameof(memoryBuffer));
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        Length = length;
    }

    public unsafe Span<T> AsSpan()
    {
        EnsureNotDisposed();
        
        // ILGPU doesn't directly expose device memory as a span
        // This would only work for unified/managed memory
        throw new NotSupportedException(
            "ILGPU device memory cannot be accessed as a span. " +
            "Use CopyToHostAsync to transfer data to host memory first.");
    }

    public async Task CopyFromHostAsync(
        T[] source,
        int sourceOffset,
        int destinationOffset,
        int count,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (source == null)
            throw new ArgumentNullException(nameof(source));

        if (sourceOffset < 0 || sourceOffset + count > source.Length)
            throw new ArgumentOutOfRangeException(nameof(sourceOffset));

        if (destinationOffset < 0 || destinationOffset + count > Length)
            throw new ArgumentOutOfRangeException(nameof(destinationOffset));

        try
        {
            _logger.LogTrace(
                "Copying {Count} elements of {TypeName} from host array to device offset {DestOffset}",
                count, typeof(T).Name, destinationOffset);

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create a view of the target region
            var targetView = _memoryBuffer.View.SubView(destinationOffset, count);

            // Copy from host array
            var sourceSpan = source.AsSpan(sourceOffset, count);
            targetView.CopyFromCPU(stream, sourceSpan);

            // Synchronize to ensure copy completes
            stream.Synchronize();

            _logger.LogTrace("Host array to device copy completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to copy from host array to device");
            throw;
        }
    }

    public async Task CopyToHostAsync(
        T[] destination,
        int sourceOffset,
        int destinationOffset,
        int count,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (destination == null)
            throw new ArgumentNullException(nameof(destination));

        if (sourceOffset < 0 || sourceOffset + count > Length)
            throw new ArgumentOutOfRangeException(nameof(sourceOffset));

        if (destinationOffset < 0 || destinationOffset + count > destination.Length)
            throw new ArgumentOutOfRangeException(nameof(destinationOffset));

        try
        {
            _logger.LogTrace(
                "Copying {Count} elements of {TypeName} from device offset {SrcOffset} to host array",
                count, typeof(T).Name, sourceOffset);

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create a view of the source region
            var sourceView = _memoryBuffer.View.SubView(sourceOffset, count);

            // Copy to host array
            var destSpan = destination.AsSpan(destinationOffset, count);
            sourceView.CopyToCPU(stream, destSpan);

            // Synchronize to ensure copy completes
            stream.Synchronize();

            _logger.LogTrace("Device to host array copy completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to copy from device to host array");
            throw;
        }
    }

    public async Task FillAsync(
        T value,
        int offset,
        int count,
        CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (offset < 0 || offset + count > Length)
            throw new ArgumentOutOfRangeException(nameof(offset), "Fill range exceeds buffer bounds");

        try
        {
            _logger.LogTrace(
                "Filling {Count} elements of {TypeName} with value at offset {Offset}",
                count, typeof(T).Name, offset);

            var accelerator = _device.Accelerator;
            var stream = accelerator.DefaultStream;

            // Create a view of the target region
            var targetView = _memoryBuffer.View.SubView(offset, count);

            // Check if we can use the optimized zero fill
            if (EqualityComparer<T>.Default.Equals(value, default(T)))
            {
                targetView.MemSetToZero(stream);
            }
            else
            {
                // For non-zero values, create a host buffer and copy
                var fillBuffer = new T[count];
                Array.Fill(fillBuffer, value);
                targetView.CopyFromCPU(stream, fillBuffer.AsSpan());
            }

            // Synchronize to ensure fill completes
            stream.Synchronize();

            _logger.LogTrace("Typed memory fill completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to fill typed device memory");
            throw;
        }
    }

    // IDeviceMemory interface implementations (delegated to byte-level operations)
    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default)
    {
        var elementSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        var elementOffset = (int)(offsetBytes / elementSize);
        var elementCount = (int)(sizeBytes / elementSize);

        unsafe
        {
            var typedPtr = (T*)hostPointer.ToPointer();
            var sourceSpan = new ReadOnlySpan<T>(typedPtr, elementCount);
            return CopyFromHostAsync(sourceSpan.ToArray(), 0, elementOffset, elementCount, cancellationToken);
        }
    }

    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default)
    {
        var elementSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        var elementOffset = (int)(offsetBytes / elementSize);
        var elementCount = (int)(sizeBytes / elementSize);

        var tempArray = new T[elementCount];
        return CopyToHostAsync(tempArray, elementOffset, 0, elementCount, cancellationToken)
            .ContinueWith(_ =>
            {
                unsafe
                {
                    var typedPtr = (T*)hostPointer.ToPointer();
                    var destSpan = new Span<T>(typedPtr, elementCount);
                    tempArray.AsSpan().CopyTo(destSpan);
                }
            }, cancellationToken);
    }

    public async Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default)
    {
        if (source is not ILGPUDeviceMemoryWrapper<T> typedSource)
        {
            throw new ArgumentException($"Source must be ILGPU device memory of type {typeof(T).Name}", nameof(source));
        }

        var elementSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        var elementSourceOffset = (int)(sourceOffset / elementSize);
        var elementDestOffset = (int)(destinationOffset / elementSize);
        var elementCount = (int)(sizeBytes / elementSize);

        var accelerator = _device.Accelerator;
        var stream = accelerator.DefaultStream;

        var sourceView = typedSource._memoryBuffer.View.SubView(elementSourceOffset, elementCount);
        var destView = _memoryBuffer.View.SubView(elementDestOffset, elementCount);

        destView.CopyFrom(stream, sourceView);
        stream.Synchronize();
    }

    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default)
    {
        // For typed buffers, byte-level fill is complex
        throw new NotSupportedException("Use FillAsync(T value, int offset, int count) for typed buffers");
    }

    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes)
    {
        var elementSize = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        var elementOffset = (int)(offsetBytes / elementSize);
        var elementCount = (int)(sizeBytes / elementSize);

        var subBuffer = _memoryBuffer.View.SubView(elementOffset, elementCount);
        
        return new ILGPUDeviceMemoryWrapper<T>(
            subBuffer,
            _device,
            elementCount,
            _allocator,
            _logger);
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ILGPUDeviceMemoryWrapper<T>));
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _logger.LogTrace(
                "Disposing ILGPU typed device memory ({Length} elements of {TypeName})",
                Length, typeof(T).Name);

            if (!_memoryBuffer.IsDisposed)
            {
                _memoryBuffer.Dispose();
            }

            // Notify allocator
            _allocator.OnAllocationDisposed(DevicePointer, SizeBytes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing ILGPU typed device memory");
        }

        _disposed = true;
    }
}