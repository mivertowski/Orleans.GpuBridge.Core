using System;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

namespace Orleans.GpuBridge.Runtime;

/// <summary>
/// CPU compute context implementation providing fallback compute capabilities
/// </summary>
public sealed class CpuComputeContext : IComputeContext
{
    private readonly ILogger _logger;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuComputeContext"/> class
    /// </summary>
    /// <param name="logger">Logger instance for diagnostics</param>
    public CpuComputeContext(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        Backend = GpuBackend.CPU;
        DeviceIndex = 0;
    }

    /// <summary>
    /// Gets the compute backend type (CPU)
    /// </summary>
    public GpuBackend Backend { get; }

    /// <summary>
    /// Gets the device index (always 0 for CPU)
    /// </summary>
    public int DeviceIndex { get; }

    /// <summary>
    /// Creates a new CPU compute buffer
    /// </summary>
    /// <typeparam name="T">The unmanaged element type</typeparam>
    /// <param name="size">Number of elements in the buffer</param>
    /// <param name="usage">Usage flags for the buffer</param>
    /// <returns>A new CPU compute buffer instance</returns>
    public IComputeBuffer<T> CreateBuffer<T>(int size, BufferUsage usage) where T : unmanaged
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeContext));
        }

        _logger.LogDebug("Creating CPU buffer: {Size} elements of type {Type} with usage {Usage}", size, typeof(T).Name, usage);
        return new CpuComputeBuffer<T>(size, usage, _logger);
    }

    /// <summary>
    /// Compiles a compute kernel for CPU execution
    /// </summary>
    /// <param name="source">Kernel source code</param>
    /// <param name="entryPoint">Entry point function name</param>
    /// <returns>A compiled CPU kernel instance</returns>
    public IComputeKernel CompileKernel(string source, string entryPoint)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeContext));
        }

        _logger.LogDebug("Compiling CPU kernel: {Source} with entry point {EntryPoint}", source, entryPoint);
        return new CpuComputeKernel(source, entryPoint, _logger);
    }

    /// <summary>
    /// Executes a CPU compute kernel
    /// </summary>
    /// <param name="kernel">The kernel to execute</param>
    /// <param name="workSize">Number of work items to execute</param>
    public void Execute(IComputeKernel kernel, int workSize)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeContext));
        }

        if (kernel is not CpuComputeKernel cpuKernel)
        {
            throw new ArgumentException("Kernel is not a CPU kernel", nameof(kernel));
        }

        _logger.LogDebug("Executing CPU kernel with work size {WorkSize}", workSize);

        // CPU kernel execution implementation
        cpuKernel.Execute(workSize);
    }

    /// <summary>
    /// Synchronizes the CPU compute context (no-op for CPU)
    /// </summary>
    public void Synchronize()
    {
        if (_disposed)
        {
            return;
        }

        _logger.LogTrace("Synchronizing CPU context");

        // CPU synchronization is typically a no-op as CPU execution is synchronous
    }

    /// <summary>
    /// Disposes the CPU compute context
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            _logger.LogInformation("Disposing CPU compute context");

            // CPU cleanup is minimal

            _disposed = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing CPU compute context");
        }
    }
}

/// <summary>
/// CPU compute buffer implementation
/// </summary>
public sealed class CpuComputeBuffer<T> : IComputeBuffer<T> where T : unmanaged
{
    private readonly ILogger _logger;
    private readonly T[] _data;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuComputeBuffer{T}"/> class
    /// </summary>
    /// <param name="size">Number of elements in the buffer</param>
    /// <param name="usage">Usage flags for the buffer</param>
    /// <param name="logger">Logger instance for diagnostics</param>
    public CpuComputeBuffer(int size, BufferUsage usage, ILogger logger)
    {
        _logger = logger;
        _data = new T[size];
        Size = size;
        Usage = usage;
    }

    /// <summary>
    /// Gets the size of the buffer in elements
    /// </summary>
    public int Size { get; }

    /// <summary>
    /// Gets the usage flags for the buffer
    /// </summary>
    public BufferUsage Usage { get; }

    /// <summary>
    /// Writes data to the buffer
    /// </summary>
    /// <param name="data">Data to write</param>
    public void Write(ReadOnlySpan<T> data)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeBuffer<T>));
        }

        if (data.Length > Size)
        {
            throw new ArgumentOutOfRangeException(nameof(data));
        }

        try
        {
            data.CopyTo(_data.AsSpan());
            _logger.LogTrace("Wrote {Count} elements to CPU buffer", data.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to write to CPU buffer");
            throw;
        }
    }

    /// <summary>
    /// Reads data from the buffer
    /// </summary>
    /// <param name="data">Span to read data into</param>
    public void Read(Span<T> data)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeBuffer<T>));
        }

        if (data.Length > Size)
        {
            throw new ArgumentOutOfRangeException(nameof(data));
        }

        try
        {
            _data.AsSpan(0, data.Length).CopyTo(data);
            _logger.LogTrace("Read {Count} elements from CPU buffer", data.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to read from CPU buffer");
            throw;
        }
    }

    /// <summary>
    /// Copies the buffer contents to another buffer
    /// </summary>
    /// <param name="destination">Destination buffer</param>
    public void CopyTo(IComputeBuffer<T> destination)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeBuffer<T>));
        }

        if (destination.Size < Size)
        {
            throw new ArgumentException("Destination buffer is too small", nameof(destination));
        }

        try
        {
            var temp = new T[Size];
            _data.AsSpan().CopyTo(temp);
            destination.Write(temp);
            _logger.LogTrace("Copied {Size} elements to destination buffer", Size);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to copy CPU buffer");
            throw;
        }
    }

    /// <summary>
    /// Disposes the buffer
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _logger.LogTrace("Disposing CPU buffer with {Size} elements", Size);
            // CPU buffer cleanup is minimal
            _disposed = true;
        }
    }
}

/// <summary>
/// CPU compute kernel implementation
/// </summary>
public sealed class CpuComputeKernel : IComputeKernel
{
    private readonly ILogger _logger;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuComputeKernel"/> class
    /// </summary>
    /// <param name="source">Kernel source code</param>
    /// <param name="name">Kernel entry point name</param>
    /// <param name="logger">Logger instance for diagnostics</param>
    public CpuComputeKernel(string source, string name, ILogger logger)
    {
        _logger = logger;
        Name = name;
    }

    /// <summary>
    /// Gets the kernel name
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Sets a float buffer argument
    /// </summary>
    /// <param name="index">Argument index</param>
    /// <param name="buffer">Buffer to set</param>
    public void SetArgument(int index, IComputeBuffer<float> buffer)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Setting CPU kernel argument {Index} to float buffer", index);
    }

    /// <summary>
    /// Sets a double buffer argument
    /// </summary>
    /// <param name="index">Argument index</param>
    /// <param name="buffer">Buffer to set</param>
    public void SetArgument(int index, IComputeBuffer<double> buffer)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Setting CPU kernel argument {Index} to double buffer", index);
    }

    /// <summary>
    /// Sets an int buffer argument
    /// </summary>
    /// <param name="index">Argument index</param>
    /// <param name="buffer">Buffer to set</param>
    public void SetArgument(int index, IComputeBuffer<int> buffer)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Setting CPU kernel argument {Index} to int buffer", index);
    }

    /// <summary>
    /// Sets a float scalar argument
    /// </summary>
    /// <param name="index">Argument index</param>
    /// <param name="value">Float value to set</param>
    public void SetArgument(int index, float value)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Setting CPU kernel argument {Index} to float value {Value}", index, value);
    }

    /// <summary>
    /// Sets a double scalar argument
    /// </summary>
    /// <param name="index">Argument index</param>
    /// <param name="value">Double value to set</param>
    public void SetArgument(int index, double value)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Setting CPU kernel argument {Index} to double value {Value}", index, value);
    }

    /// <summary>
    /// Sets an int scalar argument
    /// </summary>
    /// <param name="index">Argument index</param>
    /// <param name="value">Int value to set</param>
    public void SetArgument(int index, int value)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Setting CPU kernel argument {Index} to int value {Value}", index, value);
    }

    internal void Execute(int workSize)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CpuComputeKernel));
        }

        _logger.LogTrace("Executing CPU kernel {Name} with work size {WorkSize}", Name, workSize);

        // CPU kernel execution implementation would go here
        // For now, this is a placeholder
    }

    /// <summary>
    /// Disposes the kernel
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _logger.LogTrace("Disposing CPU kernel {Name}", Name);
            // CPU kernel cleanup is minimal
            _disposed = true;
        }
    }
}