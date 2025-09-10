using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Mock backend provider factory for testing
/// </summary>
public class BackendProviderFactory
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<BackendProviderFactory> _logger;
    private bool _initialized;

    public BackendProviderFactory(IServiceProvider serviceProvider, ILogger<BackendProviderFactory> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
    }

    public void Initialize()
    {
        _initialized = true;
        _logger.LogInformation("Backend provider factory initialized");
    }

    public IMockBackendProvider GetPrimaryProvider()
    {
        if (!_initialized)
            throw new InvalidOperationException("Factory not initialized");

        return new MockBackendProvider(_logger);
    }
}

/// <summary>
/// Mock backend provider implementation
/// </summary>
public class MockBackendProvider : IMockBackendProvider
{
    private readonly ILogger _logger;

    public MockBackendProvider(ILogger logger)
    {
        _logger = logger;
    }

    public string Name => "Mock";
    public BackendType Type => BackendType.CPU;
    public bool IsAvailable => true;

    public IMockComputeContext CreateContext()
    {
        return new MockComputeContext(_logger);
    }

    public Task<bool> ValidateAsync()
    {
        return Task.FromResult(true);
    }

    public void Dispose()
    {
        // No-op for mock
    }
}

/// <summary>
/// Mock compute context implementation
/// </summary>
public class MockComputeContext : IMockComputeContext
{
    private readonly ILogger _logger;

    public MockComputeContext(ILogger logger)
    {
        _logger = logger;
    }

    public IMockComputeBuffer<T> CreateBuffer<T>(int count, BufferUsage usage) where T : unmanaged
    {
        return new MockComputeBuffer<T>(count, _logger);
    }

    public void Dispose()
    {
        // No-op for mock
    }
}

/// <summary>
/// Mock compute buffer implementation
/// </summary>
public class MockComputeBuffer<T> : IMockComputeBuffer<T> where T : unmanaged
{
    private readonly T[] _data;
    private readonly ILogger _logger;

    public MockComputeBuffer(int count, ILogger logger)
    {
        _data = new T[count];
        _logger = logger;
        Length = count;
    }

    public int Length { get; }

    public void Write(ReadOnlySpan<T> data)
    {
        data.CopyTo(_data);
    }

    public void Read(Span<T> destination)
    {
        _data.CopyTo(destination);
    }

    public void Dispose()
    {
        // No-op for mock
    }
}

/// <summary>
/// Mock parallel kernel executor for testing
/// </summary>
public class ParallelKernelExecutor
{
    private readonly ILogger _logger;

    public ParallelKernelExecutor(ILogger logger)
    {
        _logger = logger;
    }

    public async Task<TResult> ExecuteAsync<TInput, TResult>(
        string kernelId, 
        TInput input, 
        int batchSize = 1000)
    {
        _logger.LogInformation($"Executing kernel {kernelId} with batch size {batchSize}");
        
        // Mock execution - return default for now
        await Task.Delay(10); // Simulate work
        return default(TResult)!;
    }
}

/// <summary>
/// Mock vector operation utilities
/// </summary>
public static class VectorOperationUtils
{
    public static float[] Add(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be same length");

        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    public static float[] Multiply(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be same length");

        var result = new float[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i] * b[i];
        }
        return result;
    }
}

/// <summary>
/// Mock buffer serializer for testing
/// </summary>
public static class BufferSerializer
{
    public static byte[] SerializeFloat(float[] data)
    {
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public static float[] DeserializeFloat(byte[] bytes)
    {
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    public static byte[] SerializeInt(int[] data)
    {
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public static int[] DeserializeInt(byte[] bytes)
    {
        var ints = new int[bytes.Length / sizeof(int)];
        Buffer.BlockCopy(bytes, 0, ints, 0, bytes.Length);
        return ints;
    }

    public static byte[] Serialize<T>(T[] data) where T : unmanaged
    {
        var size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
        var bytes = new byte[data.Length * size];
        
        unsafe
        {
            fixed (T* dataPtr = data)
            fixed (byte* bytesPtr = bytes)
            {
                Buffer.MemoryCopy(dataPtr, bytesPtr, bytes.Length, bytes.Length);
            }
        }
        
        return bytes;
    }

    public static T[] Deserialize<T>(byte[] bytes) where T : unmanaged
    {
        var size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
        var data = new T[bytes.Length / size];
        
        unsafe
        {
            fixed (byte* bytesPtr = bytes)
            fixed (T* dataPtr = data)
            {
                Buffer.MemoryCopy(bytesPtr, dataPtr, bytes.Length, bytes.Length);
            }
        }
        
        return data;
    }

    public static async Task<byte[]> SerializeCompressedAsync<T>(T[] data) where T : unmanaged
    {
        var serialized = Serialize(data);
        await Task.Delay(1); // Simulate async compression
        return serialized; // Mock: no actual compression
    }

    public static async Task<T[]> DeserializeCompressedAsync<T>(byte[] bytes) where T : unmanaged
    {
        await Task.Delay(1); // Simulate async decompression  
        return Deserialize<T>(bytes); // Mock: no actual decompression
    }
}