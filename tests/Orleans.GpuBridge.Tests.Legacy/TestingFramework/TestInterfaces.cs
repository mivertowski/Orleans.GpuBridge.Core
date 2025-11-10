using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Memory;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Mock backend provider interface for testing
/// </summary>
public interface IMockBackendProvider : IDisposable
{
    string Name { get; }
    BackendType Type { get; }
    bool IsAvailable { get; }
    IMockComputeContext CreateContext();
    Task<bool> ValidateAsync();
}

/// <summary>
/// Mock compute context interface for testing
/// </summary>
public interface IMockComputeContext : IDisposable
{
    IMockComputeBuffer<T> CreateBuffer<T>(int count, BufferUsage usage) where T : unmanaged;
}

/// <summary>
/// Mock compute buffer interface for testing
/// </summary>
public interface IMockComputeBuffer<T> : IDisposable where T : unmanaged
{
    int Length { get; }
    void Write(ReadOnlySpan<T> data);
    void Read(Span<T> destination);
}

/// <summary>
/// Backend type enumeration
/// </summary>
public enum BackendType
{
    CPU,
    CUDA,
    OpenCL,
    Vulkan,
    DirectX,
    Metal
}

/// <summary>
/// Vector operation enumeration
/// </summary>
public enum VectorOperation
{
    Add,
    Subtract,
    Multiply,
    Divide,
    FusedMultiplyAdd,
    Dot,
    Cross
}