using DotCompute.Core.Compute;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;
using Orleans.GpuBridge.Backends.DotCompute.Memory;

namespace Orleans.GpuBridge.Backends.DotCompute.Tests.EndToEnd;

/// <summary>
/// End-to-end tests for Phase 1.3 Memory Integration
/// Demonstrates real GPU memory allocation with native IUnifiedMemoryBuffer storage
/// </summary>
public class MemoryIntegrationTests : IDisposable
{
    private readonly DotComputeAcceleratorAdapter? _adapter;
    private readonly DotComputeMemoryAllocator _allocator;
    private bool _disposed;

    public MemoryIntegrationTests()
    {
        // Try to get a real DotCompute device
        _adapter = GetFirstAvailableAdapter().Result;
        _allocator = CreateMemoryAllocator(_adapter);
    }

    private static async Task<DotComputeAcceleratorAdapter?> GetFirstAvailableAdapter()
    {
        try
        {
            var manager = await DefaultAcceleratorManagerFactory.CreateAsync();
            var accelerators = await manager.GetAcceleratorsAsync();
            var firstAccelerator = accelerators.FirstOrDefault();

            if (firstAccelerator == null)
                return null;

            return new DotComputeAcceleratorAdapter(firstAccelerator, 0, NullLogger.Instance);
        }
        catch
        {
            return null; // No GPU available
        }
    }

    private static DotComputeMemoryAllocator CreateMemoryAllocator(DotComputeAcceleratorAdapter? adapter)
    {
        var logger = NullLogger<DotComputeMemoryAllocator>.Instance;
        // Create a simple device manager mock
        var deviceManager = new TestDeviceManager(adapter);
        var config = new Orleans.GpuBridge.Abstractions.Providers.BackendConfiguration();
        return new DotComputeMemoryAllocator(logger, deviceManager, config);
    }

    /// <summary>
    /// Test Phase 1.3: Real GPU memory allocation with native IUnifiedMemoryBuffer
    /// </summary>
    [Fact]
    public async Task MemoryAllocation_WithNativeBuffer_Success()
    {
        // Arrange
        if (_adapter == null)
        {
            // Skip test if no GPU available
            return;
        }

        const int elementCount = 1024;

        // Act - Allocate GPU memory with real IUnifiedMemoryBuffer (Phase 1.3)
        var gpuMemory = await _allocator.AllocateAsync<float>(
            elementCount,
            new MemoryAllocationOptions(),
            CancellationToken.None);

        // Assert
        gpuMemory.Should().NotBeNull("GPU memory allocation should succeed");
        gpuMemory.Length.Should().Be(elementCount, "Element count should match");
        gpuMemory.SizeBytes.Should().Be(elementCount * sizeof(float), "Size in bytes should be correct");

        // Verify Phase 1.3: Native buffer is stored
        if (gpuMemory is DotComputeDeviceMemoryWrapper<float> wrapper)
        {
            wrapper.NativeBuffer.Should().NotBeNull("Phase 1.3: Native buffer should be stored for zero-copy execution");
        }

        // Cleanup
        gpuMemory.Dispose();
    }

    /// <summary>
    /// Test Phase 1.3: Data transfer round-trip using native buffers
    /// </summary>
    [Fact]
    public async Task DataTransfer_RoundTrip_WithNativeBuffer_Success()
    {
        // Arrange
        if (_adapter == null)
        {
            return; // Skip if no GPU
        }

        const int elementCount = 256;
        var originalData = Enumerable.Range(0, elementCount).Select(i => (float)i * 2.5f).ToArray();

        // Act - Allocate and transfer data
        var gpuMemory = await _allocator.AllocateAsync<float>(
            elementCount,
            new MemoryAllocationOptions(),
            CancellationToken.None);

        // Copy to GPU
        await gpuMemory.CopyFromHostAsync(originalData, 0, 0, elementCount, CancellationToken.None);

        // Copy back from GPU
        var retrievedData = new float[elementCount];
        await gpuMemory.CopyToHostAsync(retrievedData, 0, 0, elementCount, CancellationToken.None);

        // Assert
        retrievedData.Should().NotBeNull("Retrieved data should not be null");
        retrievedData.Length.Should().Be(elementCount, "Retrieved data should have correct length");

        for (int i = 0; i < elementCount; i++)
        {
            retrievedData[i].Should().BeApproximately(
                originalData[i],
                0.0001f,
                $"Element {i} should match after round-trip: expected {originalData[i]}, got {retrievedData[i]}");
        }

        // Cleanup
        gpuMemory.Dispose();
    }

    /// <summary>
    /// Test Phase 1.3: Multiple allocations with native buffers
    /// </summary>
    [Fact]
    public async Task MultipleAllocations_WithNativeBuffers_Success()
    {
        // Arrange
        if (_adapter == null)
        {
            return; // Skip if no GPU
        }

        const int buffer1Size = 512;
        const int buffer2Size = 1024;
        const int buffer3Size = 2048;

        // Act - Allocate multiple GPU buffers
        var buffer1 = await _allocator.AllocateAsync<float>(
            buffer1Size, new MemoryAllocationOptions(), CancellationToken.None);
        var buffer2 = await _allocator.AllocateAsync<int>(
            buffer2Size, new MemoryAllocationOptions(), CancellationToken.None);
        var buffer3 = await _allocator.AllocateAsync<double>(
            buffer3Size, new MemoryAllocationOptions(), CancellationToken.None);

        // Assert - All allocations successful
        buffer1.Should().NotBeNull("Buffer 1 should be allocated");
        buffer1.Length.Should().Be(buffer1Size);

        buffer2.Should().NotBeNull("Buffer 2 should be allocated");
        buffer2.Length.Should().Be(buffer2Size);

        buffer3.Should().NotBeNull("Buffer 3 should be allocated");
        buffer3.Length.Should().Be(buffer3Size);

        // Verify all have native buffers (Phase 1.3)
        (buffer1 as DotComputeDeviceMemoryWrapper<float>)?.NativeBuffer.Should().NotBeNull();
        (buffer2 as DotComputeDeviceMemoryWrapper<int>)?.NativeBuffer.Should().NotBeNull();
        (buffer3 as DotComputeDeviceMemoryWrapper<double>)?.NativeBuffer.Should().NotBeNull();

        // Cleanup
        buffer1.Dispose();
        buffer2.Dispose();
        buffer3.Dispose();
    }

    /// <summary>
    /// Test Phase 1.3: Verify zero-copy execution readiness
    /// </summary>
    [Fact]
    public async Task NativeBuffer_IsAccessibleForZeroCopyExecution()
    {
        // Arrange
        if (_adapter == null)
        {
            return; // Skip if no GPU
        }

        const int elementCount = 100;

        // Act - Allocate GPU memory
        var gpuMemory = await _allocator.AllocateAsync<float>(
            elementCount,
            new MemoryAllocationOptions(),
            CancellationToken.None);

        // Assert - Verify native buffer is accessible for kernel execution
        var wrapper = gpuMemory as DotComputeDeviceMemoryWrapper<float>;
        wrapper.Should().NotBeNull("Memory should be DotComputeDeviceMemoryWrapper");
        wrapper!.NativeBuffer.Should().NotBeNull("Native buffer must be available for zero-copy kernel execution");

        // This native buffer can now be passed directly to kernel.ExecuteAsync()
        // No temporary buffer allocation needed (Phase 1.3 achievement)

        // Cleanup
        gpuMemory.Dispose();
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _allocator?.Dispose();
        _disposed = true;
    }

    /// <summary>
    /// Simple test device manager for isolated tests
    /// </summary>
    private class TestDeviceManager : Orleans.GpuBridge.Abstractions.Providers.IDeviceManager
    {
        private readonly DotComputeAcceleratorAdapter? _adapter;
        private readonly IReadOnlyList<Orleans.GpuBridge.Abstractions.Providers.IComputeDevice> _devices;

        public TestDeviceManager(DotComputeAcceleratorAdapter? adapter)
        {
            _adapter = adapter;
            _devices = adapter != null
                ? new List<Orleans.GpuBridge.Abstractions.Providers.IComputeDevice> { adapter }
                : new List<Orleans.GpuBridge.Abstractions.Providers.IComputeDevice>();
        }

        public Task InitializeAsync(CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public IReadOnlyList<Orleans.GpuBridge.Abstractions.Providers.IComputeDevice> GetDevices()
            => _devices;

        public Orleans.GpuBridge.Abstractions.Providers.IComputeDevice? GetDevice(int deviceIndex)
            => deviceIndex == 0 ? _adapter : null;

        public Orleans.GpuBridge.Abstractions.Providers.IComputeDevice GetDefaultDevice()
            => _adapter ?? throw new InvalidOperationException("No GPU device available");

        public Orleans.GpuBridge.Abstractions.Providers.IComputeDevice SelectDevice(
            Orleans.GpuBridge.Abstractions.Models.DeviceSelectionCriteria criteria)
            => _adapter ?? throw new InvalidOperationException("No GPU device available");

        public Task<Orleans.GpuBridge.Abstractions.Providers.IComputeContext> CreateContextAsync(
            Orleans.GpuBridge.Abstractions.Providers.IComputeDevice device,
            Orleans.GpuBridge.Abstractions.Models.ContextOptions options,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException("Context creation not needed for memory tests");

        public Task<Orleans.GpuBridge.Abstractions.Models.DeviceMetrics> GetDeviceMetricsAsync(
            Orleans.GpuBridge.Abstractions.Providers.IComputeDevice device,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException("Metrics not needed for memory tests");

        public Task ResetDeviceAsync(
            Orleans.GpuBridge.Abstractions.Providers.IComputeDevice device,
            CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public void Dispose() { }
    }
}
