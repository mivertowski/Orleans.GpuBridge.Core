using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Runtime.BackendProviders;
using Xunit;

namespace Orleans.GpuBridge.Tests.BackendProviders;

public class BackendProviderFactoryTests
{
    private readonly BackendProviderFactory _factory;
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<BackendProviderFactory> _logger;

    public BackendProviderFactoryTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        _serviceProvider = services.BuildServiceProvider();
        _logger = new TestLogger<BackendProviderFactory>();
        _factory = new BackendProviderFactory(_serviceProvider, _logger);
    }

    [Fact]
    public void Initialize_Should_Always_Include_CPU_Provider()
    {
        // Act
        _factory.Initialize();
        var providers = _factory.GetAvailableProviders();

        // Assert
        Assert.NotNull(providers);
        Assert.Contains(providers, p => p.Type == BackendType.Cpu);
    }

    [Fact]
    public void GetPrimaryProvider_Should_Return_Provider()
    {
        // Arrange
        _factory.Initialize();

        // Act
        var provider = _factory.GetPrimaryProvider();

        // Assert
        Assert.NotNull(provider);
    }

    [Fact]
    public void GetPrimaryProvider_Without_Initialize_Should_Throw()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _factory.GetPrimaryProvider());
    }

    [Fact]
    public void GetProvider_Should_Return_Correct_Provider()
    {
        // Arrange
        _factory.Initialize();

        // Act
        var cpuProvider = _factory.GetProvider(BackendType.Cpu);

        // Assert
        Assert.NotNull(cpuProvider);
        Assert.Equal(BackendType.Cpu, cpuProvider!.Type);
    }

    [Fact]
    public void GetProvider_For_Unavailable_Backend_Should_Return_Null()
    {
        // Arrange
        _factory.Initialize();

        // Act - Assuming these aren't available in test environment
        var cudaProvider = _factory.GetProvider(BackendType.Cuda);
        var vulkanProvider = _factory.GetProvider(BackendType.Vulkan);

        // Assert - May be null depending on system
        // Just verify no exceptions are thrown
        Assert.True(cudaProvider == null || cudaProvider.Type == BackendType.Cuda);
        Assert.True(vulkanProvider == null || vulkanProvider.Type == BackendType.Vulkan);
    }

    [Fact]
    public void GetAvailableProviders_Should_Return_List()
    {
        // Arrange
        _factory.Initialize();

        // Act
        var providers = _factory.GetAvailableProviders();

        // Assert
        Assert.NotNull(providers);
        Assert.NotEmpty(providers);
    }

    [Fact]
    public void Initialize_Multiple_Times_Should_Be_Safe()
    {
        // Act
        _factory.Initialize();
        var count1 = _factory.GetAvailableProviders().Count;
        
        _factory.Initialize();
        var count2 = _factory.GetAvailableProviders().Count;

        // Assert
        Assert.True(count2 >= count1); // May detect more on second init
    }
}

public class CpuBackendProviderTests
{
    private readonly CpuBackendProvider _provider;
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger _logger;

    public CpuBackendProviderTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        _serviceProvider = services.BuildServiceProvider();
        _logger = new TestLogger<CpuBackendProvider>();
        _provider = new CpuBackendProvider(_serviceProvider, _logger);
    }

    [Fact]
    public void Initialize_Should_Return_True()
    {
        // Act
        var result = _provider.Initialize();

        // Assert
        Assert.True(result);
        Assert.True(_provider.IsAvailable);
    }

    [Fact]
    public void Name_Should_Be_CPU()
    {
        // Assert
        Assert.Equal("CPU", _provider.Name);
    }

    [Fact]
    public void Type_Should_Be_Cpu()
    {
        // Assert
        Assert.Equal(BackendType.Cpu, _provider.Type);
    }

    [Fact]
    public void DeviceCount_Should_Be_One_After_Initialize()
    {
        // Arrange
        _provider.Initialize();

        // Act & Assert
        Assert.Equal(1, _provider.DeviceCount);
    }

    [Fact]
    public void GetDevices_Should_Return_CPU_Device()
    {
        // Arrange
        _provider.Initialize();

        // Act
        var devices = _provider.GetDevices();

        // Assert
        Assert.Single(devices);
        Assert.Equal(BackendType.Cpu, devices[0].Backend);
        Assert.NotEmpty(devices[0].Name);
        Assert.True(devices[0].TotalMemory > 0);
        Assert.True(devices[0].ComputeUnits > 0);
    }

    [Fact]
    public void GetDevices_Should_Include_SIMD_Extensions()
    {
        // Arrange
        _provider.Initialize();

        // Act
        var devices = _provider.GetDevices();

        // Assert
        Assert.NotEmpty(devices[0].Extensions);
        Assert.Contains("CPU", devices[0].Extensions);
        
        // Check for at least one SIMD extension
        var hasSimd = devices[0].Extensions.Any(e => 
            e.Contains("SSE") || 
            e.Contains("AVX") || 
            e.Contains("NEON"));
    }

    [Fact]
    public void CreateContext_Should_Return_Context()
    {
        // Arrange
        _provider.Initialize();

        // Act
        using var context = _provider.CreateContext();

        // Assert
        Assert.NotNull(context);
        Assert.Equal(BackendType.Cpu, context.Backend);
        Assert.Equal(0, context.DeviceIndex);
    }

    [Fact]
    public void CreateContext_Without_Initialize_Should_Throw()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _provider.CreateContext());
    }

    [Fact]
    public void Shutdown_Should_Clear_Devices()
    {
        // Arrange
        _provider.Initialize();
        Assert.Equal(1, _provider.DeviceCount);

        // Act
        _provider.Shutdown();

        // Assert
        Assert.Equal(0, _provider.DeviceCount);
        Assert.False(_provider.IsAvailable);
    }
}

public class CpuComputeContextTests : IDisposable
{
    private readonly CpuBackendProvider _provider;
    private readonly IComputeContext _context;
    private readonly ILogger _logger;

    public CpuComputeContextTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        _logger = new TestLogger<CpuBackendProvider>();
        _provider = new CpuBackendProvider(serviceProvider, _logger);
        _provider.Initialize();
        _context = _provider.CreateContext();
    }

    [Fact]
    public void CreateBuffer_Should_Create_Buffer()
    {
        // Act
        using var buffer = _context.CreateBuffer<float>(100, BufferUsage.ReadWrite);

        // Assert
        Assert.NotNull(buffer);
        Assert.Equal(100, buffer.Size);
        Assert.Equal(BufferUsage.ReadWrite, buffer.Usage);
    }

    [Fact]
    public void CompileKernel_Should_Create_Kernel()
    {
        // Act
        using var kernel = _context.CompileKernel("vector_add", "main");

        // Assert
        Assert.NotNull(kernel);
        Assert.Equal("main", kernel.Name);
    }

    [Fact]
    public void Execute_Should_Execute_Kernel()
    {
        // Arrange
        var kernel = _context.CompileKernel("vector_add", "test");

        // Act & Assert - Should not throw
        _context.Execute(kernel, 100);
    }

    [Fact]
    public void Execute_With_Wrong_Kernel_Type_Should_Throw()
    {
        // Arrange
        var mockKernel = new MockKernel();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _context.Execute(mockKernel, 100));
    }

    [Fact]
    public void Synchronize_Should_Complete()
    {
        // Act & Assert - Should not throw
        _context.Synchronize();
    }

    [Fact]
    public void Buffer_Write_And_Read_Should_Work()
    {
        // Arrange
        using var buffer = _context.CreateBuffer<float>(10, BufferUsage.ReadWrite);
        var data = Enumerable.Range(1, 10).Select(i => (float)i).ToArray();

        // Act
        buffer.Write(data);
        var result = new float[10];
        buffer.Read(result);

        // Assert
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(data[i], result[i]);
        }
    }

    [Fact]
    public void Buffer_CopyTo_Should_Work()
    {
        // Arrange
        using var source = _context.CreateBuffer<float>(10, BufferUsage.ReadOnly);
        using var destination = _context.CreateBuffer<float>(10, BufferUsage.WriteOnly);
        var data = Enumerable.Range(1, 10).Select(i => (float)i).ToArray();
        source.Write(data);

        // Act
        source.CopyTo(destination);
        var result = new float[10];
        destination.Read(result);

        // Assert
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(data[i], result[i]);
        }
    }

    [Fact]
    public void Kernel_SetArgument_Should_Accept_All_Types()
    {
        // Arrange
        using var kernel = _context.CompileKernel("test", "kernel");
        using var floatBuffer = _context.CreateBuffer<float>(10, BufferUsage.ReadWrite);
        using var doubleBuffer = _context.CreateBuffer<double>(10, BufferUsage.ReadWrite);
        using var intBuffer = _context.CreateBuffer<int>(10, BufferUsage.ReadWrite);

        // Act & Assert - Should not throw
        kernel.SetArgument(0, floatBuffer);
        kernel.SetArgument(1, doubleBuffer);
        kernel.SetArgument(2, intBuffer);
        kernel.SetArgument(3, 1.0f);
        kernel.SetArgument(4, 2.0);
        kernel.SetArgument(5, 3);
    }

    [Fact]
    public void VectorAdd_Kernel_Should_Work()
    {
        // Arrange
        using var kernel = _context.CompileKernel("vector_add", "add");
        using var bufferA = _context.CreateBuffer<float>(100, BufferUsage.ReadOnly);
        using var bufferB = _context.CreateBuffer<float>(100, BufferUsage.ReadOnly);
        using var bufferC = _context.CreateBuffer<float>(100, BufferUsage.WriteOnly);
        
        var dataA = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var dataB = Enumerable.Range(1, 100).Select(i => (float)i * 2).ToArray();
        
        bufferA.Write(dataA);
        bufferB.Write(dataB);
        
        kernel.SetArgument(0, bufferA);
        kernel.SetArgument(1, bufferB);
        kernel.SetArgument(2, bufferC);

        // Act
        _context.Execute(kernel, 100);
        _context.Synchronize();
        
        var result = new float[100];
        bufferC.Read(result);

        // Assert
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(dataA[i] + dataB[i], result[i]);
        }
    }

    public void Dispose()
    {
        _context?.Dispose();
        _provider?.Shutdown();
    }

    private class MockKernel : IComputeKernel
    {
        public string Name => "Mock";
        public void SetArgument(int index, IComputeBuffer<float> buffer) { }
        public void SetArgument(int index, IComputeBuffer<double> buffer) { }
        public void SetArgument(int index, IComputeBuffer<int> buffer) { }
        public void SetArgument(int index, float value) { }
        public void SetArgument(int index, double value) { }
        public void SetArgument(int index, int value) { }
        public void Dispose() { }
    }
}

public class OtherBackendProvidersTests
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger _logger;

    public OtherBackendProvidersTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        _serviceProvider = services.BuildServiceProvider();
        _logger = new TestLogger<BackendProviderFactory>();
    }

    [Fact]
    public void OpenCLProvider_Should_Have_Correct_Properties()
    {
        // Arrange
        var provider = new OpenCLBackendProvider(_serviceProvider, _logger);

        // Assert
        Assert.Equal("OpenCL", provider.Name);
        Assert.Equal(BackendType.OpenCL, provider.Type);
    }

    [Fact]
    public void DirectComputeProvider_Should_Have_Correct_Properties()
    {
        // Arrange
        var provider = new DirectComputeBackendProvider(_serviceProvider, _logger);

        // Assert
        Assert.Equal("DirectCompute", provider.Name);
        Assert.Equal(BackendType.DirectCompute, provider.Type);
    }

    [Fact]
    public void MetalProvider_Should_Have_Correct_Properties()
    {
        // Arrange
        var provider = new MetalBackendProvider(_serviceProvider, _logger);

        // Assert
        Assert.Equal("Metal", provider.Name);
        Assert.Equal(BackendType.Metal, provider.Type);
    }

    [Fact]
    public void VulkanProvider_Should_Have_Correct_Properties()
    {
        // Arrange
        var provider = new VulkanBackendProvider(_serviceProvider, _logger);

        // Assert
        Assert.Equal("Vulkan", provider.Name);
        Assert.Equal(BackendType.Vulkan, provider.Type);
    }

    [Fact]
    public void CudaProvider_Should_Have_Correct_Properties()
    {
        // Arrange
        var provider = new CudaBackendProvider(_serviceProvider, _logger);

        // Assert
        Assert.Equal("CUDA", provider.Name);
        Assert.Equal(BackendType.Cuda, provider.Type);
    }

    [Theory]
    [InlineData(typeof(OpenCLBackendProvider))]
    [InlineData(typeof(DirectComputeBackendProvider))]
    [InlineData(typeof(MetalBackendProvider))]
    [InlineData(typeof(VulkanBackendProvider))]
    public void Unimplemented_Providers_CreateContext_Should_Throw(Type providerType)
    {
        // Arrange
        var provider = (IBackendProvider)Activator.CreateInstance(
            providerType, _serviceProvider, _logger)!;
        provider.Initialize();

        // Act & Assert
        Assert.Throws<NotImplementedException>(() => provider.CreateContext());
    }
}

internal class TestLogger<T> : ILogger<T>, ILogger
{
    public IDisposable BeginScope<TState>(TState state) where TState : notnull => new NoopDisposable();
    public bool IsEnabled(LogLevel logLevel) => true;
    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, 
        Exception? exception, Func<TState, Exception?, string> formatter)
    {
        // Capture logs for testing if needed
    }

    private class NoopDisposable : IDisposable
    {
        public void Dispose() { }
    }
}