using FluentAssertions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;
using Xunit;

namespace Orleans.GpuBridge.Tests;

public class KernelCatalogTests
{
    [Fact]
    public async Task Should_Register_And_Resolve_Kernel()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddGpuBridge()
            .AddKernel(k => k
                .Id("test/add")
                .Input<float[]>()
                .Output<float>()
                .WithFactory(sp => new TestKernel()));
        
        var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<KernelCatalog>();
        
        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("test/add"), provider);
        
        // Assert
        kernel.Should().NotBeNull();
        kernel.Should().BeOfType<TestKernel>();
    }
    
    [Fact]
    public async Task Should_Fallback_To_Cpu_When_Kernel_Not_Found()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddGpuBridge();
        
        var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<KernelCatalog>();
        
        // Act
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("unknown"), provider);
        
        // Assert
        kernel.Should().NotBeNull();
        kernel.GetType().Name.Should().Contain("CpuPassthroughKernel");
    }
    
    [Fact]
    public async Task Should_Get_Kernel_Info()
    {
        // Arrange
        var kernel = new TestKernel();
        
        // Act
        var info = await kernel.GetInfoAsync();
        
        // Assert
        info.Should().NotBeNull();
        info.Id.Value.Should().Be("test-kernel");
        info.InputType.Should().Be(typeof(float[]));
        info.OutputType.Should().Be(typeof(float));
        info.SupportsGpu.Should().BeFalse();
        info.PreferredBatchSize.Should().Be(1024);
    }
    
    private class TestKernel : IGpuKernel<float[], float>
    {
        public ValueTask<KernelHandle> SubmitBatchAsync(
            IReadOnlyList<float[]> items,
            GpuExecutionHints? hints = null,
            CancellationToken ct = default)
        {
            return new(KernelHandle.Create());
        }
        
        public async IAsyncEnumerable<float> ReadResultsAsync(
            KernelHandle handle,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            yield return 42f;
        }
        
        public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
        {
            return new(new KernelInfo(
                new KernelId("test-kernel"),
                "Test kernel",
                typeof(float[]),
                typeof(float),
                false,
                1024));
        }
    }
}