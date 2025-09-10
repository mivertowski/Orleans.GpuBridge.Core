using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime.Providers;
using Orleans.GpuBridge.Tests.Providers;
using Orleans.GpuBridge.Tests.TestingFramework;
using Xunit;

namespace Orleans.GpuBridge.Tests.Providers;

public class GpuBridgeProviderSelectorTests
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IGpuBridgeProviderSelector _selector;
    private readonly IGpuBackendRegistry _registry;

    public GpuBridgeProviderSelectorTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.Configure<GpuBridgeOptions>(options => { });
        
        // Add test providers
        services.AddSingleton<IGpuBackendProvider, TestGpuProvider>();
        services.AddSingleton<IGpuBackendProvider, TestCpuProvider>();
        
        _serviceProvider = services.BuildServiceProvider();
        _registry = new GpuBackendRegistry(_serviceProvider, _serviceProvider.GetRequiredService<ILogger<GpuBackendRegistry>>());
        _selector = new GpuBridgeProviderSelector(
            _serviceProvider.GetRequiredService<ILogger<GpuBridgeProviderSelector>>(),
            _registry,
            _serviceProvider.GetRequiredService<IOptions<GpuBridgeOptions>>(),
            _serviceProvider);
    }

    [Fact]
    public async Task SelectProviderAsync_WithGpuPreference_ReturnsGpuProvider()
    {
        // Arrange
        await _registry.InitializeAsync();
        var requirements = new ProviderSelectionCriteria(PreferGpu: true);

        // Act
        var provider = await _selector.SelectProviderAsync(requirements);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestGpu", provider.ProviderId);
        Assert.Contains(GpuBackend.CUDA, provider.Capabilities.SupportedBackends);
    }

    [Fact]
    public async Task SelectProviderAsync_WithCpuOnly_ReturnsCpuProvider()
    {
        // Arrange
        await _registry.InitializeAsync();
        var requirements = new ProviderSelectionCriteria(PreferGpu: false);

        // Act
        var provider = await _selector.SelectProviderAsync(requirements);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestCpu", provider.ProviderId);
        Assert.Contains(GpuBackend.CPU, provider.Capabilities.SupportedBackends);
    }

    [Fact]
    public async Task SelectProviderAsync_WithSpecificBackend_ReturnsMatchingProvider()
    {
        // Arrange
        await _registry.InitializeAsync();
        var requirements = new GpuExecutionRequirements 
        { 
            PreferGpu = true,
            RequiredCapabilities = new BackendCapabilities
            {
                SupportedBackends = new[] { GpuBackend.CUDA }
            }
        };

        // Act
        var provider = await _selector.SelectProviderAsync(requirements);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestGpu", provider.ProviderId);
    }

    [Fact]
    public async Task SelectProviderAsync_WithUnsupportedBackend_FallsBackToCpu()
    {
        // Arrange
        await _registry.InitializeAsync();
        var requirements = new GpuExecutionRequirements 
        { 
            PreferGpu = true,
            RequiredCapabilities = new BackendCapabilities
            {
                SupportedBackends = new[] { GpuBackend.DirectCompute } // Not supported by test providers
            }
        };

        // Act
        var provider = await _selector.SelectProviderAsync(requirements);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestCpu", provider.ProviderId);
    }

    [Fact]
    public async Task GetAvailableProvidersAsync_ReturnsAllRegisteredProviders()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var providers = await _selector.GetAvailableProvidersAsync();

        // Assert
        Assert.Equal(2, providers.Count());
        Assert.Contains(providers, p => p.ProviderId == "TestGpu");
        Assert.Contains(providers, p => p.ProviderId == "TestCpu");
    }

    [Fact]
    public async Task CheckProviderHealthAsync_ReturnsHealthStatus()
    {
        // Arrange
        await _registry.InitializeAsync();
        var provider = await _selector.SelectProviderAsync(new GpuExecutionRequirements { PreferGpu = true });

        // Act
        var health = await _selector.CheckProviderHealthAsync(provider);

        // Assert
        Assert.True(health.IsHealthy);
        Assert.Equal("Healthy", health.Message);
    }

    [Fact]
    public async Task ValidateProviderAsync_WithValidProvider_ReturnsTrue()
    {
        // Arrange
        await _registry.InitializeAsync();
        var provider = await _selector.SelectProviderAsync(new GpuExecutionRequirements { PreferGpu = true });
        var requirements = new GpuExecutionRequirements { PreferGpu = true };

        // Act
        var isValid = await _selector.ValidateProviderAsync(provider, requirements);

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public async Task ValidateProviderAsync_WithInvalidProvider_ReturnsFalse()
    {
        // Arrange
        await _registry.InitializeAsync();
        var provider = await _selector.SelectProviderAsync(new GpuExecutionRequirements { PreferGpu = false }); // CPU provider
        var requirements = new GpuExecutionRequirements 
        { 
            PreferGpu = true,
            RequiredCapabilities = new BackendCapabilities
            {
                SupportedBackends = new[] { GpuBackend.CUDA }, // CPU provider doesn't support CUDA
                SupportsJitCompilation = false // CPU provider supports JIT
            }
        };

        // Act
        var isValid = await _selector.ValidateProviderAsync(provider, requirements);

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public async Task SelectProviderAsync_WithMemoryRequirements_ReturnsProviderWithSufficientMemory()
    {
        // Arrange
        await _registry.InitializeAsync();
        var requirements = new GpuExecutionRequirements 
        { 
            PreferGpu = true,
            MinimumMemoryBytes = 512 * 1024 * 1024 // 512 MB
        };

        // Act
        var provider = await _selector.SelectProviderAsync(requirements);

        // Assert
        Assert.NotNull(provider);
        // Both test providers should have enough memory for this test
    }

    [Theory]
    [InlineData(true, "TestGpu")]
    [InlineData(false, "TestCpu")]
    public async Task SelectProviderAsync_GpuPreferenceParameterized_ReturnsExpectedProvider(bool preferGpu, string expectedProviderId)
    {
        // Arrange
        await _registry.InitializeAsync();
        var requirements = new GpuExecutionRequirements { PreferGpu = preferGpu };

        // Act
        var provider = await _selector.SelectProviderAsync(requirements);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal(expectedProviderId, provider.ProviderId);
    }
}

