using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime.Providers;
using Orleans.GpuBridge.Tests.TestingFramework;
using Xunit;

namespace Orleans.GpuBridge.Tests.Providers;

public class GpuBackendRegistryTests
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IGpuBackendRegistry _registry;

    public GpuBackendRegistryTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        
        // Add test providers
        services.AddSingleton<IGpuBackendProvider, TestGpuProvider>();
        services.AddSingleton<IGpuBackendProvider, TestCpuProvider>();
        
        _serviceProvider = services.BuildServiceProvider();
        _registry = new GpuBackendRegistry(_serviceProvider, _serviceProvider.GetRequiredService<ILogger<GpuBackendRegistry>>());
    }

    [Fact]
    public async Task InitializeAsync_RegistersAllAvailableProviders()
    {
        // Act
        await _registry.InitializeAsync();
        
        // Assert
        await _registry.DiscoverProvidersAsync();
        var providers = _registry.GetRegisteredProviders().ToList();
        Assert.Equal(2, providers.Count());
        Assert.Contains(providers, p => p == "TestGpu");
        Assert.Contains(providers, p => p == "TestCpu");
    }

    [Fact]
    public async Task SelectProviderAsync_ReturnsProviderMatchingCriteria()
    {
        // Arrange
        await _registry.InitializeAsync();
        var criteria = new ProviderSelectionCriteria
        {
            PreferredBackends = new[] { GpuBackend.Cuda },
            RequiredCapabilities = new BackendCapabilities
            {
                SupportedBackends = new[] { GpuBackend.Cuda }
            }
        };

        // Act
        var provider = await _registry.SelectProviderAsync(criteria);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestGpu", provider.ProviderId);
    }

    [Fact]
    public async Task SelectProviderAsync_ReturnsCpuFallbackWhenNoGpuAvailable()
    {
        // Arrange
        await _registry.InitializeAsync();
        var criteria = new ProviderSelectionCriteria
        {
            PreferredBackends = new[] { GpuBackend.DirectCompute }, // Not supported by test providers
            AllowCpuFallback = true
        };

        // Act
        var provider = await _registry.SelectProviderAsync(criteria);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestCpu", provider.ProviderId);
    }

    [Fact]
    public async Task SelectProviderAsync_ReturnsNullWhenNoCpuFallback()
    {
        // Arrange
        await _registry.InitializeAsync();
        var criteria = new ProviderSelectionCriteria
        {
            PreferredBackends = new[] { GpuBackend.DirectCompute },
            AllowCpuFallback = false
        };

        // Act
        var provider = await _registry.SelectProviderAsync(criteria);

        // Assert
        Assert.Null(provider);
    }

    [Fact]
    public async Task GetProviderByIdAsync_ReturnsCorrectProvider()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var provider = await _registry.GetProviderAsync("TestGpu");

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestGpu", provider.ProviderId);
    }

    [Fact]
    public async Task IsProviderAvailableAsync_ReturnsTrueForRegisteredProvider()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var provider = await _registry.GetProviderAsync("TestGpu");
        var isAvailable = provider != null;

        // Assert
        Assert.True(isAvailable);
    }

    [Fact]
    public async Task IsProviderAvailableAsync_ReturnsFalseForUnregisteredProvider()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var provider = await _registry.GetProviderAsync("NonExistent");
        var isAvailable = provider != null;

        // Assert
        Assert.False(isAvailable);
    }
}

// Using shared test stubs from TestingFramework namespace