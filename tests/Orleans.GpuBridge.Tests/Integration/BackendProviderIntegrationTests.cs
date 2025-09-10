using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Tests.TestingFramework;
using HealthCheckResult = Orleans.GpuBridge.Abstractions.Providers.HealthCheckResult;
using Xunit;

namespace Orleans.GpuBridge.Tests.Integration;

/// <summary>
/// Integration tests for the complete backend provider system
/// </summary>
public class BackendProviderIntegrationTests
{
    [Fact]
    public async Task FullSystemIntegration_WithCpuFallback_WorksCorrectly()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        
        // Add GPU Bridge with CPU fallback backend
        services.AddGpuBridge(options => 
        {
            options.PreferGpu = true;
            options.BatchSize = 64;
        })
        .AddCpuFallbackBackend();

        var serviceProvider = services.BuildServiceProvider();
        
        // Act - Initialize the system
        var registry = serviceProvider.GetRequiredService<IGpuBackendRegistry>();
        await registry.InitializeAsync();
        
        var selector = serviceProvider.GetRequiredService<IGpuBridgeProviderSelector>();
        var criteria = new ProviderSelectionCriteria { PreferGpu = true };
        var provider = await selector.SelectProviderAsync(criteria);
        
        // Assert
        Assert.NotNull(provider);
        Assert.Equal("CpuFallback", provider.ProviderId);
        Assert.True(provider.IsAvailable());
        
        // Test provider components
        var deviceManager = provider.GetDeviceManager();
        Assert.NotNull(deviceManager);
        
        var devices = deviceManager.GetDevices();
        Assert.NotEmpty(devices);
        
        var memoryAllocator = provider.GetMemoryAllocator();
        Assert.NotNull(memoryAllocator);
        
        var kernelCompiler = provider.GetKernelCompiler();
        Assert.NotNull(kernelCompiler);
        
        var kernelExecutor = provider.GetKernelExecutor();
        Assert.NotNull(kernelExecutor);
        
        // Test health check
        var health = await provider.CheckHealthAsync();
        Assert.True(health.IsHealthy);
    }

    [Fact]
    public async Task ProviderSelection_WithMultipleProviders_SelectsCorrectOne()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        
        services.AddGpuBridge()
            .AddBackendProvider<TestGpuProvider>()
            .AddBackendProvider<TestCpuProvider>();

        var serviceProvider = services.BuildServiceProvider();
        
        // Act
        var registry = serviceProvider.GetRequiredService<IGpuBackendRegistry>();
        await registry.InitializeAsync();
        
        var selector = serviceProvider.GetRequiredService<IGpuBridgeProviderSelector>();
        
        // Test GPU preference
        var gpuCriteria = new ProviderSelectionCriteria { PreferGpu = true };
        var gpuProvider = await selector.SelectProviderAsync(gpuCriteria);
        
        // Test CPU preference  
        var cpuCriteria = new ProviderSelectionCriteria { PreferGpu = false };
        var cpuProvider = await selector.SelectProviderAsync(cpuCriteria);
        
        // Assert
        Assert.NotNull(gpuProvider);
        Assert.Equal("TestGpu", gpuProvider.ProviderId);
        
        Assert.NotNull(cpuProvider);
        Assert.Equal("TestCpu", cpuProvider.ProviderId);
        
        Assert.NotEqual(gpuProvider.ProviderId, cpuProvider.ProviderId);
    }

    [Fact]
    public async Task BackendConfiguration_WithCustomOptions_AppliesCorrectly()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        
        services.AddGpuBridge(options => 
        {
            options.PreferGpu = false; // Force CPU
            options.BatchSize = 128;
            options.MaxRetries = 5;
        })
        .AddCpuFallbackBackend()
        .ConfigureBackendSelection(selection =>
        {
            selection.PreferredBackends = new() { GpuBackend.CPU };
            selection.AllowCpuFallback = true;
            selection.MaxConcurrentDevices = 2;
        });

        var serviceProvider = services.BuildServiceProvider();
        
        // Act
        var gpuBridgeOptions = serviceProvider.GetRequiredService<Microsoft.Extensions.Options.IOptions<GpuBridgeOptions>>();
        var registry = serviceProvider.GetRequiredService<IGpuBackendRegistry>();
        await registry.InitializeAsync();
        
        // Assert
        Assert.False(gpuBridgeOptions.Value.PreferGpu);
        Assert.Equal(128, gpuBridgeOptions.Value.BatchSize);
        Assert.Equal(5, gpuBridgeOptions.Value.MaxRetries);
    }

    [Fact]
    public async Task ErrorHandling_WithUnavailableProvider_HandlesGracefully()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddSingleton<IGpuBackendProvider, UnavailableTestProvider>();
        
        services.AddGpuBridge()
            .AddCpuFallbackBackend(); // Add fallback

        var serviceProvider = services.BuildServiceProvider();
        
        // Act
        var registry = serviceProvider.GetRequiredService<IGpuBackendRegistry>();
        await registry.InitializeAsync();
        
        var selector = serviceProvider.GetRequiredService<IGpuBridgeProviderSelector>();
        var criteria = new ProviderSelectionCriteria { PreferGpu = true };
        var provider = await selector.SelectProviderAsync(criteria);
        
        // Assert - Should fall back to CPU when GPU provider is unavailable
        Assert.NotNull(provider);
        Assert.Equal("CpuFallback", provider.ProviderId);
    }

    [Fact]
    public async Task ProviderLifecycle_InitializeAndDispose_WorksCorrectly()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        
        services.AddGpuBridge()
            .AddBackendProvider<TestGpuProvider>();

        var serviceProvider = services.BuildServiceProvider();
        
        // Act & Assert - Initialize
        var registry = serviceProvider.GetRequiredService<IGpuBackendRegistry>();
        await registry.InitializeAsync();
        
        var provider = await registry.GetProviderByIdAsync("TestGpu");
        Assert.NotNull(provider);
        Assert.True(provider.IsAvailable());
        
        // Test components are accessible
        var deviceManager = provider.GetDeviceManager();
        Assert.NotNull(deviceManager);
        
        // Act & Assert - Dispose
        var act = () => serviceProvider.Dispose();
        act.Should().NotThrow();
    }
}

/// <summary>
/// Test provider that is always unavailable
/// </summary>
internal class UnavailableTestProvider : IGpuBackendProvider
{
    public string ProviderId => "UnavailableTest";
    public string DisplayName => "Unavailable Test Provider";
    public Version Version => new Version(1, 0, 0);
    public BackendCapabilities Capabilities => BackendCapabilities.CreateILGPU();

    public bool IsAvailable() => false; // Always unavailable

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(false);
    }

    public Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default)
    {
        var metrics = new Dictionary<string, object>
        {
            ["IsAvailable"] = false,
            ["LastError"] = "Provider is not available"
        };
        return Task.FromResult<IReadOnlyDictionary<string, object>>(metrics);
    }

    public Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        throw new InvalidOperationException("Provider is not available");
    }

    public IDeviceManager GetDeviceManager() => throw new InvalidOperationException("Provider not initialized");
    public IKernelCompiler GetKernelCompiler() => throw new InvalidOperationException("Provider not initialized");
    public IMemoryAllocator GetMemoryAllocator() => throw new InvalidOperationException("Provider not initialized");
    public IKernelExecutor GetKernelExecutor() => throw new InvalidOperationException("Provider not initialized");
    public ICommandQueue GetDefaultCommandQueue() => throw new InvalidOperationException("Provider not initialized");

    public Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new HealthCheckResult(false, "Provider is not available"));
    }
    
    public Task<object> CreateContext(int deviceIndex = 0)
    {
        throw new InvalidOperationException("Provider is not available");
    }

    public void Dispose() { }
}

/// <summary>
/// Test implementation for GPU execution requirements
/// </summary>
internal class TestGpuExecutionRequirements
{
    public bool PreferGpu { get; set; }
    public long MinimumMemoryBytes { get; set; } = 0;
    public BackendCapabilities? RequiredCapabilities { get; set; }
    public List<GpuBackend> PreferredBackends { get; set; } = new();
    public int MaxDevices { get; set; } = 1;
}