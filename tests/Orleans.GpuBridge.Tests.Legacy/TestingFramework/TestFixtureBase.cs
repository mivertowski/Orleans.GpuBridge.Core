using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AutoFixture;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Log entry for test logging
/// </summary>
public record LogEntry(LogLevel LogLevel, EventId EventId, string Message, Exception? Exception);

/// <summary>
/// Base class for test fixtures providing common services and utilities
/// </summary>
public abstract class TestFixtureBase : IDisposable
{
    private readonly ServiceProvider _serviceProvider;
    private readonly CancellationTokenSource _cancellationTokenSource;
    protected readonly IFixture Fixture;

    protected TestFixtureBase()
    {
        Fixture = new Fixture();
        ConfigureFixture(Fixture);

        var services = new ServiceCollection();
        ConfigureServices(services);
        
        _serviceProvider = services.BuildServiceProvider();
        _cancellationTokenSource = new CancellationTokenSource(TimeSpan.FromMinutes(5)); // Default test timeout
    }

    protected virtual void ConfigureFixture(IFixture fixture)
    {
        // Configure common test data generation
        fixture.Customize(new GpuBridgeCustomization());
    }

    protected virtual void ConfigureServices(IServiceCollection services)
    {
        services.AddLogging(builder => builder
            .AddConsole()
            .SetMinimumLevel(LogLevel.Debug));

        services.Configure<GpuBridgeOptions>(options =>
        {
            options.PreferGpu = false; // CPU fallback for tests
            options.MemoryPoolSizeMB = 128; // Small pool for tests
            options.Telemetry = new TelemetryOptions
            {
                EnableMetrics = true,
                EnableTracing = false // Disable for tests
            };
        });

        services.AddSingleton<TestLogger>();
        services.AddScoped<MockGpuDeviceFactory>();
        services.AddGpuBridge();
    }

    protected T GetService<T>() where T : notnull => _serviceProvider.GetRequiredService<T>();
    protected IServiceScope CreateScope() => _serviceProvider.CreateScope();
    protected CancellationToken CancellationToken => _cancellationTokenSource.Token;

    public virtual void Dispose()
    {
        _cancellationTokenSource?.Cancel();
        _serviceProvider?.Dispose();
        _cancellationTokenSource?.Dispose();
    }
}

/// <summary>
/// AutoFixture customization for GPU Bridge types
/// </summary>
public class GpuBridgeCustomization : ICustomization
{
    public void Customize(IFixture fixture)
    {
        // Configure KernelId generation
        fixture.Register(() => new KernelId(fixture.Create<string>()));

        // Configure device type generation
        fixture.Register(() => fixture.Create<Generator<DeviceType>>().First());

        // Configure memory size generation (reasonable sizes for tests)
        fixture.Register(() => (long)fixture.Create<Random>().Next(1024, 1024 * 1024 * 10)); // 1KB to 10MB

        // Configure GPU execution hints
        fixture.Register(() => new GpuExecutionHints
        {
            PreferredBatchSize = fixture.Create<int>() % 2048 + 1,
            PreferGpu = fixture.Create<bool>(),
            TimeoutMs = fixture.Create<int>() % 30000 + 1000
        });

        // Configure float arrays with reasonable sizes
        fixture.Register(() => fixture.CreateMany<float>(fixture.Create<int>() % 1000 + 1).ToArray());

        // Configure compute capabilities
        fixture.Register(() => fixture.CreateMany<string>(3).ToArray());
    }
}

/// <summary>
/// Test logger that captures log entries for assertions
/// </summary>
public class TestLogger : ILogger
{
    private readonly List<LogEntry> _logEntries = new();
    public IReadOnlyList<LogEntry> LogEntries => _logEntries;

    public IDisposable BeginScope<TState>(TState state) where TState : notnull
        => new NoopDisposable();

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state,
        Exception? exception, Func<TState, Exception?, string> formatter)
    {
        _logEntries.Add(new LogEntry(
            logLevel,
            eventId,
            formatter(state, exception),
            exception));
    }

    public void Clear() => _logEntries.Clear();

    public bool HasLogLevel(LogLevel level) => _logEntries.Any(e => e.LogLevel == level);
    public bool HasException<T>() where T : Exception => _logEntries.Any(e => e.Exception is T);

    public record LogEntry(LogLevel LogLevel, EventId EventId, string Message, Exception? Exception);

    private class NoopDisposable : IDisposable
    {
        public void Dispose() { }
    }
}

/// <summary>
/// Generic test logger for typed logging
/// </summary>
public class TestLogger<T> : ILogger<T>
{
    private readonly TestLogger _testLogger;

    public TestLogger(TestLogger testLogger)
    {
        _testLogger = testLogger;
    }
    
    public TestLogger() : this(new TestLogger()) 
    {
    }
    
    public IReadOnlyList<LogEntry> LogEntries => _testLogger.LogEntries;
    
    public IEnumerable<(LogLevel LogLevel, string Message)> LoggedMessages => 
        _testLogger.LogEntries.Select(entry => (entry.LogLevel, entry.Message));

    public IDisposable BeginScope<TState>(TState state) where TState : notnull
        => _testLogger.BeginScope(state);

    public bool IsEnabled(LogLevel logLevel) => _testLogger.IsEnabled(logLevel);

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state,
        Exception? exception, Func<TState, Exception?, string> formatter)
        => _testLogger.Log(logLevel, eventId, state, exception, formatter);
}

/// <summary>
/// Factory for creating mock GPU devices for testing
/// </summary>
public class MockGpuDeviceFactory
{
    private readonly IFixture _fixture;

    public MockGpuDeviceFactory(IFixture fixture)
    {
        _fixture = fixture;
    }

    public IComputeDevice CreateMockDevice(Orleans.GpuBridge.Abstractions.Enums.DeviceType deviceType = Orleans.GpuBridge.Abstractions.Enums.DeviceType.GPU, int index = 0)
    {
        var device = new TestComputeDevice
        {
            DeviceId = Guid.NewGuid().ToString(),
            Index = index,
            Name = $"Mock {deviceType} Device {index}",
            Type = deviceType,
            Vendor = "Test Vendor",
            Architecture = "Test Architecture",
            ComputeCapability = new Version(7, 5),
            TotalMemoryBytes = _fixture.Create<long>(),
            AvailableMemoryBytes = _fixture.Create<long>(),
            ComputeUnits = _fixture.Create<int>() % 64 + 1,
            MaxClockFrequencyMHz = 1500,
            MaxThreadsPerBlock = 1024,
            MaxWorkGroupDimensions = new[] { 1024, 1024, 64 },
            WarpSize = 32,
            Properties = new Dictionary<string, object> { ["Capabilities"] = _fixture.Create<string[]>() }
        };
        return device;
    }

    public List<IComputeDevice> CreateMockDevices(int count = 3)
    {
        var devices = new List<IComputeDevice>();
        
        // Always include CPU device
        devices.Add(CreateMockDevice(DeviceType.CPU, 0));
        
        // Add GPU devices
        for (int i = 1; i < count; i++)
        {
            devices.Add(CreateMockDevice(DeviceType.GPU, i));
        }

        return devices;
    }
}