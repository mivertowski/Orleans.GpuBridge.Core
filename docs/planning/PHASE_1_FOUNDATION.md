# Phase 1: Foundation Implementation Specification

## Duration: Weeks 1-2

## Objectives
Establish the foundational infrastructure for Orleans.GpuBridge with complete CPU fallback implementation, preparing for GPU integration in later phases.

## Week 1: Project Infrastructure & Core Abstractions

### Day 1-2: Project Setup

#### Build Configuration
```xml
<!-- Directory.Build.props -->
<Project>
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <EnableAOT>true</EnableAOT>
  </PropertyGroup>
</Project>
```

#### Solution Structure
```bash
dotnet new sln -n Orleans.GpuBridge
dotnet new classlib -n Orleans.GpuBridge.Abstractions -o src/Orleans.GpuBridge.Abstractions
dotnet new classlib -n Orleans.GpuBridge.Runtime -o src/Orleans.GpuBridge.Runtime
dotnet new classlib -n Orleans.GpuBridge.BridgeFX -o src/Orleans.GpuBridge.BridgeFX
dotnet new classlib -n Orleans.GpuBridge.Grains -o src/Orleans.GpuBridge.Grains
dotnet new xunit -n Orleans.GpuBridge.Tests -o tests/Orleans.GpuBridge.Tests
```

#### Package References
```xml
<!-- Orleans.GpuBridge.Abstractions.csproj -->
<ItemGroup>
  <PackageReference Include="Microsoft.Orleans.Core.Abstractions" Version="8.0.0" />
  <PackageReference Include="Microsoft.Extensions.Options" Version="9.0.0" />
</ItemGroup>

<!-- Orleans.GpuBridge.Runtime.csproj -->
<ItemGroup>
  <PackageReference Include="Microsoft.Orleans.Core" Version="8.0.0" />
  <PackageReference Include="Microsoft.Extensions.Hosting" Version="9.0.0" />
  <PackageReference Include="System.Threading.Channels" Version="9.0.0" />
</ItemGroup>
```

### Day 3-4: Core Abstractions Implementation

#### IGpuBridge Interface
```csharp
namespace Orleans.GpuBridge.Abstractions;

public interface IGpuBridge
{
    ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default);
    ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId, 
        CancellationToken ct = default) where TIn : notnull where TOut : notnull;
    ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken ct = default);
}

public sealed record GpuBridgeInfo(
    string Version,
    int DeviceCount,
    long TotalMemoryBytes,
    GpuBackend Backend,
    bool IsGpuAvailable);

public enum GpuBackend
{
    Cpu,
    Cuda,
    OpenCL,
    DirectCompute,
    Metal
}
```

#### IGpuKernel Interface
```csharp
public interface IGpuKernel<TIn, TOut> where TIn : notnull where TOut : notnull
{
    ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default);
    
    IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        CancellationToken ct = default);
    
    ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default);
}

public sealed record KernelHandle(string Id, DateTimeOffset SubmittedAt);
public sealed record KernelInfo(
    KernelId Id,
    string Description,
    Type InputType,
    Type OutputType,
    bool SupportsGpu,
    int PreferredBatchSize);
```

#### Memory Abstractions
```csharp
public interface IGpuMemory<T> : IDisposable where T : unmanaged
{
    int Length { get; }
    Memory<T> AsMemory();
    ValueTask CopyToDeviceAsync(CancellationToken ct = default);
    ValueTask CopyFromDeviceAsync(CancellationToken ct = default);
}

public interface IGpuMemoryPool<T> where T : unmanaged
{
    IGpuMemory<T> Rent(int minimumLength);
    void Return(IGpuMemory<T> memory);
}
```

### Day 5: Kernel Registration System

#### KernelDescriptor & Builder
```csharp
public sealed class KernelDescriptor
{
    public required KernelId Id { get; init; }
    public required Type InputType { get; init; }
    public required Type OutputType { get; init; }
    public string? Description { get; init; }
    public int PreferredBatchSize { get; init; } = 1024;
    public Func<IServiceProvider, object>? Factory { get; init; }
    
    public class Builder
    {
        private readonly KernelDescriptor _descriptor = new();
        
        public Builder Id(string id)
        {
            _descriptor.Id = new KernelId(id);
            return this;
        }
        
        public Builder Input<TIn>() where TIn : notnull
        {
            _descriptor.InputType = typeof(TIn);
            return this;
        }
        
        public Builder Output<TOut>() where TOut : notnull
        {
            _descriptor.OutputType = typeof(TOut);
            return this;
        }
        
        public Builder WithFactory<TKernel>(Func<IServiceProvider, TKernel> factory)
            where TKernel : class
        {
            _descriptor.Factory = sp => factory(sp);
            return this;
        }
        
        public Builder WithBatchSize(int size)
        {
            _descriptor.PreferredBatchSize = size;
            return this;
        }
        
        public KernelDescriptor Build() => _descriptor;
    }
}
```

## Week 2: Runtime Implementation & CPU Fallback

### Day 6-7: Service Registration

#### ServiceCollectionExtensions
```csharp
public static class ServiceCollectionExtensions
{
    public static IGpuBridgeBuilder AddGpuBridge(
        this IServiceCollection services,
        Action<GpuBridgeOptions>? configure = null)
    {
        services.Configure<GpuBridgeOptions>(configure ?? (_ => { }));
        
        // Core services
        services.AddSingleton<IGpuBridge, GpuBridge>();
        services.AddSingleton<KernelCatalog>();
        services.AddSingleton<DeviceBroker>();
        services.AddSingleton<PersistentKernelHost>();
        services.AddHostedService<GpuHostFeature>();
        
        // Memory management
        services.AddSingleton(typeof(IGpuMemoryPool<>), typeof(CpuMemoryPool<>));
        
        // Diagnostics
        services.AddSingleton<IGpuDiagnostics, GpuDiagnostics>();
        
        return new GpuBridgeBuilder(services);
    }
}

public interface IGpuBridgeBuilder
{
    IGpuBridgeBuilder AddKernel(Action<KernelDescriptor.Builder> configure);
    IGpuBridgeBuilder AddKernel<TKernel>() where TKernel : class;
    IGpuBridgeBuilder ConfigureOptions(Action<GpuBridgeOptions> configure);
}

internal class GpuBridgeBuilder : IGpuBridgeBuilder
{
    private readonly IServiceCollection _services;
    
    public GpuBridgeBuilder(IServiceCollection services) => _services = services;
    
    public IGpuBridgeBuilder AddKernel(Action<KernelDescriptor.Builder> configure)
    {
        var builder = new KernelDescriptor.Builder();
        configure(builder);
        var descriptor = builder.Build();
        
        _services.Configure<KernelCatalogOptions>(o => 
            o.Descriptors.Add(descriptor));
        
        return this;
    }
    
    public IGpuBridgeBuilder AddKernel<TKernel>() where TKernel : class
    {
        _services.AddTransient<TKernel>();
        // Auto-registration logic based on attributes
        return this;
    }
    
    public IGpuBridgeBuilder ConfigureOptions(Action<GpuBridgeOptions> configure)
    {
        _services.Configure(configure);
        return this;
    }
}
```

### Day 8-9: CPU Fallback Kernels

#### Base CPU Kernel Implementation
```csharp
public abstract class CpuKernelBase<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull where TOut : notnull
{
    private readonly ILogger _logger;
    private readonly Channel<WorkItem> _workChannel;
    
    protected CpuKernelBase(ILogger logger)
    {
        _logger = logger;
        _workChannel = Channel.CreateUnbounded<WorkItem>();
        _ = ProcessWorkItemsAsync();
    }
    
    public virtual ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var handle = new KernelHandle(
            Guid.NewGuid().ToString("N"),
            DateTimeOffset.UtcNow);
        
        var workItem = new WorkItem(handle, items, hints);
        
        if (!_workChannel.Writer.TryWrite(workItem))
        {
            throw new InvalidOperationException("Failed to queue work item");
        }
        
        return ValueTask.FromResult(handle);
    }
    
    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        // Wait for results to be available
        while (!_results.TryGetValue(handle.Id, out var results))
        {
            await Task.Delay(10, ct);
        }
        
        foreach (var result in results)
        {
            yield return result;
        }
    }
    
    protected abstract Task<IReadOnlyList<TOut>> ExecuteBatchAsync(
        IReadOnlyList<TIn> items,
        CancellationToken ct);
    
    private async Task ProcessWorkItemsAsync()
    {
        await foreach (var item in _workChannel.Reader.ReadAllAsync())
        {
            try
            {
                var results = await ExecuteBatchAsync(item.Items, CancellationToken.None);
                _results[item.Handle.Id] = results;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process batch");
                _results[item.Handle.Id] = Array.Empty<TOut>();
            }
        }
    }
    
    private record WorkItem(
        KernelHandle Handle,
        IReadOnlyList<TIn> Items,
        GpuExecutionHints? Hints);
    
    private readonly ConcurrentDictionary<string, IReadOnlyList<TOut>> _results = new();
}
```

#### Sample CPU Kernels
```csharp
// Vector operations
public class CpuVectorAddKernel : CpuKernelBase<VectorPair, float[]>
{
    protected override Task<IReadOnlyList<float[]>> ExecuteBatchAsync(
        IReadOnlyList<VectorPair> items,
        CancellationToken ct)
    {
        var results = new float[items.Count][];
        
        Parallel.For(0, items.Count, i =>
        {
            var pair = items[i];
            var len = Math.Min(pair.A.Length, pair.B.Length);
            var result = new float[len];
            
            // SIMD optimized
            var simdLength = Vector<float>.Count;
            var j = 0;
            
            for (; j <= len - simdLength; j += simdLength)
            {
                var va = new Vector<float>(pair.A, j);
                var vb = new Vector<float>(pair.B, j);
                (va + vb).CopyTo(result, j);
            }
            
            // Remainder
            for (; j < len; j++)
            {
                result[j] = pair.A[j] + pair.B[j];
            }
            
            results[i] = result;
        });
        
        return Task.FromResult<IReadOnlyList<float[]>>(results);
    }
}

public record VectorPair(float[] A, float[] B);

// Matrix operations
public class CpuMatrixMultiplyKernel : CpuKernelBase<MatrixPair, float[,]>
{
    protected override Task<IReadOnlyList<float[,]>> ExecuteBatchAsync(
        IReadOnlyList<MatrixPair> items,
        CancellationToken ct)
    {
        var results = new float[items.Count][,];
        
        Parallel.For(0, items.Count, i =>
        {
            var pair = items[i];
            results[i] = MultiplyMatrices(pair.A, pair.B);
        });
        
        return Task.FromResult<IReadOnlyList<float[,]>>(results);
    }
    
    private static float[,] MultiplyMatrices(float[,] a, float[,] b)
    {
        // Optimized matrix multiplication with cache blocking
        // Implementation details...
    }
}
```

### Day 10: Testing Framework

#### Unit Tests
```csharp
public class KernelCatalogTests
{
    [Fact]
    public async Task Should_Register_And_Resolve_Kernel()
    {
        var services = new ServiceCollection();
        services.AddGpuBridge()
            .AddKernel(k => k
                .Id("test/add")
                .Input<float[]>()
                .Output<float>()
                .WithFactory(sp => new TestKernel()));
        
        var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<KernelCatalog>();
        
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("test/add"), provider);
        
        Assert.NotNull(kernel);
    }
    
    [Fact]
    public async Task Should_Fallback_To_Cpu_When_Kernel_Not_Found()
    {
        var services = new ServiceCollection();
        services.AddGpuBridge();
        
        var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<KernelCatalog>();
        
        var kernel = await catalog.ResolveAsync<float[], float>(
            new KernelId("unknown"), provider);
        
        Assert.IsType<CpuPassthroughKernel<float[], float>>(kernel);
    }
}

public class CpuKernelTests
{
    [Fact]
    public async Task VectorAdd_Should_Compute_Correctly()
    {
        var kernel = new CpuVectorAddKernel(
            NullLogger<CpuVectorAddKernel>.Instance);
        
        var input = new[]
        {
            new VectorPair(
                new[] { 1f, 2f, 3f },
                new[] { 4f, 5f, 6f })
        };
        
        var handle = await kernel.SubmitBatchAsync(input);
        var results = await kernel.ReadResultsAsync(handle).ToListAsync();
        
        Assert.Single(results);
        Assert.Equal(new[] { 5f, 7f, 9f }, results[0]);
    }
    
    [Theory]
    [InlineData(100)]
    [InlineData(1000)]
    [InlineData(10000)]
    public async Task Should_Handle_Large_Batches(int batchSize)
    {
        var kernel = new CpuVectorAddKernel(
            NullLogger<CpuVectorAddKernel>.Instance);
        
        var input = Enumerable.Range(0, batchSize)
            .Select(i => new VectorPair(
                new[] { i * 1f },
                new[] { i * 2f }))
            .ToList();
        
        var handle = await kernel.SubmitBatchAsync(input);
        var results = await kernel.ReadResultsAsync(handle).ToListAsync();
        
        Assert.Equal(batchSize, results.Count);
    }
}
```

## Deliverables Checklist

### Week 1 Deliverables
- [ ] Solution and project files created
- [ ] NuGet package references configured
- [ ] Core abstractions (IGpuBridge, IGpuKernel) implemented
- [ ] Memory abstractions defined
- [ ] Kernel registration system implemented
- [ ] Basic unit test project setup

### Week 2 Deliverables
- [ ] Service registration and DI complete
- [ ] KernelCatalog with resolution logic
- [ ] CPU fallback base class implemented
- [ ] Sample CPU kernels (VectorAdd, MatrixMultiply)
- [ ] Unit tests for all components
- [ ] CI/CD pipeline configuration

## Success Metrics

### Code Quality
- 100% unit test coverage for public APIs
- Zero compiler warnings
- All code analyzers passing
- Documentation comments on all public types

### Performance Baseline
- CPU kernel latency < 10ms for 1K batch
- Memory allocation < 100KB per kernel invocation
- Successful parallel execution of 100 concurrent kernels

### Integration Readiness
- Clean separation of abstractions and implementation
- All extension points identified and documented
- Ready for Orleans integration in Phase 2

## Known Issues & Risks

### Technical Debt
- CPU kernels are not optimized for NUMA architectures
- No memory pressure handling yet
- Missing telemetry and metrics collection

### Mitigation Plan
- Document all shortcuts taken for later improvement
- Create issues for technical debt items
- Maintain list of optimization opportunities

## Next Phase Preview

Phase 2 will build upon this foundation to add:
- Orleans grain implementations
- Custom placement strategy
- Streaming support
- Integration with Orleans lifecycle

The foundation established in Phase 1 ensures all Orleans-specific features can be cleanly added without refactoring core abstractions.