using System;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.ILGPU.Kernels;
using Orleans.GpuBridge.Backends.ILGPU.Kernels.Templates;
using Xunit;

namespace Orleans.GpuBridge.Tests.Kernels;

public class VectorOperationKernelTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly KernelTemplateRegistry _registry;
    private readonly ILogger<KernelTemplateRegistry> _logger;

    public VectorOperationKernelTests()
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole());
        var provider = services.BuildServiceProvider();
        
        _logger = provider.GetRequiredService<ILogger<KernelTemplateRegistry>>();
        _context = Context.CreateDefault();
        _accelerator = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);
        _registry = new KernelTemplateRegistry(_logger, _context);
    }

    [Fact]
    public async Task VectorAdd_Should_Compute_Correctly()
    {
        // Arrange
        const int size = 1024;
        var a = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(1, size).Select(i => (float)i * 2).ToArray();
        var expected = a.Zip(b, (x, y) => x + y).ToArray();
        
        using var bufferA = _accelerator.Allocate1D<float>(size);
        using var bufferB = _accelerator.Allocate1D<float>(size);
        using var bufferC = _accelerator.Allocate1D<float>(size);
        
        bufferA.CopyFromCPU(a);
        bufferB.CopyFromCPU(b);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferA.View);
        args.Add(bufferB.View);
        args.Add(bufferC.View);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/add",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferC.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options.WithStrictOrdering());
    }

    [Fact]
    public async Task VectorMultiply_Should_Compute_Correctly()
    {
        // Arrange
        const int size = 512;
        var a = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(1, size).Select(i => (float)i * 0.5f).ToArray();
        var expected = a.Zip(b, (x, y) => x * y).ToArray();
        
        using var bufferA = _accelerator.Allocate1D<float>(size);
        using var bufferB = _accelerator.Allocate1D<float>(size);
        using var bufferC = _accelerator.Allocate1D<float>(size);
        
        bufferA.CopyFromCPU(a);
        bufferB.CopyFromCPU(b);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferA.View);
        args.Add(bufferB.View);
        args.Add(bufferC.View);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/multiply",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferC.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options
            .WithStrictOrdering()
            .Using<float>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.0001f))
            .WhenTypeIs<float>());
    }

    [Fact]
    public async Task ScalarMultiply_Should_Scale_Vector()
    {
        // Arrange
        const int size = 256;
        const float scalar = 2.5f;
        var input = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var expected = input.Select(x => x * scalar).ToArray();
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(size);
        
        bufferInput.CopyFromCPU(input);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(scalar);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/scalar_multiply",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options
            .WithStrictOrdering()
            .Using<float>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.0001f))
            .WhenTypeIs<float>());
    }

    [Fact]
    public async Task VectorClamp_Should_Limit_Values()
    {
        // Arrange
        const int size = 100;
        const float minValue = -50.0f;
        const float maxValue = 50.0f;
        var input = Enumerable.Range(-100, 200).Select(i => (float)i).Take(size).ToArray();
        var expected = input.Select(x => Math.Max(minValue, Math.Min(maxValue, x))).ToArray();
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(size);
        
        bufferInput.CopyFromCPU(input);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(minValue);
        args.Add(maxValue);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/clamp",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options.WithStrictOrdering());
    }

    [Fact]
    public async Task VectorSqrt_Should_Compute_Square_Roots()
    {
        // Arrange
        const int size = 100;
        var input = Enumerable.Range(1, size).Select(i => (float)(i * i)).ToArray();
        var expected = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(size);
        
        bufferInput.CopyFromCPU(input);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/sqrt",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options
            .WithStrictOrdering()
            .Using<float>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.001f))
            .WhenTypeIs<float>());
    }

    [Fact]
    public async Task VectorExp_Should_Compute_Exponentials()
    {
        // Arrange
        const int size = 10;
        var input = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var expected = input.Select(x => (float)Math.Exp(x)).ToArray();
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(size);
        
        bufferInput.CopyFromCPU(input);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/exp",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        for (int i = 0; i < size; i++)
        {
            output[i].Should().BeApproximately(expected[i], expected[i] * 0.01f); // 1% tolerance
        }
    }

    [Fact]
    public async Task SAXPY_Should_Compute_Correctly()
    {
        // Arrange
        const int size = 128;
        const float alpha = 2.0f;
        var x = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var y = Enumerable.Range(1, size).Select(i => (float)i * 0.5f).ToArray();
        var expected = x.Zip(y, (xi, yi) => alpha * xi + yi).ToArray();
        
        using var bufferX = _accelerator.Allocate1D<float>(size);
        using var bufferY = _accelerator.Allocate1D<float>(size);
        
        bufferX.CopyFromCPU(x);
        bufferY.CopyFromCPU(y);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferX.View);
        args.Add(bufferY.View);
        args.Add(alpha);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/saxpy",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferY.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options
            .WithStrictOrdering()
            .Using<float>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.0001f))
            .WhenTypeIs<float>());
    }

    [Fact]
    public async Task VectorFill_Should_Set_All_Elements()
    {
        // Arrange
        const int size = 100;
        const float fillValue = 42.0f;
        var expected = Enumerable.Repeat(fillValue, size).ToArray();
        
        using var buffer = _accelerator.Allocate1D<float>(size);
        
        // Act
        var args = new KernelArguments();
        args.Add(buffer.View);
        args.Add(fillValue);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/fill",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = buffer.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected);
    }

    [Fact]
    public async Task FusedMultiplyAdd_Should_Compute_Correctly()
    {
        // Arrange
        const int size = 64;
        var a = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(1, size).Select(i => (float)i * 2).ToArray();
        var c = Enumerable.Range(1, size).Select(i => (float)i * 0.5f).ToArray();
        var expected = Enumerable.Range(0, size)
            .Select(i => a[i] * b[i] + c[i])
            .ToArray();
        
        using var bufferA = _accelerator.Allocate1D<float>(size);
        using var bufferB = _accelerator.Allocate1D<float>(size);
        using var bufferC = _accelerator.Allocate1D<float>(size);
        using var bufferD = _accelerator.Allocate1D<float>(size);
        
        bufferA.CopyFromCPU(a);
        bufferB.CopyFromCPU(b);
        bufferC.CopyFromCPU(c);
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferA.View);
        args.Add(bufferB.View);
        args.Add(bufferC.View);
        args.Add(bufferD.View);
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/fma",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var output = bufferD.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output.Should().BeEquivalentTo(expected, options => options
            .WithStrictOrdering()
            .Using<float>(ctx => ctx.Subject.Should().BeApproximately(ctx.Expectation, 0.0001f))
            .WhenTypeIs<float>());
    }

    [Fact]
    public async Task LargeVector_Should_Handle_Performance()
    {
        // Arrange
        const int size = 1_000_000;
        var a = new float[size];
        var b = new float[size];
        
        // Initialize with random values
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            a[i] = (float)random.NextDouble() * 100;
            b[i] = (float)random.NextDouble() * 100;
        }
        
        using var bufferA = _accelerator.Allocate1D<float>(size);
        using var bufferB = _accelerator.Allocate1D<float>(size);
        using var bufferC = _accelerator.Allocate1D<float>(size);
        
        bufferA.CopyFromCPU(a);
        bufferB.CopyFromCPU(b);
        
        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        var args = new KernelArguments();
        args.Add(bufferA.View);
        args.Add(bufferB.View);
        args.Add(bufferC.View);
        
        var blockSize = _accelerator.MaxNumThreadsPerGroup;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "vector/add",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        stopwatch.Stop();
        
        // Assert
        result.Success.Should().BeTrue();
        stopwatch.ElapsedMilliseconds.Should().BeLessThan(1000); // Should complete within 1 second
        
        // Verify a sample of results
        var output = bufferC.GetAsArray1D();
        for (int i = 0; i < 100; i++)
        {
            var idx = random.Next(size);
            output[idx].Should().BeApproximately(a[idx] + b[idx], 0.0001f);
        }
    }

    public void Dispose()
    {
        _registry?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}