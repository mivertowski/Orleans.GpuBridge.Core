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

public class ReductionKernelTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly KernelTemplateRegistry _registry;
    private readonly ILogger<KernelTemplateRegistry> _logger;

    public ReductionKernelTests()
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
    public async Task SumReduction_Should_Compute_Correctly()
    {
        // Arrange
        const int size = 1024;
        var input = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var expected = input.Sum();
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferOutput.MemSetToZero();
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/sum",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output[0].Should().BeApproximately(expected, expected * 0.001f); // 0.1% tolerance for floating point
    }

    [Fact]
    public async Task MaxReduction_Should_Find_Maximum()
    {
        // Arrange
        const int size = 512;
        var input = new float[size];
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            input[i] = (float)(random.NextDouble() * 1000 - 500);
        }
        input[random.Next(size)] = 999.99f; // Set a known maximum
        var expected = 999.99f;
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferOutput.CopyFromCPU(new[] { float.MinValue });
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/max",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output[0].Should().BeApproximately(expected, 0.01f);
    }

    [Fact]
    public async Task MinReduction_Should_Find_Minimum()
    {
        // Arrange
        const int size = 256;
        var input = new float[size];
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            input[i] = (float)(random.NextDouble() * 1000);
        }
        input[random.Next(size)] = -999.99f; // Set a known minimum
        var expected = -999.99f;
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferOutput.CopyFromCPU(new[] { float.MaxValue });
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/min",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output[0].Should().BeApproximately(expected, 0.01f);
    }

    [Fact]
    public async Task CountNonZero_Should_Count_Correctly()
    {
        // Arrange
        const int size = 100;
        var input = new float[size];
        var nonZeroCount = 0;
        var random = new Random(42);
        
        for (int i = 0; i < size; i++)
        {
            if (random.NextDouble() > 0.3) // ~70% non-zero
            {
                input[i] = (float)(random.NextDouble() * 100);
                nonZeroCount++;
            }
        }
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<int>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferOutput.MemSetToZero();
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/count_nonzero",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        var output = bufferOutput.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        output[0].Should().Be(nonZeroCount);
    }

    [Fact]
    public async Task AverageReduction_Should_Compute_Mean()
    {
        // Arrange
        const int size = 128;
        var input = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var expected = input.Average();
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferSumOutput = _accelerator.Allocate1D<float>(1);
        using var bufferCountOutput = _accelerator.Allocate1D<int>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferSumOutput.MemSetToZero();
        bufferCountOutput.MemSetToZero();
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferSumOutput.View);
        args.Add(bufferCountOutput.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/average",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        var sum = bufferSumOutput.GetAsArray1D()[0];
        var count = bufferCountOutput.GetAsArray1D()[0];
        var average = sum / count;
        
        // Assert
        result.Success.Should().BeTrue();
        average.Should().BeApproximately(expected, 0.001f);
    }

    [Fact]
    public async Task Histogram_Should_Compute_Distribution()
    {
        // Arrange
        const int size = 1000;
        const int numBins = 10;
        const float minValue = 0.0f;
        const float maxValue = 100.0f;
        
        var input = new float[size];
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            input[i] = (float)(random.NextDouble() * 100);
        }
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferHistogram = _accelerator.Allocate1D<int>(numBins);
        
        bufferInput.CopyFromCPU(input);
        bufferHistogram.MemSetToZero();
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferHistogram.View);
        args.Add(minValue);
        args.Add(maxValue);
        args.Add(numBins);
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/histogram",
            _accelerator,
            args,
            new KernelConfig(size, 1));
        
        var histogram = bufferHistogram.GetAsArray1D();
        
        // Assert
        result.Success.Should().BeTrue();
        histogram.Sum().Should().Be(size); // All values should be binned
        histogram.All(count => count > 0).Should().BeTrue(); // All bins should have some values (with uniform distribution)
    }

    [Fact]
    public async Task ArgMaxReduction_Should_Find_Index_Of_Maximum()
    {
        // Arrange
        const int size = 100;
        var input = new float[size];
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            input[i] = (float)(random.NextDouble() * 100);
        }
        
        var maxIndex = 42;
        input[maxIndex] = 999.99f; // Set a known maximum at a specific index
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferMaxValue = _accelerator.Allocate1D<float>(1);
        using var bufferMaxIndex = _accelerator.Allocate1D<int>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferMaxValue.CopyFromCPU(new[] { float.MinValue });
        bufferMaxIndex.CopyFromCPU(new[] { -1 });
        
        // Act
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferMaxValue.View);
        args.Add(bufferMaxIndex.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/argmax",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        var maxValue = bufferMaxValue.GetAsArray1D()[0];
        var foundIndex = bufferMaxIndex.GetAsArray1D()[0];
        
        // Assert
        result.Success.Should().BeTrue();
        maxValue.Should().BeApproximately(999.99f, 0.01f);
        foundIndex.Should().Be(maxIndex);
    }

    [Fact]
    public async Task StandardDeviation_Should_Compute_Correctly()
    {
        // Arrange
        const int size = 100;
        var input = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
        var mean = input.Average();
        var variance = input.Select(x => Math.Pow(x - mean, 2)).Average();
        var expectedStdDev = (float)Math.Sqrt(variance);
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferSumOutput = _accelerator.Allocate1D<float>(1);
        using var bufferCountOutput = _accelerator.Allocate1D<int>(1);
        using var bufferVarianceOutput = _accelerator.Allocate1D<float>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferSumOutput.MemSetToZero();
        bufferCountOutput.MemSetToZero();
        bufferVarianceOutput.MemSetToZero();
        
        // Act - Pass 1: Compute mean
        var args1 = new KernelArguments();
        args1.Add(bufferInput.View);
        args1.Add(bufferSumOutput.View);
        args1.Add(bufferCountOutput.View);
        args1.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        await _registry.ExecuteTemplateAsync(
            "reduction/stddev_pass1",
            _accelerator,
            args1,
            new KernelConfig(gridSize, blockSize));
        
        var computedMean = bufferSumOutput.GetAsArray1D()[0] / bufferCountOutput.GetAsArray1D()[0];
        
        // Pass 2: Compute variance
        var args2 = new KernelArguments();
        args2.Add(bufferInput.View);
        args2.Add(bufferVarianceOutput.View);
        args2.Add(computedMean);
        args2.Add(new SpecializedValue<int>(256));
        
        await _registry.ExecuteTemplateAsync(
            "reduction/stddev_pass2",
            _accelerator,
            args2,
            new KernelConfig(gridSize, blockSize));
        
        var computedVariance = bufferVarianceOutput.GetAsArray1D()[0] / size;
        var computedStdDev = (float)Math.Sqrt(computedVariance);
        
        // Assert
        computedMean.Should().BeApproximately(mean, 0.001f);
        computedStdDev.Should().BeApproximately(expectedStdDev, expectedStdDev * 0.01f); // 1% tolerance
    }

    [Fact]
    public async Task LargeReduction_Performance_Test()
    {
        // Arrange
        const int size = 10_000_000;
        var input = new float[size];
        var random = new Random(42);
        
        for (int i = 0; i < size; i++)
        {
            input[i] = (float)random.NextDouble();
        }
        
        using var bufferInput = _accelerator.Allocate1D<float>(size);
        using var bufferOutput = _accelerator.Allocate1D<float>(1);
        
        bufferInput.CopyFromCPU(input);
        bufferOutput.MemSetToZero();
        
        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        var args = new KernelArguments();
        args.Add(bufferInput.View);
        args.Add(bufferOutput.View);
        args.Add(new SpecializedValue<int>(256));
        
        var blockSize = 256;
        var gridSize = (size + blockSize - 1) / blockSize;
        
        var result = await _registry.ExecuteTemplateAsync(
            "reduction/sum",
            _accelerator,
            args,
            new KernelConfig(gridSize, blockSize));
        
        stopwatch.Stop();
        
        var output = bufferOutput.GetAsArray1D();
        var expected = input.Sum();
        
        // Assert
        result.Success.Should().BeTrue();
        stopwatch.ElapsedMilliseconds.Should().BeLessThan(1000); // Should complete within 1 second
        output[0].Should().BeApproximately(expected, expected * 0.01f); // 1% tolerance for large sums
    }

    public void Dispose()
    {
        _registry?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}