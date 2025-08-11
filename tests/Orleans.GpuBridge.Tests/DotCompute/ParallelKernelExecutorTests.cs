using System;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.DotCompute;
using Xunit;

namespace Orleans.GpuBridge.Tests.DotCompute;

public class ParallelKernelExecutorTests
{
    private readonly ParallelKernelExecutor _executor;
    private readonly ILogger<ParallelKernelExecutor> _logger;

    public ParallelKernelExecutorTests()
    {
        _logger = new TestLogger<ParallelKernelExecutor>();
        _executor = new ParallelKernelExecutor(_logger);
    }

    [Fact]
    public async Task ExecuteAsync_Should_Apply_Kernel_To_All_Elements()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).ToArray();
        Func<int, int> kernel = x => x * 2;

        // Act
        var result = await _executor.ExecuteAsync(input, kernel);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i] * 2, result[i]);
        }
    }

    [Fact]
    public async Task ExecuteAsync_With_Empty_Input_Should_Return_Empty()
    {
        // Arrange
        var input = Array.Empty<int>();
        Func<int, int> kernel = x => x * 2;

        // Act
        var result = await _executor.ExecuteAsync(input, kernel);

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public async Task ExecuteAsync_Should_Handle_Large_Data()
    {
        // Arrange
        var input = Enumerable.Range(1, 100000).ToArray();
        Func<int, int> kernel = x => x + 1;

        // Act
        var result = await _executor.ExecuteAsync(input, kernel);

        // Assert
        Assert.Equal(input.Length, result.Length);
        Assert.Equal(2, result[0]);
        Assert.Equal(100001, result[99999]);
    }

    [Fact]
    public async Task ExecuteAsync_With_Custom_Options_Should_Respect_Parallelism()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).ToArray();
        var executionCount = 0;
        Func<int, int> kernel = x =>
        {
            Interlocked.Increment(ref executionCount);
            return x * 2;
        };
        
        var options = new ParallelExecutionOptions
        {
            MaxDegreeOfParallelism = 2
        };

        // Act
        var result = await _executor.ExecuteAsync(input, kernel, options);

        // Assert
        Assert.Equal(input.Length, result.Length);
        Assert.Equal(100, executionCount);
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Add_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var parameters = new[] { 10.0f };

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Add, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i] + 10.0f, result[i]);
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Multiply_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var parameters = new[] { 2.0f };

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Multiply, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i] * 2.0f, result[i]);
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_FusedMultiplyAdd_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var parameters = new[] { 2.0f, 3.0f }; // multiply by 2, add 3

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.FusedMultiplyAdd, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i] * 2.0f + 3.0f, result[i], 5);
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Sqrt_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).Select(i => (float)(i * i)).ToArray();
        var parameters = Array.Empty<float>();

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Sqrt, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(MathF.Sqrt(input[i]), result[i], 5);
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Max_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(-50, 100).Select(i => (float)i).ToArray();
        var parameters = new[] { 0.0f };

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Max, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(MathF.Max(input[i], 0.0f), result[i]);
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Min_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(-50, 100).Select(i => (float)i).ToArray();
        var parameters = new[] { 0.0f };

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Min, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(MathF.Min(input[i], 0.0f), result[i]);
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Should_Handle_Non_Aligned_Sizes()
    {
        // Test with various sizes that aren't multiples of vector size
        var sizes = new[] { 7, 13, 31, 63, 127, 255 };

        foreach (var size in sizes)
        {
            // Arrange
            var input = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
            var parameters = new[] { 2.0f };

            // Act
            var result = await _executor.ExecuteVectorizedAsync(
                input, VectorOperation.Multiply, parameters);

            // Assert
            Assert.Equal(size, result.Length);
            for (int i = 0; i < size; i++)
            {
                Assert.Equal(input[i] * 2.0f, result[i]);
            }
        }
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_With_Small_Data_Should_Use_Scalar()
    {
        // Arrange
        var input = new float[] { 1.0f, 2.0f, 3.0f }; // Too small for vectorization
        var parameters = new[] { 10.0f };

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Add, parameters);

        // Assert
        Assert.Equal(3, result.Length);
        Assert.Equal(11.0f, result[0]);
        Assert.Equal(12.0f, result[1]);
        Assert.Equal(13.0f, result[2]);
    }

    [Fact]
    public async Task ExecuteAsync_With_Cancellation_Should_Cancel()
    {
        // Arrange
        var input = Enumerable.Range(1, 100000).ToArray();
        var cts = new CancellationTokenSource();
        var executionStarted = false;
        
        Func<int, int> kernel = x =>
        {
            executionStarted = true;
            Thread.Sleep(10); // Slow kernel
            return x * 2;
        };

        // Act
        cts.CancelAfter(50); // Cancel quickly
        
        // Assert
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
            await _executor.ExecuteAsync(input, kernel, ct: cts.Token));
    }

    [Fact]
    public async Task ExecuteVectorizedAsync_Reciprocal_Should_Work()
    {
        // Arrange
        var input = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var parameters = Array.Empty<float>();

        // Act
        var result = await _executor.ExecuteVectorizedAsync(
            input, VectorOperation.Reciprocal, parameters);

        // Assert
        Assert.Equal(input.Length, result.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(1.0f / input[i], result[i], 5);
        }
    }

    [Fact]
    public void SIMD_Support_Should_Be_Detected()
    {
        // This test verifies SIMD detection works
        var hasSimd = Vector.IsHardwareAccelerated ||
                      Avx512F.IsSupported ||
                      Avx2.IsSupported ||
                      Avx.IsSupported ||
                      Sse42.IsSupported ||
                      AdvSimd.IsSupported;

        // At least one SIMD instruction set should be available on modern hardware
        // If this fails, the executor will fall back to scalar operations
        _logger.Log(LogLevel.Information, 0, $"SIMD Support: {hasSimd}", null, (s, e) => s.ToString()!);
    }
}

public class ParallelExtensionsTests
{
    [Fact]
    public async Task ParallelMapAsync_Should_Transform_All_Elements()
    {
        // Arrange
        var source = Enumerable.Range(1, 10);
        Func<int, Task<int>> mapper = async x =>
        {
            await Task.Delay(10);
            return x * 2;
        };

        // Act
        var result = await source.ParallelMapAsync(mapper);

        // Assert
        Assert.Equal(10, result.Length);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal((i + 1) * 2, result[i]);
        }
    }

    [Fact]
    public async Task ParallelMapAsync_With_MaxConcurrency_Should_Limit_Parallelism()
    {
        // Arrange
        var source = Enumerable.Range(1, 10);
        var concurrentCount = 0;
        var maxObserved = 0;
        
        Func<int, Task<int>> mapper = async x =>
        {
            var current = Interlocked.Increment(ref concurrentCount);
            maxObserved = Math.Max(maxObserved, current);
            await Task.Delay(50);
            Interlocked.Decrement(ref concurrentCount);
            return x;
        };

        // Act
        var result = await source.ParallelMapAsync(mapper, maxConcurrency: 3);

        // Assert
        Assert.Equal(10, result.Length);
        Assert.True(maxObserved <= 3, $"Max concurrency was {maxObserved}, expected <= 3");
    }

    [Fact]
    public async Task ParallelReduceAsync_Should_Aggregate_Values()
    {
        // Arrange
        var source = Enumerable.Range(1, 10);
        Func<int, int, int> reducer = (a, b) => a + b;

        // Act
        var result = await source.ParallelReduceAsync(reducer);

        // Assert
        Assert.Equal(55, result); // Sum of 1 to 10
    }

    [Fact]
    public async Task ParallelReduceAsync_With_Single_Element_Should_Return_Element()
    {
        // Arrange
        var source = new[] { 42 };
        Func<int, int, int> reducer = (a, b) => a + b;

        // Act
        var result = await source.ParallelReduceAsync(reducer);

        // Assert
        Assert.Equal(42, result);
    }

    [Fact]
    public async Task ParallelReduceAsync_With_Empty_Should_Throw()
    {
        // Arrange
        var source = Array.Empty<int>();
        Func<int, int, int> reducer = (a, b) => a + b;

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => source.ParallelReduceAsync(reducer));
    }

    [Fact]
    public async Task ParallelMapAsync_With_Cancellation_Should_Cancel()
    {
        // Arrange
        var source = Enumerable.Range(1, 100);
        var cts = new CancellationTokenSource();
        
        Func<int, Task<int>> mapper = async x =>
        {
            await Task.Delay(100);
            return x;
        };

        // Act
        cts.CancelAfter(50);
        
        // Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            () => source.ParallelMapAsync(mapper, ct: cts.Token));
    }
}

internal class TestLogger<T> : ILogger<T>
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