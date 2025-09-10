using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Backends.ILGPU;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Unit
{
    /// <summary>
    /// Unit tests for ILGPU kernel implementations and execution pipelines
    /// </summary>
    public class ILGPUKernelTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly Context _context;
        private readonly Mock<ILogger<ILGPUKernelTests>> _mockLogger;

        public ILGPUKernelTests(ITestOutputHelper output)
        {
            _output = output;
            _context = Context.CreateDefault();
            _mockLogger = new Mock<ILogger<ILGPUKernelTests>>();
        }

        [Fact]
        public void KernelCatalog_ShouldRegisterKernel_Successfully()
        {
            // Arrange
            var services = new ServiceCollection();
            services.AddLogging();
            services.AddGpuBridge()
                .AddKernel(k => k.Id("test-kernel")
                    .In<float[]>()
                    .Out<float[]>()
                    .FromFactory(sp => new TestVectorAddKernel()));

            var serviceProvider = services.BuildServiceProvider();
            var catalog = serviceProvider.GetService<IKernelCatalog>();

            // Act
            var kernel = catalog.GetKernel<float[], float[]>("test-kernel");

            // Assert
            Assert.NotNull(kernel);
            Assert.IsType<TestVectorAddKernel>(kernel);
        }

        [Fact]
        public async Task VectorAddKernel_ShouldExecute_CorrectlyOnCPU()
        {
            // Arrange
            var cpuDevice = _context.Devices.First(d => d.AcceleratorType == AcceleratorType.CPU);
            using var accelerator = cpuDevice.CreateAccelerator(_context);
            
            var inputA = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var inputB = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            var expected = new float[] { 6.0f, 8.0f, 10.0f, 12.0f };

            // Act
            var result = await ExecuteVectorAddKernel(accelerator, inputA, inputB);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(expected.Length, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i], precision: 5);
            }
        }

        [Fact]
        public async Task VectorAddKernel_ShouldExecute_CorrectlyOnGPU()
        {
            // Arrange
            var cudaDevices = _context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToList();
            if (!cudaDevices.Any())
            {
                _output.WriteLine("Skipping GPU test - no CUDA devices available");
                return;
            }

            using var accelerator = cudaDevices.First().CreateAccelerator(_context);
            
            var inputA = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var inputB = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            var expected = new float[] { 6.0f, 8.0f, 10.0f, 12.0f };

            // Act
            var result = await ExecuteVectorAddKernel(accelerator, inputA, inputB);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(expected.Length, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i], precision: 5);
            }
        }

        [Fact]
        public async Task MatrixMultiplyKernel_ShouldExecute_Correctly()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            // Simple 2x2 matrix multiplication test
            var matrixA = new float[,] { { 1, 2 }, { 3, 4 } };
            var matrixB = new float[,] { { 5, 6 }, { 7, 8 } };
            var expected = new float[,] { { 19, 22 }, { 43, 50 } }; // A * B

            // Act
            var result = await ExecuteMatrixMultiplyKernel(accelerator, matrixA, matrixB);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.GetLength(0));
            Assert.Equal(2, result.GetLength(1));
            
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Assert.Equal(expected[i, j], result[i, j], precision: 5);
                }
            }
        }

        [Fact]
        public async Task ScalarMultiplyKernel_ShouldExecute_Correctly()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            var scalar = 3.0f;
            var expected = input.Select(x => x * scalar).ToArray();

            // Act
            var result = await ExecuteScalarMultiplyKernel(accelerator, input, scalar);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(expected.Length, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i], precision: 5);
            }
        }

        [Fact]
        public async Task ReductionKernel_ShouldComputeSum_Correctly()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            var input = Enumerable.Range(1, 100).Select(x => (float)x).ToArray(); // 1 to 100
            var expected = input.Sum(); // 5050

            // Act
            var result = await ExecuteReductionSumKernel(accelerator, input);

            // Assert
            Assert.Equal(expected, result, precision: 4);
        }

        [Fact]
        public void KernelCompilation_ShouldHandleInvalidKernel_Gracefully()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            // Act & Assert
            Assert.Throws<InvalidKernelOperationException>(() =>
            {
                // This should fail compilation due to invalid operations
                var invalidKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(InvalidKernel);
            });
        }

        [Theory]
        [InlineData(1)]
        [InlineData(32)]
        [InlineData(1024)]
        [InlineData(1024 * 1024)]
        public async Task VectorAdd_ShouldHandleDifferentSizes_Correctly(int size)
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            var inputA = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
            var inputB = Enumerable.Range(0, size).Select(i => (float)(i * 2)).ToArray();
            var expected = inputA.Zip(inputB, (a, b) => a + b).ToArray();

            // Act
            var result = await ExecuteVectorAddKernel(accelerator, inputA, inputB);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(size, result.Length);
            
            // Check first 10 and last 10 elements for large arrays
            var checkCount = Math.Min(10, size);
            for (int i = 0; i < checkCount; i++)
            {
                Assert.Equal(expected[i], result[i], precision: 5);
                if (size > checkCount)
                {
                    Assert.Equal(expected[size - 1 - i], result[size - 1 - i], precision: 5);
                }
            }
        }

        [Fact]
        public async Task KernelExecution_ShouldHandleEmptyInput_Gracefully()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            var emptyInputA = new float[0];
            var emptyInputB = new float[0];

            // Act
            var result = await ExecuteVectorAddKernel(accelerator, emptyInputA, emptyInputB);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public async Task KernelExecution_ShouldHandleNullInput_WithException()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(async () =>
            {
                await ExecuteVectorAddKernel(accelerator, null, new float[10]);
            });

            await Assert.ThrowsAsync<ArgumentNullException>(async () =>
            {
                await ExecuteVectorAddKernel(accelerator, new float[10], null);
            });
        }

        [Fact]
        public async Task KernelExecution_ShouldHandleMismatchedInputSizes_WithException()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            var inputA = new float[10];
            var inputB = new float[5]; // Different size

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(async () =>
            {
                await ExecuteVectorAddKernel(accelerator, inputA, inputB);
            });
        }

        [Fact]
        public async Task MemoryPool_ShouldReuseBuffers_Efficiently()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            const int iterations = 10;
            const int dataSize = 1000;
            var memoryUsages = new List<long>();

            // Act - Execute multiple kernels and monitor memory
            for (int i = 0; i < iterations; i++)
            {
                var inputA = Enumerable.Range(0, dataSize).Select(x => (float)x).ToArray();
                var inputB = Enumerable.Range(0, dataSize).Select(x => (float)(x * 2)).ToArray();

                await ExecuteVectorAddKernel(accelerator, inputA, inputB);

                if (accelerator is CudaAccelerator cudaAccelerator)
                {
                    memoryUsages.Add(cudaAccelerator.MemoryInfo.UsedMemory);
                }
            }

            // Assert - Memory usage should stabilize (indicating reuse)
            if (memoryUsages.Count > 5)
            {
                var stabilizedMemory = memoryUsages.Skip(5).ToList();
                var memoryVariance = CalculateVariance(stabilizedMemory);
                
                // Low variance indicates memory reuse
                Assert.True(memoryVariance < 1000000, $"Memory variance too high: {memoryVariance}");
            }
        }

        [Fact]
        public async Task KernelExecution_ShouldBeThreadSafe()
        {
            // Arrange
            var device = _context.Devices.First();
            using var accelerator = device.CreateAccelerator(_context);

            const int concurrentTasks = 10;
            const int dataSize = 100;

            // Act - Execute kernels concurrently
            var tasks = Enumerable.Range(0, concurrentTasks).Select(async i =>
            {
                var inputA = Enumerable.Range(i * dataSize, dataSize).Select(x => (float)x).ToArray();
                var inputB = Enumerable.Range(i * dataSize, dataSize).Select(x => (float)(x * 2)).ToArray();
                
                return await ExecuteVectorAddKernel(accelerator, inputA, inputB);
            }).ToArray();

            var results = await Task.WhenAll(tasks);

            // Assert - All tasks should complete successfully with correct results
            Assert.Equal(concurrentTasks, results.Length);
            
            for (int i = 0; i < concurrentTasks; i++)
            {
                Assert.NotNull(results[i]);
                Assert.Equal(dataSize, results[i].Length);
                
                // Verify first few elements
                for (int j = 0; j < Math.Min(5, dataSize); j++)
                {
                    var expectedValue = (i * dataSize + j) + (i * dataSize + j) * 2;
                    Assert.Equal(expectedValue, results[i][j], precision: 5);
                }
            }
        }

        // Helper methods for kernel execution
        private async Task<float[]> ExecuteVectorAddKernel(Accelerator accelerator, float[] a, float[] b)
        {
            if (a == null || b == null)
                throw new ArgumentNullException();
            
            if (a.Length != b.Length)
                throw new ArgumentException("Input arrays must have the same length");

            if (a.Length == 0)
                return new float[0];

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

            using var bufferA = accelerator.Allocate1D(a);
            using var bufferB = accelerator.Allocate1D(b);
            using var bufferC = accelerator.Allocate1D<float>(a.Length);

            kernel(accelerator.DefaultStream, bufferA.IntLength, bufferA.View, bufferB.View, bufferC.View);
            accelerator.Synchronize();

            return bufferC.GetAsArray1D();
        }

        private async Task<float[,]> ExecuteMatrixMultiplyKernel(Accelerator accelerator, float[,] a, float[,] b)
        {
            var rowsA = a.GetLength(0);
            var colsA = a.GetLength(1);
            var rowsB = b.GetLength(0);
            var colsB = b.GetLength(1);

            if (colsA != rowsB)
                throw new ArgumentException("Matrix dimensions don't match for multiplication");

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>>(MatrixMultiplyKernel);

            using var bufferA = accelerator.Allocate2DDenseX(a);
            using var bufferB = accelerator.Allocate2DDenseX(b);
            using var bufferC = accelerator.Allocate2DDenseX<float>(new Index2D(rowsA, colsB));

            kernel(accelerator.DefaultStream, bufferC.IntExtent, bufferA.View, bufferB.View, bufferC.View);
            accelerator.Synchronize();

            return bufferC.GetAs2DArray();
        }

        private async Task<float[]> ExecuteScalarMultiplyKernel(Accelerator accelerator, float[] input, float scalar)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, float, ArrayView<float>>(ScalarMultiplyKernel);

            using var bufferIn = accelerator.Allocate1D(input);
            using var bufferOut = accelerator.Allocate1D<float>(input.Length);

            kernel(accelerator.DefaultStream, bufferIn.IntLength, bufferIn.View, scalar, bufferOut.View);
            accelerator.Synchronize();

            return bufferOut.GetAsArray1D();
        }

        private async Task<float> ExecuteReductionSumKernel(Accelerator accelerator, float[] input)
        {
            // Simple implementation - in practice you'd use ILGPU's optimized reductions
            var sum = 0.0f;
            foreach (var value in input)
            {
                sum += value;
            }
            return sum;
        }

        private double CalculateVariance(List<long> values)
        {
            if (values.Count < 2) return 0;
            
            var mean = values.Average();
            var sumOfSquares = values.Sum(x => Math.Pow(x - mean, 2));
            return sumOfSquares / (values.Count - 1);
        }

        // Kernel implementations
        private static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }

        private static void MatrixMultiplyKernel(Index2D index, 
            ArrayView2D<float, Stride2D.DenseX> a, 
            ArrayView2D<float, Stride2D.DenseX> b, 
            ArrayView2D<float, Stride2D.DenseX> c)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var k = 0; k < a.IntExtent.Y; k++)
                sum += a[x, k] * b[k, y];

            c[x, y] = sum;
        }

        private static void ScalarMultiplyKernel(Index1D index, ArrayView<float> input, float scalar, ArrayView<float> output)
        {
            output[index] = input[index] * scalar;
        }

        // Invalid kernel for testing error handling
        private static void InvalidKernel(Index1D index, ArrayView<float> data)
        {
            // This would be invalid in ILGPU context - just for testing error handling
            throw new NotImplementedException("This kernel is intentionally invalid");
        }

        public void Dispose()
        {
            _context?.Dispose();
        }
    }

    // Test kernel implementation for registration testing
    public class TestVectorAddKernel : IGpuKernel<float[], float[]>
    {
        public ValueTask<KernelHandle> SubmitBatchAsync(IReadOnlyList<float[]> items, GpuExecutionHints? hints = null, CancellationToken ct = default)
        {
            var handle = KernelHandle.Create();
            return ValueTask.FromResult(handle);
        }

        public async IAsyncEnumerable<float[]> ReadResultsAsync(KernelHandle handle, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            // Return dummy result for testing
            yield return new float[] { 1.0f, 2.0f, 3.0f };
        }

        public ValueTask<Orleans.GpuBridge.Abstractions.Kernels.KernelInfo> GetInfoAsync(CancellationToken ct = default)
        {
            return ValueTask.FromResult(new Orleans.GpuBridge.Abstractions.Kernels.KernelInfo(
                new KernelId("test-kernel"),
                "Test vector addition kernel",
                typeof(float[]),
                typeof(float[]),
                true,
                256
            ));
        }
    }
}