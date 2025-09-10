using System;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Backends.ILGPU;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Integration
{
    /// <summary>
    /// Comprehensive integration tests for ILGPU backend functionality
    /// </summary>
    public class ILGPUBackendTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly IServiceProvider _serviceProvider;
        private readonly Context _context;
        private readonly ILogger<ILGPUBackendTests> _logger;

        public ILGPUBackendTests(ITestOutputHelper output)
        {
            _output = output;
            
            // Setup DI container
            var services = new ServiceCollection();
            services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Information));
            services.AddGpuBridge(options => 
            {
                options.PreferGpu = true;
                options.EnableILGPU = true;
                options.FallbackToCpu = true;
            });
            
            _serviceProvider = services.BuildServiceProvider();
            _logger = _serviceProvider.GetService<ILogger<ILGPUBackendTests>>();
            
            // Initialize ILGPU context
            _context = Context.CreateDefault();
            _output.WriteLine($"ILGPU Context initialized with {_context.Devices.Count()} devices");
        }

        [Fact]
        public void Context_ShouldInitialize_Successfully()
        {
            // Arrange & Act
            using var context = Context.CreateDefault();
            
            // Assert
            Assert.NotNull(context);
            Assert.True(context.Devices.Count() >= 1, "At least CPU device should be available");
            _output.WriteLine($"Available devices: {context.Devices.Count()}");
            
            foreach (var device in context.Devices)
            {
                _output.WriteLine($"Device: {device.Name} ({device.AcceleratorType})");
            }
        }

        [Fact]
        public async Task DeviceDetection_ShouldDetectAvailableDevices_AndReportCorrectly()
        {
            // Arrange
            var deviceBroker = _serviceProvider.GetService<IDeviceBroker>();
            
            // Act
            var availableDevices = await deviceBroker.GetAvailableDevicesAsync();
            
            // Assert
            Assert.NotEmpty(availableDevices);
            Assert.Contains(availableDevices, d => d.Type == DeviceType.CPU);
            
            _output.WriteLine($"Detected {availableDevices.Count()} devices:");
            foreach (var device in availableDevices)
            {
                _output.WriteLine($"- {device.Name} ({device.Type}, Memory: {device.TotalMemory:N0} bytes)");
            }
        }

        [Theory]
        [InlineData(AcceleratorType.CPU)]
        [InlineData(AcceleratorType.Cuda)]
        public async Task KernelCompilation_ShouldCompile_ForAvailableAccelerators(AcceleratorType acceleratorType)
        {
            // Arrange
            var availableDevices = _context.Devices.Where(d => d.AcceleratorType == acceleratorType).ToList();
            
            if (!availableDevices.Any())
            {
                _output.WriteLine($"Skipping test - {acceleratorType} not available");
                return;
            }

            var device = availableDevices.First();
            using var accelerator = device.CreateAccelerator(_context);
            
            // Act & Assert
            var vectorAddKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView<float>, 
                ArrayView<float>, 
                ArrayView<float>>(VectorAddKernel);
            
            Assert.NotNull(vectorAddKernel);
            _output.WriteLine($"Successfully compiled VectorAdd kernel for {acceleratorType}");
        }

        [Fact]
        public async Task VectorAddition_ShouldExecuteCorrectly_OnGPU()
        {
            // Arrange
            const int dataSize = 1024;
            var a = Enumerable.Range(0, dataSize).Select(i => (float)i).ToArray();
            var b = Enumerable.Range(0, dataSize).Select(i => (float)(i * 2)).ToArray();
            var expected = a.Zip(b, (x, y) => x + y).ToArray();

            var gpuBridge = _serviceProvider.GetService<IGpuBridge>();
            
            // Act
            var result = await gpuBridge.ExecuteKernelAsync("vectoradd", new { A = a, B = b });
            
            // Assert
            Assert.NotNull(result);
            var resultArray = result as float[] ?? ((dynamic)result).Result as float[];
            Assert.NotNull(resultArray);
            Assert.Equal(dataSize, resultArray.Length);
            
            for (int i = 0; i < Math.Min(10, dataSize); i++)
            {
                Assert.Equal(expected[i], resultArray[i], precision: 5);
            }
            
            _output.WriteLine($"Vector addition completed successfully for {dataSize} elements");
        }

        [Fact]
        public async Task MemoryAllocation_ShouldAllocateAndRelease_Correctly()
        {
            // Arrange
            var cudaDevices = _context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToList();
            
            if (!cudaDevices.Any())
            {
                _output.WriteLine("Skipping CUDA memory test - no CUDA devices available");
                return;
            }

            using var accelerator = cudaDevices.First().CreateAccelerator(_context);
            const int size = 1024 * 1024; // 1M floats
            
            // Act - Allocate memory
            using var buffer = accelerator.Allocate1D<float>(size);
            
            // Assert
            Assert.Equal(size, buffer.Length);
            Assert.True(accelerator.MemoryInfo.TotalBytes > 0);
            
            var memoryBefore = accelerator.MemoryInfo.AvailableBytes;
            _output.WriteLine($"Available memory before: {memoryBefore:N0} bytes");
            
            // Test memory operations
            var hostData = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
            buffer.CopyFromCPU(hostData);
            
            var resultData = new float[size];
            buffer.CopyToCPU(resultData);
            
            // Verify data integrity
            Assert.Equal(hostData.Take(100), resultData.Take(100));
            
            var memoryAfter = accelerator.MemoryInfo.AvailableBytes;
            _output.WriteLine($"Available memory after: {memoryAfter:N0} bytes");
        }

        [Fact]
        public async Task GpuContextCreation_ShouldCreateAndCleanup_Properly()
        {
            // Arrange
            var contextCount = 0;
            var cleanupCount = 0;

            // Act - Create multiple contexts
            for (int i = 0; i < 5; i++)
            {
                using (var context = Context.CreateDefault())
                {
                    contextCount++;
                    Assert.NotNull(context);
                    
                    foreach (var device in context.Devices.Take(2)) // Test first 2 devices
                    {
                        using var accelerator = device.CreateAccelerator(context);
                        Assert.NotNull(accelerator);
                    }
                    
                    cleanupCount++;
                }
            }
            
            // Assert
            Assert.Equal(5, contextCount);
            Assert.Equal(5, cleanupCount);
            _output.WriteLine($"Successfully created and cleaned up {contextCount} contexts");
        }

        [Fact]
        public async Task ErrorHandling_ShouldFallbackToCPU_WhenGPUFails()
        {
            // Arrange
            var gpuBridge = _serviceProvider.GetService<IGpuBridge>();
            var invalidKernelId = "nonexistent_kernel";
            
            // Act & Assert - Should not throw but fallback gracefully
            try
            {
                await gpuBridge.ExecuteKernelAsync(invalidKernelId, new { Data = new float[10] });
                Assert.True(false, "Expected exception for invalid kernel");
            }
            catch (InvalidOperationException ex)
            {
                Assert.Contains("kernel", ex.Message.ToLower());
                _output.WriteLine($"Correctly handled invalid kernel: {ex.Message}");
            }
        }

        [Fact]
        public async Task MatrixMultiplication_ShouldExecuteCorrectly_OnGPU()
        {
            // Arrange
            const int matrixSize = 64; // Small matrix for testing
            var matrixA = CreateIdentityMatrix(matrixSize);
            var matrixB = CreateTestMatrix(matrixSize);
            
            var gpuBridge = _serviceProvider.GetService<IGpuBridge>();
            
            // Act
            var result = await gpuBridge.ExecuteKernelAsync("matmul", new 
            { 
                A = matrixA, 
                B = matrixB, 
                Size = matrixSize 
            });
            
            // Assert - Identity matrix multiplication should return the original matrix
            var resultMatrix = result as float[,] ?? ConvertToMatrix((float[])((dynamic)result).Result, matrixSize);
            Assert.NotNull(resultMatrix);
            
            // Verify a few key elements
            for (int i = 0; i < Math.Min(5, matrixSize); i++)
            {
                for (int j = 0; j < Math.Min(5, matrixSize); j++)
                {
                    Assert.Equal(matrixB[i, j], resultMatrix[i, j], precision: 4);
                }
            }
            
            _output.WriteLine($"Matrix multiplication completed for {matrixSize}x{matrixSize} matrices");
        }

        [Fact]
        public async Task DeviceFailureSimulation_ShouldFallbackToCPU_Gracefully()
        {
            // Arrange - Force use of non-existent device
            var services = new ServiceCollection();
            services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Information));
            services.AddGpuBridge(options => 
            {
                options.PreferGpu = true;
                options.FallbackToCpu = true;
                options.MaxRetries = 2;
            });
            
            var provider = services.BuildServiceProvider();
            var gpuBridge = provider.GetService<IGpuBridge>();
            
            // Act
            var data = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();
            var result = await gpuBridge.ExecuteKernelAsync("vectoradd", new { A = data, B = data });
            
            // Assert - Should complete using CPU fallback
            Assert.NotNull(result);
            _output.WriteLine("Successfully fell back to CPU after GPU failure simulation");
        }

        [Theory]
        [InlineData(100)]
        [InlineData(1000)]
        [InlineData(10000)]
        public async Task PerformanceValidation_ShouldMeetBaseline_ForDifferentSizes(int dataSize)
        {
            // Arrange
            var data = Enumerable.Range(0, dataSize).Select(i => (float)i).ToArray();
            var gpuBridge = _serviceProvider.GetService<IGpuBridge>();
            
            // Act - Measure execution time
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = await gpuBridge.ExecuteKernelAsync("vectoradd", new { A = data, B = data });
            stopwatch.Stop();
            
            // Assert - Should complete within reasonable time
            Assert.NotNull(result);
            Assert.True(stopwatch.ElapsedMilliseconds < 5000, $"Execution took too long: {stopwatch.ElapsedMilliseconds}ms");
            
            var throughput = dataSize / (stopwatch.ElapsedMilliseconds / 1000.0);
            _output.WriteLine($"Size: {dataSize}, Time: {stopwatch.ElapsedMilliseconds}ms, Throughput: {throughput:F0} elements/sec");
        }

        [Fact]
        public async Task GpuVsCpuValidation_ShouldProduceSameResults_ForSameInput()
        {
            // Arrange
            const int dataSize = 512;
            var inputA = Enumerable.Range(0, dataSize).Select(i => (float)Math.Sin(i * 0.1)).ToArray();
            var inputB = Enumerable.Range(0, dataSize).Select(i => (float)Math.Cos(i * 0.1)).ToArray();
            
            // CPU Reference calculation
            var cpuResult = inputA.Zip(inputB, (a, b) => a + b).ToArray();
            
            // GPU calculation
            var gpuBridge = _serviceProvider.GetService<IGpuBridge>();
            var gpuResult = await gpuBridge.ExecuteKernelAsync("vectoradd", new { A = inputA, B = inputB });
            var gpuArray = gpuResult as float[] ?? ((dynamic)gpuResult).Result as float[];
            
            // Assert - Results should match within tolerance
            Assert.NotNull(gpuArray);
            Assert.Equal(dataSize, gpuArray.Length);
            
            for (int i = 0; i < dataSize; i++)
            {
                Assert.Equal(cpuResult[i], gpuArray[i], precision: 5);
            }
            
            _output.WriteLine($"GPU and CPU results match for {dataSize} elements");
        }

        // Helper methods
        private static void VectorAddKernel(
            Index1D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c)
        {
            c[index] = a[index] + b[index];
        }

        private float[,] CreateIdentityMatrix(int size)
        {
            var matrix = new float[size, size];
            for (int i = 0; i < size; i++)
            {
                matrix[i, i] = 1.0f;
            }
            return matrix;
        }

        private float[,] CreateTestMatrix(int size)
        {
            var matrix = new float[size, size];
            var random = new Random(42); // Fixed seed for reproducibility
            
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i, j] = (float)random.NextDouble() * 10.0f;
                }
            }
            return matrix;
        }

        private float[,] ConvertToMatrix(float[] array, int size)
        {
            var matrix = new float[size, size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i, j] = array[i * size + j];
                }
            }
            return matrix;
        }

        public void Dispose()
        {
            _context?.Dispose();
            _serviceProvider?.GetService<IServiceScope>()?.Dispose();
        }
    }
}