using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Performance
{
    /// <summary>
    /// Comprehensive performance benchmarks for ILGPU operations
    /// </summary>
    public class ILGPUBenchmarks : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly IServiceProvider _serviceProvider;
        private readonly Context _context;
        private readonly List<BenchmarkResult> _results;

        public ILGPUBenchmarks(ITestOutputHelper output)
        {
            _output = output;
            _results = new List<BenchmarkResult>();
            
            // Setup DI container
            var services = new ServiceCollection();
            services.AddLogging(builder => builder.AddXUnit(output));
            services.AddGpuBridge(options => 
            {
                options.PreferGpu = true;
                options.EnableILGPU = true;
            });
            
            _serviceProvider = services.BuildServiceProvider();
            _context = Context.CreateDefault();
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(10000)]
        [InlineData(100000)]
        [InlineData(1000000)]
        public async Task VectorAddition_PerformanceBenchmark_GPU(int dataSize)
        {
            // Arrange
            var inputA = GenerateRandomData(dataSize);
            var inputB = GenerateRandomData(dataSize);
            
            var cudaDevices = _context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToList();
            if (!cudaDevices.Any())
            {
                _output.WriteLine("Skipping GPU benchmark - no CUDA devices available");
                return;
            }

            using var accelerator = cudaDevices.First().CreateAccelerator(_context);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

            // Warmup runs
            await RunVectorAdditionWarmup(accelerator, kernel, inputA, inputB);

            // Benchmark runs
            var times = new List<double>();
            const int iterations = 10;

            for (int i = 0; i < iterations; i++)
            {
                var time = await MeasureVectorAdditionTime(accelerator, kernel, inputA, inputB);
                times.Add(time);
            }

            // Calculate statistics
            var avgTime = times.Average();
            var minTime = times.Min();
            var maxTime = times.Max();
            var throughput = dataSize / (avgTime / 1000.0); // elements per second
            var bandwidth = (dataSize * 3 * sizeof(float)) / (avgTime / 1000.0) / (1024 * 1024 * 1024); // GB/s

            // Store results
            _results.Add(new BenchmarkResult
            {
                Operation = "VectorAddition_GPU",
                DataSize = dataSize,
                AvgTimeMs = avgTime,
                MinTimeMs = minTime,
                MaxTimeMs = maxTime,
                Throughput = throughput,
                Bandwidth = bandwidth
            });

            _output.WriteLine($"GPU VectorAdd [{dataSize:N0}]: Avg={avgTime:F2}ms, Throughput={throughput:F0} elem/s, Bandwidth={bandwidth:F2} GB/s");
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(10000)]
        [InlineData(100000)]
        [InlineData(1000000)]
        public async Task VectorAddition_PerformanceBenchmark_CPU(int dataSize)
        {
            // Arrange
            var inputA = GenerateRandomData(dataSize);
            var inputB = GenerateRandomData(dataSize);
            
            var cpuDevice = _context.Devices.First(d => d.AcceleratorType == AcceleratorType.CPU);
            using var accelerator = cpuDevice.CreateAccelerator(_context);
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

            // Warmup runs
            await RunVectorAdditionWarmup(accelerator, kernel, inputA, inputB);

            // Benchmark runs
            var times = new List<double>();
            const int iterations = 10;

            for (int i = 0; i < iterations; i++)
            {
                var time = await MeasureVectorAdditionTime(accelerator, kernel, inputA, inputB);
                times.Add(time);
            }

            // Calculate statistics
            var avgTime = times.Average();
            var minTime = times.Min();
            var maxTime = times.Max();
            var throughput = dataSize / (avgTime / 1000.0);
            var bandwidth = (dataSize * 3 * sizeof(float)) / (avgTime / 1000.0) / (1024 * 1024 * 1024);

            // Store results
            _results.Add(new BenchmarkResult
            {
                Operation = "VectorAddition_CPU",
                DataSize = dataSize,
                AvgTimeMs = avgTime,
                MinTimeMs = minTime,
                MaxTimeMs = maxTime,
                Throughput = throughput,
                Bandwidth = bandwidth
            });

            _output.WriteLine($"CPU VectorAdd [{dataSize:N0}]: Avg={avgTime:F2}ms, Throughput={throughput:F0} elem/s, Bandwidth={bandwidth:F2} GB/s");
        }

        [Theory]
        [InlineData(64)]
        [InlineData(128)]
        [InlineData(256)]
        [InlineData(512)]
        public async Task MatrixMultiplication_PerformanceBenchmark(int matrixSize)
        {
            // Arrange
            var matrixA = GenerateRandomMatrix(matrixSize);
            var matrixB = GenerateRandomMatrix(matrixSize);
            
            var availableDevices = _context.Devices.Where(d => 
                d.AcceleratorType == AcceleratorType.Cuda || 
                d.AcceleratorType == AcceleratorType.CPU).ToList();

            foreach (var device in availableDevices)
            {
                using var accelerator = device.CreateAccelerator(_context);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index2D, ArrayView2D<float, Stride2D.DenseX>, 
                    ArrayView2D<float, Stride2D.DenseX>, 
                    ArrayView2D<float, Stride2D.DenseX>>(MatrixMultiplyKernel);

                // Warmup
                await RunMatrixMultiplyWarmup(accelerator, kernel, matrixA, matrixB, matrixSize);

                // Benchmark
                var times = new List<double>();
                const int iterations = 5;

                for (int i = 0; i < iterations; i++)
                {
                    var time = await MeasureMatrixMultiplyTime(accelerator, kernel, matrixA, matrixB, matrixSize);
                    times.Add(time);
                }

                var avgTime = times.Average();
                var flops = 2.0 * matrixSize * matrixSize * matrixSize; // Matrix multiply FLOPs
                var gflops = flops / (avgTime / 1000.0) / 1e9;

                _results.Add(new BenchmarkResult
                {
                    Operation = $"MatrixMultiply_{device.AcceleratorType}",
                    DataSize = matrixSize * matrixSize,
                    AvgTimeMs = avgTime,
                    MinTimeMs = times.Min(),
                    MaxTimeMs = times.Max(),
                    GFLOPS = gflops
                });

                _output.WriteLine($"{device.AcceleratorType} MatMul [{matrixSize}x{matrixSize}]: {avgTime:F2}ms, {gflops:F2} GFLOPS");
            }
        }

        [Fact]
        public async Task MemoryBandwidth_Benchmark()
        {
            // Test different memory access patterns
            var dataSizes = new[] { 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024 };
            
            var cudaDevices = _context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToList();
            if (!cudaDevices.Any())
            {
                _output.WriteLine("Skipping memory bandwidth test - no CUDA devices available");
                return;
            }

            using var accelerator = cudaDevices.First().CreateAccelerator(_context);

            foreach (var dataSize in dataSizes)
            {
                // Test memory copy bandwidth
                var copyTime = await MeasureMemoryCopyBandwidth(accelerator, dataSize);
                var copyBandwidth = (dataSize * sizeof(float) * 2) / (copyTime / 1000.0) / (1024 * 1024 * 1024); // GB/s (read + write)

                // Test memory access kernel bandwidth
                var accessTime = await MeasureMemoryAccessBandwidth(accelerator, dataSize);
                var accessBandwidth = (dataSize * sizeof(float)) / (accessTime / 1000.0) / (1024 * 1024 * 1024); // GB/s

                _results.Add(new BenchmarkResult
                {
                    Operation = "MemoryCopy",
                    DataSize = dataSize,
                    AvgTimeMs = copyTime,
                    Bandwidth = copyBandwidth
                });

                _results.Add(new BenchmarkResult
                {
                    Operation = "MemoryAccess",
                    DataSize = dataSize,
                    AvgTimeMs = accessTime,
                    Bandwidth = accessBandwidth
                });

                _output.WriteLine($"Memory [{dataSize / (1024 * 1024)}MB]: Copy={copyBandwidth:F1} GB/s, Access={accessBandwidth:F1} GB/s");
            }
        }

        [Fact]
        public async Task KernelLaunchOverhead_Benchmark()
        {
            // Test kernel launch overhead with different work sizes
            var workSizes = new[] { 1, 32, 256, 1024, 8192 };
            
            var availableDevices = _context.Devices.ToList();

            foreach (var device in availableDevices)
            {
                using var accelerator = device.CreateAccelerator(_context);
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(SimpleKernel);

                foreach (var workSize in workSizes)
                {
                    var times = new List<double>();
                    const int iterations = 100;

                    using var buffer = accelerator.Allocate1D<float>(workSize);

                    // Warmup
                    for (int w = 0; w < 10; w++)
                    {
                        kernel(accelerator.DefaultStream, workSize, buffer.View);
                        accelerator.Synchronize();
                    }

                    // Benchmark
                    for (int i = 0; i < iterations; i++)
                    {
                        var sw = Stopwatch.StartNew();
                        kernel(accelerator.DefaultStream, workSize, buffer.View);
                        accelerator.Synchronize();
                        sw.Stop();
                        times.Add(sw.Elapsed.TotalMilliseconds);
                    }

                    var avgTime = times.Average();
                    _output.WriteLine($"{device.AcceleratorType} KernelLaunch [size={workSize}]: {avgTime:F4}ms");
                }
            }
        }

        [Fact]
        public void PrintBenchmarkSummary()
        {
            if (!_results.Any()) return;

            _output.WriteLine("\n" + new string('=', 80));
            _output.WriteLine("BENCHMARK SUMMARY");
            _output.WriteLine(new string('=', 80));

            var groupedResults = _results.GroupBy(r => r.Operation);

            foreach (var group in groupedResults)
            {
                _output.WriteLine($"\n{group.Key}:");
                _output.WriteLine(new string('-', 60));

                foreach (var result in group.OrderBy(r => r.DataSize))
                {
                    var throughputStr = result.Throughput > 0 ? $", {result.Throughput:F0} elem/s" : "";
                    var bandwidthStr = result.Bandwidth > 0 ? $", {result.Bandwidth:F2} GB/s" : "";
                    var gflopsStr = result.GFLOPS > 0 ? $", {result.GFLOPS:F2} GFLOPS" : "";
                    
                    _output.WriteLine($"  Size: {result.DataSize:N0}, Time: {result.AvgTimeMs:F2}ms{throughputStr}{bandwidthStr}{gflopsStr}");
                }
            }

            // Performance comparison (GPU vs CPU if both available)
            var gpuVectorAdd = _results.Where(r => r.Operation == "VectorAddition_GPU").ToList();
            var cpuVectorAdd = _results.Where(r => r.Operation == "VectorAddition_CPU").ToList();

            if (gpuVectorAdd.Any() && cpuVectorAdd.Any())
            {
                _output.WriteLine("\nGPU vs CPU SPEEDUP (VectorAddition):");
                _output.WriteLine(new string('-', 40));

                foreach (var gpuResult in gpuVectorAdd)
                {
                    var cpuResult = cpuVectorAdd.FirstOrDefault(c => c.DataSize == gpuResult.DataSize);
                    if (cpuResult != null)
                    {
                        var speedup = cpuResult.AvgTimeMs / gpuResult.AvgTimeMs;
                        _output.WriteLine($"  Size: {gpuResult.DataSize:N0}, Speedup: {speedup:F2}x");
                    }
                }
            }
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

        private static void SimpleKernel(Index1D index, ArrayView<float> data)
        {
            data[index] = data[index] * 2.0f + 1.0f;
        }

        private static void MemoryAccessKernel(Index1D index, ArrayView<float> data)
        {
            // Simple memory access pattern
            var value = data[index];
            data[index] = value * 1.1f;
        }

        // Helper methods
        private float[] GenerateRandomData(int size)
        {
            var random = new Random(42);
            return Enumerable.Range(0, size).Select(_ => (float)random.NextDouble()).ToArray();
        }

        private float[,] GenerateRandomMatrix(int size)
        {
            var random = new Random(42);
            var matrix = new float[size, size];
            
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i, j] = (float)random.NextDouble();
                }
            }
            return matrix;
        }

        private async Task RunVectorAdditionWarmup(Accelerator accelerator, 
            Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> kernel,
            float[] inputA, float[] inputB)
        {
            using var bufferA = accelerator.Allocate1D(inputA);
            using var bufferB = accelerator.Allocate1D(inputB);
            using var bufferC = accelerator.Allocate1D<float>(inputA.Length);

            for (int i = 0; i < 3; i++)
            {
                kernel(accelerator.DefaultStream, bufferA.IntLength, bufferA.View, bufferB.View, bufferC.View);
                accelerator.Synchronize();
            }
        }

        private async Task<double> MeasureVectorAdditionTime(Accelerator accelerator,
            Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> kernel,
            float[] inputA, float[] inputB)
        {
            using var bufferA = accelerator.Allocate1D(inputA);
            using var bufferB = accelerator.Allocate1D(inputB);
            using var bufferC = accelerator.Allocate1D<float>(inputA.Length);

            var sw = Stopwatch.StartNew();
            kernel(accelerator.DefaultStream, bufferA.IntLength, bufferA.View, bufferB.View, bufferC.View);
            accelerator.Synchronize();
            sw.Stop();

            return sw.Elapsed.TotalMilliseconds;
        }

        private async Task RunMatrixMultiplyWarmup(Accelerator accelerator,
            Action<AcceleratorStream, Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> kernel,
            float[,] matrixA, float[,] matrixB, int size)
        {
            using var bufferA = accelerator.Allocate2DDenseX(matrixA);
            using var bufferB = accelerator.Allocate2DDenseX(matrixB);
            using var bufferC = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));

            for (int i = 0; i < 2; i++)
            {
                kernel(accelerator.DefaultStream, bufferC.IntExtent, bufferA.View, bufferB.View, bufferC.View);
                accelerator.Synchronize();
            }
        }

        private async Task<double> MeasureMatrixMultiplyTime(Accelerator accelerator,
            Action<AcceleratorStream, Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> kernel,
            float[,] matrixA, float[,] matrixB, int size)
        {
            using var bufferA = accelerator.Allocate2DDenseX(matrixA);
            using var bufferB = accelerator.Allocate2DDenseX(matrixB);
            using var bufferC = accelerator.Allocate2DDenseX<float>(new Index2D(size, size));

            var sw = Stopwatch.StartNew();
            kernel(accelerator.DefaultStream, bufferC.IntExtent, bufferA.View, bufferB.View, bufferC.View);
            accelerator.Synchronize();
            sw.Stop();

            return sw.Elapsed.TotalMilliseconds;
        }

        private async Task<double> MeasureMemoryCopyBandwidth(Accelerator accelerator, int dataSize)
        {
            var hostData = GenerateRandomData(dataSize);
            var resultData = new float[dataSize];

            using var buffer = accelerator.Allocate1D<float>(dataSize);

            var sw = Stopwatch.StartNew();
            buffer.CopyFromCPU(hostData);
            buffer.CopyToCPU(resultData);
            accelerator.Synchronize();
            sw.Stop();

            return sw.Elapsed.TotalMilliseconds;
        }

        private async Task<double> MeasureMemoryAccessBandwidth(Accelerator accelerator, int dataSize)
        {
            var hostData = GenerateRandomData(dataSize);
            using var buffer = accelerator.Allocate1D(hostData);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(MemoryAccessKernel);

            var sw = Stopwatch.StartNew();
            kernel(accelerator.DefaultStream, buffer.IntLength, buffer.View);
            accelerator.Synchronize();
            sw.Stop();

            return sw.Elapsed.TotalMilliseconds;
        }

        public void Dispose()
        {
            PrintBenchmarkSummary();
            _context?.Dispose();
            _serviceProvider?.GetService<IServiceScope>()?.Dispose();
        }
    }

    public class BenchmarkResult
    {
        public string Operation { get; set; }
        public int DataSize { get; set; }
        public double AvgTimeMs { get; set; }
        public double MinTimeMs { get; set; }
        public double MaxTimeMs { get; set; }
        public double Throughput { get; set; } // elements/second
        public double Bandwidth { get; set; } // GB/s
        public double GFLOPS { get; set; } // Giga floating point operations per second
    }
}