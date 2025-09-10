using System;
using System.Diagnostics;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;

namespace Orleans.GpuBridge.Examples
{
    /// <summary>
    /// Production validation test for ILGPU integration
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Validates ILGPU can detect and use available hardware
        /// </summary>
        public static async Task<ValidationResult> ValidateILGPUAsync()
        {
            var result = new ValidationResult();
            var sw = Stopwatch.StartNew();

            try
            {
                // Initialize ILGPU context
                using var context = Context.CreateDefault();
                result.ILGPUContextCreated = true;

                Console.WriteLine("✅ ILGPU Context Created Successfully");

                // Enumerate devices
                var devices = context.Devices;
                result.DeviceCount = devices.Length;

                Console.WriteLine($"📊 Found {devices.Length} compute devices:");

                foreach (var device in devices)
                {
                    Console.WriteLine($"  - {device.Name} ({device.AcceleratorType})");
                    Console.WriteLine($"    Memory: {device.MemorySize / (1024 * 1024)} MB");
                    Console.WriteLine($"    Max Threads/Group: {device.MaxNumThreadsPerGroup}");

                    if (device.AcceleratorType == AcceleratorType.Cuda)
                    {
                        result.CudaDeviceFound = true;
                    }
                    else if (device.AcceleratorType == AcceleratorType.CPU)
                    {
                        result.CpuDeviceFound = true;
                    }
                }

                // Test GPU device if available
                if (result.CudaDeviceFound)
                {
                    await TestCudaDeviceAsync(context, result);
                }

                // Test CPU fallback
                await TestCpuFallbackAsync(context, result);

                result.ValidationTime = sw.Elapsed;
                result.Success = true;
                Console.WriteLine($"✅ ILGPU Validation Completed in {result.ValidationTime.TotalMilliseconds:F2}ms");
            }
            catch (Exception ex)
            {
                result.Error = ex.Message;
                result.Success = false;
                Console.WriteLine($"❌ ILGPU Validation Failed: {ex.Message}");
            }
            finally
            {
                sw.Stop();
            }

            return result;
        }

        private static Task TestCudaDeviceAsync(Context context, ValidationResult result)
        {
            try
            {
                using var accelerator = context.CreateCudaAccelerator(0);
                
                // Test basic GPU computation
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(VectorAddKernel);

                const int dataSize = 1024;
                using var buffer1 = accelerator.Allocate1D<float>(dataSize);
                using var buffer2 = accelerator.Allocate1D<float>(dataSize);

                // Initialize data
                var data = new float[dataSize];
                for (int i = 0; i < dataSize; i++)
                {
                    data[i] = i;
                }

                buffer1.CopyFromCPU(data);
                
                // Execute kernel
                var sw = Stopwatch.StartNew();
                kernel(dataSize, buffer1.View, buffer2.View, 2.0f);
                accelerator.Synchronize();
                sw.Stop();

                // Copy results back
                var results = buffer2.GetAsArray1D();

                // Validate results
                bool resultsValid = true;
                for (int i = 0; i < Math.Min(10, dataSize); i++)
                {
                    if (Math.Abs(results[i] - (data[i] * 2.0f)) > 1e-6f)
                    {
                        resultsValid = false;
                        break;
                    }
                }

                result.CudaComputationSuccessful = resultsValid;
                result.CudaExecutionTime = sw.Elapsed;
                
                Console.WriteLine($"  ✅ CUDA Computation: {(resultsValid ? "PASSED" : "FAILED")} ({sw.Elapsed.TotalMicroseconds:F2}μs)");
            }
            catch (Exception ex)
            {
                result.CudaError = ex.Message;
                Console.WriteLine($"  ❌ CUDA Test Failed: {ex.Message}");
            }
            return Task.CompletedTask;
        }

        private static Task TestCpuFallbackAsync(Context context, ValidationResult result)
        {
            try
            {
                using var accelerator = context.CreateCPUAccelerator(0);
                
                // Test CPU fallback computation
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(VectorAddKernel);

                const int dataSize = 256;
                using var buffer1 = accelerator.Allocate1D<float>(dataSize);
                using var buffer2 = accelerator.Allocate1D<float>(dataSize);

                // Initialize data
                var data = new float[dataSize];
                for (int i = 0; i < dataSize; i++)
                {
                    data[i] = i * 0.5f;
                }

                buffer1.CopyFromCPU(data);
                
                // Execute kernel
                var sw = Stopwatch.StartNew();
                kernel(dataSize, buffer1.View, buffer2.View, 3.0f);
                accelerator.Synchronize();
                sw.Stop();

                // Copy results back
                var results = buffer2.GetAsArray1D();

                // Validate results
                bool resultsValid = true;
                for (int i = 0; i < Math.Min(10, dataSize); i++)
                {
                    if (Math.Abs(results[i] - (data[i] * 3.0f)) > 1e-6f)
                    {
                        resultsValid = false;
                        break;
                    }
                }

                result.CpuFallbackSuccessful = resultsValid;
                result.CpuExecutionTime = sw.Elapsed;
                
                Console.WriteLine($"  ✅ CPU Fallback: {(resultsValid ? "PASSED" : "FAILED")} ({sw.Elapsed.TotalMicroseconds:F2}μs)");
            }
            catch (Exception ex)
            {
                result.CpuFallbackError = ex.Message;
                Console.WriteLine($"  ❌ CPU Fallback Test Failed: {ex.Message}");
            }
            return Task.CompletedTask;
        }

        /// <summary>
        /// Simple vector multiplication kernel for testing
        /// </summary>
        static void VectorAddKernel(Index1D index, ArrayView<float> input, ArrayView<float> output, float multiplier)
        {
            output[index] = input[index] * multiplier;
        }

        public static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 ILGPU Production Validation Starting...");
            Console.WriteLine();

            var result = await ValidateILGPUAsync();

            Console.WriteLine();
            Console.WriteLine("📊 VALIDATION SUMMARY:");
            Console.WriteLine($"Success: {(result.Success ? "✅" : "❌")}");
            Console.WriteLine($"Devices Found: {result.DeviceCount}");
            Console.WriteLine($"CUDA Available: {(result.CudaDeviceFound ? "✅" : "❌")}");
            Console.WriteLine($"CPU Available: {(result.CpuDeviceFound ? "✅" : "❌")}");
            Console.WriteLine($"CUDA Computation: {(result.CudaComputationSuccessful ? "✅" : "❌")}");
            Console.WriteLine($"CPU Fallback: {(result.CpuFallbackSuccessful ? "✅" : "❌")}");
            Console.WriteLine($"Total Time: {result.ValidationTime.TotalMilliseconds:F2}ms");

            if (!result.Success)
            {
                Console.WriteLine($"Error: {result.Error}");
                Environment.Exit(1);
            }

            Environment.Exit(0);
        }
    }

    public class ValidationResult
    {
        public bool Success { get; set; }
        public bool ILGPUContextCreated { get; set; }
        public int DeviceCount { get; set; }
        public bool CudaDeviceFound { get; set; }
        public bool CpuDeviceFound { get; set; }
        public bool CudaComputationSuccessful { get; set; }
        public bool CpuFallbackSuccessful { get; set; }
        public TimeSpan ValidationTime { get; set; }
        public TimeSpan CudaExecutionTime { get; set; }
        public TimeSpan CpuExecutionTime { get; set; }
        public string? Error { get; set; }
        public string? CudaError { get; set; }
        public string? CpuFallbackError { get; set; }
    }
}
