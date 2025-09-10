using System;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Orleans.GpuBridge.Examples
{
    /// <summary>
    /// Focused CUDA detection diagnostic
    /// </summary>
    public class CudaDetectionProgram
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("🔍 CUDA Detection Diagnostic...");
            
            try
            {
                using var context = Context.CreateDefault();
                
                Console.WriteLine("ILGPU Context created successfully");
                
                // Check available device types
                var devices = context.Devices;
                Console.WriteLine($"Found {devices.Length} device(s):");
                
                for (int i = 0; i < devices.Length; i++)
                {
                    var device = devices[i];
                    Console.WriteLine($"  Device {i}:");
                    Console.WriteLine($"    Name: {device.Name}");
                    Console.WriteLine($"    Type: {device.AcceleratorType}");
                    Console.WriteLine($"    Memory: {device.MemorySize / (1024 * 1024)} MB");
                    Console.WriteLine($"    Warp Size: {device.WarpSize}");
                    Console.WriteLine($"    Max Threads/Group: {device.MaxNumThreadsPerGroup}");
                }
                
                // Try to manually query for CUDA devices using context
                Console.WriteLine("\nDirect CUDA device enumeration:");
                try
                {
                    var cudaDevices = context.GetCudaDevices();
                    Console.WriteLine($"Found {cudaDevices.Count} CUDA device(s) via context query:");
                    
                    for (int i = 0; i < cudaDevices.Count; i++)
                    {
                        var cudaDevice = cudaDevices[i];
                        Console.WriteLine($"  CUDA Device {i}: {cudaDevice.Name}");
                        Console.WriteLine($"    Memory: {cudaDevice.MemorySize / (1024 * 1024)} MB");
                        Console.WriteLine($"    Max Threads/Group: {cudaDevice.MaxNumThreadsPerGroup}");
                        Console.WriteLine($"    Warp Size: {cudaDevice.WarpSize}");
                        Console.WriteLine($"    Max Grid Size: {cudaDevice.MaxGridSize}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ CUDA device query failed: {ex.Message}");
                    Console.WriteLine($"❌ Stack trace: {ex.StackTrace}");
                }
                
                // Try creating CUDA accelerator directly
                Console.WriteLine("\nTesting direct CUDA accelerator creation:");
                try
                {
                    using var cudaAccelerator = context.CreateCudaAccelerator(0);
                    Console.WriteLine($"✅ CUDA Accelerator created: {cudaAccelerator.Name}");
                    Console.WriteLine($"   Memory: {cudaAccelerator.MemorySize / (1024 * 1024)} MB");
                    Console.WriteLine($"   Warp Size: {cudaAccelerator.WarpSize}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Direct CUDA accelerator creation failed: {ex.Message}");
                    Console.WriteLine($"   This suggests the GPU may not be accessible or CUDA runtime issues");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Diagnostic failed: {ex.Message}");
            }
        }
    }
}