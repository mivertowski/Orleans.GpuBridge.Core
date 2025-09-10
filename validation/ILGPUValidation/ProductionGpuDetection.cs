using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Examples
{
    /// <summary>
    /// Production-ready GPU detection with comprehensive error handling
    /// Handles WSL2 limitations and provides clear user guidance
    /// </summary>
    public static class ProductionGpuDetection
    {
        public record GpuDetectionResult(
            bool GpuAvailable,
            bool NativeCudaAvailable, 
            bool IlgpuCudaAvailable,
            int CudaDeviceCount,
            int IlgpuDeviceCount,
            string Platform,
            string Recommendation,
            string DetailedStatus
        );

        [DllImport("cudart", EntryPoint = "cudaGetDeviceCount", CallingConvention = CallingConvention.Cdecl)]
        private static extern int CudaGetDeviceCount(out int count);

        /// <summary>
        /// Comprehensive GPU detection for production use
        /// </summary>
        public static GpuDetectionResult DetectGpuCapabilities()
        {
            Console.WriteLine("🔍 Production GPU Detection");
            Console.WriteLine("============================");

            var platform = DetectPlatform();
            Console.WriteLine($"Platform: {platform}");

            // Test 1: Native CUDA detection
            var (nativeCudaAvailable, cudaDeviceCount) = TestNativeCuda();
            
            // Test 2: ILGPU detection
            var (ilgpuCudaAvailable, ilgpuDeviceCount, ilgpuDetails) = TestIlgpuCuda();

            // Analysis and recommendations
            var recommendation = GenerateRecommendation(nativeCudaAvailable, ilgpuCudaAvailable, platform);
            var detailedStatus = GenerateDetailedStatus(nativeCudaAvailable, ilgpuCudaAvailable, cudaDeviceCount, ilgpuDeviceCount, platform);

            var result = new GpuDetectionResult(
                GpuAvailable: ilgpuCudaAvailable,
                NativeCudaAvailable: nativeCudaAvailable,
                IlgpuCudaAvailable: ilgpuCudaAvailable,
                CudaDeviceCount: cudaDeviceCount,
                IlgpuDeviceCount: ilgpuDeviceCount,
                Platform: platform,
                Recommendation: recommendation,
                DetailedStatus: detailedStatus
            );

            PrintSummary(result);
            return result;
        }

        private static string DetectPlatform()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return "Windows";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // Check if running in WSL
                if (System.IO.File.Exists("/proc/version"))
                {
                    var version = System.IO.File.ReadAllText("/proc/version");
                    if (version.Contains("WSL", StringComparison.OrdinalIgnoreCase) || 
                        version.Contains("Microsoft", StringComparison.OrdinalIgnoreCase))
                    {
                        return "WSL2";
                    }
                }
                return "Linux";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return "macOS";
            }
            
            return "Unknown";
        }

        private static (bool available, int deviceCount) TestNativeCuda()
        {
            Console.WriteLine("\n📋 Testing Native CUDA API...");
            
            try
            {
                int deviceCount = 0;
                int result = CudaGetDeviceCount(out deviceCount);
                
                if (result == 0 && deviceCount > 0)
                {
                    Console.WriteLine($"✅ Native CUDA: {deviceCount} device(s) detected");
                    return (true, deviceCount);
                }
                else
                {
                    Console.WriteLine($"❌ Native CUDA: Error code {result} or no devices");
                    return (false, 0);
                }
            }
            catch (DllNotFoundException)
            {
                Console.WriteLine("❌ Native CUDA: Runtime library not found");
                return (false, 0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Native CUDA: Unexpected error - {ex.Message}");
                return (false, 0);
            }
        }

        private static (bool available, int deviceCount, string details) TestIlgpuCuda()
        {
            Console.WriteLine("\n📋 Testing ILGPU CUDA Detection...");
            
            try
            {
                using var context = Context.CreateDefault();
                
                // Test general device enumeration
                var allDevices = context.Devices;
                Console.WriteLine($"ILGPU Total Devices: {allDevices.Length}");
                
                foreach (var device in allDevices)
                {
                    Console.WriteLine($"  - {device.Name} ({device.AcceleratorType})");
                }

                // Test CUDA-specific detection by checking device types
                int cudaCount = 0;
                foreach (var device in allDevices)
                {
                    if (device.AcceleratorType == AcceleratorType.Cuda)
                    {
                        cudaCount++;
                    }
                }

                if (cudaCount > 0)
                {
                    Console.WriteLine($"✅ ILGPU CUDA: {cudaCount} device(s) detected");
                    
                    // Test accelerator creation - find first CUDA device
                    try
                    {
                        Device? cudaDevice = null;
                        foreach (var device in allDevices)
                        {
                            if (device.AcceleratorType == AcceleratorType.Cuda)
                            {
                                cudaDevice = device;
                                break;
                            }
                        }
                        
                        if (cudaDevice != null)
                        {
                            using var accelerator = cudaDevice.CreateAccelerator(context);
                            Console.WriteLine($"✅ CUDA Accelerator created: {accelerator.Name}");
                            return (true, cudaCount, $"Successfully created CUDA accelerator: {accelerator.Name}");
                        }
                        else
                        {
                            return (false, cudaCount, "CUDA device found but could not create accelerator");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"❌ CUDA Accelerator creation failed: {ex.Message}");
                        return (false, cudaCount, $"Device detected but accelerator creation failed: {ex.Message}");
                    }
                }
                else
                {
                    Console.WriteLine("❌ ILGPU CUDA: No CUDA devices detected");
                    return (false, 0, "ILGPU detected no CUDA devices");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ ILGPU CUDA: Context creation failed - {ex.Message}");
                return (false, 0, $"ILGPU context creation failed: {ex.Message}");
            }
        }

        private static string GenerateRecommendation(bool nativeCuda, bool ilgpuCuda, string platform)
        {
            return (nativeCuda, ilgpuCuda, platform) switch
            {
                (true, true, _) => 
                    "✅ GPU acceleration available. Ready for production GPU workloads.",
                
                (true, false, "WSL2") => 
                    "⚠️ GPU detected but not accessible to ILGPU. WSL2 limitation. Use CPU fallback or deploy to native Linux/Windows.",
                
                (true, false, _) => 
                    "⚠️ Native CUDA works but ILGPU cannot access GPU. Check ILGPU installation and compatibility.",
                
                (false, false, "WSL2") => 
                    "❌ No GPU detected. Verify NVIDIA drivers and WSL2 GPU support configuration.",
                
                (false, false, _) => 
                    "❌ No GPU detected. Verify NVIDIA GPU is present and drivers are installed.",
                
                _ => "❓ Unexpected detection state. Manual investigation required."
            };
        }

        private static string GenerateDetailedStatus(bool nativeCuda, bool ilgpuCuda, int cudaCount, int ilgpuCount, string platform)
        {
            var status = $"Platform: {platform}\n";
            status += $"Native CUDA: {(nativeCuda ? $"✅ ({cudaCount} devices)" : "❌")}\n";
            status += $"ILGPU CUDA: {(ilgpuCuda ? $"✅ ({ilgpuCount} devices)" : "❌")}\n";
            
            if (platform == "WSL2" && nativeCuda && !ilgpuCuda)
            {
                status += "\nWSL2 Limitation Details:\n";
                status += "- CUDA runtime works but device files not accessible to .NET\n";
                status += "- /dev/nvidia* files missing for user-space access\n";
                status += "- Consider native Linux deployment for full GPU access\n";
            }
            
            return status;
        }

        private static void PrintSummary(GpuDetectionResult result)
        {
            Console.WriteLine("\n📊 DETECTION SUMMARY");
            Console.WriteLine("====================");
            Console.WriteLine($"GPU Available for ILGPU: {(result.GpuAvailable ? "✅ YES" : "❌ NO")}");
            Console.WriteLine($"Platform: {result.Platform}");
            Console.WriteLine($"Native CUDA Devices: {result.CudaDeviceCount}");
            Console.WriteLine($"ILGPU CUDA Devices: {result.IlgpuDeviceCount}");
            Console.WriteLine();
            Console.WriteLine("💡 RECOMMENDATION:");
            Console.WriteLine(result.Recommendation);
            Console.WriteLine();
            Console.WriteLine("📋 DETAILED STATUS:");
            Console.WriteLine(result.DetailedStatus);
        }

        /// <summary>
        /// Quick check for production code - returns true if GPU is available for ILGPU
        /// </summary>
        public static bool IsGpuAvailable(out string reason)
        {
            try
            {
                using var context = Context.CreateDefault();
                var allDevices = context.Devices;
                
                // Find CUDA devices
                Device? cudaDevice = null;
                foreach (var device in allDevices)
                {
                    if (device.AcceleratorType == AcceleratorType.Cuda)
                    {
                        cudaDevice = device;
                        break;
                    }
                }
                
                if (cudaDevice != null)
                {
                    // Test accelerator creation
                    using var accelerator = cudaDevice.CreateAccelerator(context);
                    reason = $"GPU available: {accelerator.Name}";
                    return true;
                }
                else
                {
                    reason = "No CUDA devices detected by ILGPU";
                    return false;
                }
            }
            catch (Exception ex)
            {
                reason = $"GPU check failed: {ex.Message}";
                return false;
            }
        }

        public static void Main(string[] args)
        {
            var result = DetectGpuCapabilities();
            
            // Exit code for automated systems
            Environment.Exit(result.GpuAvailable ? 0 : 1);
        }
    }
}