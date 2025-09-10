using System;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

namespace Orleans.GpuBridge.Examples
{
    /// <summary>
    /// Minimal ILGPU diagnostic to isolate initialization issues
    /// </summary>
    public class DiagnosticProgram
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("🔍 ILGPU Diagnostic Starting...");
            
            try
            {
                Console.WriteLine("Step 1: Checking ILGPU Assembly...");
                var assembly = typeof(Context).Assembly;
                Console.WriteLine($"✅ ILGPU Assembly: {assembly.FullName}");
                Console.WriteLine($"✅ Assembly Location: {assembly.Location}");

                Console.WriteLine("Step 2: Creating ILGPU Context...");
                Context? context = null;
                
                try
                {
                    context = Context.CreateDefault();
                    Console.WriteLine("✅ ILGPU Context Created Successfully");
                    
                    Console.WriteLine("Step 3: Enumerating Devices...");
                    var devices = context.Devices;
                    Console.WriteLine($"✅ Found {devices.Length} devices:");
                    
                    for (int i = 0; i < devices.Length; i++)
                    {
                        var device = devices[i];
                        Console.WriteLine($"  Device {i}: {device.Name}");
                        Console.WriteLine($"    Type: {device.AcceleratorType}");
                        Console.WriteLine($"    Memory: {device.MemorySize / (1024 * 1024)} MB");
                        Console.WriteLine($"    Threads/Group: {device.MaxNumThreadsPerGroup}");
                        Console.WriteLine($"    Groups: {device.MaxNumThreadsPerMultiprocessor}");
                    }

                    // Test creating accelerators
                    Console.WriteLine("Step 4: Testing Accelerator Creation...");
                    try
                    {
                        for (int i = 0; i < devices.Length; i++)
                        {
                            var device = devices[i];
                            try
                            {
                                using var accelerator = device.CreateAccelerator(context);
                                Console.WriteLine($"✅ Accelerator {i} Created: {accelerator.Name}");
                                Console.WriteLine($"   Memory: {accelerator.MemorySize / (1024 * 1024)} MB");
                                Console.WriteLine($"   Warp Size: {accelerator.WarpSize}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"❌ Accelerator {i} Failed: {ex.Message}");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"❌ Accelerator creation failed: {ex.Message}");
                        Console.WriteLine($"   Stack Trace: {ex.StackTrace}");
                    }

                    Console.WriteLine("✅ ILGPU Diagnostic Completed Successfully");
                }
                finally
                {
                    context?.Dispose();
                }
            }
            catch (TypeInitializationException ex)
            {
                Console.WriteLine($"❌ ILGPU Type Initialization Failed: {ex.Message}");
                Console.WriteLine($"   Inner Exception: {ex.InnerException?.Message}");
                Console.WriteLine($"   Stack Trace: {ex.StackTrace}");
                
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"   Inner Stack Trace: {ex.InnerException.StackTrace}");
                }
                Environment.Exit(1);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Unexpected Error: {ex.Message}");
                Console.WriteLine($"   Stack Trace: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }
    }
}