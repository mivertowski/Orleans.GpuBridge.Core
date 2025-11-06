using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using DotCompute.Abstractions.Factories;
using DotCompute.Runtime.Configuration;
using DotCompute.Runtime.Factories;

Console.WriteLine("=== DotCompute Device Discovery Test (v0.4.0-rc2) ===");
Console.WriteLine();

// Setup DI container
var services = new ServiceCollection();
services.AddLogging(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Information);
});

services.Configure<DotComputeRuntimeOptions>(options =>
{
    options.ValidateCapabilities = false;
    options.AcceleratorLifetime = DotCompute.Runtime.Configuration.ServiceLifetime.Transient;
});

services.AddSingleton<IUnifiedAcceleratorFactory, DefaultAcceleratorFactory>();

var serviceProvider = services.BuildServiceProvider();
var factory = serviceProvider.GetRequiredService<IUnifiedAcceleratorFactory>();

Console.WriteLine("Enumerating devices...");
Console.WriteLine();

try
{
    var devices = await factory.GetAvailableDevicesAsync();

    Console.WriteLine($"✅ Found {devices.Count} device(s)");
    Console.WriteLine();

    if (devices.Count == 0)
    {
        Console.WriteLine("❌ ERROR: No devices found!");
        Console.WriteLine();
        Console.WriteLine("System Configuration:");
        Console.WriteLine("  - OpenCL platform: Intel(R) OpenCL Graphics (integrated GPU)");
        Console.WriteLine("  - NVIDIA GPU: RTX 2000 Ada Generation (visible to nvidia-smi)");
        Console.WriteLine("  - CUDA: Version 13.0");
        Console.WriteLine("  - Driver: 581.15");
        Console.WriteLine("  - Environment: WSL2");
        Console.WriteLine();
    }
    else
    {
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine();

        foreach (var device in devices)
        {
            Console.WriteLine($"📱 Device: {device.Name}");
            Console.WriteLine($"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine($"  Type:              {device.DeviceType}");
            Console.WriteLine($"  Vendor:            {device.Vendor}");
            Console.WriteLine($"  Memory:            {device.TotalMemory / (1024.0 * 1024 * 1024):F2} GB");
            Console.WriteLine($"  Compute Units:     {device.MaxComputeUnits}");
            Console.WriteLine($"  Max Threads:       {device.MaxThreadsPerBlock}");

            if (device.DeviceType == "CUDA" && device.ComputeCapability != null)
            {
                Console.WriteLine($"  Compute Capability: {device.ComputeCapability}");
                Console.WriteLine($"  Architecture:      Ada Lovelace (based on RTX 2000 Ada)");
            }

            // Display additional properties if available
            if (device.Extensions != null && device.Extensions.Any())
            {
                Console.WriteLine();
                Console.WriteLine($"  Extensions ({device.Extensions.Count()}):");
                foreach (var ext in device.Extensions.Take(10))
                {
                    Console.WriteLine($"    - {ext}");
                }
                if (device.Extensions.Count() > 10)
                {
                    Console.WriteLine($"    ... and {device.Extensions.Count() - 10} more");
                }
            }

            Console.WriteLine();
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine();
        }

        // Check for specific device types
        var cpuDevices = devices.Count(d => d.DeviceType == "CPU");
        var cudaDevices = devices.Count(d => d.DeviceType == "CUDA");
        var openclDevices = devices.Count(d => d.DeviceType == "OpenCL");
        var metalDevices = devices.Count(d => d.DeviceType == "Metal");

        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine("  Summary");
        Console.WriteLine("═══════════════════════════════════════════════════════");
        Console.WriteLine($"Total devices:   {devices.Count}");
        Console.WriteLine($"CPU devices:     {cpuDevices}");
        Console.WriteLine($"CUDA devices:    {cudaDevices}");
        Console.WriteLine($"OpenCL devices:  {openclDevices}");
        Console.WriteLine($"Metal devices:   {metalDevices}");
        Console.WriteLine();

        Console.WriteLine("✅ Device discovery is working correctly!");
        Console.WriteLine();

        // Highlight if RTX 2000 Ada is detected
        var rtxDevice = devices.FirstOrDefault(d =>
            d.Name.Contains("RTX 2000", StringComparison.OrdinalIgnoreCase) ||
            d.Name.Contains("Ada", StringComparison.OrdinalIgnoreCase));

        if (rtxDevice != null)
        {
            Console.WriteLine("🎉 SUCCESS: NVIDIA RTX 2000 Ada Generation detected!");
            Console.WriteLine($"   Device: {rtxDevice.Name}");
            Console.WriteLine($"   Memory: {rtxDevice.TotalMemory / (1024.0 * 1024 * 1024):F2} GB");
            Console.WriteLine($"   Expected speedup: 92x (as per DotCompute 0.4.0-rc2 release notes)");
            Console.WriteLine();
        }
    }
}
catch (Exception ex)
{
    Console.WriteLine($"❌ ERROR during device enumeration:");
    Console.WriteLine($"   {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("Stack trace:");
    Console.WriteLine(ex.StackTrace);

    if (ex.InnerException != null)
    {
        Console.WriteLine();
        Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
    }
}
