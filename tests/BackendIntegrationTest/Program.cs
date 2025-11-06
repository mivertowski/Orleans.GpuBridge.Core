using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

Console.WriteLine("=== Orleans.GpuBridge DotCompute Backend Integration Test ===");
Console.WriteLine();

// Create logger
using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Debug);
});

var logger = loggerFactory.CreateLogger<DotComputeDeviceManager>();

// Create device manager
var deviceManager = new DotComputeDeviceManager(logger);

Console.WriteLine("Initializing device manager...");
Console.WriteLine();

try
{
    // Initialize (this should discover devices using new API)
    await deviceManager.InitializeAsync();

    Console.WriteLine("✅ Device manager initialized successfully!");
    Console.WriteLine();

    // Get discovered devices
    var devices = deviceManager.GetDevices();

    Console.WriteLine($"Discovered {devices.Count} device(s):");
    Console.WriteLine();

    foreach (var device in devices)
    {
        Console.WriteLine($"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine($"📱 Device: {device.Name}");
        Console.WriteLine($"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine($"  ID:                {device.DeviceId}");
        Console.WriteLine($"  Type:              {device.Type}");
        Console.WriteLine($"  Architecture:      {device.Architecture}");
        Console.WriteLine($"  Compute Units:     {device.ComputeUnits}");
        Console.WriteLine($"  Total Memory:      {device.TotalMemoryBytes / (1024.0 * 1024 * 1024):F2} GB");
        Console.WriteLine($"  Available Memory:  {device.AvailableMemoryBytes / (1024.0 * 1024 * 1024):F2} GB");
        Console.WriteLine($"  Max Threads:       {device.MaxThreadsPerBlock}");
        Console.WriteLine($"  Warp Size:         {device.WarpSize}");
        Console.WriteLine();
    }

    // Summary
    Console.WriteLine("═══════════════════════════════════════════════════════");
    Console.WriteLine("  Summary");
    Console.WriteLine("═══════════════════════════════════════════════════════");
    Console.WriteLine($"Total devices:   {devices.Count}");

    var cudaDevices = devices.Count(d => d.Type == Orleans.GpuBridge.Abstractions.Enums.DeviceType.CUDA);
    var openclDevices = devices.Count(d => d.Type == Orleans.GpuBridge.Abstractions.Enums.DeviceType.OpenCL);
    var cpuDevices = devices.Count(d => d.Type == Orleans.GpuBridge.Abstractions.Enums.DeviceType.CPU);

    Console.WriteLine($"CUDA devices:    {cudaDevices}");
    Console.WriteLine($"OpenCL devices:  {openclDevices}");
    Console.WriteLine($"CPU devices:     {cpuDevices}");
    Console.WriteLine();

    // Check for RTX 2000 Ada
    var rtxDevice = devices.FirstOrDefault(d =>
        d.Name.Contains("RTX 2000", StringComparison.OrdinalIgnoreCase) ||
        d.Name.Contains("Ada", StringComparison.OrdinalIgnoreCase));

    if (rtxDevice != null)
    {
        Console.WriteLine("🎉 SUCCESS: NVIDIA RTX 2000 Ada Generation detected!");
        Console.WriteLine($"   Device ID: {rtxDevice.DeviceId}");
        Console.WriteLine($"   Device Type: {rtxDevice.Type}");
        Console.WriteLine($"   Architecture: {rtxDevice.Architecture}");
        Console.WriteLine();
        Console.WriteLine("   ✅ Orleans.GpuBridge backend is ready for GPU acceleration!");
        Console.WriteLine();
    }
    else if (cudaDevices > 0)
    {
        Console.WriteLine("✅ CUDA device(s) detected");
        Console.WriteLine();
    }
    else if (devices.Count > 0)
    {
        Console.WriteLine("✅ Compute device(s) detected (CPU/OpenCL)");
        Console.WriteLine();
    }

    // Test default device selection
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("Testing default device selection...");
    var defaultDevice = deviceManager.GetDefaultDevice();
    Console.WriteLine($"✅ Default device: {defaultDevice.Name} ({defaultDevice.Type})");
    Console.WriteLine();

    // Dispose
    deviceManager.Dispose();
    Console.WriteLine("✅ Device manager disposed successfully");
    Console.WriteLine();
}
catch (Exception ex)
{
    Console.WriteLine($"❌ ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("Stack trace:");
    Console.WriteLine(ex.StackTrace);

    if (ex.InnerException != null)
    {
        Console.WriteLine();
        Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
        Console.WriteLine(ex.InnerException.StackTrace);
    }
}
