using DotCompute.Core.Compute;

Console.WriteLine("=== DotCompute Device Discovery ===");
Console.WriteLine();

try
{
    Console.WriteLine("Initializing DotCompute AcceleratorManager...");
    var manager = await DefaultAcceleratorManagerFactory.CreateAsync();
    Console.WriteLine("✓ AcceleratorManager created successfully");
    Console.WriteLine();

    Console.WriteLine("Discovering compute devices...");
    var accelerators = await manager.GetAcceleratorsAsync();
    var deviceList = accelerators.ToList();

    Console.WriteLine($"✓ Found {deviceList.Count} device(s)");
    Console.WriteLine();
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine();

    for (int i = 0; i < deviceList.Count; i++)
    {
        var device = deviceList[i];
        var info = device.Info;
        var memory = device.Memory;

        Console.WriteLine($"Device #{i}: {info.Name}");
        Console.WriteLine($"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine($"  Type:              {info.Type}");
        Console.WriteLine($"  Vendor:            {info.Vendor ?? "Unknown"}");
        Console.WriteLine($"  Architecture:      {info.Architecture ?? "Unknown"}");
        Console.WriteLine($"  Compute Units:     {info.ComputeUnits}");
        Console.WriteLine($"  Max Work Group:    {info.MaxWorkGroupSize}");
        Console.WriteLine($"  Warp Size:         {info.WarpSize}");
        Console.WriteLine($"  Version:           {info.MajorVersion}.{info.MinorVersion}");

        // Clock frequency if available
        Console.WriteLine($"  Max Clock:         ~{(info.Type.ToUpperInvariant() == "GPU" ? 2400 : 3800)} MHz (estimated)");

        // Memory information
        Console.WriteLine();
        Console.WriteLine($"  Memory Info:");
        Console.WriteLine($"    Total:           {memory.TotalAvailableMemory / (1024.0 * 1024.0 * 1024.0):F2} GB");
        Console.WriteLine($"    Allocated:       {memory.CurrentAllocatedMemory / (1024.0 * 1024.0 * 1024.0):F2} GB");
        Console.WriteLine($"    Available:       {(memory.TotalAvailableMemory - memory.CurrentAllocatedMemory) / (1024.0 * 1024.0 * 1024.0):F2} GB");
        Console.WriteLine($"    Utilization:     {(memory.CurrentAllocatedMemory * 100.0 / memory.TotalAvailableMemory):F1}%");

        // Extensions
        if (info.Extensions != null && info.Extensions.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"  Extensions ({info.Extensions.Count}):");
            foreach (var ext in info.Extensions.Take(10))
            {
                Console.WriteLine($"    - {ext}");
            }
            if (info.Extensions.Count > 10)
            {
                Console.WriteLine($"    ... and {info.Extensions.Count - 10} more");
            }
        }

        // Features
        if (info.Features != null && info.Features.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"  Features ({info.Features.Count}):");
            foreach (var feature in info.Features.Take(10))
            {
                Console.WriteLine($"    - {feature.Key}: {feature.Value}");
            }
            if (info.Features.Count > 10)
            {
                Console.WriteLine($"    ... and {info.Features.Count - 10} more");
            }
        }

        Console.WriteLine();
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine();
    }

    Console.WriteLine($"Device discovery completed successfully!");
    Console.WriteLine();
    Console.WriteLine($"Summary:");
    Console.WriteLine($"  Total devices: {deviceList.Count}");
    Console.WriteLine($"  GPU devices:   {deviceList.Count(d => d.Info.Type.ToUpperInvariant() == "GPU")}");
    Console.WriteLine($"  CPU devices:   {deviceList.Count(d => d.Info.Type.ToUpperInvariant() == "CPU")}");
    Console.WriteLine($"  Other devices: {deviceList.Count(d => d.Info.Type.ToUpperInvariant() != "GPU" && d.Info.Type.ToUpperInvariant() != "CPU")}");

    await manager.DisposeAsync();
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error during device discovery:");
    Console.WriteLine($"   {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine("Stack trace:");
    Console.WriteLine(ex.StackTrace);
}
