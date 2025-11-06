using System;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using DotCompute.Abstractions;
using DotCompute.Abstractions.Factories;
using DotCompute.Runtime;
using DotCompute.Runtime.Configuration;

Console.WriteLine("=== DotCompute API Explorer v0.4.1-rc2 ===");
Console.WriteLine();

// Setup DI and create accelerator
var hostBuilder = Host.CreateApplicationBuilder();
hostBuilder.Services.AddLogging(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Warning);
});

hostBuilder.Services.Configure<DotComputeRuntimeOptions>(options =>
{
    options.ValidateCapabilities = false;
    options.AcceleratorLifetime = DotCompute.Runtime.Configuration.ServiceLifetime.Transient;
});

hostBuilder.Services.AddDotComputeRuntime();

var host = hostBuilder.Build();
var factory = host.Services.GetRequiredService<IUnifiedAcceleratorFactory>();

try
{
    // Get first CUDA device
    var devices = await factory.GetAvailableDevicesAsync();
    var cudaDevice = devices.FirstOrDefault(d => d.DeviceType == "CUDA");

    if (cudaDevice == null)
    {
        Console.WriteLine("❌ No CUDA device found");
        return;
    }

    Console.WriteLine($"✅ Found CUDA device: {cudaDevice.Name}");
    Console.WriteLine();

    // Create accelerator
    var accelerator = await factory.CreateAsync(cudaDevice);

    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("IAccelerator Interface Methods:");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    var acceleratorType = accelerator.GetType();
    var interfaceTypes = acceleratorType.GetInterfaces();

    foreach (var interfaceType in interfaceTypes.Where(t => t.Namespace?.Contains("DotCompute") == true))
    {
        Console.WriteLine($"\n📦 Interface: {interfaceType.FullName}");
        Console.WriteLine();

        var methods = interfaceType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => !m.Name.StartsWith("get_") && !m.Name.StartsWith("set_"))
            .OrderBy(m => m.Name);

        foreach (var method in methods)
        {
            Console.WriteLine($"  ▸ {method.ReturnType.Name} {method.Name}(");

            var parameters = method.GetParameters();
            for (int i = 0; i < parameters.Length; i++)
            {
                var param = parameters[i];
                var comma = i < parameters.Length - 1 ? "," : "";
                Console.WriteLine($"      {param.ParameterType.Name} {param.Name}{comma}");
            }
            Console.WriteLine($"    )");
        }
    }

    // Check for kernel-related methods
    Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("Kernel-Related Methods:");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    var allMethods = acceleratorType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
        .Where(m => m.Name.Contains("Kernel", StringComparison.OrdinalIgnoreCase) ||
                    m.Name.Contains("Compile", StringComparison.OrdinalIgnoreCase) ||
                    m.Name.Contains("Execute", StringComparison.OrdinalIgnoreCase))
        .OrderBy(m => m.Name);

    foreach (var method in allMethods)
    {
        Console.WriteLine($"\n  ✓ {method.ReturnType.Name} {method.Name}");
        foreach (var param in method.GetParameters())
        {
            Console.WriteLine($"      • {param.ParameterType.Name} {param.Name}");
        }
    }

    // Check Properties
    Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("IAccelerator Properties:");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    var properties = acceleratorType.GetProperties(BindingFlags.Public | BindingFlags.Instance);
    foreach (var prop in properties)
    {
        Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name}");
    }

    // Check for any orchestrator-type services
    Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("Registered Services:");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    var allServices = host.Services.GetType()
        .GetProperty("ServiceDescriptors", BindingFlags.NonPublic | BindingFlags.Instance)?
        .GetValue(host.Services) as System.Collections.IEnumerable;

    if (allServices != null)
    {
        var dotComputeServices = allServices.Cast<object>()
            .Where(s => s?.ToString()?.Contains("DotCompute") == true)
            .Take(20);

        foreach (var service in dotComputeServices)
        {
            Console.WriteLine($"  • {service}");
        }
    }

    // Explore kernel-related types
    DotComputeApiExplorer.KernelApiExplorer.ExploreKernelTypes();

    // Explore kernel execution API
    var logger = host.Services.GetRequiredService<ILogger<Program>>();
    await DotComputeApiExplorer.ExecutionApiExplorer.ExploreExecutionApi(accelerator, logger);

    // Explore kernel arguments and launch configuration API
    await DotComputeApiExplorer.ArgumentsApiExplorer.ExploreArgumentsApi(accelerator, logger);

    Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("API Discovery Complete!");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
catch (Exception ex)
{
    Console.WriteLine($"\n❌ ERROR: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
}
