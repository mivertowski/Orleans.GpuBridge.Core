using System;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using Microsoft.Extensions.Logging;

namespace DotComputeApiExplorer;

public static class ArgumentsApiExplorer
{
    public static async Task ExploreArgumentsApi(IAccelerator accelerator, ILogger logger)
    {
        Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine("Kernel Arguments & Launch Configuration Discovery:");
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        try
        {
            var assembly = typeof(IAccelerator).Assembly;

            // ═══════════════════════════════════════════════════════
            // 1. Discover KernelArguments Type
            // ═══════════════════════════════════════════════════════
            Console.WriteLine("\n📦 KernelArguments Structure:");
            Console.WriteLine("─────────────────────────────────────────────────────");

            var kernelArgumentsType = assembly.GetTypes()
                .FirstOrDefault(t => t.Name == "KernelArguments");

            if (kernelArgumentsType != null)
            {
                Console.WriteLine($"Type: {kernelArgumentsType.FullName}");
                Console.WriteLine($"Is Class: {kernelArgumentsType.IsClass}");
                Console.WriteLine($"Is Struct: {kernelArgumentsType.IsValueType}");
                Console.WriteLine();

                // Constructors
                Console.WriteLine("Constructors:");
                foreach (var ctor in kernelArgumentsType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
                {
                    Console.Write($"  • new KernelArguments(");
                    var parameters = ctor.GetParameters();
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        var param = parameters[i];
                        Console.Write($"{param.ParameterType.Name} {param.Name}");
                        if (i < parameters.Length - 1) Console.Write(", ");
                    }
                    Console.WriteLine(")");
                }

                // Static factory methods
                Console.WriteLine("\nStatic Factory Methods:");
                var staticMethods = kernelArgumentsType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => !m.Name.StartsWith("get_") && !m.Name.StartsWith("set_"));
                foreach (var method in staticMethods)
                {
                    Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                    foreach (var param in method.GetParameters())
                    {
                        Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                    }
                    Console.WriteLine("    )");
                }

                // Instance methods
                Console.WriteLine("\nInstance Methods:");
                var instanceMethods = kernelArgumentsType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .Where(m => !m.Name.StartsWith("get_") && !m.Name.StartsWith("set_") &&
                               !m.DeclaringType.Name.Contains("Object"));
                foreach (var method in instanceMethods)
                {
                    Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                    foreach (var param in method.GetParameters())
                    {
                        Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                    }
                    Console.WriteLine("    )");
                }

                // Properties
                Console.WriteLine("\nProperties:");
                foreach (var prop in kernelArgumentsType.GetProperties())
                {
                    Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name}");
                }
            }
            else
            {
                Console.WriteLine("❌ KernelArguments type not found");
            }

            // ═══════════════════════════════════════════════════════
            // 2. Discover CudaLaunchConfig Type
            // ═══════════════════════════════════════════════════════
            Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("📦 CudaLaunchConfig Structure:");
            Console.WriteLine("─────────────────────────────────────────────────────");

            var cudaLaunchConfigType = assembly.GetTypes()
                .FirstOrDefault(t => t.Name == "CudaLaunchConfig");

            if (cudaLaunchConfigType != null)
            {
                Console.WriteLine($"Type: {cudaLaunchConfigType.FullName}");
                Console.WriteLine($"Is Class: {cudaLaunchConfigType.IsClass}");
                Console.WriteLine($"Is Struct: {cudaLaunchConfigType.IsValueType}");
                Console.WriteLine();

                // Constructors
                Console.WriteLine("Constructors:");
                foreach (var ctor in cudaLaunchConfigType.GetConstructors(BindingFlags.Public | BindingFlags.Instance))
                {
                    Console.Write($"  • new CudaLaunchConfig(");
                    var parameters = ctor.GetParameters();
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        var param = parameters[i];
                        Console.Write($"{param.ParameterType.Name} {param.Name}");
                        if (i < parameters.Length - 1) Console.Write(", ");
                    }
                    Console.WriteLine(")");
                }

                // Static factory methods
                Console.WriteLine("\nStatic Factory Methods:");
                var staticMethods = cudaLaunchConfigType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => !m.Name.StartsWith("get_") && !m.Name.StartsWith("set_"));
                foreach (var method in staticMethods)
                {
                    Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                    foreach (var param in method.GetParameters())
                    {
                        Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                    }
                    Console.WriteLine("    )");
                }

                // Properties
                Console.WriteLine("\nProperties:");
                foreach (var prop in cudaLaunchConfigType.GetProperties())
                {
                    Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name}");
                }
            }
            else
            {
                Console.WriteLine("❌ CudaLaunchConfig type not found");
            }

            // ═══════════════════════════════════════════════════════
            // 3. Explore Memory Buffer Types
            // ═══════════════════════════════════════════════════════
            Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("📦 Memory Buffer Types:");
            Console.WriteLine("─────────────────────────────────────────────────────");

            // Look for IDeviceMemory, IMemoryBuffer, etc.
            var memoryTypes = assembly.GetTypes()
                .Where(t => t.Name.Contains("Memory", StringComparison.OrdinalIgnoreCase) ||
                           t.Name.Contains("Buffer", StringComparison.OrdinalIgnoreCase))
                .Take(10);

            foreach (var memType in memoryTypes)
            {
                Console.WriteLine($"\n  ▸ {memType.FullName}");
                if (memType.IsInterface)
                    Console.WriteLine("    [Interface]");
                if (memType.IsClass)
                    Console.WriteLine("    [Class]");
                if (memType.IsValueType)
                    Console.WriteLine("    [Struct]");
            }

            // ═══════════════════════════════════════════════════════
            // 4. Explore IUnifiedMemoryManager Methods
            // ═══════════════════════════════════════════════════════
            Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("📦 IUnifiedMemoryManager Methods:");
            Console.WriteLine("─────────────────────────────────────────────────────");

            var memoryManager = accelerator.GetType().GetProperty("Memory")?.GetValue(accelerator);
            if (memoryManager != null)
            {
                var memManagerType = memoryManager.GetType();
                Console.WriteLine($"Type: {memManagerType.FullName}");
                Console.WriteLine();

                var allocMethods = memManagerType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .Where(m => m.Name.Contains("Alloc", StringComparison.OrdinalIgnoreCase) ||
                               m.Name.Contains("Copy", StringComparison.OrdinalIgnoreCase) ||
                               m.Name.Contains("Free", StringComparison.OrdinalIgnoreCase));

                foreach (var method in allocMethods)
                {
                    Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                    foreach (var param in method.GetParameters())
                    {
                        Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                    }
                    Console.WriteLine("    )");
                }
            }

            // ═══════════════════════════════════════════════════════
            // 5. Practical Example: Try to Create Arguments
            // ═══════════════════════════════════════════════════════
            Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("🔬 Attempting to Create Sample Arguments:");
            Console.WriteLine("─────────────────────────────────────────────────────");

            if (kernelArgumentsType != null)
            {
                try
                {
                    // Try default constructor
                    var argsCtor = kernelArgumentsType.GetConstructor(Type.EmptyTypes);
                    if (argsCtor != null)
                    {
                        var args = Activator.CreateInstance(kernelArgumentsType);
                        Console.WriteLine("✅ Created KernelArguments via default constructor");

                        // Try to add arguments
                        var addMethods = kernelArgumentsType.GetMethods()
                            .Where(m => m.Name.Contains("Add", StringComparison.OrdinalIgnoreCase));

                        Console.WriteLine($"\nFound {addMethods.Count()} Add methods:");
                        foreach (var method in addMethods)
                        {
                            Console.WriteLine($"  • {method.Name}({string.Join(", ", method.GetParameters().Select(p => p.ParameterType.Name))})");
                        }
                    }
                    else
                    {
                        Console.WriteLine("⚠️ No default constructor found for KernelArguments");

                        // Try to find factory methods
                        var createMethods = kernelArgumentsType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                            .Where(m => m.Name.Contains("Create", StringComparison.OrdinalIgnoreCase));

                        if (createMethods.Any())
                        {
                            Console.WriteLine("\nFound factory methods:");
                            foreach (var method in createMethods)
                            {
                                Console.WriteLine($"  • {method.Name}({string.Join(", ", method.GetParameters().Select(p => p.ParameterType.Name))})");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error creating arguments: {ex.Message}");
                }
            }

            Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("✅ Arguments & Launch Configuration Discovery Complete!");
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ ERROR: {ex.GetType().Name}: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
