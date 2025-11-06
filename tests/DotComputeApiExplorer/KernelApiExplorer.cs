using System;
using System.Linq;
using System.Reflection;
using DotCompute.Abstractions;

namespace DotComputeApiExplorer;

public static class KernelApiExplorer
{
    public static void ExploreKernelTypes()
    {
        Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine("KernelDefinition Structure:");
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Find KernelDefinition type
        var assembly = typeof(IAccelerator).Assembly;
        var kernelDefType = assembly.GetTypes()
            .FirstOrDefault(t => t.Name == "KernelDefinition");

        if (kernelDefType != null)
        {
            Console.WriteLine($"Type: {kernelDefType.FullName}");
            Console.WriteLine($"Is Struct: {kernelDefType.IsValueType}");
            Console.WriteLine($"Is Class: {kernelDefType.IsClass}");
            Console.WriteLine();

            Console.WriteLine("Properties:");
            foreach (var prop in kernelDefType.GetProperties())
            {
                Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name} {{ get; {(prop.CanWrite ? "set;" : "")} }}");
            }

            Console.WriteLine("\nConstructors:");
            foreach (var ctor in kernelDefType.GetConstructors())
            {
                Console.Write($"  • new KernelDefinition(");
                var parameters = ctor.GetParameters();
                for (int i = 0; i < parameters.Length; i++)
                {
                    var param = parameters[i];
                    Console.Write($"{param.ParameterType.Name} {param.Name}");
                    if (i < parameters.Length - 1) Console.Write(", ");
                }
                Console.WriteLine(")");
            }
        }
        else
        {
            Console.WriteLine("❌ KernelDefinition type not found");
        }

        Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine("CompilationOptions Structure:");
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Find CompilationOptions type
        var compilationOptionsType = assembly.GetTypes()
            .FirstOrDefault(t => t.Name == "CompilationOptions");

        if (compilationOptionsType != null)
        {
            Console.WriteLine($"Type: {compilationOptionsType.FullName}");
            Console.WriteLine();

            Console.WriteLine("Properties:");
            foreach (var prop in compilationOptionsType.GetProperties())
            {
                Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name}");
            }
        }
        else
        {
            Console.WriteLine("❌ CompilationOptions type not found");
        }

        Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine("ICompiledKernel Interface:");
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Find ICompiledKernel type
        var compiledKernelType = assembly.GetTypes()
            .FirstOrDefault(t => t.Name == "ICompiledKernel");

        if (compiledKernelType != null)
        {
            Console.WriteLine($"Type: {compiledKernelType.FullName}");
            Console.WriteLine();

            Console.WriteLine("Methods:");
            foreach (var method in compiledKernelType.GetMethods())
            {
                if (method.Name.StartsWith("get_") || method.Name.StartsWith("set_")) continue;

                Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                foreach (var param in method.GetParameters())
                {
                    Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                }
                Console.WriteLine($"    )");
            }

            Console.WriteLine("\nProperties:");
            foreach (var prop in compiledKernelType.GetProperties())
            {
                Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name}");
            }
        }
        else
        {
            Console.WriteLine("❌ ICompiledKernel type not found");
        }

        Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine("Kernel Language Enum:");
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Find KernelLanguage type
        var kernelLanguageType = assembly.GetTypes()
            .FirstOrDefault(t => t.Name == "KernelLanguage" && t.IsEnum);

        if (kernelLanguageType != null)
        {
            Console.WriteLine($"Type: {kernelLanguageType.FullName}");
            Console.WriteLine();

            Console.WriteLine("Values:");
            foreach (var value in Enum.GetValues(kernelLanguageType))
            {
                Console.WriteLine($"  • {value} = {(int)value}");
            }
        }
        else
        {
            Console.WriteLine("❌ KernelLanguage type not found");
        }
    }
}
