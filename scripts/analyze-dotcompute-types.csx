#!/usr/bin/env dotnet-script
// Script to analyze DotCompute package types using reflection

using System;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;

var nugetPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");

var packages = new Dictionary<string, string>
{
    ["DotCompute.Abstractions"] = Path.Combine(nugetPath, "dotcompute.abstractions/0.2.0-alpha/lib/net9.0/DotCompute.Abstractions.dll"),
    ["DotCompute.Core"] = Path.Combine(nugetPath, "dotcompute.core/0.2.0-alpha/lib/net9.0/DotCompute.Core.dll"),
    ["DotCompute.Runtime"] = Path.Combine(nugetPath, "dotcompute.runtime/0.2.0-alpha/lib/net9.0/DotCompute.Runtime.dll"),
    ["DotCompute.Memory"] = Path.Combine(nugetPath, "dotcompute.memory/0.2.0-alpha/lib/net9.0/DotCompute.Memory.dll"),
    ["DotCompute.Backends.CUDA"] = Path.Combine(nugetPath, "dotcompute.backends.cuda/0.2.0-alpha/lib/net9.0/DotCompute.Backends.CUDA.dll"),
    ["DotCompute.Backends.OpenCL"] = Path.Combine(nugetPath, "dotcompute.backends.opencl/0.2.0-alpha/lib/net9.0/DotCompute.Backends.OpenCL.dll"),
    ["DotCompute.Plugins"] = Path.Combine(nugetPath, "dotcompute.plugins/0.2.0-alpha/lib/net9.0/DotCompute.Plugins.dll")
};

Console.WriteLine("# DotCompute Package Type Analysis");
Console.WriteLine("## Package Version: 0.2.0-alpha\n");

foreach (var package in packages)
{
    if (!File.Exists(package.Value))
    {
        Console.WriteLine($"\n## {package.Key}");
        Console.WriteLine($"⚠️  Assembly not found: {package.Value}\n");
        continue;
    }

    try
    {
        var assembly = Assembly.LoadFrom(package.Value);
        Console.WriteLine($"\n## {package.Key}");
        Console.WriteLine($"**Assembly**: `{assembly.GetName().Name}`");
        Console.WriteLine($"**Version**: {assembly.GetName().Version}\n");

        var types = assembly.GetExportedTypes()
            .OrderBy(t => t.Namespace)
            .ThenBy(t => t.Name)
            .ToList();

        var grouped = types.GroupBy(t => t.Namespace ?? "(no namespace)");

        foreach (var group in grouped)
        {
            Console.WriteLine($"### Namespace: `{group.Key}`\n");

            var interfaces = group.Where(t => t.IsInterface).ToList();
            if (interfaces.Any())
            {
                Console.WriteLine("#### Interfaces\n");
                foreach (var iface in interfaces)
                {
                    Console.WriteLine($"- **`{iface.Name}`**");

                    var methods = iface.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);
                    if (methods.Any())
                    {
                        foreach (var method in methods.Take(5))
                        {
                            var parameters = string.Join(", ", method.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}"));
                            Console.WriteLine($"  - `{method.ReturnType.Name} {method.Name}({parameters})`");
                        }
                        if (methods.Length > 5)
                        {
                            Console.WriteLine($"  - *(+{methods.Length - 5} more methods)*");
                        }
                    }
                }
                Console.WriteLine();
            }

            var classes = group.Where(t => t.IsClass && !t.IsAbstract).ToList();
            if (classes.Any())
            {
                Console.WriteLine("#### Classes\n");
                foreach (var cls in classes)
                {
                    Console.WriteLine($"- **`{cls.Name}`**");

                    // Show static factory methods
                    var staticMethods = cls.GetMethods(BindingFlags.Public | BindingFlags.Static)
                        .Where(m => !m.IsSpecialName)
                        .Take(3);

                    foreach (var method in staticMethods)
                    {
                        var parameters = string.Join(", ", method.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}"));
                        Console.WriteLine($"  - `static {method.ReturnType.Name} {method.Name}({parameters})`");
                    }
                }
                Console.WriteLine();
            }

            var abstractClasses = group.Where(t => t.IsClass && t.IsAbstract && !t.IsSealed).ToList();
            if (abstractClasses.Any())
            {
                Console.WriteLine("#### Abstract Classes\n");
                foreach (var cls in abstractClasses)
                {
                    Console.WriteLine($"- **`{cls.Name}`**");
                }
                Console.WriteLine();
            }

            var enums = group.Where(t => t.IsEnum).ToList();
            if (enums.Any())
            {
                Console.WriteLine("#### Enums\n");
                foreach (var enumType in enums)
                {
                    var values = string.Join(", ", Enum.GetNames(enumType).Take(5));
                    Console.WriteLine($"- **`{enumType.Name}`**: {values}");
                }
                Console.WriteLine();
            }

            var structs = group.Where(t => t.IsValueType && !t.IsEnum && !t.IsPrimitive).ToList();
            if (structs.Any())
            {
                Console.WriteLine("#### Structs\n");
                foreach (var structType in structs)
                {
                    Console.WriteLine($"- **`{structType.Name}`**");
                }
                Console.WriteLine();
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Error loading assembly: {ex.Message}\n");
    }
}

Console.WriteLine("\n---\n*Analysis completed*");
