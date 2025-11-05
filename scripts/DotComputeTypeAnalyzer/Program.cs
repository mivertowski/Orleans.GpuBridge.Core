using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

class Program
{
    static void Main(string[] args)
    {
        var nugetPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");

        var packages = new (string Name, string Path)[]
        {
            ("DotCompute.Abstractions", Path.Combine(nugetPath, "dotcompute.abstractions/0.2.0-alpha/lib/net9.0/DotCompute.Abstractions.dll")),
            ("DotCompute.Core", Path.Combine(nugetPath, "dotcompute.core/0.2.0-alpha/lib/net9.0/DotCompute.Core.dll")),
            ("DotCompute.Runtime", Path.Combine(nugetPath, "dotcompute.runtime/0.2.0-alpha/lib/net9.0/DotCompute.Runtime.dll")),
            ("DotCompute.Memory", Path.Combine(nugetPath, "dotcompute.memory/0.2.0-alpha/lib/net9.0/DotCompute.Memory.dll")),
            ("DotCompute.Backends.CUDA", Path.Combine(nugetPath, "dotcompute.backends.cuda/0.2.0-alpha/lib/net9.0/DotCompute.Backends.CUDA.dll")),
            ("DotCompute.Backends.OpenCL", Path.Combine(nugetPath, "dotcompute.backends.opencl/0.2.0-alpha/lib/net9.0/DotCompute.Backends.OpenCL.dll")),
            ("DotCompute.Plugins", Path.Combine(nugetPath, "dotcompute.plugins/0.2.0-alpha/lib/net9.0/DotCompute.Plugins.dll"))
        };

        var sb = new StringBuilder();
        sb.AppendLine("# DotCompute Package Type Analysis");
        sb.AppendLine("## Package Version: 0.2.0-alpha");
        sb.AppendLine();

        foreach (var (name, path) in packages)
        {
            if (!File.Exists(path))
            {
                sb.AppendLine($"## {name}");
                sb.AppendLine($"⚠️  Assembly not found: {path}");
                sb.AppendLine();
                continue;
            }

            try
            {
                var assembly = Assembly.LoadFrom(path);
                sb.AppendLine($"## {name}");
                sb.AppendLine($"**Assembly**: `{assembly.GetName().Name}`");
                sb.AppendLine($"**Version**: {assembly.GetName().Version}");
                sb.AppendLine();

                var types = assembly.GetExportedTypes()
                    .OrderBy(t => t.Namespace)
                    .ThenBy(t => t.Name)
                    .ToList();

                var grouped = types.GroupBy(t => t.Namespace ?? "(no namespace)");

                foreach (var group in grouped)
                {
                    sb.AppendLine($"### Namespace: `{group.Key}`");
                    sb.AppendLine();

                    var interfaces = group.Where(t => t.IsInterface).ToList();
                    if (interfaces.Any())
                    {
                        sb.AppendLine("#### Interfaces");
                        sb.AppendLine();
                        foreach (var iface in interfaces)
                        {
                            sb.AppendLine($"- **`{iface.Name}`**");

                            var methods = iface.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);
                            foreach (var method in methods.Take(5))
                            {
                                var parameters = string.Join(", ", method.GetParameters().Select(p => $"{GetSimpleTypeName(p.ParameterType)} {p.Name}"));
                                sb.AppendLine($"  - `{GetSimpleTypeName(method.ReturnType)} {method.Name}({parameters})`");
                            }
                            if (methods.Length > 5)
                            {
                                sb.AppendLine($"  - *(+{methods.Length - 5} more methods)*");
                            }
                        }
                        sb.AppendLine();
                    }

                    var classes = group.Where(t => t.IsClass && !t.IsAbstract).ToList();
                    if (classes.Any())
                    {
                        sb.AppendLine("#### Classes");
                        sb.AppendLine();
                        foreach (var cls in classes)
                        {
                            sb.AppendLine($"- **`{cls.Name}`**");

                            var staticMethods = cls.GetMethods(BindingFlags.Public | BindingFlags.Static)
                                .Where(m => !m.IsSpecialName)
                                .Take(3);

                            foreach (var method in staticMethods)
                            {
                                var parameters = string.Join(", ", method.GetParameters().Select(p => $"{GetSimpleTypeName(p.ParameterType)} {p.Name}"));
                                sb.AppendLine($"  - `static {GetSimpleTypeName(method.ReturnType)} {method.Name}({parameters})`");
                            }
                        }
                        sb.AppendLine();
                    }

                    var abstractClasses = group.Where(t => t.IsClass && t.IsAbstract && !t.IsSealed).ToList();
                    if (abstractClasses.Any())
                    {
                        sb.AppendLine("#### Abstract Classes");
                        sb.AppendLine();
                        foreach (var cls in abstractClasses)
                        {
                            sb.AppendLine($"- **`{cls.Name}`**");
                        }
                        sb.AppendLine();
                    }

                    var enums = group.Where(t => t.IsEnum).ToList();
                    if (enums.Any())
                    {
                        sb.AppendLine("#### Enums");
                        sb.AppendLine();
                        foreach (var enumType in enums)
                        {
                            var values = string.Join(", ", Enum.GetNames(enumType).Take(5));
                            sb.AppendLine($"- **`{enumType.Name}`**: {values}");
                        }
                        sb.AppendLine();
                    }

                    var structs = group.Where(t => t.IsValueType && !t.IsEnum && !t.IsPrimitive).ToList();
                    if (structs.Any())
                    {
                        sb.AppendLine("#### Structs");
                        sb.AppendLine();
                        foreach (var structType in structs)
                        {
                            sb.AppendLine($"- **`{structType.Name}`**");
                        }
                        sb.AppendLine();
                    }
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine($"❌ Error loading assembly: {ex.Message}");
                sb.AppendLine();
            }
        }

        sb.AppendLine("---");
        sb.AppendLine("*Analysis completed*");

        Console.WriteLine(sb.ToString());
    }

    static string GetSimpleTypeName(Type type)
    {
        if (type.IsGenericType)
        {
            var name = type.Name.Split('`')[0];
            var args = string.Join(", ", type.GetGenericArguments().Select(GetSimpleTypeName));
            return $"{name}<{args}>";
        }
        return type.Name;
    }
}
