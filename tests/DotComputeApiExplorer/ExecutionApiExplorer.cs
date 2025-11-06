using System;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using Microsoft.Extensions.Logging;

namespace DotComputeApiExplorer;

public static class ExecutionApiExplorer
{
    public static async Task ExploreExecutionApi(IAccelerator accelerator, ILogger logger)
    {
        Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Console.WriteLine("Kernel Execution API Discovery:");
        Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // Simple CUDA vector add kernel
        const string vectorAddKernel = @"
extern ""C"" __global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
";

        try
        {
            // Get KernelDefinition type
            var assembly = typeof(IAccelerator).Assembly;
            var kernelDefType = assembly.GetTypes()
                .FirstOrDefault(t => t.Name == "KernelDefinition");

            if (kernelDefType == null)
            {
                Console.WriteLine("❌ KernelDefinition type not found");
                return;
            }

            // Get KernelLanguage enum
            var kernelLanguageType = assembly.GetTypes()
                .FirstOrDefault(t => t.Name == "KernelLanguage" && t.IsEnum);

            if (kernelLanguageType == null)
            {
                Console.WriteLine("❌ KernelLanguage type not found");
                return;
            }

            // Create KernelDefinition instance
            var cudaLanguage = Enum.Parse(kernelLanguageType, "Cuda");
            var kernelDef = Activator.CreateInstance(kernelDefType, "VectorAdd", vectorAddKernel, "vectorAdd");

            // Set Language property
            var languageProp = kernelDefType.GetProperty("Language");
            languageProp?.SetValue(kernelDef, cudaLanguage);

            Console.WriteLine("✅ Created KernelDefinition for CUDA vector add");
            Console.WriteLine($"   Name: VectorAdd");
            Console.WriteLine($"   Entry Point: vectorAdd");
            Console.WriteLine($"   Language: Cuda");
            Console.WriteLine();

            // Get CompilationOptions
            var compilationOptionsType = assembly.GetTypes()
                .FirstOrDefault(t => t.Name == "CompilationOptions");

            var releaseOptions = compilationOptionsType?.GetProperty("Release")?.GetValue(null);

            if (releaseOptions == null)
            {
                Console.WriteLine("⚠️ Using default compilation options");
            }

            // Compile the kernel
            Console.WriteLine("Compiling kernel...");
            var compileMethod = accelerator.GetType().GetMethod("CompileKernelAsync");

            if (compileMethod == null)
            {
                Console.WriteLine("❌ CompileKernelAsync method not found");
                return;
            }

            var compileTask = (dynamic)compileMethod.Invoke(accelerator, new[] { kernelDef, releaseOptions, CancellationToken.None });
            var compiledKernel = await compileTask;

            Console.WriteLine($"✅ Kernel compiled successfully!");

            // Inspect compiled kernel type
            var compiledKernelType = compiledKernel.GetType();

            // Try to get Name property
            var nameProp = compiledKernelType.GetProperty("Name");
            if (nameProp != null)
            {
                var nameValue = nameProp.GetValue(compiledKernel);
                Console.WriteLine($"   Kernel Name: {nameValue}");
            }

            // Try to get IsValid property
            var isValidProp = compiledKernelType.GetProperty("IsValid");
            if (isValidProp != null)
            {
                var isValidValue = isValidProp.GetValue(compiledKernel);
                Console.WriteLine($"   IsValid: {isValidValue}");
            }

            Console.WriteLine();
            Console.WriteLine($"Compiled Kernel Type: {compiledKernelType.FullName}");
            Console.WriteLine();

            Console.WriteLine("Available Methods:");
            foreach (var method in compiledKernelType.GetMethods(BindingFlags.Public | BindingFlags.Instance))
            {
                if (method.Name.StartsWith("get_") || method.Name.StartsWith("set_")) continue;

                Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                foreach (var param in method.GetParameters())
                {
                    Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                }
                Console.WriteLine($"    )");
            }

            Console.WriteLine("\nAvailable Properties:");
            foreach (var prop in compiledKernelType.GetProperties())
            {
                Console.WriteLine($"  • {prop.PropertyType.Name} {prop.Name}");
            }

            // Check IAccelerator for execution methods
            Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            Console.WriteLine("IAccelerator Execution Methods:");
            Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            var acceleratorExecuteMethods = accelerator.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Where(m => m.Name.Contains("Execute", StringComparison.OrdinalIgnoreCase) ||
                           m.Name.Contains("Launch", StringComparison.OrdinalIgnoreCase) ||
                           m.Name.Contains("Invoke", StringComparison.OrdinalIgnoreCase));

            foreach (var method in acceleratorExecuteMethods)
            {
                Console.WriteLine($"\n  ▸ {method.ReturnType.Name} {method.Name}(");
                foreach (var param in method.GetParameters())
                {
                    Console.WriteLine($"      {param.ParameterType.Name} {param.Name}");
                }
                Console.WriteLine($"    )");
            }

            Console.WriteLine("\n✅ Execution API Discovery Complete!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ ERROR: {ex.GetType().Name}: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
