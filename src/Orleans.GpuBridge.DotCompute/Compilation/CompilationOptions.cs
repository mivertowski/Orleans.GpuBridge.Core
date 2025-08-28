namespace Orleans.GpuBridge.DotCompute.Compilation;

/// <summary>
/// Kernel compilation options
/// </summary>
public sealed class CompilationOptions
{
    public bool EnableOptimizations { get; set; } = true;
    public bool EnableDebugInfo { get; set; }
    public string? TargetArchitecture { get; set; }
    public Dictionary<string, string> Defines { get; set; } = new();
    public int MaxRegisters { get; set; }
    public bool UseFastMath { get; set; } = true;
}