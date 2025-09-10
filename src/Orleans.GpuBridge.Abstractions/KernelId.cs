using System.Diagnostics.CodeAnalysis;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Represents a unique kernel identifier
/// </summary>
public readonly record struct KernelId(string Value)
{
    public override string ToString() => Value;
    
    public static KernelId Parse([NotNull] string s) 
    {
        if (string.IsNullOrWhiteSpace(s))
            throw new ArgumentException("Kernel ID cannot be null or whitespace", nameof(s));
        return new(s);
    }
    
    public static bool TryParse(string? s, out KernelId result)
    {
        if (string.IsNullOrWhiteSpace(s))
        {
            result = default;
            return false;
        }
        
        result = new KernelId(s);
        return true;
    }
    
    public static implicit operator string(KernelId id) => id.Value;
}
