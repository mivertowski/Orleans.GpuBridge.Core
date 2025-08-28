namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;

/// <summary>
/// Memory usage advice for unified memory
/// </summary>
public enum MemoryAdvice
{
    /// <summary>Memory will be mostly read</summary>
    ReadMostly,
    /// <summary>Memory will be accessed by a specific processor</summary>
    PreferredLocation,
    /// <summary>Memory will be accessed equally by all processors</summary>
    AccessedBy,
    /// <summary>Memory should not migrate</summary>
    NoMigration
}