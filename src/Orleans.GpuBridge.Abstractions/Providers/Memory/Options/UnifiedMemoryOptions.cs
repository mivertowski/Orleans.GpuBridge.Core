namespace Orleans.GpuBridge.Abstractions.Providers.Memory.Options;

/// <summary>
/// Options for unified memory allocation
/// </summary>
public sealed record UnifiedMemoryOptions(
    bool AttachToHost = false,
    bool PreferredLocationDevice = true,
    bool EnableMigration = true);