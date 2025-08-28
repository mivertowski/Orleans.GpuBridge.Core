using Orleans.GpuBridge.Abstractions.Enums;

namespace Orleans.GpuBridge.Runtime.Configuration;

/// <summary>
/// Configuration options for controlling backend provider selection strategy and behavior.
/// These options determine which GPU backends are preferred, fallback behavior, and resource constraints.
/// </summary>
public class ProviderSelectionOptions
{
    /// <summary>
    /// Gets or sets the preferred backend types in order of preference.
    /// The system will attempt to use backends in this order when multiple are available.
    /// Default preference: CUDA, OpenCL, CPU fallback.
    /// </summary>
    /// <value>A list of <see cref="GpuBackend"/> values representing the preferred order of backend selection.</value>
    public List<GpuBackend> PreferredBackends { get; set; } = new() { GpuBackend.Cuda, GpuBackend.OpenCL, GpuBackend.Cpu };
    
    /// <summary>
    /// Gets or sets a value indicating whether the system should fall back to CPU execution when no GPU backends are available.
    /// When <c>true</c>, operations will use CPU execution as a fallback. When <c>false</c>, operations will fail if no GPU is available.
    /// </summary>
    /// <value><c>true</c> to enable CPU fallback; otherwise, <c>false</c>. Default is <c>true</c>.</value>
    public bool AllowCpuFallback { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the maximum number of GPU devices to use concurrently.
    /// This limits the number of devices that can be utilized simultaneously to prevent resource exhaustion.
    /// </summary>
    /// <value>The maximum number of concurrent devices. Default is 4.</value>
    public int MaxConcurrentDevices { get; set; } = 4;
    
    /// <summary>
    /// Gets or sets the minimum memory required per device in bytes.
    /// Devices with less available memory than this threshold will be excluded from selection.
    /// </summary>
    /// <value>The minimum device memory in bytes. Default is 512 MB (536,870,912 bytes).</value>
    public long MinimumDeviceMemory { get; set; } = 512 * 1024 * 1024; // 512 MB
    
    /// <summary>
    /// Gets or sets the required capabilities that must be supported by selected backend providers.
    /// This allows filtering backends based on specific feature requirements.
    /// </summary>
    /// <value>A <see cref="RequiredCapabilities"/> instance specifying the required features.</value>
    public RequiredCapabilities RequiredCapabilities { get; set; } = new();
}