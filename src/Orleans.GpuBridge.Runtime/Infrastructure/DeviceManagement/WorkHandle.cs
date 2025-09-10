using System.Threading.Tasks;

namespace Orleans.GpuBridge.Runtime.Infrastructure.DeviceManagement;

/// <summary>
/// Handle for tracking work item completion
/// </summary>
internal sealed record WorkHandle(string Id, Task CompletionTask);