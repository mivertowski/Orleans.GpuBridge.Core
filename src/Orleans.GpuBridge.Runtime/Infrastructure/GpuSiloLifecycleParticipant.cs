using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions.Capacity;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.Hosting;
using Orleans.Runtime;

namespace Orleans.GpuBridge.Runtime.Infrastructure;

/// <summary>
/// Orleans lifecycle participant that registers GPU capacity with the cluster
/// and maintains periodic capacity updates
/// </summary>
/// <remarks>
/// This participant integrates with Orleans silo lifecycle to:
/// - Register GPU capacity on silo startup
/// - Update capacity every 30 seconds
/// - Unregister GPU capacity on graceful shutdown
///
/// The participant queries DeviceBroker for actual GPU metrics and reports
/// them to the centralized IGpuCapacityGrain for intelligent placement decisions.
/// </remarks>
public sealed class GpuSiloLifecycleParticipant : ILifecycleParticipant<ISiloLifecycle>
{
    private readonly ILogger<GpuSiloLifecycleParticipant> _logger;
    private readonly IGrainFactory _grainFactory;
    private readonly ILocalSiloDetails _siloDetails;
    private readonly DeviceBroker _deviceBroker;
    private IDisposable? _capacityUpdateTimer;
    private bool _isRegistered;

    /// <summary>
    /// Capacity update interval (30 seconds)
    /// </summary>
    private static readonly TimeSpan UpdateInterval = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Initializes a new instance of the GPU silo lifecycle participant
    /// </summary>
    public GpuSiloLifecycleParticipant(
        ILogger<GpuSiloLifecycleParticipant> logger,
        IGrainFactory grainFactory,
        ILocalSiloDetails siloDetails,
        DeviceBroker deviceBroker)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _grainFactory = grainFactory ?? throw new ArgumentNullException(nameof(grainFactory));
        _siloDetails = siloDetails ?? throw new ArgumentNullException(nameof(siloDetails));
        _deviceBroker = deviceBroker ?? throw new ArgumentNullException(nameof(deviceBroker));
    }

    /// <summary>
    /// Participates in the Orleans silo lifecycle
    /// </summary>
    public void Participate(ISiloLifecycle lifecycle)
    {
        // Register at ApplicationServices stage (after Orleans core services but before grains)
        lifecycle.Subscribe(
            nameof(GpuSiloLifecycleParticipant),
            ServiceLifecycleStage.ApplicationServices,
            OnStart,
            OnStop);

        _logger.LogDebug(
            "GPU silo lifecycle participant registered at stage {Stage}",
            ServiceLifecycleStage.ApplicationServices);
    }

    /// <summary>
    /// Called when the silo is starting
    /// </summary>
    private async Task OnStart(CancellationToken cancellationToken)
    {
        _logger.LogInformation(
            "Starting GPU capacity registration for silo {SiloAddress}",
            _siloDetails.SiloAddress.ToParsableString());

        try
        {
            // Get current GPU capacity from DeviceBroker
            var capacity = await GetCurrentCapacityAsync();

            // Register with the capacity grain
            var capacityGrain = _grainFactory.GetGrain<IGpuCapacityGrain>(0);
            await capacityGrain.RegisterSiloAsync(_siloDetails.SiloAddress, capacity);

            _isRegistered = true;

            _logger.LogInformation(
                "Registered GPU capacity for silo {SiloAddress}: {Capacity}",
                _siloDetails.SiloAddress.ToParsableString(),
                capacity);

            // Start periodic capacity update timer
            _capacityUpdateTimer = RegisterTimer(
                UpdateCapacityAsync,
                null,
                UpdateInterval,
                UpdateInterval);

            _logger.LogDebug(
                "Started periodic GPU capacity updates every {Interval}",
                UpdateInterval);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "Failed to register GPU capacity for silo {SiloAddress}",
                _siloDetails.SiloAddress.ToParsableString());
            throw;
        }
    }

    /// <summary>
    /// Called when the silo is stopping
    /// </summary>
    private async Task OnStop(CancellationToken cancellationToken)
    {
        _logger.LogInformation(
            "Stopping GPU capacity registration for silo {SiloAddress}",
            _siloDetails.SiloAddress.ToParsableString());

        try
        {
            // Stop the capacity update timer
            _capacityUpdateTimer?.Dispose();
            _capacityUpdateTimer = null;

            _logger.LogDebug("Stopped periodic GPU capacity updates");

            // Unregister from the capacity grain
            if (_isRegistered)
            {
                var capacityGrain = _grainFactory.GetGrain<IGpuCapacityGrain>(0);
                await capacityGrain.UnregisterSiloAsync(_siloDetails.SiloAddress);

                _isRegistered = false;

                _logger.LogInformation(
                    "Unregistered GPU capacity for silo {SiloAddress}",
                    _siloDetails.SiloAddress.ToParsableString());
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(
                ex,
                "Error during GPU capacity unregistration for silo {SiloAddress}",
                _siloDetails.SiloAddress.ToParsableString());
            // Don't throw - allow silo to shut down gracefully
        }
    }

    /// <summary>
    /// Periodic callback to update GPU capacity
    /// </summary>
    private async Task UpdateCapacityAsync(object? state)
    {
        if (!_isRegistered)
        {
            return;
        }

        try
        {
            // Get current GPU capacity
            var capacity = await GetCurrentCapacityAsync();

            // Update the capacity grain
            var capacityGrain = _grainFactory.GetGrain<IGpuCapacityGrain>(0);
            await capacityGrain.UpdateCapacityAsync(_siloDetails.SiloAddress, capacity);

            _logger.LogTrace(
                "Updated GPU capacity for silo {SiloAddress}: Available={AvailableMemory}MB, Queue={QueueDepth}",
                _siloDetails.SiloAddress.ToParsableString(),
                capacity.AvailableMemoryMB,
                capacity.QueueDepth);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(
                ex,
                "Failed to update GPU capacity for silo {SiloAddress}",
                _siloDetails.SiloAddress.ToParsableString());
            // Don't throw - continue trying on next interval
        }
    }

    /// <summary>
    /// Gets the current GPU capacity from DeviceBroker
    /// </summary>
    private async Task<GpuCapacity> GetCurrentCapacityAsync()
    {
        await Task.CompletedTask; // DeviceBroker operations are currently synchronous

        var devices = _deviceBroker.GetDevices();

        // Filter out CPU devices - we only care about actual GPUs
        var gpuDevices = devices.Where(d => d.Type != DeviceType.CPU).ToList();

        if (gpuDevices.Count == 0)
        {
            _logger.LogDebug(
                "No GPU devices found for silo {SiloAddress}, reporting no capacity",
                _siloDetails.SiloAddress.ToParsableString());

            return GpuCapacity.None;
        }

        // Calculate aggregate capacity across all GPU devices
        var totalMemoryBytes = gpuDevices.Sum(d => d.TotalMemoryBytes);
        var availableMemoryBytes = gpuDevices.Sum(d => d.AvailableMemoryBytes);
        var queueDepth = _deviceBroker.CurrentQueueDepth;

        // Determine backend from first GPU device
        var backend = gpuDevices[0].Type.ToString();

        var capacity = new GpuCapacity(
            DeviceCount: gpuDevices.Count,
            TotalMemoryMB: totalMemoryBytes / (1024 * 1024),
            AvailableMemoryMB: availableMemoryBytes / (1024 * 1024),
            QueueDepth: queueDepth,
            Backend: backend,
            LastUpdated: DateTime.UtcNow);

        _logger.LogTrace(
            "Retrieved GPU capacity: Devices={DeviceCount}, Total={TotalMemory}MB, Available={AvailableMemory}MB, Queue={QueueDepth}, Backend={Backend}",
            capacity.DeviceCount,
            capacity.TotalMemoryMB,
            capacity.AvailableMemoryMB,
            capacity.QueueDepth,
            capacity.Backend);

        return capacity;
    }

    /// <summary>
    /// Registers a timer for periodic callbacks
    /// </summary>
    private IDisposable RegisterTimer(
        Func<object?, Task> callback,
        object? state,
        TimeSpan dueTime,
        TimeSpan period)
    {
        // Use System.Threading.Timer for Orleans lifecycle integration
        var timer = new Timer(
            async _ =>
            {
                try
                {
                    await callback(state).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in timer callback");
                }
            },
            null,
            dueTime,
            period);

        return timer;
    }
}
