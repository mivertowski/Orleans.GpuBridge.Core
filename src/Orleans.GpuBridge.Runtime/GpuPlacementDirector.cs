using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.Runtime;
using Orleans.Runtime.Placement;

namespace Orleans.GpuBridge.Runtime;

public sealed class GpuPlacementDirector : IPlacementDirector
{
    private readonly ILogger<GpuPlacementDirector> _logger;
    private readonly DeviceBroker _broker;
    
    public GpuPlacementDirector(ILogger<GpuPlacementDirector> logger, DeviceBroker broker)
    {
        _logger = logger;
        _broker = broker;
    }
    
    public Task<SiloAddress> OnAddActivation(PlacementStrategy strategy, PlacementTarget target, IPlacementContext context)
    {
        var compatibleSilos = context.GetCompatibleSilos(target);
        var chosen = compatibleSilos.FirstOrDefault();
        
        if (chosen == null)
        {
            _logger.LogError("No compatible silos found for grain {GrainIdentity}", target.GrainIdentity);
            throw new InvalidOperationException($"No compatible silos found for grain {target.GrainIdentity}");
        }
        
        _logger.LogDebug("Placed {GrainIdentity} on {SiloAddress}", target.GrainIdentity, chosen);
        return Task.FromResult(chosen);
    }
}
