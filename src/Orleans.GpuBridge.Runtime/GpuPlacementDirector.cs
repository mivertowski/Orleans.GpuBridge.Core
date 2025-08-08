using System.Linq; using System.Threading.Tasks; using Microsoft.Extensions.Logging; using Orleans.Runtime; using Orleans.Runtime.Placement;
namespace Orleans.GpuBridge.Runtime;
public sealed class GpuPlacementDirector:IPlacementDirector
{
    private readonly ILogger<GpuPlacementDirector> _log; private readonly DeviceBroker _broker;
    public GpuPlacementDirector(ILogger<GpuPlacementDirector> log, DeviceBroker broker){_log=log;_broker=broker;}
    public Task<SiloAddress> OnAddActivation(PlacementStrategy s, PlacementTarget t, IPlacementContext c)
    { var chosen=c.GetCompatibleSilos(t).FirstOrDefault(); _log.LogDebug("Placed {grain} on {silo}",t.GrainIdentity,chosen); return Task.FromResult(chosen); }
}
