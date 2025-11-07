using System.Threading.Tasks;
using Xunit;

namespace Orleans.GpuBridge.Tests.Providers;

/// <summary>
/// TODO: REIMPLEMENT - These tests were written against a non-existent API
///
/// Original tests assumed:
/// - ProviderSelectionCriteria.PreferredBackends (array) - DOESN'T EXIST
/// - ProviderSelectionCriteria.AllowCpuFallback - DOESN'T EXIST
/// - GpuBackend.Cuda - WRONG (should be GpuBackend.CUDA)
/// - BackendCapabilities type - DOESN'T EXIST
///
/// Actual API (Orleans.GpuBridge.Abstractions.Providers.ProviderSelectionCriteria):
/// - PreferredProviderId (string)
/// - PreferredBackend (GpuBackend? - single enum, not array)
/// - RequiredCapabilities (IReadOnlyList&lt;string&gt; - strings, not BackendCapabilities)
/// - PreferGpu (bool - this controls CPU fallback preference)
/// - RequireJitCompilation, RequireUnifiedMemory, RequireProfiling, RequireCpuDebugging (bool flags)
/// - ExcludeProviders (IReadOnlyList&lt;string&gt;)
///
/// To reimplement:
/// 1. Study actual IGpuBackendRegistry interface in Abstractions
/// 2. Study actual GpuBackendRegistry implementation in Runtime
/// 3. Write tests that match the REAL API, not the imagined one
/// 4. Test provider selection with correct ProviderSelectionCriteria properties
/// 5. Test provider registration, discovery, and initialization
/// </summary>
public class GpuBackendRegistryTests
{
    [Fact(Skip = "Test file deleted - needs complete rewrite against actual API")]
    public async Task Placeholder_TestsNeedReimplementation()
    {
        // This test file contained 160 lines of tests against a non-existent API
        // Rather than keep broken code, it was deleted cleanly
        // Reimplement tests against the actual API as documented above
        await Task.CompletedTask;
    }
}
