# Migration Note — Orleans.GpuBridge.Core v1.0 Alignment

This note plans the upgrade of Orleans.GpuBridge.Core onto the upcoming **DotCompute v1.0.0**
release line. It is written for a maintainer who will execute the bump once the new DotCompute
NuGet packages are published.

The changes in this document **have not been applied** to the repository yet. The current
working state is still pinned to DotCompute **0.5.3** (net9.0). What has been applied already
is a small documentation scrub (see "Already applied" below).

---

## Upstream context — DotCompute v1.0.0 scope

DotCompute has the following v1.0.0 changes staged on branch
`chore/net10-migration-and-openacl-removal` (repo: `mivertowski/DotCompute`):

- **.NET 10 migration.** `TargetFramework` moved from `net9.0` to `net10.0`, `LangVersion`
  from 13 to 14, `global.json` bumped to `10.0.0`. Microsoft.Extensions.* and System.* moved
  to 10.0.6. Microsoft.CodeAnalysis.CSharp moved to 5.3.0. ILLink 10.0.6, SourceLink 10.0.202.
- **OpenCL backend dropped.** `DotCompute.Backends.OpenCL` is removed. v1.0.0 ships
  CPU / CUDA / Metal only. OpenCL code paths in `DotCompute.Linq.V2` and related generators
  were also scrubbed. DotCompute v1.0.0 does **not** expose an OpenCL backend package.
- **WSL2 workarounds removed from CUDA backend.** ~837 lines of fallback code
  (`IsRunningInWsl2`, `AsyncControlBlock`, `GetEffectiveKernelMode` overrides, non-cooperative
  launch paths, etc.) are gone. Persistent kernel mode is now driven purely by the device's
  `HostNativeAtomicSupported` capability and the user's explicit `RingKernelMode` choice.
  WSL2 is documented as dev-only for v1.0.0 — not a runtime target.
- **First-class Hopper / sm_90 support.** New namespace
  `DotCompute.Backends.CUDA.Hopper` with `HopperFeatures`, `ClusterLaunchConfig`, `TmaConfig`,
  `DsmemConfig`, and async memory pool config. Capability-gated (CC 9.0 for clusters / TMA /
  DSMEM; CC 6.0+ for async mem pool). Older devices skip these paths cleanly.
- **Classified error model.** New `CudaErrorClass` enum (`Success`, `Transient`, `Resource`,
  `Programmer`, `Fatal`) with extension methods (`.Classify()`, `.IsRecoverable()`,
  `.IsResource()`, `.IsFatal()`, `.IsProgrammerError()`, `.IsRetryable()`) on `CudaError`.
  `CudaException` now exposes `.Classification`, `.IsRecoverable`, `.IsResourceError`,
  `.IsFatal`, `.IsRetryable` properties. `CudaErrorHandler` retry / circuit-breaker policies
  delegate to the central classification.
- **Defense-in-depth finalizers.** `CudaContext` gained `~CudaContext` to release primary
  context + stream if `Dispose` was missed.
- **Hardware CI test gate, throughput soak test, OpenTelemetry `ActivitySource`.**
- **P2P idempotency bitmap.** `CudaMessageQueue` counters are now cache-line padded.

None of these changes are NuGet-published yet — the new package set
(`DotCompute.Abstractions.V2`, `DotCompute.Core.V2`, `DotCompute.Runtime.V2`,
`DotCompute.Backends.CPU.V2`, `DotCompute.Backends.CUDA.V2`, `DotCompute.Generators.V2`,
`DotCompute.Backends.Metal.V2` once it ships) will land under version
`1.0.0-preview1` (or whichever final tag) once the upstream release is cut.

---

## What to bump here, in one commit, once DotCompute ships

### 1. `Directory.Build.props`

```xml
<DotComputeVersion>1.0.0-preview1</DotComputeVersion>   <!-- or the final shipped tag -->
```

Consider also bumping the root `<Version>` at the same time — v0.4.0 is a reasonable target
for "Orleans.GpuBridge on DotCompute 1.0 + .NET 10" given the surface-area impact.

### 2. `<TargetFramework>` on every `src/` project — **except the source generator**

| Project | Current | Target |
|--------|--------|--------|
| `src/Orleans.GpuBridge.Abstractions/Orleans.GpuBridge.Abstractions.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Backends.DotCompute/Orleans.GpuBridge.Backends.DotCompute.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.BridgeFX/Orleans.GpuBridge.BridgeFX.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Diagnostics/Orleans.GpuBridge.Diagnostics.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Grains/Orleans.GpuBridge.Grains.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.HealthChecks/Orleans.GpuBridge.HealthChecks.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Logging/Orleans.GpuBridge.Logging.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Resilience/Orleans.GpuBridge.Resilience.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Runtime/Orleans.GpuBridge.Runtime.csproj` | net9.0 | net10.0 |
| `src/Orleans.GpuBridge.Generators/Orleans.GpuBridge.Generators.csproj` | netstandard2.0 | **KEEP netstandard2.0** (Roslyn source-generator requirement) |

Bump every `tests/**/*.csproj` to `net10.0` as well, plus every project under `examples/`.

### 3. `global.json`

Bump to the 10.0.x SDK in use upstream (DotCompute is on 10.0.0 for its `sdk.version`).

### 4. `LangVersion` and `Directory.Build.props`

If `<LangVersion>latest</LangVersion>` is in effect, nothing to change. If any project pins
`13`, bump to `14` to match upstream.

### 5. Bump the Microsoft.* package floor

Upstream moved to Microsoft.Extensions 10.0.6; our current pin is 10.0.1. Match upstream to
avoid transitive version resolution surprises:

```xml
<MicrosoftExtensionsVersion>10.0.6</MicrosoftExtensionsVersion>
<MicrosoftCodeAnalysisVersion>5.3.0</MicrosoftCodeAnalysisVersion>
```

ILLink and `Microsoft.DotNet.ILCompiler` hardcoded in
`src/Orleans.GpuBridge.Backends.DotCompute/Orleans.GpuBridge.Backends.DotCompute.csproj`
should move to 10.0.6 / 10.0.0 respectively (or whatever upstream settles on).

### 6. `ocl-icd-opencl-dev` in `Dockerfile`

Optional cleanup — OpenCL ICD loader is installed in the build image but is no longer needed
once the OpenCL backend is gone from DotCompute. Safe to leave in place, but a nice follow-up.

### 7. Re-enable the two `.disabled` Temporal kernel files

See "Disabled kernel file analysis" below. The rename is:

```bash
git mv src/Orleans.GpuBridge.Backends.DotCompute/Temporal/TemporalKernels.cs.disabled \
       src/Orleans.GpuBridge.Backends.DotCompute/Temporal/TemporalKernels.cs
git mv src/Orleans.GpuBridge.Backends.DotCompute/Temporal/ActorRingKernels.cs.disabled \
       src/Orleans.GpuBridge.Backends.DotCompute/Temporal/ActorRingKernels.cs
```

…**followed by code-level fixups** (see below). Do not rename the files without applying
those fixups — the files reference placeholder/stub helpers that only compile against the
new DotCompute v1.0.0 runtime.

### 8. README / CLAUDE.md / CHANGELOG

- Bump the `DotCompute-0.5.3-orange` badge in `README.md` to the shipped 1.0 version.
- Bump the .NET badge from `9.0-purple` to `10.0-purple`.
- Update `CLAUDE.md` "Current Version" and "DotCompute Integration" sections to reflect
  v1.0.0.
- Add a `CHANGELOG.md` entry for v0.4.0 (or the chosen release) covering the .NET 10 bump,
  DotCompute 1.0 upgrade, OpenCL support drop, and re-enable of the two kernel files.

---

## Already applied (this PR / branch)

The following safe, non-package-dependent changes have already been made. They compile
against the current 0.5.3 pin:

- `README.md` — dropped "OpenCL Backend | Planned" row; updated DotCompute backend package
  description from "(CUDA, CPU)" to "(CUDA, Metal, CPU)".
- `src/Orleans.GpuBridge.Backends.DotCompute/Orleans.GpuBridge.Backends.DotCompute.csproj` —
  `<Description>` no longer mentions OpenCL.
- `src/Orleans.GpuBridge.Backends.DotCompute/DotComputeBackendProvider.cs` — XML doc comment
  no longer mentions OpenCL.
- `src/Orleans.GpuBridge.Backends.DotCompute/README.md` — architecture diagram and feature
  list no longer advertise OpenCL / DirectCompute / Vulkan.
- `src/Orleans.GpuBridge.Backends.DotCompute/Extensions/ServiceCollectionExtensions.cs` —
  XML doc comments no longer mention OpenCL / DirectCompute / Vulkan; platform preference
  updated to "CUDA > Metal > CPU".
- `src/Orleans.GpuBridge.Backends.DotCompute/Configuration/DotComputeBackendConfiguration.cs`
  — top-level remarks no longer list OpenCL / DirectCompute / Vulkan.
- `src/Orleans.GpuBridge.Backends.DotCompute/Models/ComputeRequirements.cs` — XML doc
  comments no longer mention OpenCL / DirectCompute.
- `CLAUDE.md` — dropped "OpenCL backend (planned)" from implementation status.
- `SUPPORT.md` — dropped OpenCL and Vulkan Compute from the planned-backends list in the
  FAQ.

Notes:

- `DotComputeVersion` in `Directory.Build.props` is **unchanged (0.5.3)** by intention.
- `<TargetFramework>net9.0</TargetFramework>` is **unchanged** by intention.
- The two `*.cs.disabled` files are **unchanged** by intention.
- `BackendType`-style enums (`AcceleratorType`, `GpuBackend`, `DeviceType`,
  `KernelLanguage`) still have OpenCL values — these are **not** removed (wire-format /
  Orleans serialization stability). See "OpenCL enum values audit" below.

---

## OpenCL enum values audit

Four enums in `src/Orleans.GpuBridge.Abstractions/Enums/` still expose an `OpenCL` value:

| Enum | File | Note |
|------|------|------|
| `AcceleratorType` | `Enums/AcceleratorType.cs` | Used across backend providers for device selection. |
| `GpuBackend` | `Enums/GpuBackend.cs` | Referenced by `DotComputeBackendConfiguration.PreferredPlatforms` and `KernelLanguageSettings.PreferredLanguages` dictionary. |
| `DeviceType` | `Enums/DeviceType.cs` | Used inside `GpuDevice`, `DotComputeDeviceManager`, `DotComputeAcceleratorProvider`. |
| `KernelLanguage` | `Enums/Compilation/KernelLanguage.cs` | Used by `DotComputeKernelCompiler` (maps `.cl` file extension → `KernelLanguage.OpenCL`). |

Recommendation: **leave all four enum values in place** through v1.0.0. They are public
API and Orleans serializes some of them to grain state / silo messages. Switching the
public API so soon would force a breaking change on every consumer.

A future minor version may:
- mark the `OpenCL` members with `[EditorBrowsable(EditorBrowsableState.Never)]` and
  `[Obsolete("OpenCL backend not supported in DotCompute 1.0+; will be removed in a future
  release.")]`, then
- remove them after a one-cycle deprecation window.

For this upgrade: take no enum action.

---

## Disabled kernel file analysis

There are two disabled files under `src/Orleans.GpuBridge.Backends.DotCompute/Temporal/`:

### `TemporalKernels.cs.disabled`

Four static kernel methods — `ProcessActorMessageWithTimestamp`, `BatchHLCUpdate`,
`CalibrationSampleKernel`, `DetectTemporalPattern` — decorated with
`[global::DotCompute.Generators.Kernel.Attributes.Kernel(...)]`. The header comment says
"using DotCompute 0.4.2-rc2", so the file predates the current 0.5.3 pin.

**Why it's disabled (observed):**

1. The kernel bodies use the placeholder pattern `int actorId = 0; // TODO: GetGlobalId(0)
   when DotCompute kernel support is added`. That is, the DotCompute kernel runtime the
   file was written against did not yet surface a stable `Kernel.ThreadId.X` /
   `GetGlobalId` intrinsic. Every kernel in this file hardcodes actor ID to 0, which would
   give wrong results on GPU — not safe to ship.
2. Device-wide barriers are stubbed as `// TODO: DeviceBarrier()` — the runtime did not
   yet expose the barrier API.
3. The attributes reference the `DotCompute.Generators.Kernel.Attributes.{Kernel,
   MemoryOrderingMode, BarrierScope}` namespace. Active kernels in the same directory
   (`VectorAddRingKernel.cs`, `PatternMatchRingKernel.cs`) use
   `DotCompute.Abstractions.Attributes.RingKernel` — the *new*, unified namespace that
   replaces the `Generators.Kernel.Attributes.Kernel` pattern. The generator-side attribute
   namespace still exists on v0.5.3 but is considered legacy.

**What unblocks this file in DotCompute v1.0.0:**

DotCompute v1.0.0 exposes a stable `Kernel.ThreadId.X/Y/Z` intrinsic (per the DotCompute
CLAUDE.md "Add New Kernel — Modern (v0.2.0+)" guidance). To re-enable:

1. Rename `TemporalKernels.cs.disabled` → `TemporalKernels.cs`.
2. Replace every `int actorId = 0; // TODO: GetGlobalId(0)` with
   `int actorId = Kernel.ThreadId.X;` (plus a `using DotCompute.Abstractions.Kernels;` or
   whatever the final namespace lands on).
3. Replace each `[global::DotCompute.Generators.Kernel.Attributes.Kernel(...)]` attribute
   with the v1.0 equivalent — likely `[DotCompute.Abstractions.Attributes.Kernel(...)]` —
   and remap the old property names (`EnableTimestamps`, `MemoryOrdering`, `EnableBarriers`,
   `BarrierScope`) onto whatever the stable attribute exposes. Cross-check against
   `src/Core/DotCompute.Abstractions/Attributes/RingKernelAttribute.cs` in the upstream
   repo for the authoritative list of stable properties.
4. Replace the `// TODO: DeviceBarrier()` placeholders with the v1.0 barrier intrinsic
   (per DotCompute's "Barrier API (ThreadBlock, Grid, Warp, Named barriers — <20ns)"
   feature). Exact call-site will be documented in the v1.0 release notes; expect
   `Kernel.Barrier.Device()` or similar.

Smallest possible change: the attribute namespace swap and the `ThreadId` substitution.
Barrier call-site can be done in a follow-up if the v1.0 barrier API name differs from the
current expectation.

### `ActorRingKernels.cs.disabled`

Three persistent ring-kernel methods — `ActorMessageProcessorRing`,
`BatchedActorMessageProcessorRing`, `CoordinatedActorRing` — decorated with
`[global::DotCompute.Generators.Kernel.Attributes.RingKernel(...)]`.

**Why it's disabled (observed):**

1. Same `int actorId = 0; // TODO: GetGlobalId(0)` placeholders as `TemporalKernels.cs`.
2. Two private placeholder helpers at the bottom:

   ```csharp
   private static int AtomicLoad(ref int value) => value;   // Placeholder — DotCompute will
                                                              // provide proper atomic
   private static void Yield() { }                            // Placeholder — DotCompute will
                                                              // provide proper yield
   ```

   These are stubs. The file explicitly admits it is waiting for DotCompute to expose
   `AtomicLoad` and `Yield` intrinsics that map to CUDA `__atomic_load_n` and
   `__nanosleep(100)` (and OpenCL equivalents, which are no longer relevant under v1.0).
3. The attribute namespace mismatch is the same as `TemporalKernels.cs`: this file targets
   the legacy `DotCompute.Generators.Kernel.Attributes.RingKernel` namespace, while active
   production ring kernels in the same directory target the unified
   `DotCompute.Abstractions.Attributes.RingKernel` — and use an entirely different
   handler signature (`ProcessXxx(RingKernelContext ctx, TRequest request)` vs. the raw
   `Span<long> timestamps, Span<ActorMessage> messageQueue, ...` positional parameter list
   used in `ActorRingKernels.cs.disabled`).
4. The current active ring-kernel runtime goes through `CustomRingKernelRuntimeFactory`,
   which exists because the DotCompute-generated `RingKernelRuntimeFactory.g.cs` references
   unpublished types (`OpenCLDeviceManager`, `Metal.RingKernels`, etc.). Once v1.0 ships,
   `CustomRingKernelRuntimeFactory` may no longer be needed, or will need to be refactored
   to use CPU + CUDA + Metal only.

**What unblocks this file in DotCompute v1.0.0:**

The core blocker is that `ActorRingKernels.cs.disabled` is written against a **different
ring-kernel programming model** than what actually ships in v0.5.3+. The active kernels
(`VectorAddRingKernel.cs`, `PatternMatchRingKernel.cs`) use a context-object model
(`RingKernelContext` + strongly-typed message types), not raw `Span<T>` arguments +
manual ring-buffer bookkeeping.

To re-enable: this is **not** a simple rename. It is a port. Options in order of preference:

- **Port to the v1.0 ring-kernel model.** Rewrite `ActorMessageProcessorRing`,
  `BatchedActorMessageProcessorRing`, and `CoordinatedActorRing` as
  `RingKernelContext`-based handlers, mirroring the structure of
  `VectorAddRingKernel.ProcessVectorOperation`. Define an `ActorMessage` /
  `ActorStateUpdate` message-type pair with `[MemoryPackable]` attributes so DotCompute
  can codegen the (de)serializer, and move the HLC / state transitions inside the
  handler. The `AtomicLoad` and `Yield` stubs go away — the v1.0 runtime owns the
  dispatch loop.
- **Delete and rewrite from scratch.** Arguably simpler if the original design intent
  (hand-rolled ring buffer with per-actor tail) is no longer compatible with the v1.0
  single-thread-per-handler execution model that `VectorAddRingKernel.cs` illustrates.

Smallest possible change that still compiles: the attribute-namespace swap + `ThreadId`
substitution, same as `TemporalKernels.cs`. But the result will still be unusable at
runtime because the signature shape doesn't match — you'll hit "no ring-kernel runtime
accepts this signature" at dispatch time. So the disabled state is **both** an attribute
concern **and** a design concern. Document it as a port, not a flip.

---

## Ready-to-bump checklist

One commit, once DotCompute v1.0.0 NuGet packages are live:

- [ ] Bump `DotComputeVersion` in `Directory.Build.props` to the shipped 1.0 tag
- [ ] Bump `MicrosoftExtensionsVersion` to 10.0.6 (or upstream floor)
- [ ] Bump `MicrosoftCodeAnalysisVersion` to 5.3.0
- [ ] Bump hardcoded ILLink / ILCompiler versions in the DotCompute backend csproj
- [ ] Bump `global.json` to .NET 10.0.x
- [ ] Flip every src / tests / examples csproj to `net10.0` (keep Generators on netstandard2.0)
- [ ] Verify `LangVersion` is `latest` or bump pinned value to `14`
- [ ] `dotnet restore` — confirm the new DotCompute packages resolve
- [ ] `dotnet build -c Release` — expect zero errors, warnings only for pre-existing items
- [ ] Port `TemporalKernels.cs.disabled` → `TemporalKernels.cs` (attribute namespace + `ThreadId` + barrier)
- [ ] Port `ActorRingKernels.cs.disabled` → `ActorRingKernels.cs` using the v1.0 `RingKernelContext` model (not a rename — a rewrite)
- [ ] Review whether `CustomRingKernelRuntimeFactory` is still needed; if DotCompute v1.0 exposes CPU + CUDA + Metal through a clean factory, consume that directly
- [ ] Drop the `DotComputeGenerateCudaSerialization` MSBuild property and the `ExcludeGeneratedRuntimeFactory` target in the DotCompute backend csproj if v1.0 resolves the generator mismatch they work around
- [ ] `dotnet test` — confirm the full test suite still passes (adjust for any v1.0 behavioural changes — classified errors may surface differently, and WSL2 tests will skip hard now)
- [ ] Update `README.md` badge (`.NET 10.0`, DotCompute v1.0), `CLAUDE.md` "Current Version", and add a `CHANGELOG.md` entry
- [ ] Bump root `<Version>` in `Directory.Build.props` (recommended: 0.4.0 or later given the surface impact)
- [ ] Remove `ocl-icd-opencl-dev` from `Dockerfile` (optional tidy-up)
