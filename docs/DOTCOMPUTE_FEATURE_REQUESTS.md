# DotCompute Feature Requests

**From**: Orleans.GpuBridge.Core team
**Date**: 2025-11-27
**DotCompute Version**: 0.5.0

---

## 1. GenerateCudaSerializationTask - Microsoft.CodeAnalysis Version Conflict

**Priority**: High
**Type**: Bug / Compatibility

### Issue
The `DotCompute.Generators.MSBuild.GenerateCudaSerializationTask` fails to load when `Microsoft.CodeAnalysis.Common 4.14.0` is present in the project:

```
error MSB4062: The "DotCompute.Generators.MSBuild.GenerateCudaSerializationTask" task could not be loaded
from the assembly. Could not load file or assembly 'Microsoft.CodeAnalysis, Version=4.14.0.0'
```

### Workaround
Projects must add `<DotComputeGenerateCudaSerialization>false</DotComputeGenerateCudaSerialization>` to disable the task.

### Suggested Fix
- Update the MSBuild task to be compatible with Microsoft.CodeAnalysis 4.14.0
- Or make the task version-agnostic using binding redirects

---

## 2. Source Generator Missing #nullable Directive

**Priority**: Medium
**Type**: Enhancement

### Issue
Generated code (e.g., `RingKernelRegistry.g.cs`, `RingKernelRuntimeFactory.g.cs`) doesn't emit `#nullable` directives, causing CS8669 warnings:

```
warning CS8669: The annotation for nullable reference types should only be used in code
within a '#nullable' annotations context.
```

### Workaround
Projects must suppress with `<NoWarn>$(NoWarn);CS8669</NoWarn>`.

### Suggested Fix
Add `#nullable enable` at the top of all generated source files.

---

## 3. RingKernelRuntimeFactory References Non-Existent CPU.RingKernels

**Priority**: High
**Type**: Bug

### Issue
The generated `RingKernelRuntimeFactory.g.cs` references `DotCompute.Backends.CPU.RingKernels` namespace:

```csharp
// Generated code references:
DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime
```

But this namespace doesn't exist in `DotCompute.Backends.CPU` v0.5.0, causing:

```
error CS0234: The type or namespace name 'RingKernels' does not exist in the namespace
'DotCompute.Backends.CPU'
```

### Workaround
Either:
1. Add `DotCompute.Backends.CPU` as a dependency (even when not needed)
2. Exclude the generated file via MSBuild Target (doesn't work reliably for source generators)

### Suggested Fix
- Add the `RingKernels` namespace to `DotCompute.Backends.CPU` package
- Or make the generator conditionally emit CPU runtime code only when CPU backend is referenced

---

## 4. Source Generator Transitive Activation

**Priority**: Medium
**Type**: Enhancement

### Issue
`DotCompute.Generators` activates transitively through project references, even when marked with `PrivateAssets="all"`. This causes downstream projects to receive generated code they don't need.

**Scenario**:
```
ProjectA (has DotCompute.Generators)
  └─> ProjectB (references ProjectA, gets generator activated)
       └─> ProjectC (references ProjectB, also gets generator activated)
```

### Workaround
Each downstream project must either:
1. Disable serialization generation
2. Exclude generated files via Target
3. Add missing backend dependencies

### Suggested Fix
- Ensure `PrivateAssets="all"` properly prevents transitive generator activation
- Or provide a project property to opt-out of generator activation: `<DotComputeDisableGenerators>true</DotComputeDisableGenerators>`

---

## 5. Telemetry Sequence Analyzer Warnings (DC015-DC017)

**Priority**: Low
**Type**: Enhancement

### Issue
The analyzers DC015, DC016, DC017 report warnings for wrapper methods where the caller controls the telemetry sequence, not the wrapper itself.

### Workaround
Suppress with `<NoWarn>$(NoWarn);DC015;DC016;DC017</NoWarn>`.

### Suggested Fix
Consider adding an attribute like `[TelemetrySequenceControlledByCaller]` to suppress these warnings on specific methods.

---

## Summary Table

| # | Issue | Priority | Type |
|---|-------|----------|------|
| 1 | GenerateCudaSerializationTask version conflict | High | Bug |
| 2 | Missing #nullable in generated code | Medium | Enhancement |
| 3 | CPU.RingKernels namespace missing | High | Bug |
| 4 | Transitive generator activation | Medium | Enhancement |
| 5 | Telemetry sequence analyzer false positives | Low | Enhancement |

---

## Contact

For questions about these requests, please reach out to the Orleans.GpuBridge.Core team.
