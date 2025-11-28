# DotCompute Feature Requests

**From**: Orleans.GpuBridge.Core team
**Date**: 2025-11-27
**DotCompute Version**: 0.5.0 → **Resolved in 0.5.1**

---

## Status: ALL ISSUES RESOLVED IN v0.5.1

All feature requests below have been addressed by the DotCompute team in version 0.5.1. Thank you!

---

## 1. GenerateCudaSerializationTask - Microsoft.CodeAnalysis Version Conflict

**Priority**: High
**Type**: Bug / Compatibility
**Status**: ✅ **RESOLVED in v0.5.1**

### Issue
The `DotCompute.Generators.MSBuild.GenerateCudaSerializationTask` fails to load when `Microsoft.CodeAnalysis.Common 4.14.0` is present in the project:

```
error MSB4062: The "DotCompute.Generators.MSBuild.GenerateCudaSerializationTask" task could not be loaded
from the assembly. Could not load file or assembly 'Microsoft.CodeAnalysis, Version=4.14.0.0'
```

### Resolution
Fixed in DotCompute v0.5.1 - MSBuild task now compatible with Microsoft.CodeAnalysis 4.14.0.

---

## 2. Source Generator Missing #nullable Directive

**Priority**: Medium
**Type**: Enhancement
**Status**: ✅ **RESOLVED in v0.5.1**

### Issue
Generated code (e.g., `RingKernelRegistry.g.cs`, `RingKernelRuntimeFactory.g.cs`) doesn't emit `#nullable` directives, causing CS8669 warnings:

```
warning CS8669: The annotation for nullable reference types should only be used in code
within a '#nullable' annotations context.
```

### Resolution
Fixed in DotCompute v0.5.1 - Generated source files now include `#nullable enable` directive.

---

## 3. RingKernelRuntimeFactory References Non-Existent CPU.RingKernels

**Priority**: High
**Type**: Bug
**Status**: ✅ **RESOLVED in v0.5.1**

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

### Resolution
Fixed in DotCompute v0.5.1 - `RingKernels` namespace added to `DotCompute.Backends.CPU` package.

---

## 4. Source Generator Transitive Activation

**Priority**: Medium
**Type**: Enhancement
**Status**: ✅ **RESOLVED in v0.5.1**

### Issue
`DotCompute.Generators` activates transitively through project references, even when marked with `PrivateAssets="all"`. This causes downstream projects to receive generated code they don't need.

**Scenario**:
```
ProjectA (has DotCompute.Generators)
  └─> ProjectB (references ProjectA, gets generator activated)
       └─> ProjectC (references ProjectB, also gets generator activated)
```

### Resolution
Fixed in DotCompute v0.5.1 - `PrivateAssets="all"` now properly prevents transitive generator activation.

---

## 5. Telemetry Sequence Analyzer Warnings (DC015-DC017)

**Priority**: Low
**Type**: Enhancement
**Status**: ✅ **RESOLVED in v0.5.1**

### Issue
The analyzers DC015, DC016, DC017 report warnings for wrapper methods where the caller controls the telemetry sequence, not the wrapper itself.

### Resolution
Fixed in DotCompute v0.5.1 - Analyzer now correctly identifies wrapper methods and suppresses false positives.

---

## Summary Table

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 1 | GenerateCudaSerializationTask version conflict | High | ✅ Resolved |
| 2 | Missing #nullable in generated code | Medium | ✅ Resolved |
| 3 | CPU.RingKernels namespace missing | High | ✅ Resolved |
| 4 | Transitive generator activation | Medium | ✅ Resolved |
| 5 | Telemetry sequence analyzer false positives | Low | ✅ Resolved |

---

## Acknowledgements

Thank you to the DotCompute team for the rapid turnaround on these issues!
