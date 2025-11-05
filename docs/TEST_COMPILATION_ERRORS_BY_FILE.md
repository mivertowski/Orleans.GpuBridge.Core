# Test Compilation Errors by File

**Total Errors:** 336
**Status:** BLOCKING - No tests can run

---

## File: ILGPUBackendTests.cs
**Location:** `/tests/Orleans.GpuBridge.Tests/Integration/ILGPUBackendTests.cs`
**Error Count:** 4

### Errors:
1. **Line 246** - `CS1061`: `IGpuBridge` does not contain definition for `ExecuteKernelAsync`
2. **Line 287** - `CS1061`: `IGpuBridge` does not contain definition for `ExecuteKernelAsync`
3. **Line 306** - `CS1061`: `IGpuBridge` does not contain definition for `ExecuteKernelAsync`
4. **Line 330** - `CS1061`: `IGpuBridge` does not contain definition for `ExecuteKernelAsync`

**Fix Required:**
- Add `ExecuteKernelAsync` method to `IGpuBridge` interface, OR
- Update test code to use the correct method name (possibly `ExecuteAsync`?)

---

## File: TestFixtureBase.cs
**Location:** `/tests/Orleans.GpuBridge.Tests/TestingFramework/TestFixtureBase.cs`
**Error Count:** 2

### Errors:
1. **Line 103** - `CS0200`: Property `GpuExecutionHints.PreferredBatchSize` is read-only
2. **Line 105** - `CS0200`: Property `GpuExecutionHints.TimeoutMs` is read-only

**Fix Required:**
- Change `GpuExecutionHints` properties from `{ get; init; }` to `{ get; set; }`, OR
- Use constructor/with-expression initialization in tests

---

## File: GpuBridgePropertyTests.cs
**Location:** `/tests/Orleans.GpuBridge.Tests/PropertyBased/GpuBridgePropertyTests.cs`
**Error Count:** 7

### Errors:
1. **Line 207** - `CS0103`: Name `KernelHandle` does not exist
2. **Line 249** - `CS1061`: `KernelInfo` does not contain `DisplayName` property
3. **Line 275** - `CS0117`: `Arb` does not contain `Generate` method
4. **Line 279** - `CS0200`: Property `GpuExecutionHints.PreferredBatchSize` is read-only
5. **Line 281** - `CS0200`: Property `GpuExecutionHints.TimeoutMs` is read-only
6. **Line 290** - `CS1503`: Cannot convert `float` to `int` (argument 1)
7. **Line 290** - `CS1503`: Cannot convert `float` to `int` (argument 2)
8. **Line 298** - `CS0117`: `Arb` does not contain `Register` method

**Fix Required:**
- Restore `KernelHandle` type or update to use current type
- Check `KernelInfo` properties - use correct property name
- Update FsCheck API usage (v3.x breaking changes)
- Fix `GpuExecutionHints` initialization pattern
- Fix type conversions for numeric arguments

---

## File: FaultToleranceIntegrationTests.cs
**Location:** `/tests/Orleans.GpuBridge.Tests/Integration/FaultToleranceIntegrationTests.cs`
**Error Count:** 3

### Errors:
1. **Line 347** - `CS0311`: `IGpuBatchGrain<float[], float[]>` cannot be used as `TGrainInterface` - missing `IGrainWithStringKey` constraint
2. **Line 362** - `CS0311`: `IGpuBatchGrain<float[], float[]>` cannot be used as `TGrainInterface` - missing `IGrainWithStringKey` constraint
3. **Line 392** - `CS1501`: No overload for `GetGrain` takes 4 arguments

**Fix Required:**
- Make `IGpuBatchGrain` implement `IGrainWithStringKey`, OR
- Change grain retrieval code to use correct Orleans API pattern
- Check Orleans `GetGrain` signature changes

---

## File: OrleansClusterIntegrationTests.cs
**Location:** `/tests/Orleans.GpuBridge.Tests/Integration/OrleansClusterIntegrationTests.cs`
**Error Count:** 2

### Errors:
1. **Line 209** - `CS1061`: `TestCluster` does not contain `Host` property
2. **Line 276** - `CS1061`: `SimpleTimeSpanAssertions` does not contain `BeLessOrEqualTo` method

**Fix Required:**
- Use correct property name for Orleans TestCluster (possibly `Primary`?)
- Update FluentAssertions API usage (v7.x changes)

---

## File: EnhancedDeviceBrokerTests.cs
**Location:** `/tests/Orleans.GpuBridge.Tests/Unit/EnhancedDeviceBrokerTests.cs`
**Error Count:** 1

### Errors:
1. **Line 329** - `CS0103`: Name `DeviceType` does not exist

**Fix Required:**
- Restore `DeviceType` enum/type, OR
- Update to use current device type system

---

## Summary by Error Type

### CS1061 - Member Not Found (7 occurrences)
- `IGpuBridge.ExecuteKernelAsync` - 4 times
- `TestCluster.Host` - 1 time
- `SimpleTimeSpanAssertions.BeLessOrEqualTo` - 1 time
- `KernelInfo.DisplayName` - 1 time

### CS0200 - Read-Only Property (4 occurrences)
- `GpuExecutionHints.PreferredBatchSize` - 2 times
- `GpuExecutionHints.TimeoutMs` - 2 times

### CS0103 - Name Does Not Exist (2 occurrences)
- `DeviceType` - 1 time
- `KernelHandle` - 1 time

### CS0311 - Type Constraint Violation (2 occurrences)
- `IGpuBatchGrain` missing Orleans interface - 2 times

### CS0117 - Missing Static Method (2 occurrences)
- `Arb.Register` - 1 time
- `Arb.Generate` - 1 time

### CS1501 - Overload Not Found (1 occurrence)
- `GetGrain` wrong parameter count - 1 time

### CS1503 - Type Conversion (2 occurrences)
- `float` to `int` conversion - 2 times

---

## Recommended Fix Order

1. **Phase 1 - Core API Issues** (Highest Priority)
   - Fix `IGpuBridge.ExecuteKernelAsync` (4 errors)
   - Fix `GpuExecutionHints` read-only properties (4 errors)
   - Fix `IGpuBatchGrain` Orleans constraints (2 errors)

2. **Phase 2 - Missing Types**
   - Restore or replace `DeviceType` (1 error)
   - Restore or replace `KernelHandle` (1 error)
   - Fix `KernelInfo.DisplayName` (1 error)

3. **Phase 3 - External Library Updates**
   - Update FsCheck API usage (2 errors)
   - Update FluentAssertions API usage (1 error)
   - Fix Orleans TestCluster API (1 error)

4. **Phase 4 - Minor Fixes**
   - Fix type conversions (2 errors)
   - Fix Orleans GetGrain overload (1 error)

---

## Expected Impact After Fixes

After resolving these 20+ unique issues (causing 336 total errors):
- ✅ Test project should compile
- ✅ Can run test suite
- ✅ Can verify production code quality
- ✅ Can provide QA sign-off for housekeeping

**Current Blocker:** Cannot execute ANY tests until compilation succeeds.
