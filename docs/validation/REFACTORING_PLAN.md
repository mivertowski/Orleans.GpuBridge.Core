# God Files Refactoring Plan

## Overview

This document provides a concrete refactoring plan for the 65 identified god files, with detailed file structure recommendations for the most critical cases.

---

## Priority 1: Critical God Files (16 & 11 Definitions)

### 🔴 CpuFallbackProvider.cs (16 Definitions)

**Current:** `/src/Orleans.GpuBridge.Runtime/Infrastructure/Backends/Providers/CpuFallbackProvider.cs` (709 lines)

**Proposed Structure:**

```
/src/Orleans.GpuBridge.Runtime/Infrastructure/Backends/Providers/CpuFallback/
├── CpuFallbackProvider.cs                    # Main provider (50 lines)
├── DeviceManagement/
│   ├── CpuDeviceManager.cs                   # Device manager (50 lines)
│   ├── CpuDevice.cs                          # Device implementation (30 lines)
│   ├── CpuContext.cs                         # Compute context (20 lines)
│   └── CpuCommandQueue.cs                    # Command queue (30 lines)
├── Compilation/
│   └── CpuKernelCompiler.cs                  # Kernel compiler (80 lines)
├── Memory/
│   ├── CpuMemoryAllocator.cs                 # Memory allocator (60 lines)
│   ├── CpuDeviceMemory.cs                    # Basic device memory (30 lines)
│   ├── CpuDeviceMemoryGeneric.cs             # Generic device memory (40 lines)
│   ├── CpuPinnedMemory.cs                    # Pinned memory (30 lines)
│   └── CpuUnifiedMemory.cs                   # Unified memory (40 lines)
└── Execution/
    ├── CpuKernelExecutor.cs                  # Kernel executor (80 lines)
    ├── CpuKernelExecution.cs                 # Execution handle (30 lines)
    ├── CpuKernelGraph.cs                     # Kernel graph (40 lines)
    ├── CpuGraphNode.cs                       # Graph node (30 lines)
    └── CpuCompiledGraph.cs                   # Compiled graph (30 lines)
```

**Benefits:**
- Clear separation of concerns
- Easy to navigate and maintain
- Follows backend provider pattern used by ILGPU
- Each file < 100 lines

---

### 🟠 GpuResiliencePolicyOptions.cs (11 Definitions)

**Current:** `/src/Orleans.GpuBridge.Resilience/Configuration/GpuResiliencePolicyOptions.cs` (336 lines)

**Proposed Structure:**

```
/src/Orleans.GpuBridge.Resilience/Configuration/
├── GpuResiliencePolicyOptions.cs             # Main options aggregator (50 lines)
├── Policies/
│   ├── RetryPolicyOptions.cs                 # Retry configuration (35 lines)
│   ├── CircuitBreakerPolicyOptions.cs        # Circuit breaker config (30 lines)
│   ├── TimeoutPolicyOptions.cs               # Timeout configuration (50 lines)
│   ├── BulkheadPolicyOptions.cs              # Bulkhead config (30 lines)
│   └── RateLimitingOptions.cs                # Rate limiting config (35 lines)
├── Chaos/
│   ├── ChaosEngineeringOptions.cs            # Main chaos options (30 lines)
│   ├── LatencyInjectionOptions.cs            # Latency injection (25 lines)
│   ├── ExceptionInjectionOptions.cs          # Exception injection (25 lines)
│   └── ResourceExhaustionOptions.cs          # Resource simulation (25 lines)
└── Enums/
    └── RateLimitingAlgorithm.cs              # Rate limiting algorithms (15 lines)
```

**Benefits:**
- Grouped by policy category
- Easy to find specific configuration
- Aligns with Polly's resilience patterns
- Each file < 60 lines

---

## Priority 2: High Priority God Files (9 Definitions)

### 🟠 HighPerformanceMemoryPool.cs (9 Definitions)

**Current:** `/src/Orleans.GpuBridge.Performance/HighPerformanceMemoryPool.cs` (385 lines)

**Proposed Structure:**

```
/src/Orleans.GpuBridge.Performance/Memory/
├── HighPerformanceMemoryPool.cs              # Main pool (120 lines)
├── HighPerformanceMemoryOwner.cs             # Memory owner (40 lines)
├── UnmanagedMemoryManager.cs                 # NUMA manager (50 lines)
├── Models/
│   ├── MemoryPoolBucket.cs                   # Bucket tracker (30 lines)
│   ├── MemoryPoolStats.cs                    # Pool statistics (20 lines)
│   └── BucketStats.cs                        # Bucket stats (15 lines)
└── Enums/
    ├── AllocationType.cs                     # P/Invoke allocation type (20 lines)
    ├── MemoryProtection.cs                   # P/Invoke protection (15 lines)
    └── FreeType.cs                           # P/Invoke free type (15 lines)
```

---

### 🟠 MemoryPool.cs (9 Definitions)

**Current:** `/src/Orleans.GpuBridge.Runtime/MemoryPool.cs` (559 lines)

**Proposed Structure:**

```
/src/Orleans.GpuBridge.Runtime/Memory/
├── Pooling/
│   ├── AdvancedMemoryPool.cs                 # Main pool (180 lines)
│   ├── PooledGpuMemory.cs                    # Pooled memory (60 lines)
│   ├── PooledSegment.cs                      # Segment tracker (30 lines)
│   └── AllocationInfo.cs                     # Allocation metadata (20 lines)
├── Management/
│   ├── MemoryPoolManager.cs                  # Cross-pool manager (120 lines)
│   └── MemoryPoolOptions.cs                  # Configuration (40 lines)
├── Health/
│   └── MemoryPoolHealth.cs                   # Health monitoring (30 lines)
├── Models/
│   └── GpuMemoryStats.cs                     # Statistics (30 lines)
└── Enums/
    └── HealthStatus.cs                       # Move to Abstractions!
```

**Important:** `HealthStatus` should be moved to:
- `/src/Orleans.GpuBridge.Abstractions/Enums/HealthStatus.cs`

---

## Priority 3: Medium Priority Files (7-8 Definitions)

### Pattern: Performance Monitoring

**Files:**
- `VectorizedKernelExecutor.cs` (7 definitions)
- `PerformanceBenchmarkSuite.cs` (7 definitions)

**Recommended Structure:**
```
/src/Orleans.GpuBridge.Performance/
├── Vectorization/
│   ├── VectorizedKernelExecutor.cs
│   ├── VectorOperations.cs
│   ├── VectorizationStrategy.cs
│   └── VectorizedMemory.cs
└── Benchmarking/
    ├── PerformanceBenchmarkSuite.cs
    ├── BenchmarkConfiguration.cs
    ├── BenchmarkResults.cs
    └── BenchmarkMetrics.cs
```

---

### Pattern: Resilience & Fallback

**Files:**
- `GpuFallbackChain.cs` (7 definitions)
- `FallbackMetricsCollector.cs` (7 definitions)

**Recommended Structure:**
```
/src/Orleans.GpuBridge.Resilience/Fallback/
├── GpuFallbackChain.cs
├── FallbackExecutor.cs
├── FallbackStrategy.cs
├── FallbackLevel.cs (enum)
└── Metrics/
    ├── FallbackMetricsCollector.cs
    ├── FallbackMetrics.cs
    └── MetricType.cs (enum)
```

---

### Pattern: Logging Infrastructure

**Files:**
- `LoggingConfiguration.cs` (8 definitions)
- `LoggerFactory.cs` (7 definitions)
- `LogBuffer.cs` (6 definitions)

**Recommended Structure:**
```
/src/Orleans.GpuBridge.Logging/
├── Configuration/
│   ├── LoggingConfiguration.cs
│   ├── FileLoggerConfiguration.cs
│   ├── ConsoleLoggerConfiguration.cs
│   ├── TelemetryLoggerConfiguration.cs
│   └── LogRotationPolicy.cs
├── Core/
│   ├── LoggerFactory.cs
│   ├── GpuBridgeLogger.cs
│   └── LoggerDelegateManager.cs
└── Buffering/
    ├── LogBuffer.cs
    ├── LogBufferSegment.cs
    ├── BufferEntry.cs
    └── BufferStatistics.cs
```

---

## Priority 4: Lower Priority Files (5-6 Definitions)

### Pattern: Backend Execution

**Files:**
- `DotComputeKernelExecutor.cs` (6 definitions)
- `ILGPUKernelExecution.cs` (5 definitions)

**Recommended Structure:**
```
/src/Orleans.GpuBridge.Backends.{Backend}/Execution/
├── {Backend}KernelExecutor.cs
├── {Backend}KernelExecution.cs
├── {Backend}ExecutionContext.cs
├── Models/
│   ├── ExecutionParameters.cs
│   └── ExecutionResult.cs
└── Enums/
    └── ExecutionStatus.cs
```

---

## Refactoring Guidelines

### Step-by-Step Process

1. **Create New Directory Structure**
   ```bash
   mkdir -p src/Orleans.GpuBridge.Runtime/Infrastructure/Backends/Providers/CpuFallback/{DeviceManagement,Compilation,Memory,Execution}
   ```

2. **Extract Each Type to New File**
   - Copy type definition
   - Add appropriate `using` statements
   - Ensure namespace matches directory structure

3. **Update Original File**
   - Remove extracted types
   - Keep only main class
   - Update internal references

4. **Update .csproj if Needed**
   - Usually auto-detected by SDK
   - Verify with `dotnet build`

5. **Run Tests**
   ```bash
   dotnet test
   ```

6. **Commit Incrementally**
   ```bash
   git add .
   git commit -m "refactor: split CpuFallbackProvider into 16 files"
   ```

---

## Naming Conventions

### File Names
- Match the primary type name exactly
- Use PascalCase
- Include generic parameters: `ClassName<T>.cs` or `ClassName.Generic.cs`

### Namespace Structure
```csharp
namespace Orleans.GpuBridge.{Component}.{Feature}.{SubFeature};
```

Examples:
```csharp
namespace Orleans.GpuBridge.Runtime.Providers.CpuFallback.Memory;
namespace Orleans.GpuBridge.Resilience.Configuration.Policies;
namespace Orleans.GpuBridge.Performance.Memory.Models;
```

---

## Impact Analysis

### Before Refactoring
- 65 god files
- Average ~8-10 types per god file
- Difficult navigation
- Merge conflicts common

### After Refactoring
- ~350 focused files
- 1 primary type per file
- Clear organization
- Easy to find components

### Build Time Impact
- Minimal (modern SDKs handle many files efficiently)
- Potential for better incremental compilation

### Code Review Impact
- Smaller, focused PRs
- Easier to review changes
- Clearer git history

---

## Automation Opportunities

### Script to Find God Files
```bash
#!/bin/bash
# find-god-files.sh
find src -name "*.cs" -type f ! -path "*/obj/*" ! -path "*/bin/*" | \
while read file; do
    count=$(grep -c "^\s*\(public\|internal\)\s\+\(class\|struct\|enum\|interface\|record\)" "$file" 2>/dev/null || echo 0)
    if [ "$count" -gt 2 ]; then
        echo "$file: $count definitions"
    fi
done | sort -t: -k2 -rn
```

### Script to Extract Type
```bash
#!/bin/bash
# extract-type.sh <source-file> <type-name> <destination-file>
# TODO: Implement C# type extraction
```

---

## Migration Checklist

### For Each God File:

- [ ] Identify all type definitions
- [ ] Create directory structure
- [ ] Extract types to new files
- [ ] Update namespaces
- [ ] Fix `using` statements
- [ ] Update internal access modifiers if needed
- [ ] Run `dotnet build`
- [ ] Run `dotnet test`
- [ ] Commit changes
- [ ] Update this checklist

---

## Estimated Effort

| Priority | Files | Types | Estimated Hours |
|----------|-------|-------|-----------------|
| Priority 1 | 2 | 27 | 8-10 hours |
| Priority 2 | 10 | 80 | 16-20 hours |
| Priority 3 | 8 | 48 | 10-12 hours |
| Priority 4 | 45 | 180 | 30-40 hours |
| **Total** | **65** | **335** | **64-82 hours** |

### Sprint Planning
- **Sprint 1:** Priority 1 + start Priority 2 (12-15 hours)
- **Sprint 2:** Complete Priority 2 + start Priority 3 (12-15 hours)
- **Sprint 3:** Complete Priority 3 + start Priority 4 (12-15 hours)
- **Sprint 4-6:** Complete Priority 4 (30-40 hours across 3 sprints)

---

## Success Criteria

- [ ] All files have ≤ 2 type definitions
- [ ] Clear directory structure by feature
- [ ] All tests passing
- [ ] Zero build warnings related to file organization
- [ ] Documentation updated
- [ ] Team trained on new structure

---

**Next Steps:**
1. Review this plan with team
2. Get approval for directory structure
3. Start with CpuFallbackProvider.cs
4. Track progress in project board
5. Update this document as patterns emerge

---

**Document Version:** 1.0
**Last Updated:** October 27, 2025
**Author:** CodebaseAnalyst Agent
