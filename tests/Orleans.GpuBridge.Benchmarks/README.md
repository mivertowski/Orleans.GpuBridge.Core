# Orleans.GpuBridge.Benchmarks

Comprehensive performance benchmarks for Orleans.GpuBridge.Core using BenchmarkDotNet.

## Benchmark Suites

### 1. GpuAccelerationBenchmarks
**Purpose**: Compare CPU and GPU performance for vector addition operations.

**Current Implementation**:
- ✅ CPU Scalar baseline (simple loop)
- ✅ CPU SIMD optimization (AVX2/AVX512)
- 🚧 GPU CUDA benchmarks (placeholders - waiting for DotCompute stabilization)

**Test Sizes**:
- 1K elements (1,000 floats)
- 100K elements (100,000 floats)
- 1M elements (1,000,000 floats)

**Expected Results** (when GPU implementation is complete):
```
1K Elements:
  CPU Scalar: ~2-10μs
  CPU SIMD:   ~0.5-2μs (4-8× faster)
  GPU:        ~10-50μs (SLOWER due to kernel launch overhead)

100K Elements:
  CPU Scalar: ~200-1,000μs
  CPU SIMD:   ~50-250μs (4-8× faster)
  GPU:        ~20-100μs (5-20× faster than scalar)

1M Elements:
  CPU Scalar: ~2-10ms
  CPU SIMD:   ~0.5-2.5ms (4-8× faster)
  GPU:        ~0.1-0.5ms (10-50× faster than scalar)
```

**Key Insights**:
- Small data (<10K): CPU faster due to kernel launch overhead
- Medium data (10K-1M): GPU 5-20× faster
- Large data (>1M): GPU 10-50× faster (bandwidth-limited)
- GPU-native actors: 100-500ns latency (no kernel launch overhead)

### 2. HybridLogicalClockBenchmarks
**Purpose**: Measure HLC timestamp generation and operations.

**Benchmarks**:
- `Now()` - Single timestamp generation (baseline)
- `Update()` - Timestamp update with remote timestamp
- `CompareTo()` - Timestamp comparison operation
- `Now_Batch1000()` - Throughput test (1000 sequential timestamps)
- `Update_Batch1000()` - Update throughput test
- `Now_AllocationTest()` - Memory allocation verification
- `Now_LogicalCounterIncrement()` - Worst-case scenario

**Target Performance**:
- Timestamp generation: <50ns (CPU baseline)
- Timestamp update: <70ns
- Comparison: <5ns (struct comparison)
- Memory allocation: 0 bytes (stack-only structs)

### 3. IntervalTreeBenchmarks
**Purpose**: Measure IntervalTree performance for temporal queries.

**Test Sizes**:
- 1K intervals
- 10K intervals
- 100K intervals
- 1M intervals

**Target Performance**:
- Single insertion: <1μs
- Query (1K intervals): O(log 1000) ≈ 10 comparisons
- Query (1M intervals): O(log 1000000) ≈ 20 comparisons

### 4. TemporalGraphStorageBenchmarks
**Purpose**: Measure TemporalGraphStorage performance for temporal graph operations.

**Benchmarks**:
- AddEdge operations
- Time-range queries
- Path search (various graph sizes)

**Target Performance**:
- AddEdge: <5μs
- Time-range query: <10μs
- Path search (small graph): <10μs
- Path search (medium graph): <100μs

### 5. MemoryAllocationBenchmarks
**Purpose**: Validate zero-allocation design principles.

**Target Performance**:
- HLC Now(): 0 bytes (stack-only structs)
- IntervalTree node: ~64 bytes per node
- GC Gen0: Minimize collections

## Running Benchmarks

### Run All Benchmarks
```bash
cd tests/Orleans.GpuBridge.Benchmarks
dotnet run -c Release
```

### Run Specific Benchmark Suite
```bash
# GPU Acceleration benchmarks only
dotnet run -c Release --filter "*GpuAccelerationBenchmarks*"

# HLC benchmarks only
dotnet run -c Release --filter "*HybridLogicalClockBenchmarks*"

# IntervalTree benchmarks only
dotnet run -c Release --filter "*IntervalTreeBenchmarks*"

# TemporalGraph benchmarks only
dotnet run -c Release --filter "*TemporalGraphStorageBenchmarks*"

# Memory benchmarks only
dotnet run -c Release --filter "*MemoryAllocationBenchmarks*"
```

### Run Specific Benchmark Method
```bash
# Run only 1M element GPU benchmark
dotnet run -c Release --filter "*VectorAdd_Gpu_1M*"

# Run only HLC Now() benchmark
dotnet run -c Release --filter "*HybridLogicalClockBenchmarks.Now"
```

### List Available Benchmarks
```bash
dotnet run -c Release --list flat
```

### Export Results
```bash
# Export to multiple formats
dotnet run -c Release --exporters markdown html csv

# Results will be saved in BenchmarkDotNet.Artifacts/results/
```

## Output Files

After running benchmarks, results are saved to:
```
BenchmarkDotNet.Artifacts/
  results/
    Orleans.GpuBridge.Benchmarks.*.md      # Markdown reports
    Orleans.GpuBridge.Benchmarks.*.html    # HTML reports
    Orleans.GpuBridge.Benchmarks.*.csv     # CSV data
  logs/
    *.log                                  # Execution logs
```

## Performance Targets

### Phase 7 Week 16 Goals
- **HLC**: <50ns timestamp generation (CPU baseline)
- **IntervalTree**: <1μs insertion, O(log n) queries
- **TemporalGraph**: <5μs edge addition, <10μs queries
- **GPU Acceleration**: Establish CPU baseline for future GPU comparison

### Future GPU Implementation Goals
- **VectorAdd 100K**: 5-20× speedup over CPU scalar
- **VectorAdd 1M**: 10-50× speedup over CPU scalar
- **Memory Bandwidth**: >1,935 GB/s (RTX 4090)
- **GPU-Native Actors**: 100-500ns message latency

## System Requirements

### Minimum Requirements
- .NET 9.0 SDK
- BenchmarkDotNet 0.14.0+
- Linux/Windows/macOS

### Recommended for GPU Benchmarks (Future)
- CUDA-capable GPU (RTX 3000/4000 series)
- CUDA Toolkit 12.0+
- DotCompute backend (when stable)

## Architecture Notes

### CPU Baseline Implementation
The current implementation provides CPU-only benchmarks as a baseline:
- **Scalar**: Simple loop without SIMD
- **SIMD**: AVX2/AVX512 vectorization (4-8 floats per iteration)

### GPU Implementation (Planned)
GPU benchmarks will be added once DotCompute backend stabilizes:
- **CUDA kernel compilation** via DotCompute.Abstractions
- **Memory management** via IUnifiedMemoryManager
- **Kernel execution** via ICompiledKernel
- **GPU-to-CPU transfers** via IUnifiedMemoryBuffer

### GPU-Native Actor Model
The revolutionary GPU-native actor model provides:
- **Ring kernels**: Persistent GPU threads (launched once)
- **GPU-resident message queues**: Lock-free on GPU
- **Temporal alignment**: HLC/Vector Clocks on GPU (20ns)
- **Hypergraph actors**: Multi-way relationships with GPU acceleration
- **100-500ns latency**: No kernel launch overhead

## Troubleshooting

### Build Warnings
Package version warnings for Microsoft.CodeAnalysis.Common can be safely ignored - they don't affect benchmark execution.

### GPU Benchmarks Not Running
GPU benchmarks are currently placeholders. They will be implemented once:
1. DotCompute.Abstractions API stabilizes
2. IUnifiedMemoryManager allocation methods are clarified
3. IUnifiedMemoryBuffer read/write API is finalized

### Performance Variance
BenchmarkDotNet automatically runs multiple iterations and warmups to minimize variance. If results are unstable:
- Close background applications
- Run in Release mode (`-c Release`)
- Check CPU governor settings (Linux)
- Ensure adequate cooling for sustained loads

## Contributing

When adding new benchmarks:
1. Use `[MemoryDiagnoser]` attribute for memory profiling
2. Include baseline benchmark with `[Benchmark(Baseline = true)]`
3. Use descriptive method names following pattern: `Operation_Variant_Size`
4. Add comprehensive XML documentation
5. Include expected performance targets in comments

## References

- [BenchmarkDotNet Documentation](https://benchmarkdotnet.org/)
- [Orleans.GpuBridge Architecture](../../docs/starter-kit/DESIGN.md)
- [GPU-Native Actor Model](../../docs/gpu-native-actors/)
- [Phase 7 Implementation Guide](../../docs/temporal/PHASE7-IMPLEMENTATION.md)
