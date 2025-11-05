/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/Program.cs(41,32): warning IL2026: Using member 'System.Reflection.Assembly.LoadFrom(String)' which has 'RequiresUnreferencedCodeAttribute' can break functionality when trimming application code. Types and members the loaded assembly depends on might be removed. [/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/DotComputeTypeAnalyzer.csproj]
/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/Program.cs(47,29): warning IL2026: Using member 'System.Reflection.Assembly.GetExportedTypes()' which has 'RequiresUnreferencedCodeAttribute' can break functionality when trimming application code. Types might be removed. [/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/DotComputeTypeAnalyzer.csproj]
/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/Program.cs(68,43): warning IL2075: 'this' argument does not satisfy 'DynamicallyAccessedMemberTypes.PublicMethods' in call to 'System.Type.GetMethods(BindingFlags)'. The return value of method 'System.Collections.Generic.List<T>.Enumerator.Current.get' does not have matching annotations. The source value must declare at least the same requirements as those declared on the target location it is assigned to. [/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/DotComputeTypeAnalyzer.csproj]
/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/Program.cs(91,49): warning IL2075: 'this' argument does not satisfy 'DynamicallyAccessedMemberTypes.PublicMethods' in call to 'System.Type.GetMethods(BindingFlags)'. The return value of method 'System.Collections.Generic.List<T>.Enumerator.Current.get' does not have matching annotations. The source value must declare at least the same requirements as those declared on the target location it is assigned to. [/home/mivertowski/GpuBridgeCore/Orleans.GpuBridge.Core/scripts/DotComputeTypeAnalyzer/DotComputeTypeAnalyzer.csproj]
# DotCompute Package Type Analysis
## Package Version: 0.2.0-alpha

## DotCompute.Abstractions
**Assembly**: `DotCompute.Abstractions`
**Version**: 0.2.0.0

### Namespace: `DotCompute.Abstractions`

#### Interfaces

- **`IAccelerator`**
  - `AcceleratorInfo get_Info()`
  - `AcceleratorType get_Type()`
  - `String get_DeviceType()`
  - `IUnifiedMemoryManager get_Memory()`
  - `IUnifiedMemoryManager get_MemoryManager()`
  - *(+4 more methods)*
- **`IAcceleratorManager`**
  - `IAccelerator get_DefaultAccelerator()`
  - `IReadOnlyList<IAccelerator> get_AvailableAccelerators()`
  - `Int32 get_Count()`
  - `ValueTask InitializeAsync(CancellationToken cancellationToken)`
  - `IAccelerator GetAccelerator(Int32 index)`
  - *(+9 more methods)*
- **`IAcceleratorProvider`**
  - `String get_Name()`
  - `IReadOnlyList<AcceleratorType> get_SupportedTypes()`
  - `ValueTask<IEnumerable<IAccelerator>> DiscoverAsync(CancellationToken cancellationToken)`
  - `ValueTask<IAccelerator> CreateAsync(AcceleratorInfo info, CancellationToken cancellationToken)`
- **`ICompiledKernel`**
  - `Guid get_Id()`
  - `String get_Name()`
  - `ValueTask ExecuteAsync(KernelArguments arguments, CancellationToken cancellationToken)`
- **`IKernel`**
  - `String get_Name()`
  - `String get_Source()`
  - `String get_EntryPoint()`
  - `Int32 get_RequiredSharedMemory()`
- **`IKernelSource`**
  - `String get_Name()`
  - `String get_Code()`
  - `KernelLanguage get_Language()`
  - `String get_EntryPoint()`
  - `IReadOnlyList<String> get_Dependencies()`
- **`IMemoryMapping`1`**
  - `Span<T> get_Span()`
  - `MemoryMapMode get_Mode()`
  - `Boolean get_IsValid()`
  - `Void Flush()`
- **`IMemoryStatistics`**
  - `Int64 get_TotalAllocatedBytes()`
  - `Int64 get_AvailableBytes()`
  - `Int64 get_PeakUsageBytes()`
  - `Int32 get_AllocationCount()`
  - `Double get_FragmentationPercentage()`
  - *(+1 more methods)*
- **`ISyncMemoryBuffer`**
  - `Void* GetHostPointer()`
  - `Span<T> AsSpan()`
  - `ISyncMemoryBuffer Slice(Int64 offset, Int64 length)`
  - `Boolean get_IsDisposed()`
- **`ISyncMemoryManager`**
  - `ISyncMemoryBuffer Allocate(Int64 sizeInBytes, MemoryOptions options)`
  - `ISyncMemoryBuffer AllocateAligned(Int64 sizeInBytes, Int32 alignment, MemoryOptions options)`
  - `Void Copy(ISyncMemoryBuffer source, ISyncMemoryBuffer destination, Int64 sizeInBytes, Int64 sourceOffset, Int64 destinationOffset)`
  - `Void CopyFromHost(Void* source, ISyncMemoryBuffer destination, Int64 sizeInBytes, Int64 destinationOffset)`
  - `Void CopyToHost(ISyncMemoryBuffer source, Void* destination, Int64 sizeInBytes, Int64 sourceOffset)`
  - *(+5 more methods)*
- **`IUnifiedKernelCompiler`**
  - `Task<ICompiledKernel> CompileAsync(KernelDefinition kernelDefinition, IAccelerator accelerator, CancellationToken cancellationToken)`
  - `Task<Boolean> CanCompileAsync(KernelDefinition kernelDefinition, IAccelerator accelerator)`
  - `CompilationOptions GetSupportedOptions(IAccelerator accelerator)`
  - `Task<IDictionary<String, ICompiledKernel>> BatchCompileAsync(IEnumerable<KernelDefinition> kernelDefinitions, IAccelerator accelerator, CancellationToken cancellationToken)`
- **`IUnifiedKernelCompiler`2`**
  - `String get_Name()`
  - `IReadOnlyList<KernelLanguage> get_SupportedSourceTypes()`
  - `IReadOnlyDictionary<String, Object> get_Capabilities()`
  - `ValueTask<TCompiled> CompileAsync(TSource source, CompilationOptions options, CancellationToken cancellationToken)`
  - `UnifiedValidationResult Validate(TSource source)`
  - *(+2 more methods)*
- **`IUnifiedMemoryBuffer`**
  - `Int64 get_SizeInBytes()`
  - `MemoryOptions get_Options()`
  - `Boolean get_IsDisposed()`
  - `BufferState get_State()`
  - `ValueTask CopyFromAsync(ReadOnlyMemory<T> source, Int64 offset, CancellationToken cancellationToken)`
  - *(+1 more methods)*
- **`IUnifiedMemoryBuffer`1`**
  - `Int32 get_Length()`
  - `Int32 get_ElementCount()`
  - `IAccelerator get_Accelerator()`
  - `Boolean get_IsOnHost()`
  - `Boolean get_IsOnDevice()`
  - *(+25 more methods)*
- **`IUnifiedMemoryManager`**
  - `IAccelerator get_Accelerator()`
  - `MemoryStatistics get_Statistics()`
  - `Int64 get_MaxAllocationSize()`
  - `Int64 get_TotalAvailableMemory()`
  - `Int64 get_CurrentAllocatedMemory()`
  - *(+21 more methods)*

#### Classes

- **`AcceleratorException`**
- **`AcceleratorInfo`**
- **`AcceleratorSelectionCriteria`**
- **`CompilationException`**
- **`CompilationOptions`**
- **`MemoryException`**

#### Abstract Classes

- **`AcceleratorEvent`**
- **`ComputeStream`**

#### Enums

- **`AcceleratorType`**: None, CPU, CUDA, ROCm, OneAPI
- **`KernelSourceType`**: CSharp, Cuda, OpenCL, Metal, HLSL
- **`MemoryAccess`**: ReadOnly, WriteOnly, ReadWrite, HostAccess
- **`MemoryLocation`**: Host, Device, HostPinned, Unified, Managed
- **`MemoryMapMode`**: Read, Write, ReadWrite

#### Structs

- **`AcceleratorContext`**
- **`DeviceMemory`**

### Namespace: `DotCompute.Abstractions.Accelerators`

#### Enums

- **`AcceleratorFeature`**: None, Float16, DoublePrecision, LongInteger, TensorCores

### Namespace: `DotCompute.Abstractions.Analysis`

#### Interfaces

- **`IAdvancedComplexityMetrics`**
  - `ComplexityClass get_ComputationalComplexityClass()`
  - `ComplexityClass get_SpaceComplexity()`
  - `Int64 get_MemoryAccesses()`
  - `IReadOnlyDictionary<String, Double> get_OperationComplexity()`
  - `IReadOnlyList<MemoryAccessComplexity> get_MemoryAccessPatterns()`
  - *(+3 more methods)*
- **`IComplexityMetrics`**
  - `Int32 get_ComputationalComplexity()`
  - `Int32 get_MemoryComplexity()`
  - `Int32 get_ParallelizationComplexity()`
  - `Int32 get_OverallComplexity()`
  - `Int64 get_OperationCount()`
  - *(+7 more methods)*

#### Classes

- **`MemoryAccessComplexity`**
- **`MemoryConflict`**
- **`MemoryHotspot`**
- **`MemoryLocation`**
- **`MemoryRegion`**

#### Enums

- **`ComplexityClass`**: Constant, Logarithmic, Linear, Linearithmic, Quadratic
- **`ComputeComplexity`**: Low, Medium, High, VeryHigh
- **`ConflictType`**: BankConflict, CacheLineConflict, FalseSharing, CoalescingConflict, WriteAfterRead
- **`MemoryAccessPattern`**: Sequential, Random, Strided, Coalesced, Broadcast
- **`MemoryRegionType`**: Global, Shared, Local, Constant, Texture

### Namespace: `DotCompute.Abstractions.Attributes`

#### Classes

- **`RingKernelAttribute`**

### Namespace: `DotCompute.Abstractions.Compute.Enums`

#### Enums

- **`ComputeBackendType`**: CPU, CUDA, OpenCL, Metal, Vulkan

### Namespace: `DotCompute.Abstractions.Compute.Options`

#### Classes

- **`ExecutionOptions`**

#### Enums

- **`ExecutionPriority`**: Low, Normal, High, Critical

### Namespace: `DotCompute.Abstractions.Configuration`

#### Classes

- **`KernelExecutionConfig`**
  - `static KernelExecutionConfig CreateDebugConfig()`
  - `static KernelExecutionConfig CreatePerformanceConfig()`
  - `static KernelExecutionConfig CreateHighPrecisionConfig()`

### Namespace: `DotCompute.Abstractions.Debugging`

#### Interfaces

- **`IKernelDebugService`**
  - `Task<KernelValidationResult> ValidateKernelAsync(String kernelName, Object[] inputs, Single tolerance)`
  - `Task<KernelExecutionResult> ExecuteOnBackendAsync(String kernelName, String backendType, Object[] inputs)`
  - `Task<ResultComparisonReport> CompareResultsAsync(IEnumerable<KernelExecutionResult> results, ComparisonStrategy comparisonStrategy)`
  - `Task<KernelExecutionTrace> TraceKernelExecutionAsync(String kernelName, Object[] inputs, String[] tracePoints)`
  - `Task<DeterminismReport> ValidateDeterminismAsync(String kernelName, Object[] inputs, Int32 iterations)`
  - *(+3 more methods)*

#### Classes

- **`AcceleratorComparisonResult`**
- **`AcceleratorPerformanceSummary`**
- **`BottleneckAnalysis`**
- **`ComparisonIssue`**
- **`CpuProfilingData`**
- **`CrossValidationResult`**
- **`DebugConfiguration`**
- **`DebugData`**
- **`DebugReport`**
- **`DebugServiceOptions`**
- **`DebugValidationIssue`**
- **`DeterminismAnalysisResult`**
- **`DeterminismReport`**
- **`DeterminismTestResult`**
- **`ErrorAnalysis`**
- **`ExecutionStatistics`**
- **`InputValidationResult`**
- **`KernelDebugInfo`**
- **`KernelExecutionTrace`**
- **`KernelValidationResult`**
- **`MemoryAccessPattern`**
- **`MemoryAnalysis`**
- **`MemoryAnalysisReport`**
- **`MemoryIssue`**
- **`MemoryPatternAnalysis`**
- **`MemoryProfile`**
- **`MemoryProfilingData`**
- **`MetricAnomaly`**
- **`MetricPoint`**
- **`MetricsReport`**
- **`MetricsSeries`**
- **`MetricsSummary`**
- **`PerformanceAnalysis`**
- **`PerformanceAnalysisResult`**
- **`PerformanceAnomaly`**
- **`PerformanceBottleneck`**
- **`PerformanceOptimization`**
- **`ProfilingData`**
- **`ProfilingSession`**
- **`ReportAttachment`**
- **`ResourceUsage`**
- **`ResultComparison`**
- **`ResultComparisonReport`**
- **`ResultDifference`**
- **`SystemMetrics`**
- **`TracePoint`**
- **`ValidationProfile`**

#### Enums

- **`AnomalySeverity`**: Low, Medium, High, Critical
- **`AnomalyType`**: PerformanceSpike, MemorySpike, ExecutionTime, MemoryUsage, ThroughputDrop
- **`BottleneckSeverity`**: Low, Medium, High, Critical
- **`ComparisonSeverity`**: Info, Warning, Error
- **`ComparisonStrategy`**: Exact, Tolerance, Statistical, Relative
- **`LogLevel`**: Trace, Debug, Information, Warning, Error
- **`MemoryIssueSeverity`**: Info, Warning, Error, Critical
- **`MemoryIssueType`**: BufferOverflow, LargeAllocation, MemoryLeak, InefficientAccess, EmptyArray
- **`ReportFormat`**: Markdown, Html, Json, PlainText, Xml
- **`TrendDirection`**: Unknown, None, Improving, Degrading, Stable
- **`ValidationLevel`**: Minimal, Basic, Standard, Comprehensive

### Namespace: `DotCompute.Abstractions.Debugging.Types`

#### Classes

- **`BackendAnalysisSummary`**
- **`BackendDebugStats`**
- **`BackendInfo`**
- **`ComprehensiveDebugReport`**
- **`CoreKernelDebugOrchestrator`**
- **`CpuUtilizationStats`**
- **`DebugServiceStatistics`**
- **`DeviceQueueStatistics`**
- **`GpuUtilizationStats`**
- **`IoUtilizationStats`**
- **`MemoryUsageAnalysis`**
- **`MemoryUtilizationStats`**
- **`PerformancePrediction`**
- **`PerformanceProfile`**
- **`PerformanceReport`**
- **`PerformanceTrends`**
- **`ResourceUtilizationReport`**
- **`TrendAnalysis`**

#### Enums

- **`ComparisonStrategy`**: Exact, Tolerance, Statistical, Relative
- **`LogLevel`**: Trace, Debug, Information, Warning, Error

### Namespace: `DotCompute.Abstractions.Enums`

#### Enums

- **`PrecisionMode`**: Single, Double, Half, Mixed

### Namespace: `DotCompute.Abstractions.Execution`

#### Classes

- **`DeviceStealingStats`**
- **`KernelExecutionContext`**
  - `static IUnifiedMemoryBuffer<Byte> GetBuffer(KernelExecutionContext context, Int32 index)`
  - `static KernelExecutionContext CreateDefault()`
- **`StealingStatistics`**
- **`WorkStealingStatistics`**

#### Enums

- **`ExecutionPriority`**: Idle, Low, BelowNormal, Normal, AboveNormal
- **`StreamFlags`**: None, NonBlocking, HighPriority, LowPriority, Synchronized

### Namespace: `DotCompute.Abstractions.Extensions`

### Namespace: `DotCompute.Abstractions.Factories`

#### Interfaces

- **`IUnifiedAcceleratorFactory`**
  - `ValueTask<IAccelerator> CreateAsync(AcceleratorType type, AcceleratorConfiguration configuration, IServiceProvider serviceProvider, CancellationToken cancellationToken)`
  - `ValueTask<IAccelerator> CreateAsync(String backendName, AcceleratorConfiguration configuration, IServiceProvider serviceProvider, CancellationToken cancellationToken)`
  - `ValueTask<IAccelerator> CreateAsync(AcceleratorInfo acceleratorInfo, IServiceProvider serviceProvider, CancellationToken cancellationToken)`
  - `ValueTask<TProvider> CreateProviderAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)`
  - `ValueTask<IReadOnlyList<AcceleratorType>> GetAvailableTypesAsync(CancellationToken cancellationToken)`
  - *(+4 more methods)*

#### Classes

- **`AcceleratorConfiguration`**
- **`WorkloadProfile`**

#### Enums

- **`MemoryAllocationStrategy`**: Default, Pooled, OptimizedPooled, Unified, AggressiveCaching
- **`PerformanceProfile`**: Balanced, MaxPerformance, LowLatency, PowerSaver

### Namespace: `DotCompute.Abstractions.Interfaces`

#### Interfaces

- **`IComputeOrchestrator`**
  - `Task<T> ExecuteAsync(String kernelName, Object[] args)`
  - `Task<T> ExecuteAsync(String kernelName, String preferredBackend, Object[] args)`
  - `Task<T> ExecuteAsync(String kernelName, IAccelerator accelerator, Object[] args)`
  - `Task<T> ExecuteWithBuffersAsync(String kernelName, IEnumerable<IUnifiedMemoryBuffer> buffers, Object[] scalarArgs)`
  - `Task<IAccelerator> GetOptimalAcceleratorAsync(String kernelName)`
  - *(+5 more methods)*
- **`IKernelExecutionParameters`**
  - `IReadOnlyList<Object> get_Arguments()`
  - `String get_PreferredBackend()`
  - `IDictionary<String, Object> get_Options()`
  - `CancellationToken get_CancellationToken()`

#### Structs

- **`GridDimensions`**

### Namespace: `DotCompute.Abstractions.Interfaces.Compute`

#### Interfaces

- **`ICompilationMetadata`**
  - `DateTimeOffset get_CompilationTime()`
  - `CompilationOptions get_Options()`
  - `IReadOnlyList<String> get_Warnings()`
  - `OptimizationLevel get_OptimizationLevel()`
- **`IComputeEngine`**
  - `ValueTask<ICompiledKernel> CompileKernelAsync(String kernelSource, String entryPoint, CompilationOptions options, CancellationToken cancellationToken)`
  - `ValueTask ExecuteAsync(ICompiledKernel kernel, Object[] arguments, ComputeBackendType backendType, ExecutionOptions options, CancellationToken cancellationToken)`
  - `IReadOnlyList<ComputeBackendType> get_AvailableBackends()`
  - `ComputeBackendType get_DefaultBackend()`

### Namespace: `DotCompute.Abstractions.Interfaces.Device`

#### Interfaces

- **`ICacheSizes`**
  - `Int64 get_L1Size()`
  - `Int64 get_L2Size()`
  - `Int64 get_L3Size()`
  - `Int64 get_TextureCacheSize()`
  - `Int64 get_ConstantCacheSize()`
- **`ICommandQueue`**
  - `String get_Id()`
  - `IComputeDevice get_Device()`
  - `ValueTask EnqueueKernelAsync(ICompiledKernel kernel, KernelExecutionContext context, CancellationToken cancellationToken)`
  - `ValueTask EnqueueCopyAsync(IDeviceMemory source, IDeviceMemory destination, Int64 sourceOffset, Int64 destinationOffset, Nullable<Int64> sizeInBytes, CancellationToken cancellationToken)`
  - `ValueTask EnqueueBarrierAsync(CancellationToken cancellationToken)`
  - *(+2 more methods)*
- **`IComputeDevice`**
  - `String get_Id()`
  - `String get_Name()`
  - `ComputeDeviceType get_Type()`
  - `IDeviceCapabilities get_Capabilities()`
  - `DeviceStatus get_Status()`
  - *(+6 more methods)*
- **`IDeviceCapabilities`**
  - `Version get_ComputeCapability()`
  - `Int32 get_MaxWorkGroupSize()`
  - `Int32 get_MaxWorkItemDimensions()`
  - `IReadOnlyList<Int64> get_MaxWorkItemSizes()`
  - `Int32 get_ComputeUnits()`
  - *(+4 more methods)*
- **`IDeviceMemory`**
  - `Int64 get_SizeInBytes()`
  - `IComputeDevice get_Device()`
  - `MemoryAccess get_AccessMode()`
  - `ValueTask WriteAsync(ReadOnlyMemory<T> source, Int64 offset, CancellationToken cancellationToken)`
  - `ValueTask ReadAsync(Memory<T> destination, Int64 offset, CancellationToken cancellationToken)`
  - *(+2 more methods)*
- **`IDeviceMemoryInfo`**
  - `Int64 get_TotalGlobalMemory()`
  - `Int64 get_AvailableGlobalMemory()`
  - `Int64 get_LocalMemoryPerWorkGroup()`
  - `Double get_MemoryBandwidth()`
  - `ICacheSizes get_CacheSizes()`
  - *(+1 more methods)*
- **`IDeviceMetrics`**
  - `Double get_Utilization()`
  - `Double get_MemoryUsage()`
  - `Nullable<Double> get_Temperature()`
  - `Nullable<Double> get_PowerConsumption()`
  - `Int64 get_KernelExecutionCount()`
  - *(+3 more methods)*
- **`IMemoryTransferStats`**
  - `Int64 get_BytesToDevice()`
  - `Int64 get_BytesFromDevice()`
  - `Double get_AverageRateToDevice()`
  - `Double get_AverageRateFromDevice()`
  - `TimeSpan get_TotalTransferTime()`

### Namespace: `DotCompute.Abstractions.Interfaces.Kernels`

#### Interfaces

- **`ICompiledKernel`**
  - `String get_Name()`
  - `Boolean get_IsReady()`
  - `String get_BackendType()`
  - `Task ExecuteAsync(Object[] parameters, CancellationToken cancellationToken)`
  - `Object GetMetadata()`
- **`IKernelExecutor`**
  - `IAccelerator get_Accelerator()`
  - `ValueTask<KernelExecutionResult> ExecuteAsync(CompiledKernel kernel, KernelArgument[] arguments, KernelExecutionConfig executionConfig, CancellationToken cancellationToken)`
  - `ValueTask<KernelExecutionResult> ExecuteAndWaitAsync(CompiledKernel kernel, KernelArgument[] arguments, KernelExecutionConfig executionConfig, CancellationToken cancellationToken)`
  - `KernelExecutionHandle EnqueueExecution(CompiledKernel kernel, KernelArgument[] arguments, KernelExecutionConfig executionConfig)`
  - `ValueTask<KernelExecutionResult> WaitForCompletionAsync(KernelExecutionHandle handle, CancellationToken cancellationToken)`
  - *(+2 more methods)*
- **`IKernelGenerator`**
  - `AcceleratorType get_AcceleratorType()`
  - `GeneratedKernel GenerateKernel(Expression expression, KernelGenerationContext context)`
  - `GeneratedKernel GenerateOperationKernel(String operation, Type[] inputTypes, Type outputType, KernelGenerationContext context)`
  - `Boolean CanCompile(Expression expression)`
  - `KernelOptimizationHints GetOptimizationHints(KernelGenerationContext context)`
- **`IKernelManager`**
  - `Void RegisterGenerator(AcceleratorType acceleratorType, IKernelGenerator generator)`
  - `Void RegisterCompiler(AcceleratorType acceleratorType, IUnifiedKernelCompiler compiler)`
  - `Void RegisterExecutor(AcceleratorType acceleratorType, IKernelExecutor executor)`
  - `ValueTask<ManagedCompiledKernel> GetOrCompileKernelAsync(Expression expression, IAccelerator accelerator, KernelGenerationContext context, CompilationOptions options, CancellationToken cancellationToken)`
  - `ValueTask<ManagedCompiledKernel> GetOrCompileOperationKernelAsync(String operation, Type[] inputTypes, Type outputType, IAccelerator accelerator, KernelGenerationContext context, CompilationOptions options, CancellationToken cancellationToken)`
  - *(+4 more methods)*

#### Classes

- **`BottleneckAnalysis`**
- **`GeneratedKernel`**
- **`KernelArgument`**
- **`KernelExecutionConfig`**
- **`KernelExecutionHandle`**
- **`KernelExecutionResult`**
  - `static KernelExecutionResult CreateSuccess(KernelExecutionHandle handle)`
  - `static KernelExecutionResult CreateFailure(KernelExecutionHandle handle, Exception error)`
- **`KernelExecutionTimings`**
- **`KernelGenerationContext`**
- **`KernelOptimizationHints`**
- **`KernelProfilingResult`**

#### Enums

- **`CacheConfiguration`**: Default, PreferL1, PreferShared, Equal
- **`KernelExecutionFlags`**: None, PreferSharedMemory, PreferL1Cache, DisableCache, CooperativeKernel
- **`MemorySpace`**: Global, Shared, Constant, Private
- **`PrecisionMode`**: Half, Single, Double, Mixed

### Namespace: `DotCompute.Abstractions.Interfaces.Linq`

#### Interfaces

- **`IComputeLinqProvider`**
  - `IQueryable<T> CreateQueryable(IEnumerable<T> source, IAccelerator accelerator)`
  - `IQueryable<T> CreateQueryable(T[] source, IAccelerator accelerator)`
  - `Task<T> ExecuteAsync(Expression expression, CancellationToken cancellationToken)`
  - `Task<T> ExecuteAsync(Expression expression, IAccelerator preferredAccelerator, CancellationToken cancellationToken)`
  - `IEnumerable<OptimizationSuggestion> GetOptimizationSuggestions(Expression expression)`
  - *(+2 more methods)*
- **`IOperatorAnalyzer`**
  - `OperatorAnalysisResult AnalyzeOperator(Expression expression)`
  - `OperatorInfo GetOperatorInfo(ExpressionType operatorType, Type[] operandTypes, BackendType backend)`
  - `VectorizationAnalysis AnalyzeVectorization(ExpressionType operatorType, Type[] operandTypes)`
  - `FusionAnalysisResult AnalyzeFusion(IEnumerable<Expression> operators)`
  - `Double EstimateComputationalCost(ExpressionType operatorType, Type[] operandTypes, BackendType backend)`
  - *(+2 more methods)*

#### Classes

- **`AccuracyInfo`**
- **`FusionAnalysisResult`**
- **`FusionOpportunity`**
- **`OperatorAnalysisResult`**
- **`OperatorCompatibility`**
- **`OperatorInfo`**
- **`OptimizationSuggestion`**
- **`PrecisionAnalysisResult`**
- **`VectorizationAnalysis`**

#### Enums

- **`AccuracyLevel`**: Exact, High, Standard, Reduced, Low
- **`BackendType`**: CPU, CUDA, Metal, OpenCL, Vulkan
- **`ComputationalComplexity`**: Constant, Linear, Quadratic, Logarithmic, Exponential
- **`FusionPattern`**: ElementWise, MultiplyAdd, Reduction, Matrix, Conditional
- **`ImplementationMethod`**: Native, Intrinsic, Library, Emulated, Custom
- **`NumericalPrecision`**: Exact, High, Standard, Reduced, Low
- **`SuggestionSeverity`**: Info, Warning, High, Critical
- **`SupportLevel`**: Full, Partial, Basic, None

### Namespace: `DotCompute.Abstractions.Interfaces.Pipelines`

#### Interfaces

- **`IKernelChainBuilder`**
  - `IKernelChainBuilder Kernel(String kernelName, Object[] args)`
  - `IKernelChainBuilder ThenExecute(String kernelName, Object[] args)`
  - `IKernelChainBuilder Then(String kernelName, Object[] args)`
  - `IKernelChainBuilder Parallel(ValueTuple`2[] kernels)`
  - `IKernelChainBuilder Branch(Func<T, Boolean> condition, Func<IKernelChainBuilder, IKernelChainBuilder> truePath, Func<IKernelChainBuilder, IKernelChainBuilder> falsePath)`
  - *(+10 more methods)*
- **`IKernelPipeline`**
  - `String get_Id()`
  - `String get_Name()`
  - `IReadOnlyList<IPipelineStage> get_Stages()`
  - `PipelineOptimizationSettings get_OptimizationSettings()`
  - `IReadOnlyDictionary<String, Object> get_Metadata()`
  - *(+4 more methods)*
- **`IKernelPipelineBuilder`**
  - `IKernelPipelineBuilder WithName(String name)`
  - `IKernelPipelineBuilder AddKernel(String name, ICompiledKernel kernel, Action<IKernelStageBuilder> configure)`
  - `IKernelPipelineBuilder AddParallel(Action<IParallelStageBuilder> configure)`
  - `IKernelPipelineBuilder AddBranch(Func<PipelineExecutionContext, Boolean> condition, Action<IKernelPipelineBuilder> trueBranch, Action<IKernelPipelineBuilder> falseBranch)`
  - `IKernelPipelineBuilder AddLoop(Func<PipelineExecutionContext, Int32, Boolean> condition, Action<IKernelPipelineBuilder> body)`
  - *(+6 more methods)*
- **`IKernelStageBuilder`**
  - `IKernelStageBuilder WithParameters(Object[] parameters)`
  - `IKernelStageBuilder PreferBackend(String backendName, BackendFallbackStrategy fallbackStrategy)`
  - `IKernelStageBuilder WithTimeout(TimeSpan timeout)`
  - `IKernelStageBuilder WithMemoryHints(MemoryHint[] hints)`
  - `IKernelStageBuilder WithRetryPolicy(Int32 maxRetries, RetryStrategy retryStrategy)`
  - *(+10 more methods)*
- **`IParallelStageBuilder`**
  - `IParallelStageBuilder AddKernel(String kernelName, Action<IKernelStageBuilder> stageBuilder)`
  - `IParallelStageBuilder AddKernels(IEnumerable<ParallelKernelConfig> kernelConfigs)`
  - `IParallelStageBuilder WithMaxDegreeOfParallelism(Int32 maxDegreeOfParallelism)`
  - `IParallelStageBuilder WithSynchronization(SynchronizationMode mode)`
  - `IParallelStageBuilder WithLoadBalancing(LoadBalancingStrategy strategy)`
  - *(+11 more methods)*
- **`IPipeline`**
  - `String get_Name()`
  - `IReadOnlyList<IPipelineStage> get_Stages()`
- **`IPipelineMemory`1`**
  - `String get_Id()`
  - `Int64 get_ElementCount()`
  - `Int64 get_SizeInBytes()`
  - `Boolean get_IsLocked()`
  - `MemoryAccess get_AccessMode()`
  - *(+6 more methods)*
- **`IPipelineMemoryManager`**
  - `ValueTask<IPipelineMemory<T>> AllocateAsync(Int64 elementCount, MemoryHint hint, CancellationToken cancellationToken)`
  - `ValueTask<IPipelineMemory<T>> AllocateSharedAsync(String key, Int64 elementCount, MemoryHint hint, CancellationToken cancellationToken)`
  - `IPipelineMemory<T> GetShared(String key)`
  - `ValueTask TransferAsync(IPipelineMemory<T> memory, String fromStage, String toStage, CancellationToken cancellationToken)`
  - `IPipelineMemoryView<T> CreateView(IPipelineMemory<T> memory, Int64 offset, Nullable<Int64> length)`
  - *(+4 more methods)*
- **`IPipelineMemoryView`1`**
  - `Int64 get_Offset()`
  - `Int64 get_Length()`
  - `IPipelineMemory<T> get_Parent()`
  - `IPipelineMemoryView<T> Slice(Int64 offset, Int64 length)`
- **`IPipelineMemoryWithDirectAccess`1`**
  - `Span<T> GetSpan()`
  - `ReadOnlySpan<T> GetReadOnlySpan()`
- **`IPipelineOptimizer`**
  - `Task<PipelineAnalysisResult> AnalyzeAsync(IKernelPipeline pipeline, CancellationToken cancellationToken)`
  - `Task<IKernelPipeline> OptimizeAsync(IKernelPipeline pipeline, OptimizationType optimizationTypes, PipelineOptimizationSettings settings, CancellationToken cancellationToken)`
  - `Task<IKernelPipeline> ApplyKernelFusionAsync(IKernelPipeline pipeline, FusionCriteria fusionCriteria, CancellationToken cancellationToken)`
  - `Task<IKernelPipeline> OptimizeMemoryAccessAsync(IKernelPipeline pipeline, MemoryConstraints memoryConstraints, CancellationToken cancellationToken)`
  - `Task<IKernelPipeline> OptimizeForBackendsAsync(IKernelPipeline pipeline, IEnumerable<String> targetBackends, CancellationToken cancellationToken)`
  - *(+8 more methods)*
- **`IPipelineStage`**
  - `String get_Id()`
  - `String get_Name()`
  - `PipelineStageType get_Type()`
  - `IReadOnlyList<String> get_Dependencies()`
  - `IReadOnlyDictionary<String, Object> get_Metadata()`
  - *(+3 more methods)*

#### Classes

- **`KernelChainExecutionResult`**
- **`KernelChainMemoryMetrics`**
- **`KernelChainValidationResult`**
- **`KernelStepMetrics`**
- **`MemoryManagerStats`**
- **`MemoryPoolOptions`**

#### Enums

- **`MemoryLayoutHint`**: None, Sequential, Strided, Tiled2D, Blocked3D
- **`MemoryLockMode`**: ReadOnly, ReadWrite, Exclusive
- **`PoolRetentionPolicy`**: KeepAll, TimeBasedRelease, LeastRecentlyUsed, Adaptive

#### Structs

- **`MemoryLock`1`**

### Namespace: `DotCompute.Abstractions.Interfaces.Pipelines.Interfaces`

#### Interfaces

- **`IPipelineMetrics`**
  - `String get_PipelineId()`
  - `Int64 get_ExecutionCount()`
  - `Int64 get_SuccessfulExecutionCount()`
  - `Int64 get_FailedExecutionCount()`
  - `TimeSpan get_AverageExecutionTime()`
  - *(+15 more methods)*
- **`IStageMetrics`**
  - `String get_StageId()`
  - `String get_StageName()`
  - `Int64 get_ExecutionCount()`
  - `TimeSpan get_AverageExecutionTime()`
  - `TimeSpan get_MinExecutionTime()`
  - *(+6 more methods)*

### Namespace: `DotCompute.Abstractions.Interfaces.Pipelines.Profiling`

#### Interfaces

- **`IPipelineProfiler`**
  - `Void StartPipelineExecution(String pipelineId, String executionId)`
  - `Void EndPipelineExecution(String executionId)`
  - `Void StartStageExecution(String executionId, String stageId)`
  - `Void EndStageExecution(String executionId, String stageId)`
  - `Void RecordMemoryAllocation(String executionId, Int64 bytes, String purpose)`
  - *(+6 more methods)*

### Namespace: `DotCompute.Abstractions.Interfaces.Plugins`

#### Interfaces

- **`IAlgorithmPlugin`**
  - `String get_Id()`
  - `String get_Name()`
  - `Version get_Version()`
  - `String get_Description()`
  - `IReadOnlyList<Type> get_SupportedInputTypes()`
  - *(+4 more methods)*
- **`IBackendPlugin`**
  - `String get_Id()`
  - `String get_Name()`
  - `Version get_Version()`
  - `String get_Description()`
  - `String get_Author()`
  - *(+17 more methods)*
- **`IPluginExecutor`**
  - `Task<Object> ExecutePluginAsync(String pluginId, Object[] inputs, Dictionary<String, Object> parameters, CancellationToken cancellationToken)`
  - `Task<Object> ExecuteWithRetryAsync(IAlgorithmPlugin plugin, Object[] inputs, Dictionary<String, Object> parameters, CancellationToken cancellationToken)`
  - `Task<PluginExecutionStatistics> GetStatisticsAsync(String pluginId)`
  - `Task<ValidationResult> ValidateInputsAsync(String pluginId, Object[] inputs, Dictionary<String, Object> parameters)`

#### Classes

- **`AlgorithmMetadata`**
- **`PluginErrorEventArgs`**
- **`PluginExecutionStatistics`**
- **`PluginHealthChangedEventArgs`**
- **`PluginMetrics`**
- **`PluginStateChangedEventArgs`**
- **`PluginValidationResult`**
- **`ValidationResult`**

#### Enums

- **`PluginCapabilities`**: None, ComputeBackend, StorageProvider, NetworkProvider, SecurityProvider
- **`PluginHealth`**: Unknown, Healthy, Degraded, Unhealthy, Critical
- **`PluginState`**: Unknown, Loading, Loaded, Initializing, Initialized

### Namespace: `DotCompute.Abstractions.Interfaces.Recovery`

#### Interfaces

- **`IKernelExecutionMonitor`**
  - `String get_KernelId()`
  - `String get_DeviceId()`
  - `TimeSpan get_ExecutionTime()`
  - `Boolean get_IsHanging()`
  - `Boolean get_IsCompleted()`
  - *(+3 more methods)*
- **`IRecoveryStrategy`1`**
  - `RecoveryCapability get_Capability()`
  - `Int32 get_Priority()`
  - `Boolean CanHandle(Exception exception, TContext context)`
  - `Task<RecoveryResult> RecoverAsync(Exception exception, TContext context, RecoveryOptions options, CancellationToken cancellationToken)`

#### Classes

- **`RecoveryMetrics`**
- **`RecoveryOptions`**
- **`RecoveryResult`**
  - `static RecoveryResult CreateSuccess(String message, String strategy)`
  - `static RecoveryResult CreateFailure(String message, String strategy)`
  - `static RecoveryResult CreateFailure(String message, String strategy, Exception exception)`

#### Abstract Classes

- **`BaseRecoveryStrategy`1`**

#### Enums

- **`RecoveryCapability`**: None, GpuErrors, MemoryErrors, CompilationErrors, NetworkErrors

### Namespace: `DotCompute.Abstractions.Interfaces.Services`

#### Interfaces

- **`IKernelCacheService`**
  - `Task<ICompiledKernel> GetAsync(String cacheKey)`
  - `Task StoreAsync(String cacheKey, ICompiledKernel kernel)`
  - `String GenerateCacheKey(KernelDefinition definition, IAccelerator accelerator, CompilationOptions options)`
  - `Task ClearAsync()`
  - `KernelCacheStatistics GetStatistics()`
  - *(+1 more methods)*
- **`IKernelCompilerService`**
  - `Task<ICompiledKernel> CompileAsync(KernelDefinition definition, IAccelerator accelerator, CompilationOptions options)`
  - `Task PrecompileAsync(IEnumerable<KernelDefinition> definitions, IAccelerator accelerator)`
  - `KernelCompilationStatistics GetStatistics()`
  - `Task<KernelDefinition> OptimizeAsync(KernelDefinition definition, IAccelerator accelerator)`
  - `Task<KernelValidationResult> ValidateAsync(KernelDefinition definition, IAccelerator accelerator)`

#### Classes

- **`KernelCacheStatistics`**
- **`KernelCompilationStatistics`**
- **`KernelResourceRequirements`**
- **`KernelValidationResult`**
  - `static KernelValidationResult Success(KernelResourceRequirements resourceRequirements, Dictionary<String, Double> performancePredictions)`
  - `static KernelValidationResult Failure(IEnumerable<String> errors, IEnumerable<String> warnings)`

### Namespace: `DotCompute.Abstractions.Interfaces.Telemetry`

#### Interfaces

- **`ILogSink`**
  - `Task WriteAsync(StructuredLogEntry entry)`
- **`IOperationTimer`**
  - `ITimerHandle StartOperation(String operationName, String operationId)`
  - `IDisposable StartOperationScope(String operationName, String operationId)`
  - `ValueTuple<T, TimeSpan> TimeOperation(String operationName, Func<T> operation)`
  - `Task<ValueTuple<T, TimeSpan>> TimeOperationAsync(String operationName, Func<Task<T>> operation)`
  - `TimeSpan TimeOperation(String operationName, Action operation)`
  - *(+13 more methods)*
- **`ITelemetryExporter`**
  - `ValueTask ExportMetricsAsync(IEnumerable<MetricData> metrics, CancellationToken cancellationToken)`
  - `ValueTask ExportTracesAsync(IEnumerable<Activity> activities, CancellationToken cancellationToken)`
  - `ValueTask FlushAsync(CancellationToken cancellationToken)`
- **`ITelemetryProvider`**
  - `Void RecordMetric(String name, Double value, IDictionary<String, Object> tags)`
  - `Void IncrementCounter(String name, Int64 increment, IDictionary<String, Object> tags)`
  - `Void RecordHistogram(String name, Double value, IDictionary<String, Object> tags)`
  - `Activity StartActivity(String name, ActivityKind kind)`
  - `Void RecordEvent(String name, IDictionary<String, Object> attributes)`
  - *(+7 more methods)*
- **`ITelemetryService`**
  - `Void RecordKernelExecution(String kernelName, String deviceId, TimeSpan executionTime, TelemetryKernelPerformanceMetrics metrics, String correlationId, Exception exception)`
  - `Void RecordMemoryOperation(String operationType, String deviceId, Int64 bytes, TimeSpan duration, MemoryAccessMetrics metrics, String correlationId, Exception exception)`
  - `TraceContext StartDistributedTrace(String operationName, String correlationId, Dictionary<String, Object> tags)`
  - `Task<TraceData> FinishDistributedTraceAsync(String correlationId, TraceStatus status)`
  - `Task<PerformanceProfile> CreatePerformanceProfileAsync(String correlationId, ProfileOptions options, CancellationToken cancellationToken)`
  - *(+2 more methods)*
- **`ITimerHandle`**
  - `String get_OperationName()`
  - `String get_OperationId()`
  - `DateTime get_StartTime()`
  - `TimeSpan get_Elapsed()`
  - `TimeSpan StopTimer(IDictionary<String, Object> metadata)`
  - *(+2 more methods)*

#### Classes

- **`MetricData`**
- **`OpenTelemetryIntegration`**
  - `static Void Initialize()`
- **`OperationStatistics`**
- **`OperationTimingEventArgs`**
- **`PrometheusOptions`**
- **`TelemetryOptions`**
- **`TelemetryServiceOptions`**

#### Enums

- **`MetricType`**: Counter, Gauge, Histogram, Summary

### Namespace: `DotCompute.Abstractions.Kernels`

#### Classes

- **`BytecodeKernelSource`**
- **`CompiledKernel`**
- **`KernelArgument`**
  - `static KernelArgument Create(String name, T value)`
  - `static KernelArgument CreateBuffer(String name, Object buffer, ParameterDirection direction)`
- **`KernelArguments`**
  - `static KernelArguments Create(Int32 capacity)`
  - `static KernelArguments Create(Object[] arguments)`
  - `static KernelArguments Create(IEnumerable<IUnifiedMemoryBuffer> buffers, IEnumerable<Object> scalars)`
- **`KernelCompilationOptions`**
  - `static KernelCompilationOptions Debug()`
  - `static KernelCompilationOptions Release()`
  - `static KernelCompilationOptions Balanced()`
- **`KernelConfiguration`**
- **`KernelDefinition`**
- **`KernelExecutionOptions`**
- **`KernelLaunchConfiguration`**
- **`KernelParameter`**
- **`TextKernelSource`**

#### Enums

- **`MemorySpace`**: Global, Local, Shared, Constant, Private
- **`ParameterDirection`**: In, Out, InOut

#### Structs

- **`WorkDimensions`**
- **`WorkGroupSize`**

### Namespace: `DotCompute.Abstractions.Kernels.Compilation`

#### Abstract Classes

- **`ManagedCompiledKernel`**

### Namespace: `DotCompute.Abstractions.Kernels.Types`

#### Classes

- **`KernelCacheStatistics`**
- **`KernelMetadata`**
- **`KernelParameterInfo`**

#### Enums

- **`KernelLanguage`**: Auto, Cuda, OpenCL, Ptx, HLSL

### Namespace: `DotCompute.Abstractions.Logging`

#### Classes

- **`KernelPerformanceMetrics`**
- **`MemoryAccessMetrics`**
- **`StructuredLogEntry`**

### Namespace: `DotCompute.Abstractions.Memory`

#### Interfaces

- **`IMemoryBuffer`1`**
- **`IUnifiedMemoryPool`**
  - `String get_PoolId()`
  - `IAccelerator get_Accelerator()`
  - `Int64 get_TotalSize()`
  - `Int64 get_AllocatedSize()`
  - `Int64 get_AvailableSize()`
  - *(+11 more methods)*

#### Classes

- **`MappedMemory`1`**
- **`MemoryInfo`**
- **`MemoryPoolStatistics`**
- **`MemoryStatistics`**

#### Enums

- **`BufferState`**: Uninitialized, Allocated, HostAccess, DeviceAccess, Transferring
- **`MapMode`**: Read, Write, ReadWrite, WriteDiscard, WriteNoOverwrite
- **`MemoryOptions`**: None, Pinned, Mapped, WriteCombined, Portable
- **`MemoryType`**: Host, Device, Unified, Pinned, Shared

#### Structs

- **`DeviceMemory`1`**

### Namespace: `DotCompute.Abstractions.Models`

#### Classes

- **`DeviceCapabilities`**
- **`MemoryOptions`**

### Namespace: `DotCompute.Abstractions.Models.Device`

#### Classes

- **`CommandQueueOptions`**

#### Enums

- **`ComputeDeviceType`**: CPU, GPU, FPGA, Accelerator, Virtual
- **`DataTypeSupport`**: Int8, Int16, Int32, Int64, Float16
- **`DeviceFeature`**: DoublePrecision, HalfPrecision, Atomics, LocalMemory, Images
- **`DeviceFeatures`**: None, DoublePrecision, HalfPrecision, Atomics, LocalMemory
- **`DeviceStatus`**: Available, Busy, Offline, Error, Initializing
- **`QueuePriority`**: Low, Normal, High

### Namespace: `DotCompute.Abstractions.Models.Pipelines`

#### Interfaces

- **`IBranchCondition`**
  - `IList<KernelChainStep> get_TruePath()`
  - `IList<KernelChainStep> get_FalsePath()`
  - `Boolean EvaluateCondition(Object previousResult)`
- **`IPipelineMetrics`**
  - `String get_PipelineId()`
  - `TimeSpan get_TotalExecutionTime()`
  - `Int64 get_PeakMemoryUsage()`
  - `Int32 get_StageCount()`
  - `IReadOnlyList<IStageMetrics> get_StageMetrics()`

#### Classes

- **`BranchCondition`1`**
- **`DefaultStageMetrics`**
- **`KernelChainStep`**
- **`PipelineError`**
- **`PipelineExecutionMetrics`**
- **`PipelineValidationResult`**
- **`StageExecutionResult`**
- **`StageValidationResult`**
  - `static StageValidationResult Success()`
  - `static StageValidationResult Failure(String[] errors)`

#### Abstract Classes

- **`PipelineExecutionContext`**

#### Enums

- **`KernelChainStepType`**: Sequential, Parallel, Branch, Loop

### Namespace: `DotCompute.Abstractions.Performance`

#### Interfaces

- **`IUnifiedPerformanceMetrics`**
  - `String get_ComponentName()`
  - `Void RecordKernelExecution(KernelExecutionMetrics metrics)`
  - `Void RecordMemoryOperation(MemoryOperationMetrics metrics)`
  - `Void RecordDataTransfer(DataTransferMetrics metrics)`
  - `PerformanceSnapshot GetSnapshot()`
  - *(+2 more methods)*

#### Classes

- **`DataTransferMetrics`**
- **`KernelExecutionMetrics`**
- **`KernelStatistics`**
- **`MemoryOperationMetrics`**
- **`PerformanceMetrics`**
  - `static PerformanceMetrics FromStopwatch(Stopwatch stopwatch, String operation)`
  - `static PerformanceMetrics FromTimeSpan(TimeSpan elapsed, String operation)`
  - `static PerformanceMetrics Aggregate(IEnumerable<PerformanceMetrics> metrics)`
- **`PerformanceSnapshot`**
- **`PerformanceTimer`**

#### Enums

- **`MemoryOperationType`**: Allocate, Deallocate, Resize, Defragment, Pool
- **`MetricsExportFormat`**: Json, Csv, Prometheus, StatsD, OpenTelemetry
- **`TransferDirection`**: HostToDevice, DeviceToHost, DeviceToDevice, HostToHost, PeerToPeer

### Namespace: `DotCompute.Abstractions.Pipelines`

#### Interfaces

- **`ICacheEntry`**
  - `String get_Key()`
  - `Object get_Value()`
  - `Int64 get_Size()`
  - `DateTimeOffset get_CreatedAt()`
  - `DateTimeOffset get_LastAccessedAt()`
  - *(+4 more methods)*
- **`IKernelPipelineBuilder`**
  - `IKernelPipelineBuilder AddStage(String kernelName, Object[] parameters)`
  - `IKernelPipelineBuilder Transform(Func<TInput, TOutput> transform)`
  - `Task<T> ExecuteAsync(T input, CancellationToken cancellationToken)`
  - `Task<T> ExecutePipelineAsync(CancellationToken cancellationToken)`
  - `Object Create()`
  - *(+2 more methods)*
- **`IPipelineConfiguration`**
  - `String get_Name()`
  - `String get_Description()`
  - `Version get_Version()`
  - `Nullable<TimeSpan> get_GlobalTimeout()`
  - `ExecutionPriority get_DefaultPriority()`
  - *(+5 more methods)*
- **`IPipelineExecutionContext`**
  - `Guid get_ContextId()`
  - `IPipelineConfiguration get_Configuration()`
  - `CancellationToken get_CancellationToken()`
  - `IServiceProvider get_ServiceProvider()`
  - `IReadOnlyDictionary<String, Object> get_Properties()`
- **`IPipelineExecutionResult`1`**
  - `TOutput get_Result()`
  - `Boolean get_IsSuccess()`
  - `IReadOnlyList<Exception> get_Exceptions()`
  - `TimeSpan get_ExecutionTime()`
  - `IPipelineExecutionContext get_ExecutionContext()`
  - *(+2 more methods)*
- **`IStreamingExecutionContext`**
  - `StreamingConfiguration get_StreamingConfig()`

#### Classes

- **`LRUCachePolicy`**
- **`MemoryAllocationHints`**
- **`OptimizationConstraints`**
- **`OptimizationContext`**
- **`PerformanceGoals`**
- **`PerformanceWeights`**
- **`PipelineEvent`**
- **`PipelineExecutionCompletedEvent`**
- **`PipelineExecutionStartedEvent`**
- **`PipelineStageOptions`**
- **`ResourceAllocationPreferences`**
- **`RetryConfiguration`**
- **`StageExecutionCompletedEvent`**
- **`StageExecutionStartedEvent`**
- **`StreamingConfiguration`**
- **`TTLCachePolicy`**

#### Abstract Classes

- **`CachePolicy`**

#### Enums

- **`BackoffStrategy`**: Fixed, Linear, Exponential, ExponentialWithJitter
- **`BackpressureStrategy`**: Buffer, DropOldest, DropNewest, Block, Fail
- **`ErrorHandlingStrategy`**: StopOnFirstError, ContinueOnError, AutoRecover, Custom, Retry
- **`EventSeverity`**: Verbose, Information, Warning, Error, Critical
- **`MemoryAllocationStrategy`**: OnDemand, PreAllocated, Pooled, Adaptive, Optimal
- **`OptimizationStrategy`**: Conservative, Balanced, Aggressive, Adaptive, MemoryOptimal
- **`PipelineEventType`**: Started, Completed, Failed, StageStarted, StageCompleted
- **`PipelineState`**: Created, Ready, Executing, Paused, Completed
- **`ResourceSharingPolicy`**: Fair, PriorityBased, Exclusive, Adaptive
- **`StreamingErrorHandling`**: Skip, Stop, Retry, DeadLetter

### Namespace: `DotCompute.Abstractions.Pipelines.Enums`

#### Enums

- **`ComputeDeviceType`**: Unknown, CPU, CUDA, ROCm, Metal
- **`DataTransferType`**: HostToDevice, DeviceToHost, DeviceToDevice, PeerToPeer
- **`ErrorHandlingAction`**: Stop, Continue, Retry, Fallback, LogAndContinue
- **`ErrorHandlingStrategy`**: Continue, Retry, Skip, Abort, Fallback
- **`ExecutionPriority`**: Idle, Low, BelowNormal, Normal, AboveNormal
- **`MemoryHint`**: None, SequentialAccess, RandomAccess, ReadHeavy, WriteHeavy
- **`MemoryLockMode`**: None, SharedRead, ExclusiveWrite, ReadWrite, Optimistic
- **`MetricsExportFormat`**: Json, Xml, Csv, Text, Binary
- **`OptimizationType`**: None, KernelFusion, MemoryAccess, LoopOptimization, Parallelization
- **`PipelineErrorType`**: ExecutionError, ConfigurationError, MemoryError, ValidationError, CompilationError
- **`PipelineEvent`**: PipelineStarted, PipelineCompleted, PipelineFailed, PipelineCancelled, PipelinePaused
- **`PipelineEventType`**: Started, StageStarted, StageCompleted, StageFailed, Completed
- **`PipelineStageType`**: Computation, DataTransformation, MemoryTransfer, Synchronization, ConditionalBranch
- **`RetryStrategy`**: None, Immediate, FixedDelay, ExponentialBackoff, LinearBackoff

### Namespace: `DotCompute.Abstractions.Pipelines.Metrics`

#### Classes

- **`TimeSeriesMetric`**

#### Structs

- **`TimestampedValue`1`**

### Namespace: `DotCompute.Abstractions.Pipelines.Models`

#### Interfaces

- **`IOptimizationPass`**
  - `String get_Name()`
  - `OptimizationType get_OptimizationType()`
  - `Task<IKernelPipeline> ApplyAsync(IKernelPipeline pipeline, CancellationToken cancellationToken)`
- **`IOptimizationStrategy`**
  - `String get_Name()`
  - `OptimizationType get_SupportedOptimizations()`
  - `OptimizationType get_Type()`
  - `Boolean CanOptimize(IKernelPipeline pipeline)`
  - `Boolean CanApply(IKernelPipeline pipeline)`
  - *(+2 more methods)*

#### Classes

- **`AffinityRule`**
- **`BackendCompatibility`**
- **`CacheMetrics`**
- **`ComputeResourceMetrics`**
- **`DataLayoutPreferences`**
- **`DataTransferMetrics`**
- **`ErrorHandlingResult`**
  - `static ErrorHandlingResult CreateSuccess(ErrorHandlingStrategy strategy, ErrorHandlingAction action, Object result)`
  - `static ErrorHandlingResult Failure(Exception originalException, String errorMessage)`
  - `static ErrorHandlingResult Retry(Exception originalException, Int32 retryCount, Int32 maxRetries, Nullable<TimeSpan> retryDelay)`
- **`FusionCriteria`**
- **`LoopOptimizations`**
- **`MemoryConstraints`**
- **`MemoryPoolOptions`**
  - `static MemoryPoolOptions Performance()`
  - `static MemoryPoolOptions MemoryEfficient()`
  - `static MemoryPoolOptions Debug()`
- **`MemoryUsageMetrics`**
- **`OptimizationCacheSettings`**
- **`OptimizationImpactEstimate`**
- **`OptimizationMetrics`**
- **`OptimizationProfilingConfig`**
- **`OptimizationValidationResult`**
- **`ParallelismGoals`**
- **`ParallelizationMetrics`**
- **`ParallelKernelConfig`**
- **`PerformanceEstimate`**
- **`PerformanceTargets`**
- **`PipelineAnalysisResult`**
- **`PipelineExecutionMetrics`**
- **`PipelineOptimizationSettings`**
- **`QualityMetrics`**
- **`ResourceEstimate`**
- **`ResourceRequirements`**
  - `static ResourceRequirements Minimal()`
  - `static ResourceRequirements GpuOptimized()`
- **`StageComputeMetrics`**
- **`StageDataTransferMetrics`**
- **`StageExecutionMetrics`**
- **`StageMemoryMetrics`**
- **`StageOptimizationMetrics`**
- **`StagePerformanceMetrics`**
- **`StageQualityMetrics`**
- **`StageSynchronizationMetrics`**
- **`StageValidationResult`**
  - `static StageValidationResult Success(String stageName)`
  - `static StageValidationResult Failure(String stageName, String errorMessage, String errorCode)`
- **`TransferTypeMetrics`**
- **`ValidationResult`**
  - `static ValidationResult Success(String message)`
  - `static ValidationResult Failure(String message, ErrorSeverity severity)`
- **`ValidationSuggestion`**

#### Enums

- **`AdaptationPolicy`**: Conservative, Aggressive, Balanced, Custom
- **`AllocationFailurePolicy`**: ThrowException, GrowPool, CleanupAndRetry, FallbackToSystem, ReturnNull
- **`BackendFallbackStrategy`**: None, NextAvailable, Auto, Manual
- **`CompatibilityLevel`**: None, Basic, Good, Full
- **`DataTransferType`**: HostToDevice, DeviceToHost, DeviceToDevice, HostToHost, PeerToPeer
- **`ErrorCategory`**: Unknown, Hardware, Software, Network, Memory
- **`ErrorHandlingAction`**: None, Failed, Retry, Skip, Abort
- **`ErrorImpact`**: None, Minor, Moderate, Major, Critical
- **`ImplementationEffort`**: Low, Medium, High, VeryHigh
- **`LoadBalancingStrategy`**: RoundRobin, LeastLoaded, Random, Weighted, Adaptive
- **`MemorySharingMode`**: None, ReadOnly, Shared, Exclusive
- **`OptimizationFailureStrategy`**: FallbackToOriginal, RetryWithLowerLevel, ThrowException, UseBestPartial
- **`OptimizationLevel`**: None, Basic, Balanced, Aggressive, Maximum
- **`ParallelErrorStrategy`**: FailFast, ContinueOnError, RetryFailures, Aggregate
- **`PoolAllocationStrategy`**: FirstFit, BestFit, WorstFit, NextFit, BuddySystem
- **`PoolDeallocationStrategy`**: Immediate, Deferred, Batched, ReferenceCounted
- **`ResourceAllocationStrategy`**: FirstFit, BestFit, Proportional, Dynamic
- **`SuggestionCategory`**: Performance, Memory, Quality, Security, Maintainability
- **`SuggestionPriority`**: Low, Medium, High, Critical

### Namespace: `DotCompute.Abstractions.Pipelines.Results`

#### Classes

- **`AggregatedProfilingResults`**
- **`DeviceMemoryMetrics`**
- **`ExecutionTimelineEvent`**
- **`KernelChainExecutionResult`**
  - `static KernelChainExecutionResult CreateSuccess(Object result, TimeSpan executionTime, IReadOnlyList<KernelStepMetrics> stepMetrics, String backend, KernelChainMemoryMetrics memoryMetrics, String chainId, IReadOnlyDictionary<String, Object> metadata)`
  - `static KernelChainExecutionResult CreateFailure(IReadOnlyList<Exception> errors, TimeSpan executionTime, IReadOnlyList<KernelStepMetrics> stepMetrics, String backend, Object partialResult, String chainId, IReadOnlyDictionary<String, Object> metadata)`
  - `static KernelChainExecutionResult CreateFailure(Exception exception, TimeSpan executionTime, IReadOnlyList<KernelStepMetrics> stepMetrics, String backend, Object partialResult, String chainId, IReadOnlyDictionary<String, Object> metadata)`
- **`KernelChainMemoryMetrics`**
  - `static KernelChainMemoryMetrics Create(Int64 peakMemoryUsage, Int64 totalMemoryAllocated, Int64 totalMemoryFreed, Int32 garbageCollections, Boolean memoryPoolingUsed, Nullable<Int64> initialMemoryUsage, Nullable<Int64> finalMemoryUsage, Nullable<Int32> allocationCount, IReadOnlyList<String> optimizationRecommendations)`
  - `static KernelChainMemoryMetrics CreateDetailed(Int64 peakMemoryUsage, Int64 totalMemoryAllocated, Int64 totalMemoryFreed, Int32 garbageCollections, Boolean memoryPoolingUsed, IReadOnlyDictionary<String, DeviceMemoryMetrics> deviceMetrics, IReadOnlyList<StepMemoryUsage> stepMemoryUsages, IReadOnlyList<String> optimizationRecommendations)`
- **`KernelChainValidationResult`**
  - `static KernelChainValidationResult Success()`
  - `static KernelChainValidationResult SuccessWithWarnings(IEnumerable<String> warnings)`
  - `static KernelChainValidationResult Failure(IEnumerable<String> errors)`
- **`KernelStepMetrics`**
  - `static KernelStepMetrics CreateSuccess(String kernelName, Int32 stepIndex, TimeSpan executionTime, String backend, Int64 memoryUsed, Boolean wasCached, String stepId, String cacheKey, IReadOnlyDictionary<String, Object> metadata)`
  - `static KernelStepMetrics CreateFailure(String kernelName, Int32 stepIndex, TimeSpan executionTime, String backend, Exception error, Int64 memoryUsed, String stepId, IReadOnlyDictionary<String, Object> metadata)`
- **`MemoryUsageStats`**
- **`PerformanceRecommendation`**
- **`PipelineExecutionResult`**
  - `static PipelineExecutionResult CreateSuccess(String pipelineId, String pipelineName, IReadOnlyDictionary<String, Object> outputs, PipelineExecutionMetrics metrics, IReadOnlyList<StageExecutionResult> stageResults, Nullable<DateTime> startTime, Nullable<DateTime> endTime, IReadOnlyList<String> warnings, IReadOnlyList<String> optimizationRecommendations, PipelineResourceUsage resourceUsage, IReadOnlyDictionary<String, Object> metadata)`
  - `static PipelineExecutionResult CreateFailure(String pipelineId, String pipelineName, IReadOnlyList<PipelineError> errors, PipelineExecutionMetrics metrics, IReadOnlyList<StageExecutionResult> stageResults, IReadOnlyDictionary<String, Object> partialOutputs, Nullable<DateTime> startTime, Nullable<DateTime> endTime, IReadOnlyList<String> warnings, PipelineResourceUsage resourceUsage, IReadOnlyDictionary<String, Object> metadata)`
  - `static PipelineExecutionResult CreateFailure(String pipelineId, String pipelineName, Exception exception, PipelineExecutionMetrics metrics, IReadOnlyList<StageExecutionResult> stageResults, IReadOnlyDictionary<String, Object> partialOutputs, Nullable<DateTime> startTime, Nullable<DateTime> endTime, IReadOnlyDictionary<String, Object> metadata)`
- **`PipelineResourceUsage`**
- **`ProfilingResults`**
- **`StepMemoryUsage`**

### Namespace: `DotCompute.Abstractions.Pipelines.Statistics`

#### Classes

- **`KernelExecutionStats`**

### Namespace: `DotCompute.Abstractions.RingKernels`

#### Interfaces

- **`IMessageQueue`1`**
  - `Int32 get_Capacity()`
  - `Boolean get_IsEmpty()`
  - `Boolean get_IsFull()`
  - `Int32 get_Count()`
  - `IUnifiedMemoryBuffer GetBuffer()`
  - *(+9 more methods)*
- **`IRingKernelRuntime`**
  - `Task LaunchAsync(String kernelId, Int32 gridSize, Int32 blockSize, CancellationToken cancellationToken)`
  - `Task ActivateAsync(String kernelId, CancellationToken cancellationToken)`
  - `Task DeactivateAsync(String kernelId, CancellationToken cancellationToken)`
  - `Task TerminateAsync(String kernelId, CancellationToken cancellationToken)`
  - `Task SendMessageAsync(String kernelId, KernelMessage<T> message, CancellationToken cancellationToken)`
  - *(+5 more methods)*

#### Enums

- **`KernelBackends`**: None, CPU, CUDA, OpenCL, Metal
- **`MessagePassingStrategy`**: SharedMemory, AtomicQueue, P2P, NCCL
- **`MessageType`**: Data, Control, Terminate, Activate, Deactivate
- **`RingKernelDomain`**: General, GraphAnalytics, SpatialSimulation, ActorModel
- **`RingKernelMode`**: Persistent, EventDriven

#### Structs

- **`KernelMessage`1`**
- **`MessageQueueStatistics`**
- **`RingKernelMetrics`**
- **`RingKernelStatus`**

### Namespace: `DotCompute.Abstractions.Security`

#### Classes

- **`MalwareScanningOptions`**
- **`SecurityEvaluationContext`**

#### Enums

- **`SecurityLevel`**: None, Basic, Low, Standard, Medium
- **`SecurityOperation`**: FileRead, FileWrite, NetworkAccess, ReflectionAccess, UnmanagedCode
- **`SecurityZone`**: Unknown, LocalMachine, LocalIntranet, TrustedSites, Internet
- **`ThreatLevel`**: None, Low, Medium, High, Critical
- **`TrustLevel`**: None, Unknown, Untrusted, Low, PartiallyTrusted

### Namespace: `DotCompute.Abstractions.Statistics`

#### Classes

- **`AcceleratorCompilationStats`**

### Namespace: `DotCompute.Abstractions.Telemetry`

#### Classes

- **`TelemetryConfiguration`**
- **`TelemetryExportTarget`**

#### Enums

- **`TelemetrySeverity`**: Verbose, Debug, Information, Warning, Error

### Namespace: `DotCompute.Abstractions.Telemetry.Context`

#### Classes

- **`TraceContext`**

### Namespace: `DotCompute.Abstractions.Telemetry.Options`

#### Classes

- **`DistributedTracingOptions`**
- **`LogBufferOptions`**
- **`PerformanceProfilerOptions`**
- **`StructuredLoggingOptions`**
- **`TelemetryOptions`**

#### Enums

- **`LogLevel`**: Trace, Debug, Information, Warning, Error

### Namespace: `DotCompute.Abstractions.Telemetry.Profiles`

#### Classes

- **`AllocationPatternData`**
- **`CpuProfileData`**
- **`GpuProfileData`**
- **`MemoryProfileData`**
- **`PerformanceProfile`**

#### Structs

- **`TimestampedValue`1`**

### Namespace: `DotCompute.Abstractions.Telemetry.Providers`

#### Abstract Classes

- **`DistributedTracer`**
- **`PerformanceProfiler`**
- **`StructuredLogger`**
- **`TelemetryProvider`**

### Namespace: `DotCompute.Abstractions.Telemetry.Traces`

#### Classes

- **`SpanData`**
- **`TraceData`**

### Namespace: `DotCompute.Abstractions.Telemetry.Types`

#### Classes

- **`ProfileOptions`**
- **`SystemHealthMetrics`**
- **`TelemetryKernelPerformanceMetrics`**

#### Enums

- **`TelemetryExportFormat`**: Prometheus, OpenTelemetry, Json
- **`TraceStatus`**: Ok, Error, Cancelled

### Namespace: `DotCompute.Abstractions.TypeConverters`

#### Classes

- **`Dim3TypeConverter`**

### Namespace: `DotCompute.Abstractions.Types`

#### Classes

- **`CoalescingIssue`**
- **`MemoryAccessInfo`**
- **`PerformanceTrend`**
- **`ThroughputMetrics`**
- **`TileAnalysis`**

#### Enums

- **`AccessOrder`**: RowMajor, ColumnMajor, Tiled
- **`BottleneckType`**: None, CPU, GPU, Memory, MemoryBandwidth
- **`CacheConfig`**: PreferNone, PreferShared, PreferL1, PreferEqual
- **`ErrorSeverity`**: None, Info, Warning, Error, Critical
- **`ExecutionStatus`**: Pending, Running, Completed, Failed, Cancelled
- **`ExecutionStrategyType`**: Sequential, Parallel, DataParallel, ModelParallel, PipelineParallel
- **`FloatingPointMode`**: Strict, Relaxed, Fast, Default
- **`IssueSeverity`**: Low, Medium, High, Critical
- **`IssueType`**: None, Misalignment, StridedAccess, SmallElements, RandomAccess
- **`KernelType`**: Generic, ElementWise, VectorAdd, VectorMultiply, VectorScale
- **`MemoryAccessPattern`**: Sequential, Strided, Coalesced, Random, Mixed
- **`MemoryOptimizationLevel`**: None, Conservative, Balanced, Aggressive
- **`MemoryTransferType`**: HostToDevice, DeviceToHost, DeviceToDevice, HostToHost, UnifiedMemory
- **`OptimizationHint`**: None, Balanced, MemoryBound, ComputeBound, Latency
- **`OptimizationLevel`**: None, O1, O2, Default, O3
- **`SynchronizationMode`**: WaitAll, WaitAny, FireAndForget, Sequential, Barrier
- **`TrendDirection`**: Unknown, None, Stable, Improving, Degrading
- **`WorkloadType`**: Compute, Memory, IO, Mixed, Graphics
- **`WorkStatus`**: Pending, Queued, Executing, Completed, Failed

#### Structs

- **`Dim3`**
- **`MemoryAccessMetrics`**

### Namespace: `DotCompute.Abstractions.Utilities`

### Namespace: `DotCompute.Abstractions.Validation`

#### Classes

- **`AcceleratorPerformanceMetrics`**
- **`AcceleratorValidationException`**
- **`AcceleratorValidationResult`**
  - `static AcceleratorValidationResult Success(AcceleratorType acceleratorType, Int32 deviceIndex, IReadOnlyList<String> supportedFeatures, AcceleratorPerformanceMetrics performanceMetrics)`
  - `static AcceleratorValidationResult Success(IReadOnlyList<String> supportedFeatures, AcceleratorPerformanceMetrics performanceMetrics)`
  - `static AcceleratorValidationResult Failure(IEnumerable<String> errors, IEnumerable<String> warnings, AcceleratorType acceleratorType, Int32 deviceIndex)`
- **`KernelValidationResult`**
- **`ResourceUsageEstimate`**
- **`ResultComparison`**
- **`UnifiedValidationResult`**
  - `static UnifiedValidationResult Success()`
  - `static UnifiedValidationResult Failure(String errorMessage, String code)`
  - `static UnifiedValidationResult FromException(Exception exception)`
- **`ValidationException`**
- **`ValidationIssue`**
  - `static ValidationIssue Error(String code, String message)`
  - `static ValidationIssue Warning(String code, String message)`
- **`ValidationWarning`**

#### Enums

- **`ValidationSeverity`**: Info, Warning, Error, Critical
- **`WarningSeverity`**: Low, Medium, High

### Namespace: `DotCompute.Backends.CPU.Utilities`

#### Structs

- **`Point3D`**

### Namespace: `DotCompute.Backends.CUDA.Advanced.Types`

#### Enums

- **`DataType`**: FP8_E4M3, FP8_E5M2, FP16, BF16, TF32
- **`SharedMemoryCarveout`**: Default, Prefer100KB, PreferL1
- **`TensorCoreArchitecture`**: None, Volta, Turing, Ampere, Ada

### Namespace: `DotCompute.Backends.CUDA.ErrorHandling.Types`

#### Enums

- **`ErrorCategory`**: Memory, Device, Kernel, Stream, Api

### Namespace: `DotCompute.Backends.CUDA.Execution.Types`

#### Enums

- **`StreamCreationFlags`**: None, NonBlocking, DisableTiming, GraphCapture, PriorityScheduling
- **`StreamPriority`**: Normal, High, Highest, Lowest, Low
- **`WarpSchedulingMode`**: Default, Persistent, Dynamic

#### Structs

- **`EventId`**

### Namespace: `DotCompute.Backends.CUDA.Optimization.Types`

#### Structs

- **`Dim2`**

### Namespace: `DotCompute.Backends.CUDA.Profiling.Types`

#### Classes

- **`TransferTypeStats`**

### Namespace: `DotCompute.Backends.CUDA.Types`

#### Enums

- **`ExportFormat`**: Json, Markdown, Html, Text, Xml
- **`ManagedMemoryOptions`**: None, PreferDevice, PreferHost, ReadMostly, SingleDevice
- **`MemoryResidence`**: Unknown, Host, Device, Split, Migrating
- **`ValidationStatus`**: NotSet, Passed, Failed, Warning, Skipped

### Namespace: `DotCompute.Core.Kernels`

#### Interfaces

- **`IAnalyzableGeneratedKernel`**
  - `IExpressionAnalysisResult get_Analysis()`
  - `IReadOnlyList<String> get_Optimizations()`
  - `IComplexityMetrics get_ComplexityMetrics()`
- **`ICompiledKernel`**
  - `String get_Name()`
  - `Boolean get_IsValid()`
- **`IExecutableGeneratedKernel`**
  - `ICompiledKernel get_CompiledKernel()`
  - `Boolean get_IsCompiled()`
  - `IReadOnlyList<IKernelParameter> get_Parameters()`
  - `Task ExecuteAsync(Object[] parameters)`
- **`IExpressionAnalysisResult`**
  - `DateTimeOffset get_AnalysisTimestamp()`
  - `IComplexityMetrics get_ComplexityMetrics()`
- **`IFullGeneratedKernel`**
  - `IGpuMemoryManager get_MemoryManager()`
  - `DateTimeOffset get_CompiledAt()`
  - `Version get_Version()`
- **`IGeneratedKernel`**
  - `String get_Name()`
  - `String get_SourceCode()`
  - `String get_Language()`
  - `String get_TargetBackend()`
  - `String get_EntryPoint()`
  - *(+1 more methods)*
- **`IGpuMemoryManager`**
  - `Int64 get_TotalMemory()`
  - `Int64 get_AvailableMemory()`
- **`IKernelParameter`**
  - `String get_Name()`
  - `Type get_Type()`
  - `Boolean get_IsPointer()`
  - `Boolean get_IsInput()`
  - `Boolean get_IsOutput()`

### Namespace: `DotCompute.Core.Types`

#### Enums

- **`QueuePriority`**: Low, Normal, High

#### Structs

- **`Half`**
- **`SystemPerformanceSnapshot`**

## DotCompute.Core
**Assembly**: `DotCompute.Core`
**Version**: 0.2.0.0

❌ Error loading assembly: Could not load file or assembly 'Microsoft.Extensions.Logging.Abstractions, Version=9.0.0.0, Culture=neutral, PublicKeyToken=adb9793829ddae60'. The system cannot find the file specified.


## DotCompute.Runtime
**Assembly**: `DotCompute.Runtime`
**Version**: 0.2.0.0

❌ Error loading assembly: Could not load file or assembly 'Microsoft.Extensions.Hosting.Abstractions, Version=9.0.0.0, Culture=neutral, PublicKeyToken=adb9793829ddae60'. The system cannot find the file specified.


## DotCompute.Memory
**Assembly**: `DotCompute.Memory`
**Version**: 0.2.0.0

❌ Error loading assembly: Could not load file or assembly 'Microsoft.Extensions.ObjectPool, Version=9.0.0.0, Culture=neutral, PublicKeyToken=adb9793829ddae60'. The system cannot find the file specified.


## DotCompute.Backends.CUDA
**Assembly**: `DotCompute.Backends.CUDA`
**Version**: 0.2.0.0

❌ Error loading assembly: Could not load file or assembly 'DotCompute.Plugins, Version=0.2.0.0, Culture=neutral, PublicKeyToken=null'. The system cannot find the file specified.


## DotCompute.Backends.OpenCL
**Assembly**: `DotCompute.Backends.OpenCL`
**Version**: 0.2.0.0

❌ Error loading assembly: Could not load file or assembly 'DotCompute.Plugins, Version=0.2.0.0, Culture=neutral, PublicKeyToken=null'. The system cannot find the file specified.


## DotCompute.Plugins
**Assembly**: `DotCompute.Plugins`
**Version**: 0.2.0.0

❌ Error loading assembly: Could not load file or assembly 'Microsoft.Extensions.DependencyInjection.Abstractions, Version=9.0.0.0, Culture=neutral, PublicKeyToken=adb9793829ddae60'. The system cannot find the file specified.


---
*Analysis completed*

