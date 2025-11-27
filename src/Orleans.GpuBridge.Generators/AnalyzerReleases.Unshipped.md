### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
GPUGEN001 | GpuGeneration | Error | Handler method must return Task or Task<T>
GPUGEN002 | GpuGeneration | Error | Parameter type is not blittable (cannot be used in GPU kernel)
GPUGEN003 | GpuGeneration | Error | Message payload exceeds 228 bytes maximum
GPUGEN004 | GpuGeneration | Error | Reference type in payload (must be value type)
GPUGEN005 | GpuGeneration | Warning | Chunking required for large payload but not enabled
GPUGEN006 | GpuGeneration | Info | Struct layout could be more efficient
GPUGEN007 | GpuGeneration | Error | Missing grain interface inheritance
GPUGEN008 | GpuGeneration | Warning | No GPU handlers defined
GPUGEN010 | GpuGeneration | Error | Non-blittable state type
GPUGEN020 | GpuGeneration | Error | K2K target is not a GPU actor
GPUGEN021 | GpuGeneration | Error | K2K message size mismatch
GPUGEN030 | GpuGeneration | Error | Missing temporal dependency
GPUGEN099 | GpuGeneration | Error | Internal generator error
