// Forward references to maintain backward compatibility
// Types have been split into separate files under Memory/ folder structure
global using IMemoryAllocator = Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators.IMemoryAllocator;
global using IDeviceMemory = Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces.IDeviceMemory;
global using IPinnedMemory = Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces.IPinnedMemory;
global using IUnifiedMemory = Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces.IUnifiedMemory;
global using MemoryAllocationOptions = Orleans.GpuBridge.Abstractions.Providers.Memory.Options.MemoryAllocationOptions;
global using UnifiedMemoryOptions = Orleans.GpuBridge.Abstractions.Providers.Memory.Options.UnifiedMemoryOptions;
global using MemoryPoolStatistics = Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics.MemoryPoolStatistics;
global using MemoryType = Orleans.GpuBridge.Abstractions.Providers.Memory.Enums.MemoryType;
global using MemoryAdvice = Orleans.GpuBridge.Abstractions.Providers.Memory.Enums.MemoryAdvice;

// This file is kept for backward compatibility
// All memory-related types have been moved to appropriate subfolders:
// - Memory/Allocators/IMemoryAllocator.cs
// - Memory/Interfaces/IDeviceMemory.cs
// - Memory/Interfaces/IDeviceMemoryGeneric.cs
// - Memory/Interfaces/IPinnedMemory.cs
// - Memory/Interfaces/IUnifiedMemory.cs
// - Memory/Options/MemoryAllocationOptions.cs
// - Memory/Options/UnifiedMemoryOptions.cs
// - Memory/Statistics/MemoryPoolStatistics.cs
// - Memory/Enums/MemoryType.cs
// - Memory/Enums/MemoryAdvice.cs