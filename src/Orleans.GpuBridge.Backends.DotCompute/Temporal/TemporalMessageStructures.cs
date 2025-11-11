// This file has been moved to Orleans.GpuBridge.Abstractions.Temporal
// to avoid circular dependencies between Runtime and Backends projects.
//
// These structures represent shared contracts used by both layers:
// - Runtime: RingKernelManager, GpuClockCalibrator
// - Backends: GPU kernel implementations
// - Grains: GpuResidentActorGrain
//
// Please use: using Orleans.GpuBridge.Abstractions.Temporal;
