// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Orleans;

namespace Orleans.GpuBridge.Abstractions.Placement;

/// <summary>
/// Grain interface for managing persistent affinity group state.
/// </summary>
/// <remarks>
/// <para>
/// Affinity groups enable co-location of related grains on the same GPU device.
/// This grain persists the affinity group state across silo restarts, ensuring
/// that grains are re-placed on appropriate devices after recovery.
/// </para>
/// <para>
/// <strong>Usage:</strong>
/// </para>
/// <code>
/// // Get or create an affinity group
/// var affinityGrain = grainFactory.GetGrain&lt;IAffinityGroupGrain&gt;("my-affinity-group");
/// await affinityGrain.RegisterMemberAsync("grain-123", deviceIndex: 0);
///
/// // Query preferred device for new grains in the group
/// var preferredDevice = await affinityGrain.GetPreferredDeviceAsync();
/// </code>
/// </remarks>
public interface IAffinityGroupGrain : IGrainWithStringKey
{
    /// <summary>
    /// Registers a grain as a member of this affinity group.
    /// </summary>
    /// <param name="grainId">Grain identity to register.</param>
    /// <param name="deviceIndex">GPU device index where the grain is placed.</param>
    /// <param name="grainType">Optional grain type name.</param>
    /// <returns>True if registration succeeded.</returns>
    Task<bool> RegisterMemberAsync(string grainId, int deviceIndex, string? grainType = null);

    /// <summary>
    /// Unregisters a grain from this affinity group.
    /// </summary>
    /// <param name="grainId">Grain identity to unregister.</param>
    /// <returns>True if unregistration succeeded.</returns>
    Task<bool> UnregisterMemberAsync(string grainId);

    /// <summary>
    /// Gets the preferred GPU device for new grains in this affinity group.
    /// </summary>
    /// <returns>Preferred device index, or -1 if no preference.</returns>
    Task<int> GetPreferredDeviceAsync();

    /// <summary>
    /// Gets all members currently registered in this affinity group.
    /// </summary>
    /// <returns>Dictionary of grain IDs to their device index.</returns>
    Task<IReadOnlyDictionary<string, AffinityMemberInfo>> GetMembersAsync();

    /// <summary>
    /// Gets the current state of the affinity group.
    /// </summary>
    /// <returns>Affinity group state.</returns>
    Task<AffinityGroupState> GetStateAsync();

    /// <summary>
    /// Updates the preferred device for this affinity group.
    /// </summary>
    /// <param name="deviceIndex">New preferred device index.</param>
    Task SetPreferredDeviceAsync(int deviceIndex);

    /// <summary>
    /// Migrates all members to a new device.
    /// </summary>
    /// <param name="targetDeviceIndex">Target device index.</param>
    /// <returns>Number of members updated.</returns>
    Task<int> MigrateAllAsync(int targetDeviceIndex);

    /// <summary>
    /// Gets metrics for this affinity group.
    /// </summary>
    /// <returns>Affinity group metrics.</returns>
    Task<AffinityGroupMetrics> GetMetricsAsync();
}

/// <summary>
/// Information about an affinity group member.
/// </summary>
[GenerateSerializer]
public sealed class AffinityMemberInfo
{
    /// <summary>
    /// Grain identity.
    /// </summary>
    [Id(0)]
    public required string GrainId { get; init; }

    /// <summary>
    /// GPU device index where the grain is placed.
    /// </summary>
    [Id(1)]
    public required int DeviceIndex { get; set; }

    /// <summary>
    /// Grain type name (optional).
    /// </summary>
    [Id(2)]
    public string? GrainType { get; set; }

    /// <summary>
    /// When the grain joined the affinity group (nanoseconds since Unix epoch).
    /// </summary>
    [Id(3)]
    public long JoinedAtNanos { get; set; }

    /// <summary>
    /// Last activity timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(4)]
    public long LastActivityNanos { get; set; }
}

/// <summary>
/// Persistent state for an affinity group grain.
/// </summary>
[GenerateSerializer]
public sealed class AffinityGroupState
{
    /// <summary>
    /// Affinity group name/identifier.
    /// </summary>
    [Id(0)]
    public string GroupId { get; set; } = string.Empty;

    /// <summary>
    /// Preferred GPU device index for this group.
    /// </summary>
    [Id(1)]
    public int PreferredDeviceIndex { get; set; } = -1;

    /// <summary>
    /// Members of this affinity group.
    /// </summary>
    [Id(2)]
    public Dictionary<string, AffinityMemberInfo> Members { get; set; } = new();

    /// <summary>
    /// When the affinity group was created (nanoseconds since Unix epoch).
    /// </summary>
    [Id(3)]
    public long CreatedAtNanos { get; set; }

    /// <summary>
    /// Last modification timestamp (nanoseconds since Unix epoch).
    /// </summary>
    [Id(4)]
    public long ModifiedAtNanos { get; set; }

    /// <summary>
    /// Version number for optimistic concurrency.
    /// </summary>
    [Id(5)]
    public long Version { get; set; }

    /// <summary>
    /// Whether the affinity group has been initialized.
    /// </summary>
    [Id(6)]
    public bool IsInitialized { get; set; }

    /// <summary>
    /// Maximum number of members allowed (0 = unlimited).
    /// </summary>
    [Id(7)]
    public int MaxMembers { get; set; } = 0;

    /// <summary>
    /// Description of the affinity group (optional).
    /// </summary>
    [Id(8)]
    public string? Description { get; set; }

    /// <summary>
    /// Custom properties for the affinity group.
    /// </summary>
    [Id(9)]
    public Dictionary<string, object> Properties { get; set; } = new();

    /// <summary>
    /// Metrics tracking state.
    /// </summary>
    [Id(10)]
    public AffinityGroupMetricsState Metrics { get; set; } = new();
}

/// <summary>
/// Persistent metrics state for affinity groups.
/// </summary>
[GenerateSerializer]
public sealed class AffinityGroupMetricsState
{
    /// <summary>
    /// Total members registered since creation.
    /// </summary>
    [Id(0)]
    public long TotalRegistrations { get; set; }

    /// <summary>
    /// Total members unregistered since creation.
    /// </summary>
    [Id(1)]
    public long TotalUnregistrations { get; set; }

    /// <summary>
    /// Total migrations performed.
    /// </summary>
    [Id(2)]
    public long TotalMigrations { get; set; }

    /// <summary>
    /// Number of device changes.
    /// </summary>
    [Id(3)]
    public long DeviceChanges { get; set; }
}

/// <summary>
/// Runtime metrics for an affinity group.
/// </summary>
public readonly record struct AffinityGroupMetrics
{
    /// <summary>
    /// Number of current members.
    /// </summary>
    public required int MemberCount { get; init; }

    /// <summary>
    /// Total registrations since creation.
    /// </summary>
    public required long TotalRegistrations { get; init; }

    /// <summary>
    /// Total unregistrations since creation.
    /// </summary>
    public required long TotalUnregistrations { get; init; }

    /// <summary>
    /// Total migrations performed.
    /// </summary>
    public required long TotalMigrations { get; init; }

    /// <summary>
    /// Preferred device index.
    /// </summary>
    public required int PreferredDeviceIndex { get; init; }

    /// <summary>
    /// Distribution of members across devices.
    /// </summary>
    public required IReadOnlyDictionary<int, int> DeviceDistribution { get; init; }
}
