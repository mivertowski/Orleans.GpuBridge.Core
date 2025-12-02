// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Runtime;
using Orleans.GpuBridge.Abstractions.Placement;

namespace Orleans.GpuBridge.Grains.Placement;

/// <summary>
/// Orleans grain implementation for managing persistent affinity group state.
/// </summary>
/// <remarks>
/// <para>
/// This grain provides cluster-wide persistent storage for GPU affinity groups,
/// enabling grains to be co-located on the same GPU device across silo restarts.
/// </para>
/// <para>
/// <strong>Persistence:</strong>
/// State is persisted using Orleans grain storage with the "AffinityStore" or "Default" provider.
/// </para>
/// </remarks>
public sealed class AffinityGroupGrain : Grain, IAffinityGroupGrain
{
    private readonly ILogger<AffinityGroupGrain> _logger;
    private readonly IPersistentState<AffinityGroupState> _state;
    private string _groupId = string.Empty;

    /// <summary>
    /// Creates a new affinity group grain with persistence support.
    /// </summary>
    public AffinityGroupGrain(
        ILogger<AffinityGroupGrain> logger,
        [PersistentState("affinityGroup", "AffinityStore")]
        IPersistentState<AffinityGroupState> state)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _state = state ?? throw new ArgumentNullException(nameof(state));
    }

    /// <inheritdoc />
    public override Task OnActivateAsync(CancellationToken cancellationToken)
    {
        _groupId = this.GetPrimaryKeyString();

        if (!_state.State.IsInitialized)
        {
            // Initialize new affinity group
            var now = GetNanoseconds();
            _state.State.GroupId = _groupId;
            _state.State.CreatedAtNanos = now;
            _state.State.ModifiedAtNanos = now;
            _state.State.IsInitialized = true;
        }

        _logger.LogInformation(
            "Affinity group {GroupId} activated (IsNew={IsNew}, Members={MemberCount})",
            _groupId,
            !_state.State.IsInitialized,
            _state.State.Members.Count);

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override async Task OnDeactivateAsync(DeactivationReason reason, CancellationToken cancellationToken)
    {
        // Ensure state is persisted on deactivation
        if (_state.State.IsInitialized)
        {
            await _state.WriteStateAsync();
        }

        _logger.LogInformation(
            "Affinity group {GroupId} deactivating (Reason={Reason})",
            _groupId,
            reason);
    }

    /// <inheritdoc />
    public async Task<bool> RegisterMemberAsync(string grainId, int deviceIndex, string? grainType = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(grainId);

        if (_state.State.MaxMembers > 0 && _state.State.Members.Count >= _state.State.MaxMembers)
        {
            _logger.LogWarning(
                "Cannot register grain {GrainId} in affinity group {GroupId}: max members ({Max}) reached",
                grainId,
                _groupId,
                _state.State.MaxMembers);
            return false;
        }

        var now = GetNanoseconds();

        if (_state.State.Members.TryGetValue(grainId, out var existingMember))
        {
            // Update existing member
            if (existingMember.DeviceIndex != deviceIndex)
            {
                existingMember.DeviceIndex = deviceIndex;
                _state.State.Metrics.DeviceChanges++;
            }
            existingMember.LastActivityNanos = now;
            existingMember.GrainType = grainType ?? existingMember.GrainType;
        }
        else
        {
            // Add new member
            _state.State.Members[grainId] = new AffinityMemberInfo
            {
                GrainId = grainId,
                DeviceIndex = deviceIndex,
                GrainType = grainType,
                JoinedAtNanos = now,
                LastActivityNanos = now
            };
            _state.State.Metrics.TotalRegistrations++;
        }

        // Update preferred device if not set
        if (_state.State.PreferredDeviceIndex < 0)
        {
            _state.State.PreferredDeviceIndex = deviceIndex;
        }

        _state.State.ModifiedAtNanos = now;
        _state.State.Version++;

        await _state.WriteStateAsync();

        _logger.LogDebug(
            "Registered grain {GrainId} in affinity group {GroupId} on device {DeviceIndex}",
            grainId,
            _groupId,
            deviceIndex);

        return true;
    }

    /// <inheritdoc />
    public async Task<bool> UnregisterMemberAsync(string grainId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(grainId);

        if (!_state.State.Members.Remove(grainId))
        {
            _logger.LogDebug(
                "Grain {GrainId} not found in affinity group {GroupId}",
                grainId,
                _groupId);
            return false;
        }

        _state.State.Metrics.TotalUnregistrations++;
        _state.State.ModifiedAtNanos = GetNanoseconds();
        _state.State.Version++;

        await _state.WriteStateAsync();

        _logger.LogDebug(
            "Unregistered grain {GrainId} from affinity group {GroupId}",
            grainId,
            _groupId);

        return true;
    }

    /// <inheritdoc />
    public Task<int> GetPreferredDeviceAsync()
    {
        // If we have a preferred device set, use it
        if (_state.State.PreferredDeviceIndex >= 0)
        {
            return Task.FromResult(_state.State.PreferredDeviceIndex);
        }

        // Otherwise, find the most common device among members
        if (_state.State.Members.Count == 0)
        {
            return Task.FromResult(-1);
        }

        var deviceCounts = _state.State.Members.Values
            .GroupBy(m => m.DeviceIndex)
            .OrderByDescending(g => g.Count())
            .FirstOrDefault();

        return Task.FromResult(deviceCounts?.Key ?? -1);
    }

    /// <inheritdoc />
    public Task<IReadOnlyDictionary<string, AffinityMemberInfo>> GetMembersAsync()
    {
        return Task.FromResult<IReadOnlyDictionary<string, AffinityMemberInfo>>(
            _state.State.Members);
    }

    /// <inheritdoc />
    public Task<AffinityGroupState> GetStateAsync()
    {
        return Task.FromResult(_state.State);
    }

    /// <inheritdoc />
    public async Task SetPreferredDeviceAsync(int deviceIndex)
    {
        if (_state.State.PreferredDeviceIndex != deviceIndex)
        {
            _state.State.PreferredDeviceIndex = deviceIndex;
            _state.State.ModifiedAtNanos = GetNanoseconds();
            _state.State.Version++;
            _state.State.Metrics.DeviceChanges++;

            await _state.WriteStateAsync();

            _logger.LogInformation(
                "Updated preferred device for affinity group {GroupId} to {DeviceIndex}",
                _groupId,
                deviceIndex);
        }
    }

    /// <inheritdoc />
    public async Task<int> MigrateAllAsync(int targetDeviceIndex)
    {
        var now = GetNanoseconds();
        int migratedCount = 0;

        foreach (var member in _state.State.Members.Values)
        {
            if (member.DeviceIndex != targetDeviceIndex)
            {
                member.DeviceIndex = targetDeviceIndex;
                member.LastActivityNanos = now;
                migratedCount++;
            }
        }

        if (migratedCount > 0)
        {
            _state.State.PreferredDeviceIndex = targetDeviceIndex;
            _state.State.ModifiedAtNanos = now;
            _state.State.Version++;
            _state.State.Metrics.TotalMigrations += migratedCount;
            _state.State.Metrics.DeviceChanges++;

            await _state.WriteStateAsync();

            _logger.LogInformation(
                "Migrated {Count} members of affinity group {GroupId} to device {DeviceIndex}",
                migratedCount,
                _groupId,
                targetDeviceIndex);
        }

        return migratedCount;
    }

    /// <inheritdoc />
    public Task<AffinityGroupMetrics> GetMetricsAsync()
    {
        // Calculate device distribution
        var deviceDistribution = _state.State.Members.Values
            .GroupBy(m => m.DeviceIndex)
            .ToDictionary(g => g.Key, g => g.Count());

        return Task.FromResult(new AffinityGroupMetrics
        {
            MemberCount = _state.State.Members.Count,
            TotalRegistrations = _state.State.Metrics.TotalRegistrations,
            TotalUnregistrations = _state.State.Metrics.TotalUnregistrations,
            TotalMigrations = _state.State.Metrics.TotalMigrations,
            PreferredDeviceIndex = _state.State.PreferredDeviceIndex,
            DeviceDistribution = deviceDistribution
        });
    }

    private static long GetNanoseconds()
    {
        return Stopwatch.GetTimestamp() * 1_000_000_000L / Stopwatch.Frequency;
    }
}
