// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

namespace Orleans.GpuBridge.Abstractions.Placement;

/// <summary>
/// Interface for migrating GPU-native grains between silos/devices.
/// </summary>
/// <remarks>
/// <para>
/// The grain migrator coordinates the movement of grains between GPU devices
/// to achieve load balancing, handle device failures, or respond to affinity
/// group changes.
/// </para>
/// <para>
/// <strong>Migration Process:</strong>
/// <list type="number">
/// <item><description>Validate migration feasibility (target capacity, affinity rules)</description></item>
/// <item><description>Persist current grain state via Orleans storage</description></item>
/// <item><description>Deactivate grain on source silo</description></item>
/// <item><description>Activate grain on target silo (auto-restores from storage)</description></item>
/// <item><description>Update affinity group registrations</description></item>
/// </list>
/// </para>
/// <para>
/// <strong>State Transfer:</strong>
/// Migration leverages Orleans' built-in persistence (IPersistentState&lt;TState&gt;)
/// for state transfer. Grains must use persistent state to be migratable.
/// </para>
/// </remarks>
public interface IGrainMigrator
{
    /// <summary>
    /// Migrates a grain to a different device/silo.
    /// </summary>
    /// <param name="request">Migration request details.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Result of the migration operation.</returns>
    Task<MigrationResult> MigrateGrainAsync(
        MigrationRequest request,
        CancellationToken ct = default);

    /// <summary>
    /// Checks if a grain can be migrated to the specified target.
    /// </summary>
    /// <param name="grainId">Grain identity to check.</param>
    /// <param name="targetDeviceIndex">Target GPU device index.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Feasibility result with reason if not feasible.</returns>
    Task<MigrationFeasibility> CanMigrateAsync(
        string grainId,
        int targetDeviceIndex,
        CancellationToken ct = default);

    /// <summary>
    /// Executes multiple migrations from a rebalance recommendation.
    /// </summary>
    /// <param name="recommendation">Rebalance recommendation with migration suggestions.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Batch migration results.</returns>
    Task<BatchMigrationResult> ExecuteRebalanceAsync(
        RebalanceRecommendation recommendation,
        CancellationToken ct = default);

    /// <summary>
    /// Gets the current status of a migration operation.
    /// </summary>
    /// <param name="migrationId">Migration operation ID.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Migration status, or null if not found.</returns>
    Task<MigrationStatus?> GetMigrationStatusAsync(
        string migrationId,
        CancellationToken ct = default);

    /// <summary>
    /// Cancels an in-progress migration if possible.
    /// </summary>
    /// <param name="migrationId">Migration operation ID to cancel.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>True if cancellation was successful.</returns>
    Task<bool> CancelMigrationAsync(
        string migrationId,
        CancellationToken ct = default);

    /// <summary>
    /// Gets metrics about migration operations.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Migration metrics.</returns>
    Task<MigrationMetrics> GetMetricsAsync(CancellationToken ct = default);

    /// <summary>
    /// Subscribes to migration events.
    /// </summary>
    /// <param name="callback">Event callback.</param>
    /// <returns>Subscription handle.</returns>
    IDisposable SubscribeToEvents(Action<MigrationEvent> callback);
}

/// <summary>
/// Request to migrate a grain.
/// </summary>
public sealed class MigrationRequest
{
    /// <summary>
    /// Unique migration operation ID.
    /// </summary>
    public string MigrationId { get; init; } = Guid.NewGuid().ToString("N");

    /// <summary>
    /// Grain identity to migrate.
    /// </summary>
    public required string GrainId { get; init; }

    /// <summary>
    /// Grain type name.
    /// </summary>
    public required string GrainType { get; init; }

    /// <summary>
    /// Source device index.
    /// </summary>
    public required int SourceDeviceIndex { get; init; }

    /// <summary>
    /// Target device index.
    /// </summary>
    public required int TargetDeviceIndex { get; init; }

    /// <summary>
    /// Target silo identifier (optional - uses placement director if not specified).
    /// </summary>
    public string? TargetSiloId { get; init; }

    /// <summary>
    /// Priority of migration (higher = more urgent).
    /// </summary>
    public int Priority { get; init; } = 0;

    /// <summary>
    /// Reason for migration.
    /// </summary>
    public MigrationReason Reason { get; init; } = MigrationReason.LoadBalancing;

    /// <summary>
    /// Affinity group to update after migration (optional).
    /// </summary>
    public string? AffinityGroup { get; init; }

    /// <summary>
    /// Maximum time to wait for migration to complete.
    /// </summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Whether to force migration even if target is suboptimal.
    /// </summary>
    public bool Force { get; init; } = false;
}

/// <summary>
/// Result of a migration operation.
/// </summary>
public sealed class MigrationResult
{
    /// <summary>
    /// Migration operation ID.
    /// </summary>
    public required string MigrationId { get; init; }

    /// <summary>
    /// Whether migration succeeded.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Grain identity that was migrated.
    /// </summary>
    public required string GrainId { get; init; }

    /// <summary>
    /// Source device index.
    /// </summary>
    public required int SourceDeviceIndex { get; init; }

    /// <summary>
    /// Target device index (actual, may differ from requested if fallback used).
    /// </summary>
    public required int TargetDeviceIndex { get; init; }

    /// <summary>
    /// Target silo where grain was activated.
    /// </summary>
    public string? TargetSiloId { get; init; }

    /// <summary>
    /// Time taken for migration (nanoseconds).
    /// </summary>
    public required long DurationNanos { get; init; }

    /// <summary>
    /// Error message if migration failed.
    /// </summary>
    public string? ErrorMessage { get; init; }

    /// <summary>
    /// Migration phase where failure occurred (if failed).
    /// </summary>
    public MigrationPhase? FailedPhase { get; init; }

    /// <summary>
    /// Whether state was successfully transferred.
    /// </summary>
    public required bool StateTransferred { get; init; }

    /// <summary>
    /// Timestamp when migration completed.
    /// </summary>
    public required long CompletedAtNanos { get; init; }
}

/// <summary>
/// Result of checking migration feasibility.
/// </summary>
public readonly record struct MigrationFeasibility
{
    /// <summary>
    /// Whether migration is feasible.
    /// </summary>
    public required bool CanMigrate { get; init; }

    /// <summary>
    /// Reason if migration is not feasible.
    /// </summary>
    public string? Reason { get; init; }

    /// <summary>
    /// Target device's current load score.
    /// </summary>
    public required double TargetLoadScore { get; init; }

    /// <summary>
    /// Available memory on target device (bytes).
    /// </summary>
    public required long TargetAvailableMemory { get; init; }

    /// <summary>
    /// Whether grain uses persistent state (required for migration).
    /// </summary>
    public required bool HasPersistentState { get; init; }

    /// <summary>
    /// Estimated migration time.
    /// </summary>
    public required TimeSpan EstimatedDuration { get; init; }

    /// <summary>
    /// Risk level of migration.
    /// </summary>
    public required MigrationRisk Risk { get; init; }
}

/// <summary>
/// Result of batch migration operations.
/// </summary>
public sealed class BatchMigrationResult
{
    /// <summary>
    /// Batch operation ID.
    /// </summary>
    public required string BatchId { get; init; }

    /// <summary>
    /// Total migrations attempted.
    /// </summary>
    public required int TotalAttempted { get; init; }

    /// <summary>
    /// Successful migrations.
    /// </summary>
    public required int SuccessCount { get; init; }

    /// <summary>
    /// Failed migrations.
    /// </summary>
    public required int FailedCount { get; init; }

    /// <summary>
    /// Skipped migrations (not feasible).
    /// </summary>
    public required int SkippedCount { get; init; }

    /// <summary>
    /// Individual migration results.
    /// </summary>
    public required IReadOnlyList<MigrationResult> Results { get; init; }

    /// <summary>
    /// Total batch duration (nanoseconds).
    /// </summary>
    public required long TotalDurationNanos { get; init; }

    /// <summary>
    /// Load improvement achieved (0.0-1.0).
    /// </summary>
    public required double LoadImprovementAchieved { get; init; }
}

/// <summary>
/// Status of an in-progress or completed migration.
/// </summary>
public sealed class MigrationStatus
{
    /// <summary>
    /// Migration operation ID.
    /// </summary>
    public required string MigrationId { get; init; }

    /// <summary>
    /// Grain being migrated.
    /// </summary>
    public required string GrainId { get; init; }

    /// <summary>
    /// Current phase of migration.
    /// </summary>
    public required MigrationPhase Phase { get; init; }

    /// <summary>
    /// Current state of migration.
    /// </summary>
    public required MigrationState State { get; init; }

    /// <summary>
    /// Progress percentage (0-100).
    /// </summary>
    public required int ProgressPercent { get; init; }

    /// <summary>
    /// When migration started.
    /// </summary>
    public required long StartedAtNanos { get; init; }

    /// <summary>
    /// When migration completed (if finished).
    /// </summary>
    public long? CompletedAtNanos { get; init; }

    /// <summary>
    /// Error message if failed.
    /// </summary>
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Metrics about migration operations.
/// </summary>
public readonly record struct MigrationMetrics
{
    /// <summary>
    /// Total migrations attempted.
    /// </summary>
    public required long TotalMigrations { get; init; }

    /// <summary>
    /// Successful migrations.
    /// </summary>
    public required long SuccessfulMigrations { get; init; }

    /// <summary>
    /// Failed migrations.
    /// </summary>
    public required long FailedMigrations { get; init; }

    /// <summary>
    /// Cancelled migrations.
    /// </summary>
    public required long CancelledMigrations { get; init; }

    /// <summary>
    /// Currently in-progress migrations.
    /// </summary>
    public required int InProgressCount { get; init; }

    /// <summary>
    /// Average migration duration (nanoseconds).
    /// </summary>
    public required double AvgDurationNanos { get; init; }

    /// <summary>
    /// Total data transferred (bytes).
    /// </summary>
    public required long TotalDataTransferredBytes { get; init; }

    /// <summary>
    /// Success rate (0.0-1.0).
    /// </summary>
    public double SuccessRate => TotalMigrations > 0
        ? (double)SuccessfulMigrations / TotalMigrations
        : 1.0;
}

/// <summary>
/// Migration event for subscribers.
/// </summary>
public readonly record struct MigrationEvent
{
    /// <summary>
    /// Migration operation ID.
    /// </summary>
    public required string MigrationId { get; init; }

    /// <summary>
    /// Grain being migrated.
    /// </summary>
    public required string GrainId { get; init; }

    /// <summary>
    /// Event type.
    /// </summary>
    public required MigrationEventType EventType { get; init; }

    /// <summary>
    /// Event timestamp.
    /// </summary>
    public required long TimestampNanos { get; init; }

    /// <summary>
    /// Current migration phase.
    /// </summary>
    public required MigrationPhase Phase { get; init; }

    /// <summary>
    /// Event message.
    /// </summary>
    public required string Message { get; init; }

    /// <summary>
    /// Additional event data.
    /// </summary>
    public IReadOnlyDictionary<string, object>? Data { get; init; }
}

/// <summary>
/// Phases of the migration process.
/// </summary>
public enum MigrationPhase
{
    /// <summary>
    /// Not started.
    /// </summary>
    NotStarted = 0,

    /// <summary>
    /// Validating migration feasibility.
    /// </summary>
    Validating = 1,

    /// <summary>
    /// Persisting grain state.
    /// </summary>
    PersistingState = 2,

    /// <summary>
    /// Deactivating grain on source.
    /// </summary>
    Deactivating = 3,

    /// <summary>
    /// Activating grain on target.
    /// </summary>
    Activating = 4,

    /// <summary>
    /// Updating affinity groups.
    /// </summary>
    UpdatingAffinity = 5,

    /// <summary>
    /// Migration completed.
    /// </summary>
    Completed = 6,

    /// <summary>
    /// Migration failed.
    /// </summary>
    Failed = 7,

    /// <summary>
    /// Migration cancelled.
    /// </summary>
    Cancelled = 8
}

/// <summary>
/// State of a migration operation.
/// </summary>
public enum MigrationState
{
    /// <summary>
    /// Migration is pending.
    /// </summary>
    Pending = 0,

    /// <summary>
    /// Migration is in progress.
    /// </summary>
    InProgress = 1,

    /// <summary>
    /// Migration completed successfully.
    /// </summary>
    Completed = 2,

    /// <summary>
    /// Migration failed.
    /// </summary>
    Failed = 3,

    /// <summary>
    /// Migration was cancelled.
    /// </summary>
    Cancelled = 4
}

/// <summary>
/// Reasons for grain migration.
/// </summary>
public enum MigrationReason
{
    /// <summary>
    /// Migration for load balancing.
    /// </summary>
    LoadBalancing = 0,

    /// <summary>
    /// Migration due to device failure.
    /// </summary>
    DeviceFailure = 1,

    /// <summary>
    /// Migration to satisfy affinity requirements.
    /// </summary>
    AffinityRequirement = 2,

    /// <summary>
    /// Migration requested by user/admin.
    /// </summary>
    UserRequested = 3,

    /// <summary>
    /// Migration due to memory pressure.
    /// </summary>
    MemoryPressure = 4,

    /// <summary>
    /// Migration for maintenance/upgrade.
    /// </summary>
    Maintenance = 5
}

/// <summary>
/// Risk level of migration.
/// </summary>
public enum MigrationRisk
{
    /// <summary>
    /// Low risk - standard migration.
    /// </summary>
    Low = 0,

    /// <summary>
    /// Medium risk - some conditions suboptimal.
    /// </summary>
    Medium = 1,

    /// <summary>
    /// High risk - migration may cause issues.
    /// </summary>
    High = 2,

    /// <summary>
    /// Critical - migration not recommended.
    /// </summary>
    Critical = 3
}

/// <summary>
/// Types of migration events.
/// </summary>
public enum MigrationEventType
{
    /// <summary>
    /// Migration started.
    /// </summary>
    Started = 0,

    /// <summary>
    /// Migration phase changed.
    /// </summary>
    PhaseChanged = 1,

    /// <summary>
    /// Migration progress updated.
    /// </summary>
    ProgressUpdated = 2,

    /// <summary>
    /// Migration completed successfully.
    /// </summary>
    Completed = 3,

    /// <summary>
    /// Migration failed.
    /// </summary>
    Failed = 4,

    /// <summary>
    /// Migration cancelled.
    /// </summary>
    Cancelled = 5,

    /// <summary>
    /// Migration warning.
    /// </summary>
    Warning = 6
}
