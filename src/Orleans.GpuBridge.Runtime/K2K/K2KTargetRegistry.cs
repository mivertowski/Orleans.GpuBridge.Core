// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Runtime.Routing;

namespace Orleans.GpuBridge.Runtime.K2K;

/// <summary>
/// Registry for K2K target configurations.
/// Stores metadata about K2K messaging targets for routing decisions.
/// </summary>
public sealed class K2KTargetRegistry
{
    private readonly ILogger<K2KTargetRegistry> _logger;
    private readonly ConcurrentDictionary<K2KTargetKey, K2KTargetConfig> _targets;

    /// <summary>
    /// Initializes a new instance of the K2K target registry.
    /// </summary>
    public K2KTargetRegistry(ILogger<K2KTargetRegistry> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _targets = new ConcurrentDictionary<K2KTargetKey, K2KTargetConfig>();
    }

    /// <summary>
    /// Registers a K2K target configuration.
    /// </summary>
    public void RegisterTarget(
        string sourceActorType,
        string sourceMethod,
        string targetActorType,
        string targetMethod,
        K2KRoutingStrategy routingStrategy,
        bool allowCpuFallback = true)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourceActorType);
        ArgumentException.ThrowIfNullOrEmpty(sourceMethod);
        ArgumentException.ThrowIfNullOrEmpty(targetActorType);
        ArgumentException.ThrowIfNullOrEmpty(targetMethod);

        var key = new K2KTargetKey(sourceActorType, sourceMethod);
        var config = new K2KTargetConfig(
            targetActorType,
            targetMethod,
            routingStrategy,
            allowCpuFallback);

        _targets[key] = config;

        _logger.LogDebug(
            "Registered K2K target: {SourceType}.{SourceMethod} -> {TargetType}.{TargetMethod} ({Strategy})",
            sourceActorType, sourceMethod, targetActorType, targetMethod, routingStrategy);
    }

    /// <summary>
    /// Gets the K2K target configuration for a source handler.
    /// </summary>
    public K2KTargetConfig? GetTarget(string sourceActorType, string sourceMethod)
    {
        var key = new K2KTargetKey(sourceActorType, sourceMethod);
        return _targets.TryGetValue(key, out var config) ? config : null;
    }

    /// <summary>
    /// Gets all registered targets for an actor type.
    /// </summary>
    public IReadOnlyList<K2KTargetConfig> GetTargetsForActorType(string sourceActorType)
    {
        var results = new List<K2KTargetConfig>();

        foreach (var kvp in _targets)
        {
            if (kvp.Key.SourceActorType.Equals(sourceActorType, StringComparison.Ordinal))
            {
                results.Add(kvp.Value);
            }
        }

        return results;
    }

    /// <summary>
    /// Checks if a handler has K2K targets configured.
    /// </summary>
    public bool HasK2KTargets(string sourceActorType, string sourceMethod)
    {
        var key = new K2KTargetKey(sourceActorType, sourceMethod);
        return _targets.ContainsKey(key);
    }

    /// <summary>
    /// Unregisters all targets for an actor type.
    /// </summary>
    public void UnregisterActorType(string actorType)
    {
        var keysToRemove = new List<K2KTargetKey>();

        foreach (var key in _targets.Keys)
        {
            if (key.SourceActorType.Equals(actorType, StringComparison.Ordinal))
            {
                keysToRemove.Add(key);
            }
        }

        foreach (var key in keysToRemove)
        {
            _targets.TryRemove(key, out _);
        }

        _logger.LogDebug("Unregistered all K2K targets for {ActorType}", actorType);
    }

    /// <summary>
    /// Key for K2K target lookup.
    /// </summary>
    private readonly record struct K2KTargetKey(string SourceActorType, string SourceMethod);
}

/// <summary>
/// Configuration for a K2K target.
/// </summary>
public readonly record struct K2KTargetConfig(
    string TargetActorType,
    string TargetMethod,
    K2KRoutingStrategy RoutingStrategy,
    bool AllowCpuFallback);
