// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Collections.Concurrent;
using System.Reflection;

namespace Orleans.GpuBridge.Abstractions.RingKernels;

/// <summary>
/// Registry for CPU fallback handlers that provide equivalent logic to GPU ring kernel handlers.
/// </summary>
/// <remarks>
/// <para>
/// The registry maintains a collection of CPU handlers indexed by kernel ID and handler ID.
/// When GPU execution is unavailable, the bridge falls back to these CPU implementations.
/// </para>
/// <para>
/// Handlers can be registered in two ways:
/// </para>
/// <list type="bullet">
/// <item><description>Explicitly via RegisterHandler methods</description></item>
/// <item><description>Automatically via <see cref="DiscoverHandlers"/> from assemblies</description></item>
/// </list>
/// <para>
/// <b>Thread Safety:</b> This class is thread-safe for concurrent registration and lookup.
/// </para>
/// </remarks>
public sealed class CpuFallbackHandlerRegistry
{
    /// <summary>
    /// Handlers indexed by (KernelId, HandlerId) tuple.
    /// </summary>
    private readonly ConcurrentDictionary<(string KernelId, int HandlerId), ICpuFallbackHandler> _handlers;

    /// <summary>
    /// Type-erased handler delegates for execution.
    /// </summary>
    private readonly ConcurrentDictionary<(string KernelId, int HandlerId), Delegate> _handlerDelegates;

    /// <summary>
    /// Type-erased fire-and-forget handler delegates.
    /// </summary>
    private readonly ConcurrentDictionary<(string KernelId, int HandlerId), Delegate> _fireAndForgetDelegates;

    /// <summary>
    /// Statistics for handler usage.
    /// </summary>
    private readonly ConcurrentDictionary<(string KernelId, int HandlerId), long> _invocationCounts;

    /// <summary>
    /// Creates a new CPU fallback handler registry.
    /// </summary>
    public CpuFallbackHandlerRegistry()
    {
        _handlers = new ConcurrentDictionary<(string, int), ICpuFallbackHandler>();
        _handlerDelegates = new ConcurrentDictionary<(string, int), Delegate>();
        _fireAndForgetDelegates = new ConcurrentDictionary<(string, int), Delegate>();
        _invocationCounts = new ConcurrentDictionary<(string, int), long>();
    }

    /// <summary>
    /// Gets the number of registered handlers.
    /// </summary>
    public int HandlerCount => _handlers.Count;

    /// <summary>
    /// Gets all registered handler descriptions.
    /// </summary>
    public IReadOnlyCollection<CpuFallbackHandlerInfo> RegisteredHandlers =>
        _handlers.Values
            .Select(h => new CpuFallbackHandlerInfo(
                h.KernelId,
                h.HandlerId,
                h.Description,
                h.GetType(),
                _invocationCounts.GetValueOrDefault((h.KernelId, h.HandlerId))))
            .ToList()
            .AsReadOnly();

    /// <summary>
    /// Registers a CPU fallback handler for request/response processing.
    /// </summary>
    /// <typeparam name="TRequest">Request type.</typeparam>
    /// <typeparam name="TResponse">Response type.</typeparam>
    /// <typeparam name="TState">State type.</typeparam>
    /// <param name="handler">The handler implementation.</param>
    /// <returns>True if registered, false if a handler already exists for this kernel/handler ID.</returns>
    public bool RegisterHandler<TRequest, TResponse, TState>(
        ICpuFallbackHandler<TRequest, TResponse, TState> handler)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handler);

        var key = (handler.KernelId, handler.HandlerId);

        if (!_handlers.TryAdd(key, handler))
        {
            return false;
        }

        // Store type-erased delegate for execution
        Func<TRequest, TState, (TResponse, TState)> del = handler.Execute;
        _handlerDelegates[key] = del;
        _invocationCounts[key] = 0;

        return true;
    }

    /// <summary>
    /// Registers a fire-and-forget CPU fallback handler.
    /// </summary>
    /// <typeparam name="TRequest">Request type.</typeparam>
    /// <typeparam name="TState">State type.</typeparam>
    /// <param name="handler">The handler implementation.</param>
    /// <returns>True if registered, false if a handler already exists.</returns>
    public bool RegisterFireAndForgetHandler<TRequest, TState>(
        ICpuFallbackFireAndForgetHandler<TRequest, TState> handler)
        where TRequest : unmanaged
        where TState : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handler);

        var key = (handler.KernelId, handler.HandlerId);

        if (!_handlers.TryAdd(key, handler))
        {
            return false;
        }

        // Store type-erased delegate for execution
        Func<TRequest, TState, TState> del = handler.Execute;
        _fireAndForgetDelegates[key] = del;
        _invocationCounts[key] = 0;

        return true;
    }

    /// <summary>
    /// Registers a stateless CPU fallback handler.
    /// </summary>
    /// <typeparam name="TRequest">Request type.</typeparam>
    /// <typeparam name="TResponse">Response type.</typeparam>
    /// <param name="handler">The handler implementation.</param>
    /// <returns>True if registered, false if a handler already exists.</returns>
    public bool RegisterStatelessHandler<TRequest, TResponse>(
        IStatelessCpuFallbackHandler<TRequest, TResponse> handler)
        where TRequest : unmanaged
        where TResponse : unmanaged
    {
        ArgumentNullException.ThrowIfNull(handler);

        var key = (handler.KernelId, handler.HandlerId);

        if (!_handlers.TryAdd(key, handler))
        {
            return false;
        }

        // Wrap stateless handler as stateful (ignoring state)
        Func<TRequest, object, (TResponse, object)> del = (req, state) =>
            (handler.Execute(req), state);
        _handlerDelegates[key] = del;
        _invocationCounts[key] = 0;

        return true;
    }

    /// <summary>
    /// Registers a handler using a lambda function.
    /// </summary>
    /// <typeparam name="TRequest">Request type.</typeparam>
    /// <typeparam name="TResponse">Response type.</typeparam>
    /// <typeparam name="TState">State type.</typeparam>
    /// <param name="kernelId">Kernel ID.</param>
    /// <param name="handlerId">Handler ID within the kernel.</param>
    /// <param name="handler">Handler function.</param>
    /// <param name="description">Optional description.</param>
    /// <returns>True if registered.</returns>
    public bool RegisterHandler<TRequest, TResponse, TState>(
        string kernelId,
        int handlerId,
        Func<TRequest, TState, (TResponse Response, TState NewState)> handler,
        string? description = null)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelId);
        ArgumentNullException.ThrowIfNull(handler);

        var key = (kernelId, handlerId);

        var lambdaHandler = new LambdaCpuFallbackHandler<TRequest, TResponse, TState>(
            kernelId, handlerId, handler, description ?? $"Lambda handler for {kernelId}:{handlerId}");

        if (!_handlers.TryAdd(key, lambdaHandler))
        {
            return false;
        }

        _handlerDelegates[key] = handler;
        _invocationCounts[key] = 0;

        return true;
    }

    /// <summary>
    /// Registers a fire-and-forget handler using a lambda function.
    /// </summary>
    public bool RegisterFireAndForgetHandler<TRequest, TState>(
        string kernelId,
        int handlerId,
        Func<TRequest, TState, TState> handler,
        string? description = null)
        where TRequest : unmanaged
        where TState : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelId);
        ArgumentNullException.ThrowIfNull(handler);

        var key = (kernelId, handlerId);

        var lambdaHandler = new LambdaCpuFallbackFireAndForgetHandler<TRequest, TState>(
            kernelId, handlerId, handler, description ?? $"Lambda F&F handler for {kernelId}:{handlerId}");

        if (!_handlers.TryAdd(key, lambdaHandler))
        {
            return false;
        }

        _fireAndForgetDelegates[key] = handler;
        _invocationCounts[key] = 0;

        return true;
    }

    /// <summary>
    /// Tries to get a handler for the specified kernel and handler ID.
    /// </summary>
    /// <param name="kernelId">Kernel ID.</param>
    /// <param name="handlerId">Handler ID.</param>
    /// <param name="handler">The handler if found.</param>
    /// <returns>True if a handler was found.</returns>
    public bool TryGetHandler(string kernelId, int handlerId, out ICpuFallbackHandler? handler)
    {
        return _handlers.TryGetValue((kernelId, handlerId), out handler);
    }

    /// <summary>
    /// Executes a CPU fallback handler.
    /// </summary>
    /// <typeparam name="TRequest">Request type.</typeparam>
    /// <typeparam name="TResponse">Response type.</typeparam>
    /// <typeparam name="TState">State type.</typeparam>
    /// <param name="kernelId">Kernel ID.</param>
    /// <param name="handlerId">Handler ID.</param>
    /// <param name="request">The request.</param>
    /// <param name="currentState">Current state.</param>
    /// <returns>Response and updated state, or null if no handler found.</returns>
    public (TResponse Response, TState NewState)? ExecuteHandler<TRequest, TResponse, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        TState currentState)
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        var key = (kernelId, handlerId);

        if (!_handlerDelegates.TryGetValue(key, out var del))
        {
            return null;
        }

        // Track invocation
        _invocationCounts.AddOrUpdate(key, 1, (_, count) => count + 1);

        // Try to cast to the expected delegate type
        if (del is Func<TRequest, TState, (TResponse, TState)> typedDelegate)
        {
            return typedDelegate(request, currentState);
        }

        // Handler exists but types don't match
        return null;
    }

    /// <summary>
    /// Executes a fire-and-forget CPU fallback handler.
    /// </summary>
    /// <typeparam name="TRequest">Request type.</typeparam>
    /// <typeparam name="TState">State type.</typeparam>
    /// <param name="kernelId">Kernel ID.</param>
    /// <param name="handlerId">Handler ID.</param>
    /// <param name="request">The request.</param>
    /// <param name="currentState">Current state.</param>
    /// <returns>Updated state, or null if no handler found.</returns>
    public TState? ExecuteFireAndForgetHandler<TRequest, TState>(
        string kernelId,
        int handlerId,
        TRequest request,
        TState currentState)
        where TRequest : unmanaged
        where TState : struct
    {
        var key = (kernelId, handlerId);

        if (!_fireAndForgetDelegates.TryGetValue(key, out var del))
        {
            return null;
        }

        // Track invocation
        _invocationCounts.AddOrUpdate(key, 1, (_, count) => count + 1);

        if (del is Func<TRequest, TState, TState> typedDelegate)
        {
            return typedDelegate(request, currentState);
        }

        return null;
    }

    /// <summary>
    /// Checks if a handler is registered for the specified kernel and handler ID.
    /// </summary>
    public bool HasHandler(string kernelId, int handlerId)
    {
        return _handlers.ContainsKey((kernelId, handlerId));
    }

    /// <summary>
    /// Removes a handler from the registry.
    /// </summary>
    /// <param name="kernelId">Kernel ID.</param>
    /// <param name="handlerId">Handler ID.</param>
    /// <returns>True if removed.</returns>
    public bool RemoveHandler(string kernelId, int handlerId)
    {
        var key = (kernelId, handlerId);
        _handlerDelegates.TryRemove(key, out _);
        _fireAndForgetDelegates.TryRemove(key, out _);
        _invocationCounts.TryRemove(key, out _);
        return _handlers.TryRemove(key, out _);
    }

    /// <summary>
    /// Discovers and registers CPU fallback handlers from the specified assemblies.
    /// </summary>
    /// <param name="assemblies">Assemblies to scan for handlers.</param>
    /// <returns>Number of handlers discovered and registered.</returns>
    public int DiscoverHandlers(params Assembly[] assemblies)
    {
        var count = 0;

        foreach (var assembly in assemblies)
        {
            try
            {
                foreach (var type in assembly.GetTypes())
                {
                    var attr = type.GetCustomAttribute<CpuFallbackHandlerAttribute>();
                    if (attr == null)
                    {
                        continue;
                    }

                    // Check if type implements ICpuFallbackHandler
                    if (!typeof(ICpuFallbackHandler).IsAssignableFrom(type))
                    {
                        continue;
                    }

                    // Create instance and register
                    try
                    {
                        var instance = Activator.CreateInstance(type);
                        if (instance is ICpuFallbackHandler handler)
                        {
                            var key = (handler.KernelId, handler.HandlerId);
                            if (_handlers.TryAdd(key, handler))
                            {
                                _invocationCounts[key] = 0;
                                count++;

                                // Try to create delegate based on interface type
                                RegisterDelegateFromHandler(handler);
                            }
                        }
                    }
                    catch
                    {
                        // Skip handlers that can't be instantiated
                    }
                }
            }
            catch
            {
                // Skip assemblies that can't be scanned
            }
        }

        return count;
    }

    private void RegisterDelegateFromHandler(ICpuFallbackHandler handler)
    {
        // Use reflection to find the Execute method and create appropriate delegate
        var handlerType = handler.GetType();
        var interfaces = handlerType.GetInterfaces();

        foreach (var iface in interfaces)
        {
            if (!iface.IsGenericType)
            {
                continue;
            }

            var genericDef = iface.GetGenericTypeDefinition();

            if (genericDef == typeof(ICpuFallbackHandler<,,>))
            {
                var typeArgs = iface.GetGenericArguments();
                var executeMethod = iface.GetMethod("Execute");
                if (executeMethod != null)
                {
                    var key = (handler.KernelId, handler.HandlerId);
                    var delegateType = typeof(Func<,,,>).MakeGenericType(
                        typeArgs[0], typeArgs[2],
                        typeof(ValueTuple<,>).MakeGenericType(typeArgs[1], typeArgs[2]));

                    try
                    {
                        var del = Delegate.CreateDelegate(delegateType, handler, executeMethod);
                        _handlerDelegates[key] = del;
                    }
                    catch
                    {
                        // Unable to create delegate
                    }
                }
                break;
            }

            if (genericDef == typeof(ICpuFallbackFireAndForgetHandler<,>))
            {
                var typeArgs = iface.GetGenericArguments();
                var executeMethod = iface.GetMethod("Execute");
                if (executeMethod != null)
                {
                    var key = (handler.KernelId, handler.HandlerId);
                    var delegateType = typeof(Func<,,>).MakeGenericType(
                        typeArgs[0], typeArgs[1], typeArgs[1]);

                    try
                    {
                        var del = Delegate.CreateDelegate(delegateType, handler, executeMethod);
                        _fireAndForgetDelegates[key] = del;
                    }
                    catch
                    {
                        // Unable to create delegate
                    }
                }
                break;
            }
        }
    }

    /// <summary>
    /// Clears all registered handlers.
    /// </summary>
    public void Clear()
    {
        _handlers.Clear();
        _handlerDelegates.Clear();
        _fireAndForgetDelegates.Clear();
        _invocationCounts.Clear();
    }

    /// <summary>
    /// Gets statistics for a specific handler.
    /// </summary>
    public long GetInvocationCount(string kernelId, int handlerId)
    {
        return _invocationCounts.GetValueOrDefault((kernelId, handlerId));
    }

    #region Lambda Handler Implementations

    private sealed class LambdaCpuFallbackHandler<TRequest, TResponse, TState>
        : ICpuFallbackHandler<TRequest, TResponse, TState>
        where TRequest : unmanaged
        where TResponse : unmanaged
        where TState : unmanaged
    {
        private readonly Func<TRequest, TState, (TResponse, TState)> _handler;

        public string KernelId { get; }
        public int HandlerId { get; }
        public string Description { get; }

        public LambdaCpuFallbackHandler(
            string kernelId,
            int handlerId,
            Func<TRequest, TState, (TResponse, TState)> handler,
            string description)
        {
            KernelId = kernelId;
            HandlerId = handlerId;
            Description = description;
            _handler = handler;
        }

        public (TResponse Response, TState NewState) Execute(TRequest request, TState currentState)
        {
            return _handler(request, currentState);
        }
    }

    private sealed class LambdaCpuFallbackFireAndForgetHandler<TRequest, TState>
        : ICpuFallbackFireAndForgetHandler<TRequest, TState>
        where TRequest : unmanaged
        where TState : unmanaged
    {
        private readonly Func<TRequest, TState, TState> _handler;

        public string KernelId { get; }
        public int HandlerId { get; }
        public string Description { get; }

        public LambdaCpuFallbackFireAndForgetHandler(
            string kernelId,
            int handlerId,
            Func<TRequest, TState, TState> handler,
            string description)
        {
            KernelId = kernelId;
            HandlerId = handlerId;
            Description = description;
            _handler = handler;
        }

        public TState Execute(TRequest request, TState currentState)
        {
            return _handler(request, currentState);
        }
    }

    #endregion
}

/// <summary>
/// Information about a registered CPU fallback handler.
/// </summary>
/// <param name="KernelId">The kernel ID.</param>
/// <param name="HandlerId">The handler ID.</param>
/// <param name="Description">Handler description.</param>
/// <param name="HandlerType">The handler implementation type.</param>
/// <param name="InvocationCount">Number of times the handler has been invoked.</param>
public readonly record struct CpuFallbackHandlerInfo(
    string KernelId,
    int HandlerId,
    string Description,
    Type HandlerType,
    long InvocationCount);
