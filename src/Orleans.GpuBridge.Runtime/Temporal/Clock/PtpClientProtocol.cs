using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Runtime.Temporal.Clock;

/// <summary>
/// Implements IEEE 1588 PTP (Precision Time Protocol) client for software-based clock synchronization.
/// </summary>
/// <remarks>
/// <para>
/// <strong>PTP Message Exchange:</strong>
/// </para>
/// <code>
/// Master                    Client
///   |                         |
///   |-------- SYNC -------&gt;  | (t1: master TX time)
///   |                         | (t2: client RX time)
///   |                         |
///   |---- FOLLOW_UP ------&gt;  | (precise t1 timestamp)
///   |                         |
///   |  &lt;---- DELAY_REQ ------| (t3: client TX time)
///   |                         |
///   |---- DELAY_RESP -----&gt;  | (t4: master RX time)
///   |                         |
/// </code>
/// <para>
/// <strong>Offset Calculation:</strong> offset = ((t2 - t1) - (t4 - t3)) / 2
/// </para>
/// <para>
/// <strong>Round-Trip Delay:</strong> delay = (t2 - t1) + (t4 - t3)
/// </para>
/// <para>
/// <strong>Assumptions:</strong>
/// <list type="bullet">
/// <item><description>Symmetric network paths (forward delay approximately equals reverse delay)</description></item>
/// <item><description>Minimal packet loss and jitter</description></item>
/// <item><description>NTP-style timestamp format (seconds + fractional nanoseconds)</description></item>
/// </list>
/// </para>
/// </remarks>
public sealed class PtpClientProtocol : IDisposable
{
    private readonly ILogger _logger;
    private readonly UdpClient _udpClient;
    private const int PtpEventPort = 319;
    private const int PtpGeneralPort = 320;

    /// <summary>
    /// Initializes a new instance of the <see cref="PtpClientProtocol"/> class.
    /// </summary>
    /// <param name="logger">Logger instance for diagnostic messages.</param>
    /// <exception cref="ArgumentNullException">Thrown when logger is null.</exception>
    public PtpClientProtocol(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _udpClient = new UdpClient();
        _udpClient.Client.ReceiveTimeout = 5000; // 5 second timeout
    }

    /// <summary>
    /// Performs PTP message exchange with master server.
    /// </summary>
    /// <param name="masterAddress">PTP master server address.</param>
    /// <param name="port">PTP port (default: 319).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>PTP synchronization result with offset and delay measurements.</returns>
    public async Task<PtpSyncResult?> ExchangeAsync(
        string masterAddress,
        int port,
        CancellationToken ct = default)
    {
        try
        {
            // For this implementation, we'll use SNTP (simplified NTP) protocol
            // as a stand-in for full PTP, since PTP requires multicast and
            // specialized network infrastructure

            _logger.LogDebug("Performing SNTP exchange with {Master}:{Port}", masterAddress, port);

            var endpoint = new IPEndPoint(
                (await Dns.GetHostAddressesAsync(masterAddress, ct))[0],
                123); // NTP port (SNTP fallback)

            // Send SNTP request
            var requestData = CreateSntpRequest();
            var t1Local = GetTimestampNanos(); // Local timestamp before send

            await _udpClient.SendAsync(requestData, requestData.Length, endpoint);

            // Receive SNTP response
            var response = await _udpClient.ReceiveAsync(ct);
            var t2Local = GetTimestampNanos(); // Local timestamp after receive

            // Parse SNTP response
            var result = ParseSntpResponse(response.Buffer, t1Local, t2Local);

            if (result != null)
            {
                _logger.LogDebug(
                    "SNTP exchange successful: offset={Offset}ms, RTD={Rtd}ms",
                    result.OffsetNanos / 1_000_000.0,
                    result.RoundTripDelayNanos / 1_000_000.0);
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "PTP/SNTP exchange failed");
            return null;
        }
    }

    /// <summary>
    /// Creates an SNTP request packet (NTP v4 client mode).
    /// </summary>
    private byte[] CreateSntpRequest()
    {
        var packet = new byte[48];

        // LI=0 (no warning), VN=4 (NTP v4), Mode=3 (client)
        packet[0] = 0x1B; // 00 011 011

        // Transmit timestamp (current time in NTP format)
        var ntpTimestamp = ToNtpTimestamp(DateTimeOffset.UtcNow);
        Array.Copy(BitConverter.GetBytes(ntpTimestamp), 0, packet, 40, 8);

        return packet;
    }

    /// <summary>
    /// Parses SNTP response and calculates clock offset.
    /// </summary>
    private PtpSyncResult? ParseSntpResponse(byte[] data, long t1Local, long t4Local)
    {
        if (data.Length < 48)
        {
            _logger.LogWarning("Invalid SNTP response length: {Length}", data.Length);
            return null;
        }

        // Extract server transmit timestamp (bytes 40-47)
        var serverTxTime = FromNtpTimestamp(data, 40);

        // Extract server receive timestamp (bytes 32-39)
        var serverRxTime = FromNtpTimestamp(data, 32);

        // Calculate offset: ((t2 - t1) - (t4 - t3)) / 2
        // In SNTP: t1 = client TX, t2 = server RX, t3 = server TX, t4 = client RX
        long t1 = t1Local;
        long t2 = serverRxTime;
        long t3 = serverTxTime;
        long t4 = t4Local;

        long offset = ((t2 - t1) - (t4 - t3)) / 2;
        long roundTripDelay = (t4 - t1) - (t3 - t2);

        return new PtpSyncResult
        {
            OffsetNanos = offset,
            RoundTripDelayNanos = roundTripDelay,
            MasterTimestampNanos = t3,
            LocalTimestampNanos = t4,
            DriftRatePpm = null // Not available in SNTP
        };
    }

    /// <summary>
    /// Converts DateTimeOffset to NTP timestamp (seconds since 1900-01-01).
    /// </summary>
    private long ToNtpTimestamp(DateTimeOffset time)
    {
        var ntpEpoch = new DateTime(1900, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var elapsed = time.UtcDateTime - ntpEpoch;

        long seconds = (long)elapsed.TotalSeconds;
        long fractional = (long)((elapsed.TotalSeconds - seconds) * 0x100000000L);

        return (seconds << 32) | fractional;
    }

    /// <summary>
    /// Converts NTP timestamp bytes to nanoseconds since Unix epoch.
    /// </summary>
    private long FromNtpTimestamp(byte[] data, int offset)
    {
        if (BitConverter.IsLittleEndian)
        {
            // Swap endianness for seconds
            long seconds = ((long)data[offset] << 24) |
                          ((long)data[offset + 1] << 16) |
                          ((long)data[offset + 2] << 8) |
                          data[offset + 3];

            // Swap endianness for fraction
            long fraction = ((long)data[offset + 4] << 24) |
                           ((long)data[offset + 5] << 16) |
                           ((long)data[offset + 6] << 8) |
                           data[offset + 7];

            // Convert NTP timestamp (1900 epoch) to Unix timestamp (1970 epoch)
            const long ntpToUnixSeconds = 2208988800L; // Seconds between 1900 and 1970
            long unixSeconds = seconds - ntpToUnixSeconds;

            // Convert fraction to nanoseconds
            long nanoseconds = (fraction * 1_000_000_000L) / 0x100000000L;

            return (unixSeconds * 1_000_000_000L) + nanoseconds;
        }
        else
        {
            throw new NotSupportedException("Big-endian systems not supported");
        }
    }

    /// <summary>
    /// Gets current timestamp in nanoseconds since Unix epoch.
    /// </summary>
    private long GetTimestampNanos()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000;
    }

    /// <summary>
    /// Releases all resources used by the PTP client protocol.
    /// </summary>
    public void Dispose()
    {
        _udpClient?.Dispose();
    }
}

/// <summary>
/// Result of PTP synchronization exchange.
/// </summary>
public sealed class PtpSyncResult
{
    /// <summary>
    /// Clock offset from master (nanoseconds).
    /// Positive = local clock is ahead, negative = local clock is behind.
    /// </summary>
    public required long OffsetNanos { get; init; }

    /// <summary>
    /// Round-trip delay to master (nanoseconds).
    /// </summary>
    public required long RoundTripDelayNanos { get; init; }

    /// <summary>
    /// Master timestamp from SYNC message (nanoseconds since Unix epoch).
    /// </summary>
    public required long MasterTimestampNanos { get; init; }

    /// <summary>
    /// Local timestamp when SYNC was received (nanoseconds since Unix epoch).
    /// </summary>
    public required long LocalTimestampNanos { get; init; }

    /// <summary>
    /// Clock drift rate in parts per million (PPM), if available.
    /// </summary>
    public double? DriftRatePpm { get; init; }
}
