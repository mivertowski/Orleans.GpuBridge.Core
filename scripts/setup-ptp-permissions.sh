#!/bin/bash
# Setup PTP Hardware Permissions for Orleans.GpuBridge.Core
# Enables non-root access to /dev/ptp* devices for Phase 6 Physical Time Precision

set -e

echo "=== Orleans.GpuBridge.Core - PTP Hardware Setup ==="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "ERROR: This script is for Linux only"
    echo "PTP hardware clock support requires Linux kernel with PTP_1588_CLOCK"
    exit 1
fi

# Check for PTP devices
if ! ls /dev/ptp* 1> /dev/null 2>&1; then
    echo "WARNING: No PTP devices found at /dev/ptp*"
    echo ""
    echo "Possible reasons:"
    echo "  1. No PTP-capable network interface card (NIC)"
    echo "  2. PTP kernel module not loaded"
    echo "  3. Running in virtual machine without PTP passthrough"
    echo ""
    echo "Checking for Hyper-V PTP..."
    if lsmod | grep -q hv_utils; then
        echo "✓ Hyper-V utils module loaded"
        echo "  Hyper-V synthetic NIC may provide PTP at /dev/ptp0"
    fi
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# List available PTP devices
echo "Available PTP devices:"
ls -l /dev/ptp* 2>/dev/null || echo "  (none found)"
echo ""

# Check if user already has access
if [ -r /dev/ptp0 ] && [ -w /dev/ptp0 ]; then
    echo "✓ Current user already has PTP access"
    echo "  No configuration needed"
    exit 0
fi

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script requires sudo privileges"
    echo "       Run with: sudo $0"
    exit 1
fi

echo "Step 1: Creating 'ptp' group..."
if getent group ptp > /dev/null 2>&1; then
    echo "  ✓ Group 'ptp' already exists"
else
    groupadd ptp
    echo "  ✓ Created group 'ptp'"
fi

echo ""
echo "Step 2: Adding current user to 'ptp' group..."
REAL_USER="${SUDO_USER:-$USER}"
if id -nG "$REAL_USER" | grep -qw ptp; then
    echo "  ✓ User '$REAL_USER' already in 'ptp' group"
else
    usermod -aG ptp "$REAL_USER"
    echo "  ✓ Added user '$REAL_USER' to 'ptp' group"
    echo "  NOTE: User must log out and back in for group changes to take effect"
fi

echo ""
echo "Step 3: Creating udev rules for PTP devices..."
UDEV_RULE_FILE="/etc/udev/rules.d/99-ptp.rules"

cat > "$UDEV_RULE_FILE" << 'EOF'
# Orleans.GpuBridge.Core - PTP Hardware Clock Permissions
# Allow 'ptp' group access to PTP devices without sudo

# PTP hardware clocks
SUBSYSTEM=="ptp", GROUP="ptp", MODE="0660"

# PTP character devices (alternative match)
KERNEL=="ptp[0-9]*", GROUP="ptp", MODE="0660"
EOF

echo "  ✓ Created $UDEV_RULE_FILE"

echo ""
echo "Step 4: Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger --subsystem-match=ptp
echo "  ✓ udev rules reloaded"

echo ""
echo "Step 5: Verifying permissions..."
sleep 1 # Give udev time to apply rules

if ls /dev/ptp* 1> /dev/null 2>&1; then
    for ptp_dev in /dev/ptp*; do
        ptp_group=$(stat -c '%G' "$ptp_dev")
        ptp_mode=$(stat -c '%a' "$ptp_dev")

        if [ "$ptp_group" = "ptp" ] && [ "$ptp_mode" = "660" ]; then
            echo "  ✓ $ptp_dev: group=$ptp_group, mode=$ptp_mode"
        else
            echo "  ⚠ $ptp_dev: group=$ptp_group, mode=$ptp_mode (expected: ptp, 660)"
        fi
    done
else
    echo "  ⚠ No PTP devices found for verification"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Log out and log back in (for group membership to take effect)"
echo "  2. Verify access with: ls -l /dev/ptp*"
echo "  3. Test PTP clock with Orleans.GpuBridge.Core:"
echo "     cd tests/Orleans.GpuBridge.Temporal.Tests"
echo "     dotnet test --filter PtpClockSource"
echo ""
echo "Troubleshooting:"
echo "  • If PTP still not accessible, try: sudo chmod 660 /dev/ptp*"
echo "  • Check group membership: groups"
echo "  • Verify udev rules: udevadm test /sys/class/ptp/ptp0"
echo "  • For Hyper-V: Ensure Enhanced Session Mode is enabled"
echo ""
