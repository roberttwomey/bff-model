#!/usr/bin/env bash

# --- CONFIG ---
MAC="A0:0C:E2:12:BC:82"   # replace with your device's MAC

# --- BLUETOOTH CONNECT ---
bluetoothctl << EOF
power on
agent on
default-agent
trust $MAC
connect $MAC
EOF

# --- SELECT HEADSET PROFILE ---
# (forces microphone-capable headset mode)
CARD=$(pactl list cards short | grep "$MAC" | awk '{print $1}')

if [ -n "$CARD" ]; then
    pactl set-card-profile "$CARD" headset-head-unit
fi

echo "Connected $MAC in headset (HFP/HSP) mode."
