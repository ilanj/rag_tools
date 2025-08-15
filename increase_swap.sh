#!/bin/bash
# Script to permanently set swap size in Ubuntu
# Usage: sudo ./set_swap.sh 8G

SWAP_SIZE=$1

if [ -z "$SWAP_SIZE" ]; then
    echo "❌ Please provide swap size. Example: sudo $0 8G"
    exit 1
fi

echo "🔍 Current swap:"
swapon --show
free -h

echo "🚫 Turning off old swap..."
sudo swapoff -a

# Remove old swap file if exists
if [ -f /swapfile ]; then
    echo "🗑 Removing old /swapfile..."
    sudo rm /swapfile
fi

echo "📦 Creating new swap file of size $SWAP_SIZE..."
sudo fallocate -l $SWAP_SIZE /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=$(( ${SWAP_SIZE//[!0-9]/} * 1024 )) status=progress

echo "🔒 Setting permissions..."
sudo chmod 600 /swapfile

echo "⚙️ Making swap..."
sudo mkswap /swapfile

echo "✅ Enabling swap..."
sudo swapon /swapfile

echo "📌 Updating /etc/fstab for persistence..."
sudo sed -i '/\/swapfile/d' /etc/fstab
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab

echo "🔧 Setting swappiness to 10..."
sudo sysctl vm.swappiness=10
grep -q "vm.swappiness" /etc/sysctl.conf || echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf

echo "🔍 New swap status:"
swapon --show
free -h

echo "🎉 Swap size set to $SWAP_SIZE permanently!"
