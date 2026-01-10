#!/bin/bash
# Installation script for Podcasts TLDR systemd service

set -e

echo "🚀 Installing Podcasts TLDR as a systemd service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

# Configuration
INSTALL_DIR="/opt/podcasts-tldr"
SERVICE_USER="podcasts"
SERVICE_FILE="podcasts-tldr.service"

# Create service user if doesn't exist
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "👤 Creating service user: $SERVICE_USER"
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
fi

# Create installation directory
echo "📁 Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Copy files
echo "📋 Copying application files..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/" || {
    echo "❌ Failed to copy files. Make sure you're running this from the repo directory."
    exit 1
}

# Create required directories
echo "📁 Creating required directories..."
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/output"
mkdir -p "$INSTALL_DIR/cache"
mkdir -p "$INSTALL_DIR/backups"

# Set ownership
echo "🔐 Setting ownership..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
su - "$SERVICE_USER" -s /bin/bash -c "cd $INSTALL_DIR && python3 -m venv venv"
su - "$SERVICE_USER" -s /bin/bash -c "cd $INSTALL_DIR && venv/bin/pip install -r requirements.txt"

# Check if .env exists
if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo "⚙️  Creating .env from template..."
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"
    echo "⚠️  IMPORTANT: Edit $INSTALL_DIR/.env with your API keys before starting the service!"
fi

# Install systemd service
echo "⚙️  Installing systemd service..."
cp "$INSTALL_DIR/deployment/systemd/$SERVICE_FILE" "/etc/systemd/system/$SERVICE_FILE"

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Instructions
echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit the configuration file with your API keys:"
echo "   sudo nano $INSTALL_DIR/.env"
echo ""
echo "2. Enable the service to start on boot:"
echo "   sudo systemctl enable podcasts-tldr"
echo ""
echo "3. Start the service:"
echo "   sudo systemctl start podcasts-tldr"
echo ""
echo "4. Check service status:"
echo "   sudo systemctl status podcasts-tldr"
echo ""
echo "5. View logs:"
echo "   sudo journalctl -u podcasts-tldr -f"
echo ""
echo "🎯 The service will run continuously in the background!"
