#!/bin/bash
# NeuroReader Setup Script
# Run this to install and start the backend server

set -e

echo "╔══════════════════════════════════════════╗"
echo "║       🧠 NeuroReader Setup               ║"
echo "║   Brain activation from articles          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/backend"

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'" 2>/dev/null || {
    echo "❌ Python 3.10+ is required"
    exit 1
}

# Install base dependencies
echo "📦 Installing base dependencies..."
pip install -r requirements.txt --quiet

# Try installing TRIBE v2
echo ""
echo "📦 Attempting to install TRIBE v2..."
echo "   (This requires HuggingFace access to LLaMA 3.2-3B)"
echo ""

if pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2.git" --quiet 2>/dev/null; then
    echo "✅ TRIBE v2 installed successfully"
    echo ""
    echo "⚠️  Make sure you've authenticated with HuggingFace:"
    echo "   huggingface-cli login"
    echo ""
    MODE="tribev2"
else
    echo "⚠️  TRIBE v2 installation failed (this is OK)"
    echo "   Running in heuristic mode (keyword-based analysis)"
    MODE="heuristic"
fi

# Start server
echo ""
echo "🚀 Starting NeuroReader backend on port 8421..."
echo "   Mode: $MODE"
echo ""
echo "   Next steps:"
echo "   1. Open chrome://extensions/"
echo "   2. Enable Developer mode"
echo "   3. Click 'Load unpacked' → select the extension/ folder"
echo "   4. Navigate to any article and click 🧠"
echo ""
echo "   Press Ctrl+C to stop the server"
echo "────────────────────────────────────────────"
echo ""

python3 server.py
