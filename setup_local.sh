#!/bin/bash
# Local setup script for FREE YouTube transcription

echo "🚀 AI-tweets Local Setup - FREE YouTube Transcription"
echo "=================================================="
echo

# Check if we're already in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the AI-tweets directory"
    echo "   cd AI-tweets && bash setup_local.sh"
    exit 1
fi

echo "📝 Step 1: Setting up Python virtual environment..."
if command -v python3 &> /dev/null; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "❌ Python3 not found. Please install Python 3.8+ first"
    exit 1
fi

echo
echo "📦 Step 2: Installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt
echo "✅ Dependencies installed"

echo
echo "⚙️  Step 3: Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file"
else
    echo "ℹ️  .env file already exists"
fi

echo
echo "🧪 Step 4: Testing YouTube transcript access..."
python -c "
from youtube_transcript_api import YouTubeTranscriptApi
try:
    result = YouTubeTranscriptApi().fetch('dQw4w9WgXcQ', ['en'])
    print('✅ YouTube transcripts working! Found {} segments'.format(len(result.transcript)))
    print('💰 This means FREE transcription for most podcasts!')
except Exception as e:
    print('⚠️  YouTube transcripts blocked: {}'.format(str(e)[:100]))
    print('💡 You can still use Whisper API (small cost)')
"

echo
echo "📋 Step 5: Configuration checklist..."
echo "   1. ✅ Virtual environment: Created"
echo "   2. ✅ Dependencies: Installed"
echo "   3. ✅ Config files: Ready"
echo "   4. ❓ OpenAI API Key: NEEDS SETUP"

echo
echo "🔧 NEXT STEPS:"
echo "=============="
echo "1. Add your OpenAI API key to .env:"
echo "   nano .env  # (or use your preferred editor)"
echo "   Add: OPENAI_API_KEY=your_key_here"
echo
echo "2. Test the pipeline:"
echo "   python main.py --dry-run --limit 1"
echo
echo "3. Check generated threads:"
echo "   ls output/threads/"
echo "   cat output/threads/*.md"
echo
echo "🎯 Expected Results:"
echo "• FREE YouTube transcription (90%+ cost savings)"
echo "• High-quality Twitter threads generated"
echo "• Threads saved to markdown for easy posting"
echo
echo "💡 Pro tip: Start with --dry-run to test before posting!"
echo
echo "✅ Setup complete! Happy tweeting! 🐦"