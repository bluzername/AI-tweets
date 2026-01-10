@echo off
REM Local setup script for FREE YouTube transcription (Windows)

echo 🚀 AI-tweets Local Setup - FREE YouTube Transcription
echo ==================================================
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ❌ Please run this script from the AI-tweets directory
    echo    cd AI-tweets ^&^& setup_local.bat
    pause
    exit /b 1
)

echo 📝 Step 1: Setting up Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ first
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Virtual environment created
echo.

echo 📦 Step 2: Installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt
echo ✅ Dependencies installed
echo.

echo ⚙️  Step 3: Setting up configuration...
if not exist ".env" (
    copy .env.example .env
    echo ✅ Created .env file
) else (
    echo ℹ️  .env file already exists
)
echo.

echo 🧪 Step 4: Testing YouTube transcript access...
python -c "from youtube_transcript_api import YouTubeTranscriptApi; import sys; result = YouTubeTranscriptApi().fetch('dQw4w9WgXcQ', ['en']) if True else sys.exit(1); print(f'✅ YouTube transcripts working! Found {len(result.transcript)} segments'); print('💰 This means FREE transcription for most podcasts!')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  YouTube transcripts may be blocked on this network
    echo 💡 You can still use Whisper API ^(small cost^)
)
echo.

echo 📋 Step 5: Configuration checklist...
echo    1. ✅ Virtual environment: Created
echo    2. ✅ Dependencies: Installed  
echo    3. ✅ Config files: Ready
echo    4. ❓ OpenAI API Key: NEEDS SETUP
echo.

echo 🔧 NEXT STEPS:
echo ==============
echo 1. Add your OpenAI API key to .env:
echo    notepad .env  ^(or use your preferred editor^)
echo    Add: OPENAI_API_KEY=your_key_here
echo.
echo 2. Test the pipeline:
echo    python main.py --dry-run --limit 1
echo.
echo 3. Check generated threads:
echo    dir output\threads\
echo    type output\threads\*.md
echo.
echo 🎯 Expected Results:
echo • FREE YouTube transcription ^(90%% cost savings^)
echo • High-quality Twitter threads generated
echo • Threads saved to markdown for easy posting
echo.
echo 💡 Pro tip: Start with --dry-run to test before posting!
echo.
echo ✅ Setup complete! Happy tweeting! 🐦
echo.
pause