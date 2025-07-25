#!/bin/bash
# Squash Stats App Startup Script

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate virtual environment and run the app
source venv/bin/activate
streamlit run "Welcome_to_Pointless_Squash_Stats.py"
