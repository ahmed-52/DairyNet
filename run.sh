#!/bin/bash

# Quick start script for FlavorGraph web app

echo "🚀 Starting FlavorGraph web application..."
echo ""

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Starting Flask server..."
echo "🌐 Open your browser and navigate to: http://localhost:5000"
echo ""

python app.py

