#!/bin/bash

# Setup script for the Data Science Pipeline application
# This script sets up the environment and initializes the application

set -e

echo "🚀 Setting up Data Science Pipeline Application"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p data/train data/test data/processed models logs config

# Set up configuration
echo "⚙️  Setting up configuration..."
if [ ! -f "config/config.yaml" ]; then
    echo "📝 Configuration file already exists"
else
    echo "✅ Using existing configuration"
fi

# Initialize Prefect
echo "🌊 Initializing Prefect..."
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

# Create sample data if it doesn't exist
echo "📊 Setting up sample data..."
python scripts/create_sample_data.py

# Set up pre-commit hooks (if in development)
if [ "$1" = "--dev" ]; then
    echo "🔗 Setting up pre-commit hooks..."
    pre-commit install
fi

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Start Prefect server: prefect server start"
echo "2. Run the API server: python -m uvicorn src.api.api_server:app --reload"
echo "3. Deploy workflows: python src/workflows/training_flow.py"
echo "4. Access the API at: http://localhost:8000"
echo "5. Access Prefect UI at: http://localhost:4200"
echo ""
echo "For Docker deployment:"
echo "docker-compose up -d"