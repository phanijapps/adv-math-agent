#!/bin/bash

# Advanced Math Agent Setup Script

echo "🔢 Setting up Advanced Math Agent..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your API keys before running the agent!"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs exports

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenRouter API key"
echo "2. Optionally add Wolfram Alpha API key for advanced calculations"
echo "3. Run: python main.py --help"
echo "4. Or run: python main.py for interactive mode"
