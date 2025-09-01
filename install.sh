#!/bin/bash

# Installation script for SHAP and LIME Model Interpretability Examples
# This script will install all required dependencies with compatible versions

echo "🚀 Installing SHAP and LIME Model Interpretability Examples"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is required but not installed. Please install pip first."
    exit 1
fi

echo "✓ Python and pip are available"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing required packages..."
echo "   This may take a few minutes as some packages are large (TensorFlow, etc.)"

if pip install -r requirements.txt; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Installation failed. Please check the error messages above."
    echo ""
    echo "💡 Common solutions:"
    echo "   1. Try running with: pip install --user -r requirements.txt"
    echo "   2. Create a virtual environment first:"
    echo "      python -m venv venv"
    echo "      source venv/bin/activate  # Linux/Mac"
    echo "      # or venv\\Scripts\\activate  # Windows"
    echo "      pip install -r requirements.txt"
    echo "   3. If you have conda: conda create -n shap_lime python=3.10"
    exit 1
fi

# Test the installation
echo ""
echo "🧪 Testing installation..."
if python -c "import tensorflow, shap, lime, prophet; print('✅ All packages imported successfully!')"; then
    echo "✅ Installation test passed!"
else
    echo "❌ Installation test failed. Some packages may not be working correctly."
    exit 1
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📚 Quick Start:"
echo "   1. Basic demo:           python basic_shap_lime_demo.py"
echo "   2. LSTM example:         python lstm_shap_lime_examples.py"
echo "   3. Prophet example:      python prophet_shap_lime_examples.py"
echo "   4. Complete demo:        python comprehensive_interpretability_demo.py"
echo ""
echo "📁 Generated files will be saved to the current directory."
echo "============================================================"
