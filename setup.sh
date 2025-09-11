#!/bin/bash
# Setup script for specialisation project
# This script creates a virtual environment and installs the required Python packages

echo "Setting up specialisation project dependencies..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install requirements
echo "Installing Python packages in virtual environment..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify key packages are installed
echo "Verifying installation..."
python3 -c "import numpy, pandas, h5py, nibabel; print('✓ Core packages installed successfully')"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To test the configuration system:"
echo "  source venv/bin/activate && python3 config/paths.py"
echo ""
echo "To run the TCP preprocessing script:"
echo "  source venv/bin/activate && python3 tcp/preprocessing/extract_subjects.py"
echo ""
echo "To deactivate the virtual environment when done:"
echo "  deactivate"