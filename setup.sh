#!/bin/bash
# Setup script for Temporal GFN project

# Parse command line arguments
DISABLE_WANDB=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --disable-wandb)
      DISABLE_WANDB=true
      shift
      ;;
    *)
      echo "Unknown option: $key"
      echo "Usage: ./setup.sh [--disable-wandb]"
      exit 1
      ;;
  esac
done

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py
chmod +x scripts/*.sh

# Create necessary directories
echo "Creating directories..."
mkdir -p results
mkdir -p data
mkdir -p logs
mkdir -p src/temporal_gfn/utils

# Run test script
echo "Running functionality tests..."
if [ "$DISABLE_WANDB" = true ]; then
    echo "Running tests with W&B logging disabled"
    python scripts/test_functionality.py --skip_venv_check
else
    echo "Running tests with W&B logging enabled"
    python scripts/test_functionality.py --use_wandb --skip_venv_check
fi

echo -e "\n====================================="
echo "✓ Setup completed successfully!"
echo "====================================="
echo -e "\n⚠️  IMPORTANT: For future sessions, always activate the virtual environment first:"
echo -e "   source venv/bin/activate"
echo -e "\nThis ensures all dependencies are available when running scripts."
echo -e "\nCurrent session already has the virtual environment activated." 