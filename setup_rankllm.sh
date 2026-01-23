#!/bin/bash
# Setup script for all_in_one_rankllm.py
# This script helps set up the environment and run initial tests

set -e  # Exit on error

echo "================================"
echo "RankLLM Pipeline Setup Script"
echo "================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "   Error: pip3 not found. Please install pip3."
    exit 1
fi
echo "   ✓ pip3 is available"
echo ""

# Install dependencies
echo "2. Installing dependencies..."
echo "   Note: This may take several minutes..."

if [ -f "requirements_rankllm.txt" ]; then
    pip3 install -r requirements_rankllm.txt --quiet
    echo "   ✓ Core dependencies installed"
else
    echo "   Warning: requirements_rankllm.txt not found"
fi

# Try to install rank_llm from source
echo ""
echo "3. Installing rank_llm library..."
echo "   Note: This will clone and install from GitHub..."

if [ -d "rank_llm" ]; then
    echo "   rank_llm directory already exists, skipping clone"
else
    if git clone https://github.com/castorini/rank_llm.git 2>/dev/null; then
        echo "   ✓ Cloned rank_llm repository"
        cd rank_llm
        pip3 install -e . --quiet 2>/dev/null || echo "   Warning: rank_llm installation failed (optional)"
        cd ..
    else
        echo "   Warning: Could not clone rank_llm (optional dependency)"
    fi
fi

echo ""
echo "4. Creating necessary directories..."
mkdir -p datasets
mkdir -p rundicts
mkdir -p rankllm_results
mkdir -p rankllm_results/emissions
echo "   ✓ Directories created"

echo ""
echo "5. Verifying installation..."
python3 -c "
import sys
missing = []
try:
    import torch
    print('   ✓ PyTorch installed')
except ImportError:
    missing.append('torch')
    print('   ✗ PyTorch not installed')

try:
    import transformers
    print('   ✓ Transformers installed')
except ImportError:
    missing.append('transformers')
    print('   ✗ Transformers not installed')

try:
    from ranx import evaluate
    print('   ✓ ranx installed')
except ImportError:
    missing.append('ranx')
    print('   ✗ ranx not installed')

try:
    from codecarbon import OfflineEmissionsTracker
    print('   ✓ CodeCarbon installed')
except ImportError:
    missing.append('codecarbon')
    print('   ✗ CodeCarbon not installed')

try:
    from beir.datasets.data_loader import GenericDataLoader
    print('   ✓ BEIR installed')
except ImportError:
    missing.append('beir')
    print('   ✗ BEIR not installed')

if missing:
    print(f'\\n   Warning: Missing packages: {missing}')
    print('   Run: pip3 install ' + ' '.join(missing))
    sys.exit(1)
"

echo ""
echo "6. Downloading sample dataset (SciFact)..."
python3 -c "
from beir import util
import os

dataset = 'scifact'
url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
data_path = 'datasets'

if not os.path.exists(f'{data_path}/{dataset}'):
    print(f'   Downloading {dataset}...')
    try:
        util.download_and_unzip(url, data_path)
        print(f'   ✓ {dataset} dataset downloaded')
    except Exception as e:
        print(f'   Warning: Could not download dataset: {e}')
        print('   You may need to download datasets manually')
else:
    print(f'   ✓ {dataset} dataset already exists')
" 2>/dev/null || echo "   Warning: Could not download dataset automatically"

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Ensure you have pre-computed BM25 results in rundicts/"
echo "   - Format: rundicts/rundict_{dataset}_bm25.json"
echo "   - Or use pyserini to compute them"
echo ""
echo "2. Ensure you have model files in models/ directory"
echo "   - Or modify model paths in the script"
echo ""
echo "3. Run the pipeline:"
echo "   python3 all_in_one_rankllm.py --dataset scifact --filter_method bert_index"
echo ""
echo "4. Or run the example script:"
echo "   python3 example_rankllm_usage.py"
echo ""
echo "For more information, see all_in_one_rankllm_README.md"
echo "================================"
