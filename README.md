# Specialisation Project - Enhanced Cross-Platform Configuration

This repository contains analysis scripts for neuroimaging data that work across multiple platforms and environments with flexible output organization.

## Configuration System

The project uses an automatic cross-platform configuration system with **organized output categories** that detects your environment and sets appropriate paths for:

- **macOS** (M1 MacBook)
- **Windows 11** (Desktop)
- **IDUN HPC Cluster** (CentOS)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Quick Setup
```bash
# Automated setup (recommended)
./setup.sh

# Or manual setup with virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test the configuration
source venv/bin/activate && python3 config/paths.py
```

### System Installation (if you prefer)
```bash
pip3 install --user numpy pandas h5py nibabel matplotlib seaborn scipy scikit-learn
```

### Virtual Environment Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Run your scripts
python3 config/paths.py
python3 tcp/preprocessing/extract_subjects.py

# Deactivate when done
deactivate
```

## 🎯 Quick Start

### Basic Usage
All scripts automatically detect your platform and use appropriate paths:

```python
from config.paths import get_data_path, get_analysis_path

# These automatically use the correct paths for your platform
dataset_path = get_data_path('ds005237')
output_dir = get_analysis_path('connectivity_study')
```

### Flexible Output Organization
Different script types automatically use appropriate output directories:

```python
from config.paths import (
    get_preprocessing_path,    # For data preprocessing
    get_analysis_path,        # For analysis results
    get_figures_path,         # For plots and visualizations
    get_models_path,          # For trained models
    get_reports_path,         # For documentation
    get_script_output_path    # For script-specific organization
)

# Organized by output type
preprocessing_dir = get_preprocessing_path('tcp_clean_data')
analysis_dir = get_analysis_path('mdd_connectivity')
figures_dir = get_figures_path('brain_plots')
models_dir = get_models_path('neural_networks')

# Script-specific intelligent organization
tcp_output = get_script_output_path('tcp_preprocessing', 'extract_subjects')
mdd_results = get_script_output_path('mdd_analysis', 'connectivity', 'results')
plot_output = get_script_output_path('visualization', 'brain_plots')
```

## 📁 Output Organization

### Current Platform (macOS) Paths:
- **Data**: `/Users/ipeglin/Documents/Data/`
- **Preprocessing**: `/Users/ipeglin/Documents/Preprocessing/`
- **Analysis**: `/Users/ipeglin/Documents/Analysis/`
- **Figures**: `/Users/ipeglin/Documents/Figures/`
- **Models**: `/Users/ipeglin/Documents/Models/`
- **Reports**: `/Users/ipeglin/Documents/Reports/`

### Platform-Specific Organization:
- **macOS**: `Documents/[Category]/`
- **Windows**: `Documents/[Category]/`
- **IDUN Cluster**: `/cluster/work/ianpe/[category]/`

### Script-Type Detection:
The system automatically routes scripts to appropriate output categories:
- Scripts with "preprocess" → **Preprocessing** directory
- Scripts with "analysis" → **Analysis** directory
- Scripts with "plot", "vis", "fig" → **Figures** directory
- Scripts with "model" → **Models** directory
- Scripts with "report" → **Reports** directory

## 🔧 Configuration Options

### 1. Environment Variables
Create a `.env` file (see `.env.example`):
```bash
PROJECT_PREPROCESSING_BASE=/path/to/preprocessing
PROJECT_ANALYSIS_BASE=/path/to/analysis
PROJECT_FIGURES_BASE=/path/to/figures
```

### 2. Direct Path Specification
```python
# Override for specific use cases
output = get_script_output_path('custom_analysis', 'my_experiment', 'trial_1')
```

### 3. Check Current Configuration
```bash
python3 config/paths.py
```

## Project Structure

```
specialisation-project/
├── config/
│   ├── paths.py              # Core configuration module
│   ├── default_config.json   # Platform-specific defaults
│   └── .env.example          # Template for overrides
├── tcp/
│   └── preprocessing/
│       ├── extract_subjects.py    # Extract MDD subjects
│       ├── utils/
│       │   └── data_loader.py     # Data loading utilities
│       └── examples/
│           └── mdd_analysis.py    # MDD analysis workflow
└── README.md
```

## 💡 Usage Examples

### TCP Dataset Analysis
```bash
# Extract subjects - automatically routed to preprocessing directory
python3 tcp/preprocessing/extract_subjects.py

# Run MDD analysis - automatically routed to analysis directory
python3 tcp/preprocessing/examples/mdd_analysis.py
```

### Writing New Scripts

#### Simple Analysis Script
```python
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root))

from config.paths import get_data_path, get_analysis_path

# Your script automatically works on all platforms!
data_dir = get_data_path('your_dataset')
output_dir = get_analysis_path('connectivity_analysis')
```

#### Preprocessing Script
```python
from config.paths import get_script_output_path, ensure_path_exists

# Automatically goes to preprocessing directory with organized subdirs
output_dir = get_script_output_path('tcp_preprocessing', 'clean_data')
ensure_path_exists(output_dir)  # Creates directory if needed
```

#### Visualization Script
```python
from config.paths import get_script_output_path

# Automatically goes to figures directory
plot_dir = get_script_output_path('visualization', 'brain_connectivity')
model_plot_dir = get_script_output_path('model_visualization', 'training_curves')
```

### Advanced Usage
```python
from config.paths import *

# Different scripts, organized outputs
preprocessing_output = get_script_output_path('data_cleaning', 'tcp_subjects')
analysis_output = get_script_output_path('mdd_analysis', 'results', 'final')
figure_output = get_script_output_path('brain_plots', 'connectivity_maps')

# Create multiple related outputs
base_path = get_analysis_path('mdd_connectivity_study')
results_path = base_path / 'results'
intermediate_path = base_path / 'intermediate'
```

## 🖥️ Platform Details

### macOS M1 MacBook
```
Code:           /Users/ipeglin/Git/
Data:           /Users/ipeglin/Documents/Data/
Preprocessing:  /Users/ipeglin/Documents/Preprocessing/
Analysis:       /Users/ipeglin/Documents/Analysis/
Figures:        /Users/ipeglin/Documents/Figures/
Models:         /Users/ipeglin/Documents/Models/
Reports:        /Users/ipeglin/Documents/Reports/
```

### Windows 11 Desktop
```
Code:           C:/Users/ipeglin/Git/
Data:           C:/Users/ipeglin/Documents/Data/
Preprocessing:  C:/Users/ipeglin/Documents/Preprocessing/
Analysis:       C:/Users/ipeglin/Documents/Analysis/
Figures:        C:/Users/ipeglin/Documents/Figures/
Models:         C:/Users/ipeglin/Documents/Models/
Reports:        C:/Users/ipeglin/Documents/Reports/
```

### IDUN HPC Cluster
```
Code:           /cluster/home/ianpe/
Data:           /cluster/work/ianpe/
Preprocessing:  /cluster/work/ianpe/preprocessing/
Analysis:       /cluster/work/ianpe/analysis/
Figures:        /cluster/work/ianpe/figures/
Models:         /cluster/work/ianpe/models/
Reports:        /cluster/work/ianpe/reports/
```

## 🔧 Environment Variables Reference

### All Available Variables:
```bash
PROJECT_CODE_BASE=/path/to/code           # Code/repositories
PROJECT_DATA_BASE=/path/to/data           # Datasets
PROJECT_PREPROCESSING_BASE=/path/to/prep  # Data preprocessing outputs
PROJECT_ANALYSIS_BASE=/path/to/analysis   # Analysis results
PROJECT_FIGURES_BASE=/path/to/figures     # Plots and visualizations
PROJECT_MODELS_BASE=/path/to/models       # Trained models
PROJECT_REPORTS_BASE=/path/to/reports     # Documentation
PROJECT_TEMP_BASE=/path/to/temp           # Temporary files
PROJECT_OUTPUT_BASE=/path/to/output       # Backward compatibility
```
