#!/bin/bash
# TCP Preprocessing Pipeline Runner
# This script dynamically resolves paths based on the current environment (.env file)

conda activate masters_thesis

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Dynamically resolve HCP paths using Python config system
HCP_PARCELLATED_OUTPUT=$(python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from config.paths import get_data_path; print(get_data_path('hcp_output'))")

# Default HCP root path (override with environment variable if needed)
HCP_ROOT="${HCP_ROOT:-/cluster/projects/itea_lille-ie/Transdiagnostic/output/hcp_output}"

echo "Running TCP Preprocessing Pipeline"
echo "Project root: ${PROJECT_ROOT}"
echo "HCP root: ${HCP_ROOT}"
echo "HCP parcellated output: ${HCP_PARCELLATED_OUTPUT}"
echo ""

cd "${PROJECT_ROOT}" || exit 1

python3 tcp/preprocessing/run_pipeline.py \
--data-source-type combined \
--hcp-root "${HCP_ROOT}" \
--hcp-parcellated-output "${HCP_PARCELLATED_OUTPUT}" \
--duplicate-resolution prefer_hcp \
--ignore-completed \
--analysis-group primary \
--data-type timeseries
