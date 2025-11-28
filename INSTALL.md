# Installation Guide - TCP Preprocessing Pipeline

This guide covers environment setup across Windows 11, macOS, and Linux/CentOS platforms.

## Quick Start

Choose your operating system and follow the corresponding section below.

### Windows 11

```powershell
# 1. Install scoop (package manager for Windows)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# 2. Install git-annex and 7zip via scoop
scoop install git-annex 7zip

# 3. Create conda environment
conda env create -f environment-windows.yml
conda activate masters_thesis

# 4. (Optional) If scipy is needed
pip install scipy
```

### macOS (Intel or Apple Silicon)

```bash
# Create conda environment - works for both architectures
conda env create -f environment-macos.yml
conda activate masters_thesis
```

### Linux/CentOS (including IDUN cluster)

```bash
# On IDUN cluster, load Anaconda module first
module load Anaconda3

# Create conda environment
conda env create -f environment-linux.yml
conda activate masters_thesis
```

---

## Detailed Platform Instructions

### Windows 11

#### Prerequisites

1. **Miniconda or Anaconda**
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Install with default settings

2. **Scoop Package Manager**
   - Open PowerShell as Administrator
   - Run:
     ```powershell
     Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
     irm get.scoop.sh | iex
     ```

#### Installation Steps

1. **Install Windows-Specific Tools**
   ```powershell
   scoop install git-annex 7zip
   ```
   
   These tools are not available via conda on Windows:
   - `git-annex`: Required for datalad dataset management
   - `7zip`: Required for archive extraction

2. **Create Conda Environment**
   ```powershell
   conda env create -f environment-windows.yml
   ```

3. **Activate Environment**
   ```powershell
   conda activate masters_thesis
   ```

4. **Optional: Install scipy**
   ```powershell
   # Only if you need scipy (excluded due to Windows compatibility)
   pip install scipy
   ```

#### Verification

```powershell
# Check Python version
python --version  # Should show Python 3.11.x

# Check critical packages
python -c "import numpy, pandas, h5py, datalad; print('Success!')"

# Check git-annex
git annex version
```

---

### macOS

#### Prerequisites

1. **Miniconda or Anaconda**
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Choose the appropriate installer:
     - **Apple Silicon (M1/M2/M3)**: arm64 installer
     - **Intel Mac**: x86_64 installer

#### Installation Steps

1. **Create Conda Environment**
   ```bash
   conda env create -f environment-macos.yml
   ```

2. **Activate Environment**
   ```bash
   conda activate masters_thesis
   ```

#### Verification

```bash
# Check Python version
python --version  # Should show Python 3.11.x

# Check critical packages
python -c "import numpy, pandas, scipy, h5py, datalad; print('Success!')"

# Check git-annex
git annex version
```

#### Notes

- The environment file works for both Intel and Apple Silicon Macs
- All packages are available via conda-forge on macOS
- `git-annex` installs successfully on both architectures

---

### Linux/CentOS

#### Prerequisites

1. **Conda** (Miniconda or Anaconda)
   - On IDUN cluster: Available via `module load Anaconda3`
   - On personal Linux: Download from https://docs.conda.io/en/latest/miniconda.html

#### Installation Steps

**On IDUN Cluster:**

```bash
# Load Anaconda module
module load Anaconda3

# Create environment
conda env create -f environment-linux.yml

# Activate environment
conda activate masters_thesis
```

**On Personal Linux:**

```bash
# Create environment
conda env create -f environment-linux.yml

# Activate environment  
conda activate masters_thesis
```

#### Verification

```bash
# Check Python version
python --version  # Should show Python 3.11.x

# Check critical packages
python -c "import numpy, pandas, scipy, h5py, datalad; print('Success!')"

# Check git-annex
git annex version
```

#### Notes

- Includes `secretstorage` for Linux keyring functionality
- Compatible with CentOS, Ubuntu, Debian, and other distributions
- Tested on IDUN cluster

---

## Adding New Dependencies

When you need to add a new package to the project:

### Option 1: Add to Base Environment (Cross-Platform)

1. **Edit `environment.yml`**
   ```yaml
   dependencies:
     # ... existing packages ...
     - your-new-package
   ```

2. **Test on your platform**
   ```bash
   conda env update -f environment.yml --prune
   ```

3. **Update platform-specific files if needed**
   - If the package doesn't work on a specific platform, exclude it in that platform's file
   - Add a comment explaining why it's excluded

4. **Commit changes**
   ```bash
   git add environment.yml environment-*.yml
   git commit -m "deps: add your-new-package"
   ```

### Option 2: Platform-Specific Package

If a package only works/is needed on one platform:

1. **Edit only the relevant platform file**
   - `environment-windows.yml` for Windows
   - `environment-macos.yml` for macOS  
   - `environment-linux.yml` for Linux

2. **Add a comment explaining the platform specificity**

3. **Test and commit**

---

## Troubleshooting

### Issue: "PackagesNotFoundError"

**Symptom**: Conda can't find a package during environment creation.

**Solution**:
1. Check if you're using the correct platform file
2. Try updating conda: `conda update conda`
3. Clear conda cache: `conda clean --all`
4. Try again

### Issue: git-annex not working on Windows

**Symptom**: `git annex` command not found.

**Solution**:
1. Ensure scoop is installed
2. Install git-annex via scoop: `scoop install git-annex`
3. Restart terminal/PowerShell
4. Verify: `git annex version`

### Issue: scipy import error on Windows

**Symptom**: `ImportError: cannot import name 'scipy'`

**Solution**:
scipy is excluded from the Windows conda environment due to compatibility issues.
Install via pip instead:
```powershell
conda activate masters_thesis
pip install scipy
```

### Issue: Environment activation fails

**Symptom**: `conda activate masters_thesis` doesn't work.

**Solution**:
1. Initialize conda for your shell:
   ```bash
   conda init bash  # Or: zsh, powershell, cmd.exe
   ```
2. Restart your shell
3. Try activating again

### Issue: Slow environment creation

**Symptom**: `conda env create` takes a very long time.

**Solution**:
1. Use mamba for faster dependency resolution:
   ```bash
   conda install -c conda-forge mamba
   mamba env create -f environment-<platform>.yml
   ```

### Issue: "Permission denied" on Linux/cluster

**Symptom**: Cannot write to conda directories.

**Solution**:
1. Create environment in your home directory:
   ```bash
   conda env create -f environment-linux.yml --prefix ~/envs/masters_thesis
   conda activate ~/envs/masters_thesis
   ```

---

## Environment Files Reference

| File | Platform | Description |
|------|----------|-------------|
| `environment.yml` | Cross-platform | Base specification |
| `environment-windows.yml` | Windows 11 | Excludes git-annex, p7zip, scipy, secretstorage |
| `environment-macos.yml` | macOS | Includes all packages |
| `environment-linux.yml` | Linux/CentOS | Includes secretstorage for keyring |
| `conda-lock-macos-arm64.txt` | macOS ARM64 | Locked snapshot (reference only) |

---

## Updating an Existing Environment

If you already have the `masters_thesis` environment and need to update it:

```bash
# Activate environment
conda activate masters_thesis

# Update from file
conda env update -f environment-<platform>.yml --prune

# The --prune flag removes packages not in the file
```

---

## Removing the Environment

If you need to start fresh:

```bash
# Deactivate if currently active
conda deactivate

# Remove environment
conda env remove -n masters_thesis

# Recreate from scratch
conda env create -f environment-<platform>.yml
```

---

## Platform-Specific Notes

### Windows
- **git-annex** and **7zip** MUST be installed via scoop (not conda)
- **scipy** may need pip installation if required
- Use PowerShell or Git Bash for commands

### macOS
- Works identically on Intel and Apple Silicon
- All packages available via conda-forge
- Homebrew NOT required (everything via conda)

### Linux/CentOS
- On IDUN: Use `module load Anaconda3` first
- Includes `secretstorage` for proper keyring functionality
- Compatible with systemd-based distributions

---

## Support

For issues:
1. Check this troubleshooting section
2. Verify you're using the correct platform-specific file
3. Try removing and recreating the environment
4. Check conda version: `conda --version` (should be ≥4.10)

---

**Last Updated**: 2025-01-28  
**Conda Version**: Tested with conda 23.x+  
**Python Version**: 3.11.x
