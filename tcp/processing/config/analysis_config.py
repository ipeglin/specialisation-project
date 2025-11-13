"""
Configuration classes for functional connectivity analysis.

This module defines configuration dataclasses that support dependency injection
and make the analysis pipeline more testable and maintainable.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class SignalProcessingConfig:
    """Configuration for signal processing parameters.

    Attributes:
        tr_seconds: Repetition time in seconds (fMRI temporal resolution)
        filter_order: Butterworth filter order for low-pass filtering
        cutoff_frequency_hz: Cutoff frequency in Hz for low-pass filter
    """
    tr_seconds: float = 0.8
    filter_order: int = 2
    cutoff_frequency_hz: float = 0.2

    @property
    def sampling_rate_hz(self) -> float:
        """Calculate sampling rate from TR."""
        return 1.0 / self.tr_seconds

    @property
    def nyquist_frequency_hz(self) -> float:
        """Calculate Nyquist frequency (half of sampling rate)."""
        return 0.5 * self.sampling_rate_hz


@dataclass
class AtlasConfig:
    """Configuration for atlas paths and ROI selections.

    Attributes:
        cortical_lut_path: Path to cortical atlas lookup table
        subcortical_lut_path: Path to subcortical atlas lookup table
        cortical_rois: List of cortical ROI names to extract
        subcortical_rois: List of subcortical ROI names to extract
    """
    cortical_lut_path: Path
    subcortical_lut_path: Path
    cortical_rois: List[str] = field(default_factory=lambda: ['PFCm', 'PFCv'])
    subcortical_rois: List[str] = field(default_factory=lambda: ['AMY'])

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if not isinstance(self.cortical_lut_path, Path):
            self.cortical_lut_path = Path(self.cortical_lut_path)
        if not isinstance(self.subcortical_lut_path, Path):
            self.subcortical_lut_path = Path(self.subcortical_lut_path)


@dataclass
class OutputConfig:
    """Configuration for output paths and formats.

    Attributes:
        base_output_dir: Base directory for all analysis outputs
        csv_export_dir: Directory for CSV exports (relative to base)
        figures_dir: Directory for saved figures (relative to base)
        save_figures: Whether to save figures to disk
        figure_format: Format for saved figures (svg, png, pdf)
        figure_dpi: DPI for saved figures
    """
    base_output_dir: Path
    csv_export_dir: str = 'fc_analysis/static_fc'
    figures_dir: str = 'fc_analysis/figures'
    save_figures: bool = False
    figure_format: str = 'svg'
    figure_dpi: int = 300

    def __post_init__(self):
        """Ensure base_output_dir is a Path object."""
        if not isinstance(self.base_output_dir, Path):
            self.base_output_dir = Path(self.base_output_dir)

    @property
    def csv_path(self) -> Path:
        """Get full path to CSV export directory."""
        return self.base_output_dir / self.csv_export_dir

    @property
    def figures_path(self) -> Path:
        """Get full path to figures directory."""
        return self.base_output_dir / self.figures_dir


@dataclass
class PlottingConfig:
    """Configuration for plotting behavior.

    Attributes:
        create_plots: Whether to create plots
        show_plots: Whether to display plots interactively
        mask_nonsignificant: Whether to mask non-significant correlations in FC matrix
        significance_alpha: Significance threshold for statistical tests
    """
    create_plots: bool = True
    show_plots: bool = True
    mask_nonsignificant: bool = False
    significance_alpha: float = 0.05


@dataclass
class ProcessingConfig:
    """Master configuration for the analysis pipeline.

    This class aggregates all configuration objects needed for the
    functional connectivity analysis pipeline.

    Note: Signal processing parameters (TR, filter settings) are dataset-specific
    constants and are hardcoded in the analysis functions rather than configured here.

    Attributes:
        atlas_config: Configuration for atlases and ROI selection
        output_config: Configuration for output paths and formats
        plotting_config: Configuration for plotting behavior
        verbose: Whether to print verbose output during processing
    """
    atlas_config: AtlasConfig
    output_config: OutputConfig
    plotting_config: PlottingConfig
    verbose: bool = True

    @classmethod
    def create_default(cls, script_dir: Path, output_base: Path) -> 'ProcessingConfig':
        """Create a default configuration with sensible defaults.

        Args:
            script_dir: Directory containing the fc_mvp.py script (for relative atlas paths)
            output_base: Base directory for analysis outputs

        Returns:
            ProcessingConfig with default settings
        """
        atlas_config = AtlasConfig(
            cortical_lut_path=script_dir / 'parcellations/cortical/yeo17/400Parcels_Yeo2011_17Networks_info.txt',
            subcortical_lut_path=script_dir / 'parcellations/subcortical/tian/Tian_Subcortex_S2_3T_label.txt',
            cortical_rois=['PFCm', 'PFCv'],
            subcortical_rois=['AMY']
        )

        output_config = OutputConfig(
            base_output_dir=output_base,
            save_figures=False
        )

        plotting_config = PlottingConfig(
            create_plots=True,
            show_plots=True,
            mask_nonsignificant=False
        )

        return cls(
            atlas_config=atlas_config,
            output_config=output_config,
            plotting_config=plotting_config,
            verbose=True
        )
