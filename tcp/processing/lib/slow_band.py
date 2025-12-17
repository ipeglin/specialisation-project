def get_frequency_range(band_key):
    """
    Return frequency range string for slow-band visualization.

    Args:
        band_key: str, band identifier ('1', '2', '3', '4', '5', or '6')

    Returns:
        str: Frequency range in Hz (e.g., '0.027-0.073 Hz')
    """
    ranges = {
        '6': '0.000-0.010 Hz',
        '5': '0.010-0.027 Hz',
        '4': '0.027-0.073 Hz',
        '3': '0.073-0.198 Hz',
        '2': '0.198-0.500 Hz',
        '1': '0.500-0.750 Hz',
    }
    return ranges.get(band_key, 'Unknown')
