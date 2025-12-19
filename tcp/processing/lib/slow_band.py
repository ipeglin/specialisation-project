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

def get_band_number(frequency):
    """Determine which slow-band a frequency belongs to."""
    if 0.0 < frequency <= 0.01:
        return "6"
    elif 0.01 < frequency <= 0.027:
        return "5"
    elif 0.027 < frequency <= 0.073:
        return "4"
    elif 0.073 < frequency <= 0.198:
        return "3"
    elif 0.198 < frequency <= 0.5:
        return "2"
    elif 0.5 < frequency <= 0.75:
        return "1"
    else:
        return None  # Outside slow-band range