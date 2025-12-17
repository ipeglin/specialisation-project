import numpy as np

def fisher_r_to_z(r_matrix):
    """
    Apply Fisher r-to-z transformation to correlation matrix.

    The Fisher transformation normalizes the sampling distribution of correlation
    coefficients, making them suitable for averaging and statistical testing.

    Formula: z = 0.5 * ln((1 + r) / (1 - r)) = arctanh(r)

    Args:
        r_matrix: Correlation matrix (Pearson r values)

    Returns:
        z_matrix: Fisher z-transformed matrix
    """
    # Clip values to avoid infinity at r = ±1
    # Use conservative bounds to maintain numerical stability
    r_clipped = np.clip(r_matrix, -0.9999, 0.9999)

    # Apply Fisher transformation (arctanh is more numerically stable than the log form)
    z_matrix = np.arctanh(r_clipped)

    return z_matrix


def fisher_z_to_r(z_matrix):
    """
    Apply inverse Fisher transformation to convert z-scores back to correlations.

    Formula: r = (exp(2z) - 1) / (exp(2z) + 1) = tanh(z)

    Args:
        z_matrix: Fisher z-transformed matrix

    Returns:
        r_matrix: Correlation matrix (Pearson r values)
    """
    # Apply inverse Fisher transformation (tanh is more numerically stable)
    r_matrix = np.tanh(z_matrix)

    return r_matrix
