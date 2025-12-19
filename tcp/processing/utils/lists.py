import numpy as np


# from https://stackoverflow.com/a/312464 under licence CC BY-SA 4.0
# By: Ned Batchelder
# Edited by: Mateen Ulhaq & Ian Philip Eglin
# Edits made: Changed parameter names to be more descriptive
def chunks(lst, num_chunks):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), num_chunks):
       yield lst[i:i + num_chunks]

def split_by_sizes(data, sizes, axis=0):
    """
    Splits a list or numpy array into chunks based on a list of sizes.

    Args:
        data: The input list or np.ndarray.
    	sizes: A list of integers representing the size of each chunk.
        axis: The axis along which to split (relevant for multidimensional data).

    Returns:
        A list of chunks (sub-arrays or sub-lists).
    """
    # Convert to numpy array for consistent multidimensional handling
    obj = np.asanyarray(data)

    # Calculate the split indices (cumulative sum of sizes)
    # We exclude the last sum because np.split uses indices, not sizes
    indices = np.cumsum(sizes)[:-1]

    # Perform the split
    chunks = np.split(obj, indices, axis=axis)

    # Check for Edge Cases:
    # 1. Total sizes < length of data: np.split keeps the remainder in the last chunk.
    # 2. Total sizes > length of data: np.split creates empty arrays for the overflow.

    # If the input was originally a list and is 1D, you might want to convert back:
    if isinstance(data, list) and obj.ndim == 1:
        return [chunk.tolist() for chunk in chunks]

    return chunks

if __name__ == '__main__':
    # --- Examples ---

    # 1. Simple List
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sizes = [1, 2, 3, 4]
    print(f"List Result: {split_by_sizes(x, sizes)}")

    # 2. 2D Numpy Array along Axis 1 (Columns)
    y = np.ones((3, 10))
    # Split 10 columns into sizes [2, 5, 3]
    y_split = split_by_sizes(y, [2, 5, 3], axis=1)
    print(f"NumPy Shape 1: {y_split[0].shape}\n\nResult: {y_split}") # (3, 2)
