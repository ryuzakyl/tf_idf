import numpy as np


def euclidean(x, y):
    """Euclidean distance among two vectors.

    Args:
        x (list): The first vector.
        y (list): The second vector.

    Returns:
        float: The euclidean distance between vectors x and y.

    """

    # converting python lists to numpy arrays
    x_arr = np.array(x)
    y_arr = np.array(y)

    # computing euclidean distance
    return np.linalg.norm(x_arr - y_arr)
