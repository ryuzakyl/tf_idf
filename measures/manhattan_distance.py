import numpy as np


def manhattan(x, y):
    """Manhattan distance among two vectors.

    Args:
        x (list): The first vector.
        y (list): The second vector.

    Returns:
        float: The manhattan distance between vectors x and y.

    """

    # converting python lists to numpy arrays
    x_arr = np.array(x)
    y_arr = np.array(y)

    # computing manhattan distance
    return np.linalg.norm(x_arr - y_arr)
