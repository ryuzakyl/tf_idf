import numpy as np


def cosine(x, y):
    """Cosine distance among two vectors.

    Args:
        x (list): The first vector.
        y (list): The second vector.

    Returns:
        float: The cosine distance between vectors x and y.

    """

    # converting python lists to numpy arrays
    x_arr = np.array(x)
    y_arr = np.array(y)

    # computing norms of both vectors
    x_norm = np.linalg.norm(x_arr)
    y_norm = np.linalg.norm(y_arr)

    # computing cosine between x and y
    cos = np.dot(x_arr, y_arr) / (x_norm * y_norm)

    # converting cosine in a distance/dissimilarity
    return 1 - cos
