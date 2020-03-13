import pandas as pd


def build_matrix(data, samples_labels, features_labels, extra_cols=None):
    """Builds a data set from raw data information.

    Args:
        data: The samples data (vector of features).
        samples_labels: The samples names or labels.
        features_labels: Labels for every feature in the feature vector.
        extra_cols: Extra columns for the data set (e.g. classes, properties, etc.)

    Returns:
        DataFrame: A Pandas DataFrame with the data set (samples as rows and features as columns).

    """
    # validating features labels
    features_count = len(features_labels)
    if features_count <= 0:
        raise ValueError('The amount of features must be positive.')

    # validating data
    if len(data) <= 0 or not all(len(li) == features_count for li in data):
        raise ValueError('All samples must have the same amount of features.')

    # creating the data frame
    df = pd.DataFrame(data, index=samples_labels, columns=features_labels)

    # checking for extra_cols param
    if extra_cols is None:
        return df

    # validating extra columns
    cols = extra_cols.values()
    if len(extra_cols) < 1 or not all(len(x) == len(list(cols)[0]) for x in cols):
        raise ValueError('Invalid extra columns.')

    # appending extra columns to data frame
    for c_new in extra_cols.keys():
        df[c_new] = pd.Series(extra_cols[c_new], index=df.index)

    # returning the built data set
    return df