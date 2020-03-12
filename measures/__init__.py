import enum

from .euclidean_distance import euclidean
from .manhattan_distance import manhattan
from .minkowski_distance import minkowski
from .cosine_distance import cosine

# ------------------------------------------------------


class Distance(enum.Enum):
    euclidean = 1
    manhattan = 2
    minkowski = 3
    cosine = 4


# map of distances ids and their names
distances_names = {
    Distance.euclidean: 'Euc',
    Distance.manhattan: 'Man',
    Distance.minkowski: 'Min',
    Distance.cosine: 'Cos',
}

# map of distances id and the corresponding callable
distance_to_function = {
    Distance.euclidean: euclidean,
    Distance.manhattan: manhattan,
    Distance.minkowski: minkowski,
    Distance.cosine: cosine,
}
