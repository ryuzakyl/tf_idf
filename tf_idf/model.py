import os
import pickle
import pandas as pd


DATASET_REL_PATH = f"{os.path.dirname(__file__)}/../rs-cour-dataset"
DATASET_ABS_PATH = os.path.abspath(DATASET_REL_PATH)

MODEL_ABS_PATH = f"{os.path.dirname(__file__)}/model/tfidf.pickle"

# --------------------------------------------

def load_model():
    # load dataset model from pickle file
    dataset = None
    with open(MODEL_ABS_PATH, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def load_data():
    # load movies data
    movies_data_path = os.path.join(DATASET_ABS_PATH, 'movie-titles.csv')
    df_movies = pd.read_csv(movies_data_path, names=['iditem', 'title'])

    # load tags data
    tags_path = os.path.join(DATASET_ABS_PATH, 'movie-tags.csv')
    df_tags = pd.read_csv(tags_path, encoding="ISO-8859-1", names=['iditem', 'tag'])  # i had to use another encoding for this file, as it was not in utf-8 format

    # load ratings data
    ratings_path = os.path.join(DATASET_ABS_PATH, 'ratings.csv')
    df_ratings = pd.read_csv(ratings_path, names=['iduser', 'iditem', 'rating'])

    # load users data
    users_path = os.path.join(DATASET_ABS_PATH, 'users.csv')
    df_users = pd.read_csv(users_path, names=['iduser', 'username'])

    return {
        'movies': df_movies,
        'tags': df_tags,
        'ratings': df_ratings,
        'users': df_users
    }


def build_model(dataset):
    raise NotImplementedError()


def save_model(dataset):
    # save dataset model to pickle file
    with open(MODEL_ABS_PATH, 'wb') as f:
        pickle.dump(dataset, f)
