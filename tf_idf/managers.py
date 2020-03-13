import os
import math
import pickle

import numpy as np
import pandas as pd

from .utils import build_matrix

# ------------------------------------------------------------


class MoviesDatasetManager(object):
    def __init__(
            self,
            dataset_path=None,
            df_movies=None,
            df_tags=None,
            df_ratings=None,
            df_users=None):

        # path of serialized dataset provided
        if dataset_path:
            try:
                # load dataset from disk
                ds = MoviesDatasetManager.load_dataset(dataset_path)

                # if failed to load dataset
                if not ds:
                    raise IOError('Failed to load dataset from disk')

                self.df_movies = ds['movies']
                self.df_tags = ds['tags']
                self.df_ratings = ds['ratings']
                self.df_users = ds['users']
            except IOError as ioe:
                raise ioe
        else:
            # validate movies data
            if df_movies is None:
                raise ValueError('Movies data not provided')

            # validate tags data
            if df_tags is None:
                raise ValueError('Tags data not provided')

            # validate ratings data
            if df_ratings is None:
                raise ValueError('Ratings data not provided')

            # validate users data
            if df_users is None:
                raise ValueError('Users data not provided')

            self.df_movies = df_movies
            self.df_tags = df_tags
            self.df_ratings = df_ratings
            self.df_users = df_users

    def save_dataset(self, dataset_path):
        try:
            # build container dataset
            ds = {
                'movies': self.df_movies,
                'tags': self.df_tags,
                'ratings': self.df_ratings,
                'users': self.df_users
            }

            with open(dataset_path, 'wb') as f:
                pickle.dump(ds, f)

            success = True
        except IOError as ioe:
            print(ioe)
            success = False

        return success

    @staticmethod
    def load_dataset(dataset_path):
        try:
            with open(dataset_path, 'rb') as f:
                ds = pickle.load(f)
        except IOError as ioe:
            print(ioe)
            ds = None

        return ds

    @staticmethod
    def from_dataset_file(dataset_path):
        return MoviesDatasetManager(dataset_path)

    @staticmethod
    def __load_movies_from_file(movies_data_path):
        # load movies data
        df_movies = pd.read_csv(movies_data_path, names=['iditem', 'title'])

        return df_movies

    @staticmethod
    def __load_tags_from_file(tags_path):
        # load tags data
        df_tags = pd.read_csv(tags_path, encoding="ISO-8859-1", names=['iditem', 'tag'])  # i had to use another encoding for this file, as it was not in utf-8 format

        return df_tags

    @staticmethod
    def __load_ratings_from_file(ratings_path):
        # load ratings data
        df_ratings = pd.read_csv(ratings_path, names=['iduser', 'iditem', 'rating'])

        return df_ratings

    @staticmethod
    def __load_users_from_file(users_path):
        # load users data
        df_users = pd.read_csv(users_path, names=['iduser', 'username'])

        return df_users

    @staticmethod
    def from_csv_folder(csv_folder_path):
        # validate provided folder
        if not os.path.isdir(csv_folder_path):
            raise ValueError('Invalid folder path provided')

        # load movies data from .csv file
        movies_path = os.path.join(csv_folder_path, 'movie-titles.csv')
        movies = MoviesDatasetManager.__load_movies_from_file(movies_path)

        # load tags data from .csv file
        tags_path = os.path.join(csv_folder_path, 'movie-tags.csv')
        tags = MoviesDatasetManager.__load_tags_from_file(tags_path)

        # load ratings data from .csv file
        ratings_path = os.path.join(csv_folder_path, 'ratings.csv')
        ratings = MoviesDatasetManager.__load_ratings_from_file(ratings_path)

        # load reatings data from .csv file
        users_path = os.path.join(csv_folder_path, 'users.csv')
        users = MoviesDatasetManager.__load_users_from_file(users_path)

        return MoviesDatasetManager(
            df_movies=movies,
            df_tags=tags,
            df_ratings=ratings,
            df_users=users
        )

    # -------------------------------------------------

    @staticmethod
    def __filter_by_column_value(df, column_name, value):
        return df.loc[df[column_name] == value]

    def __build_product_profiles(self):
        # get unique tags and movies
        unique_tags = self.df_tags['tag'].unique()
        unique_movies = self.df_movies['iditem'].unique()[:2]

        # build empty products profile matrix
        n_movies = len(unique_movies)

        # compute idf
        idf = np.array([
            # amount of movies with given tag
            math.log(n_movies / len(self.df_tags.loc[self.df_tags['tag'] == t, 'iditem'].unique()))
            for t in unique_tags
        ])

        # compute tf matrix
        tf = np.array([[
                # amount of times the movie was tagged with a specific tag
                self.df_tags.loc[(self.df_tags['tag'] == t) & (self.df_tags['iditem'] == m)].shape[0]
                for t in unique_tags
            ]
            for m in unique_movies
        ])

        # compute TF-IDF matrix
        tf_idf = np.array([
            # pairwise product (same as foreach etiqueta TFproducto,etiqueta * IDFetiqueta)
            np.multiply(tf[i, :], idf)
            for i in range(n_movies)
        ])

        # normalizing TF-IDF matrix
        sum_of_rows = tf_idf.sum(axis=1)
        tf_idf_norm = tf_idf / sum_of_rows[:, np.newaxis]

        # build matrix
        matrix = build_matrix(tf_idf_norm, unique_movies, unique_tags)

        return matrix

    def __build_user_profiles(self):
        users_profile = None

        # get unique tags
        # unique_tags = self.df_tags['tag'].unique()

        return users_profile

    def build_profiles(self):
        # build the products profile matrix
        products_profile = self.__build_product_profiles()

        # build the users profile matrix
        users_profile = self.__build_user_profiles()

        return products_profile, users_profile
