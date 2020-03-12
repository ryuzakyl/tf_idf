import os
import pickle

import numpy as np
import pandas as pd

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
    def __filter_details_by_column_value(df, column_name, value):
        return df.loc[df[column_name] == value]

    def build_profiles(self):
        product_profiles = None
        user_profiles = None

        return product_profiles, user_profiles
