import os
import pickle
from tf_idf.managers import MoviesDatasetManager

# ------------------------------------------------------------


DATASET_REL_PATH = f"{os.path.dirname(__file__)}/rs-cour-dataset"
DATASET_ABS_PATH = os.path.abspath(DATASET_REL_PATH)

DATASET_OUTPUT_PATH = f"{os.path.dirname(__file__)}/model/tfidf.pickle"
TF_IDF_MODEL_PATH = f"{os.path.dirname(__file__)}/model/tf_idf_model.pickle"

# ------------------------------------------------------------


if __name__ == '__main__':
    # ------------ load data and save it to .pickle file ------------
    # ds_mgr = MoviesDatasetManager.from_csv_folder(DATASET_ABS_PATH)
    # ds_mgr.save_dataset(DATASET_OUTPUT_PATH)

    # ------------ load .pickle dataset ------------
    ds_mgr = MoviesDatasetManager(dataset_path=DATASET_OUTPUT_PATH)
    products_profile, users_profile = ds_mgr.build_profiles()
    model = {
        'products_profile': products_profile,
        'users_profile': users_profile
    }

    with open(TF_IDF_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print('Done!!!')