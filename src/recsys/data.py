from pathlib import Path

import numpy as np
import pandas as pd

def load_ratings(data_dir) -> pd.DataFrame:
    """Возвращает таблицу оценок """
    path = Path(data_dir) / 'ml-1m' / 'ratings.dat'
    return pd.read_csv(
        path, sep='::', engine = 'python',
        names = ['user_id', 'item_id', 'rating', 'timestamp'],
    )

def make_positive(ratings: pd.DataFrame, threshold: int = 4,
                  min_user_interactions: int = 5) -> pd.DataFrame:
    #Оставляет положительные взаимодействия (оценка >= threshold) и пользователей, у которых их не меньше min_user_interactions.
    pos = ratings[ratings['rating'] >= threshold]
    counts = pos.groupby('user_id')['item_id'].transform('size')
    return pos[counts >= min_user_interactions].reset_index(drop=True)


def temporal_split(interactions: pd.DataFrame, test_fraction: float = 0.2):
    """Для каждого пользователя последние test_fraction его взаимодействий
    по времени уходят в test, остальные в train."""
    df = interactions.sort_values(['user_id', 'timestamp', 'item_id']).reset_index(drop=True)
    rank = df.groupby('user_id').cumcount()
    total = df.groupby('user_id')['item_id'].transform('size')
    n_test = np.ceil(total*test_fraction)
    is_test = rank >= (total -n_test)
    return df[~is_test].copy(), df[is_test].copy()


def user_items(interactions: pd.DataFrame) -> dict:
    return interactions.groupby('user_id')['item_id'].apply(set).to_dict()
