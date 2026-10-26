"""Двухэтапная схема: кандидаты от ALS и признаки для ранкера."""
import numpy as np
import pandas as pd

from recsys.models import ALSModel, ItemKNN


def build_candidates(history, users, als_params, n_candidates=100, target=None):
    """Строит таблицу «пользователь - фильм-кандидат» с признаками.

    history -- взаимодействия, на которых учатся модели и считаются все признаки
    target  -- если задан, добавляется колонка label: 1, если пара есть в target
    """
    als = ALSModel(**als_params).fit(history)
    knn = ItemKNN().fit(history)

    recs = als.recommend_with_scores(users, k=n_candidates)
    user_col, item_col, rank_col, score_col = [], [], [], []
    for u in users:
        items, scores = recs[u]
        user_col += [u] * len(items)
        item_col += items
        rank_col += range(1, len(items) + 1)
        score_col += scores
    df = pd.DataFrame({'user_id': user_col, 'item_id': item_col,
                       'als_rank': rank_col, 'als_score': score_col})

    df['knn_score'] = knn.score_pairs(df['user_id'], df['item_id'])

    item_counts = history['item_id'].value_counts()
    user_counts = history['user_id'].value_counts()
    df['item_pop'] = np.log1p(df['item_id'].map(item_counts).fillna(0))
    df['user_activity'] = np.log1p(df['user_id'].map(user_counts).fillna(0))

    if target is not None:
        pairs = set(zip(target['user_id'], target['item_id']))
        df['label'] = [int((u, i) in pairs) for u, i in zip(df['user_id'], df['item_id'])]
    return df

FEATURES = ['als_rank', 'als_score', 'knn_score', 'knn_rank', 'item_pop', 'user_activity']


def add_rank_features(df):
    """Добавляет knn_rank: место кандидата по оценке ItemKNN внутри списка своего пользователя."""
    df = df.copy()
    df['knn_rank'] = df.groupby('user_id')['knn_score'].rank(ascending=False, method='first')
    return df


def rerank(df, scores):
    """Сортирует кандидатов каждого пользователя по убыванию оценки.

    Возвращает {user_id: [item_id, ...]} - готовый вход для evaluate.
    """
    tmp = df[['user_id', 'item_id']].copy()
    tmp['score'] = np.asarray(scores)
    tmp = tmp.sort_values(['user_id', 'score'], ascending=[True, False], kind='stable')
    return tmp.groupby('user_id')['item_id'].apply(list).to_dict()