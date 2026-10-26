import numpy as np
import pandas as pd
import pytest

from recsys.models import ALSModel, ItemKNN
from recsys.ranking import add_rank_features, build_candidates, rerank


def make_knn_train():
    users = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10]
    items = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 3, 3, 1]
    return pd.DataFrame({'user_id': users, 'item_id': items})


def test_als_scores_are_sorted_and_match_dot_product():
    model = ALSModel(factors=3, iterations=5).fit(make_knn_train())
    items, scores = model.recommend_with_scores([10], k=3)[10]
    assert scores == sorted(scores, reverse=True)
    j = list(model.item_ids).index(items[0])
    expected = model.user_factors[model.user_index[10]] @ model.item_factors[j]
    assert scores[0] == pytest.approx(expected)


def test_item_knn_score_pairs():
    model = ItemKNN().fit(make_knn_train())
    scores = model.score_pairs([10, 10, 1], [2, 3, 1])
    assert scores == pytest.approx([3 / 12 ** 0.5, 0.0, 3 / 12 ** 0.5], rel=1e-5)


PARAMS = {'factors': 2, 'iterations': 5, 'reg': 0.1, 'alpha': 10.0}


def test_build_candidates_labels_and_ranks():
    target = pd.DataFrame({'user_id': [10], 'item_id': [2]})
    df = build_candidates(make_knn_train(), [10], PARAMS, n_candidates=3, target=target)
    assert set(df['user_id']) == {10}
    assert 1 not in set(df['item_id'])                  # просмотренное не может быть кандидатом
    assert list(df['als_rank']) == list(range(1, len(df) + 1))
    assert df.loc[df['item_id'] == 2, 'label'].tolist() == [1]
    assert df['label'].sum() == 1


def test_build_candidates_features_use_only_history():
    # Добавим в target три фильма: признаки от них измениться не должны
    target = pd.DataFrame({'user_id': [10, 10, 10], 'item_id': [2, 3, 4]})
    df = build_candidates(make_knn_train(), [10], PARAMS, n_candidates=3, target=target)
    assert df['user_activity'].iloc[0] == pytest.approx(np.log1p(1))   # у пользователя 10 один фильм в истории
    row = df[df['item_id'] == 2].iloc[0]
    assert row['item_pop'] == pytest.approx(np.log1p(3))               # фильм 2 оценили трое в истории


def test_add_rank_features():
    df = pd.DataFrame({'user_id': [1, 1, 1, 2, 2], 'item_id': [10, 11, 12, 10, 11],
                       'knn_score': [0.5, 2.0, 1.0, 3.0, 4.0]})
    out = add_rank_features(df)
    assert out['knn_rank'].tolist() == [3, 1, 2, 2, 1]


def test_rerank_sorts_by_score_within_user():
    df = pd.DataFrame({'user_id': [1, 1, 1, 2, 2], 'item_id': [10, 11, 12, 10, 11]})
    recs = rerank(df, [0.1, 0.9, 0.5, 0.7, 0.2])
    assert recs == {1: [11, 12, 10], 2: [10, 11]}