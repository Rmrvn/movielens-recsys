import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.models import ALSModel, _solve_factors


def make_knn_train():
    users = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10]
    items = [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 3, 3, 1]
    return pd.DataFrame({'user_id': users, 'item_id': items})


def test_solve_factors_matches_direct_formula():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(5, 3))
    alpha, reg = 10.0, 0.5
    matrix = sp.csr_matrix(np.array([[1, 0, 1, 0, 0]], dtype=np.float32))
    got = _solve_factors(matrix, Y, alpha, reg)[0]

    # Та же задача «в лоб»: полная формула со всеми пятью фильмами
    p = np.array([1, 0, 1, 0, 0], dtype=float)     # предпочтения
    c = 1 + alpha * p                              # уверенность
    C = np.diag(c)
    expected = np.linalg.solve(Y.T @ C @ Y + reg * np.eye(3), Y.T @ C @ p)
    assert got == pytest.approx(expected)


def test_als_factor_shapes():
    model = ALSModel(factors=4, iterations=2).fit(make_knn_train())
    assert model.user_factors.shape == (10, 4)
    assert model.item_factors.shape == (4, 4)


def test_als_filters_seen_items():
    train = make_knn_train()
    model = ALSModel(factors=2, iterations=5).fit(train)
    recs = model.recommend([1, 10], k=4)
    assert not set(recs[1]) & {1, 2}
    assert 1 not in recs[10]


def test_als_is_reproducible():
    train = make_knn_train()
    a = ALSModel(factors=3, iterations=5, seed=1).fit(train).recommend([10], k=3)
    b = ALSModel(factors=3, iterations=5, seed=1).fit(train).recommend([10], k=3)
    assert a == b


def test_als_prefers_similar_item():
    model = ALSModel(factors=2, iterations=10, alpha=10.0, reg=0.1).fit(make_knn_train())
    assert model.recommend([10], k=3)[10][0] == 2


def test_als_unknown_user_gets_nothing():
    model = ALSModel(factors=2, iterations=2).fit(make_knn_train())
    assert model.recommend([999], k=3) == {999: []}


