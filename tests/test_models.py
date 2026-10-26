import pandas as pd
import pytest

from recsys.models import ItemKNN, PopularModel

def make_train():
    return pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'item_id': [10, 20, 10, 30, 10],
    })

def test_popular_order_without_filter():
    model = PopularModel(filter_seen=False).fit(make_train())
    assert model.recommend([3], k=2) == {3: [10,20]}


def test_popular_skips_seen_items():
    model = PopularModel(). fit(make_train())
    recs = model.recommend([1,2,3], k=2)
    assert recs[1] == [30]
    assert recs[2] == [20]
    assert recs[3] == [20, 30]


def test_unknown_user_gets_popular():
    model = PopularModel().fit(make_train())
    assert model.recommend([99], k=2) == {99: [10,20]}

def make_knn_train():
    pairs = [
        (1,1), (1,2), (2,1), (2, 2), (3, 1), (3, 2),
        (4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4),
        (7, 3), (8, 3), (9, 3), (10, 1),
    ]
    return pd.DataFrame(pairs, columns=['user_id', 'item_id'])

def test_item_knn_similarity_value():
    model = ItemKNN().fit(make_knn_train())
    i1 = list(model.item_ids).index(1)
    i2 = list(model.item_ids).index(2)

    assert model.sim[i1, i2] == pytest.approx(3 / 12 **0.5, rel=1e-5)
    assert model.sim[i1, i1] == 0


def test_item_knn_recommend_similar_not_popular():
    train = make_knn_train()
    assert ItemKNN().fit(train).recommend([10], k=2) =={10: [2]}
    assert PopularModel().fit(train).recommend([10], k=1) == {10: [3]}

def test_item_knn_filters_seen_items():
    train = make_knn_train()
    with_filter = ItemKNN(filter_seen=True).fit(train).recommend([1], k=5)[1]
    without_filter = ItemKNN(filter_seen=False).fit(train).recommend([1], k=5)[1]
    assert with_filter == []
    assert set(without_filter) == {1,2}


def test_item_knn_unknown_user_gets_nothing():
    model = ItemKNN().fit(make_knn_train())
    assert model.recommend([999], k=3) == {999: []}