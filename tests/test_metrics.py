import pytest
from recsys.metrics import coverage, evaluate, mean_popularity, ndcg_at_k, recall_at_k


def test_recall_example():
    assert recall_at_k(['X', 'A', 'Y', 'B', 'Z'], {'A', 'B', 'C'}, k=5) == pytest.approx(2/3)



def test_ndcg_example():
    assert ndcg_at_k(['X', 'A', 'Y', 'B', 'Z'], {'A', 'B', 'C'}, k=5) == pytest.approx(0.498, abs=1e-3)


def test_perfect_ranking_gives_one():
    assert ndcg_at_k(['A', 'B', 'C'], {'A', 'B', 'C'}, k=3) == pytest.approx(1.0)


def test_no_hits_gives_zero():
    assert recall_at_k(['X', 'Y'], {'A'}, k=2) == 0.0
    assert ndcg_at_k(['X', 'Y'], {'A'}, k=2) == 0.0


def test_order_matters_for_ndcg_but_not_for_recall():
    early = ndcg_at_k(['A', 'X', 'Y'], {'A'}, k=3)
    late = ndcg_at_k(['X', 'Y', 'A'], {'A'}, k=3)
    assert early > late
    assert recall_at_k(['A', 'X', 'Y'], {'A'}, k=3) == recall_at_k(['X', 'Y', 'A'], {'A'}, k=3)


def test_evaluate_counts_users_without_recommendations_as_zero():
    result = evaluate(recs={1: ['A']}, truth={1: {'A'}, 2: {'B'}}, k=1)
    assert result['recall@1'] == pytest.approx(0.5)


def test_coverage():
    recs = {1: ['A', 'B'], 2: ['B', 'C']}
    assert coverage(recs, n_items=10, k=2) == pytest.approx(0.3)   # A, B, C из 10


def test_coverage_respects_k():
    recs = {1: ['A', 'B'], 2: ['B', 'C']}
    assert coverage(recs, n_items=10, k=1) == pytest.approx(0.2)   # только A и B


def test_mean_popularity():
    counts = {'A': 10, 'B': 2}
    assert mean_popularity({1: ['A', 'B']}, counts, k=2) == pytest.approx(6.0)