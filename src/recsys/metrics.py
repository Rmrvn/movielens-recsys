import math

def recall_at_k(recommended, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits/ len(relevant)

def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = sum( 
        1.0 / math.log2(pos + 2)
        for pos, item in enumerate(recommended[:k])
        if item in relevant
    )
    n_ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range (n_ideal))
    return dcg / idcg

def evaluate(recs, truth, k):
    """Средние Recall@k и NDCG@k по всем пользователям из truth.

    recs  -- словарь {user_id: список рекомендованных item_id по убыванию уверенности}
    truth -- словарь {user_id: коллекция item_id из теста}
    Пользователь без рекомендаций получает 0, чтобы модель не выигрывала,
    просто пропуская сложных пользователей.
    """
    users = list(truth)
    recall = sum(recall_at_k(recs.get(u,[]), truth[u], k) for u in users) / len(users)
    ndcg = sum(ndcg_at_k(recs.get(u, []), truth[u], k) for u in users) / len(users)
    return {f'recall@{k}': recall, f'ndcg@{k}': ndcg}


def coverage(recs, n_items, k):
    """Доля каталога, которая хотя бы раз попала в топ-k рекомендаций."""
    shown = set()
    for items in recs.values():
        shown.update(items[:k])
    return len(shown) / n_items


def mean_popularity(recs, item_counts, k):
    """Средняя популярность (число оценок в train) рекомендованных объектов."""
    values = [item_counts.get(i, 0) for items in recs.values() for i in items[:k]]
    return sum(values) / len(values) if values else 0.0