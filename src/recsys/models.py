import numpy as np
import scipy.sparse as sp

from recsys.data import user_items

class PopularModel:


    def __init__(self, filter_seen=True):
        self.filter_seen = filter_seen
        self.ranking = []
        self.seen = {}

    def fit(self, train):
        counts = train.groupby('item_id').size().sort_values(ascending=False, kind='stable')
        self.ranking = counts.index.tolist()
        self.seen = user_items(train)
        return self

    def recommend(self, user_ids, k):
        recs = {}
        for user in user_ids:
            seen = self.seen.get(user, set()) if self.filter_seen else set()
            items = []
            for item in self.ranking:
                if item not in seen:
                    items.append(item)
                    if len(items) == k:
                        break
            recs[user] = items
        return recs


class ItemKNN:
    def __init__(self, filter_seen=True, batch_size=1000):
        self.filter_seen  = filter_seen
        self.batch_size = batch_size
       
       
    def fit(self, train):
        self.user_ids = np.sort(train['user_id'].unique())
        self.item_ids = np.sort(train['item_id'].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        item_index = {it: i for i, it in enumerate(self.item_ids)}

        rows = train['user_id'].map(self.user_index).to_numpy()
        cols = train['item_id'].map(item_index).to_numpy()

        values = np.ones(len(train), dtype=np.float32)
        self.matrix = sp.csr_matrix(
            (values, (rows, cols)),
            shape = (len(self.user_ids), len(self.item_ids)),
        )

        co = (self.matrix.T @ self.matrix).toarray()
        norms = np.sqrt(np.diag(co))
        co /= norms[:, None]
        co /= norms[None, :]
        np.fill_diagonal(co, 0.0)
        self.sim = co
        return self

    def recommend(self, user_ids, k):
        k = min(k, len(self.item_ids))
        recs = {u: [] for u in user_ids}
        known = [u for u in user_ids if u in self.user_index]

        for start in range(0, len(known), self.batch_size):
            batch = known[start: start + self.batch_size]
            history = self.matrix[[self.user_index[u] for u in batch]]

            scores = history @ self.sim

            if self.filter_seen:
                r, c = history.nonzero()
                scores[r,c] = -np.inf

            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1, kind='stable')
            top = np.take_along_axis(top, order, axis=1)
            top_scores = np.take_along_axis(top_scores, order, axis=1)


            for u, idx, sc in zip(batch, top, top_scores):
                recs[u] = [int(self.item_ids[j]) for j, s in zip(idx, sc) if s > 0]
            
        return recs


    def score_pairs(self, user_ids, item_ids):
        """Оценка ItemKNN для конкретных пар (пользователь, фильм)."""
        users = np.asarray(user_ids)
        item_index = {it: i for i, it in enumerate(self.item_ids)}
        cols = np.array([item_index[i] for i in item_ids])
        out = np.zeros(len(users), dtype=np.float32)

        unique_users = np.unique(users)
        for start in range(0, len(unique_users), self.batch_size):
            batch = unique_users[start:start + self.batch_size]
            scores = self.matrix[[self.user_index[u] for u in batch]] @ self.sim
            local = {u: i for i, u in enumerate(batch)}
            mask = np.isin(users, batch)
            local_rows = np.array([local[u] for u in users[mask]])
            out[mask] = scores[local_rows, cols[mask]]
        return out


def _build_matrix(train):
    user_ids = np.sort(train['user_id'].unique())
    item_ids = np.sort(train['item_id'].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {it: i for i , it in enumerate(item_ids)}
    rows = train['user_id'].map(user_index).to_numpy()
    cols = train['item_id'].map(item_index).to_numpy()
    values = np.ones(len(train), dtype = np.float32)
    matrix = sp.csr_matrix((values, (rows, cols)), shape =(len(user_ids), len(item_ids)))
    return user_ids, item_ids, user_index, matrix


def _top_k(scores, k):
    top = np.argpartition(-scores, k-1, axis=1)[:, :k]
    top_scores  = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-top_scores, axis=1, kind='stable')
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(top_scores, order, axis=1)\


def _solve_factors(matrix, fixed, alpha, reg):
    n_factors = fixed.shape[1]
    base = fixed.T @ fixed + reg * np.eye(n_factors)
    result = np.zeros((matrix.shape[0], n_factors))
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        cols = matrix.indices[start:end]
        if len(cols) == 0:
            continue
        y = fixed[cols]
        a = base + alpha * (y.T @ y)
        b = (1.0 + alpha) * y.sum(axis=0)
        result[row] = np.linalg.solve(a, b)
    return result


class ALSModel:
    def __init__(self, factors=32, iterations=15, alpha=20.0, reg=100.0,
                 filter_seen=True, batch_size=1000, seed=42):
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.reg = reg
        self.filter_seen = filter_seen
        self.batch_size = batch_size
        self.seed = seed


    def fit(self, train):
        self.user_ids, self.item_ids, self.user_index, self.matrix = _build_matrix(train)
        n_users, n_items = self.matrix.shape

        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0.0, 0.01, size=(n_items, self.factors))
        self.user_factors = np.zeros((n_users, self.factors))

        item_matrix = self.matrix.T.tocsr()
        for  _ in range(self.iterations):
            self.user_factors = _solve_factors(self.matrix, self.item_factors, self.alpha, self.reg)
            self.item_factors = _solve_factors(item_matrix, self.user_factors, self.alpha, self.reg)
        return self

    def recommend_with_scores(self, user_ids, k):
        """Как recommend, но возвращает {user_id: (список фильмов, список оценок)}."""
        k = min(k, len(self.item_ids))
        recs = {u: ([], []) for u in user_ids}
        known = [u for u in user_ids if u in self.user_index]

        for start in range(0, len(known), self.batch_size):
            batch = known[start:start + self.batch_size]
            rows = [self.user_index[u] for u in batch]
            scores = self.user_factors[rows] @ self.item_factors.T

            if self.filter_seen:
                r, c = self.matrix[rows].nonzero()
                scores[r, c] = -np.inf

            top, top_scores = _top_k(scores, k)
            for u, idx, sc in zip(batch, top, top_scores):
                keep = [(int(self.item_ids[j]), float(s)) for j, s in zip(idx, sc) if np.isfinite(s)]
                recs[u] = ([item for item, _ in keep], [score for _, score in keep])
        return recs

    def recommend(self, user_ids, k):
        return {u: items for u, (items, _) in self.recommend_with_scores(user_ids, k).items()}