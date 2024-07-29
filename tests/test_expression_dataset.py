import numpy as np
from scipy.sparse import csr_matrix
from persist import ExpressionDataset


def test_expression_dataset():
    """Tests ExpressionDataset for sparse inputs."""
    np.random.seed(0)
    labels = np.array([1, 2, 3]).astype(np.int_)
    x = np.random.rand(3, 6)
    x[x < 0.5] = 0
    sparse_x = csr_matrix(x)

    # check dense entries can be recovered
    for i in range(x.shape[0]):
        assert np.all(x[i] == sparse_x[i].toarray()), f"Row {i} is not the same"

    dense_dataset = ExpressionDataset(x, labels)
    sparse_dataset = ExpressionDataset(sparse_x, labels)

    # ExpressionDataset flags:
    assert not (dense_dataset.data_issparse), "expecting non-sparse data"
    assert sparse_dataset.data_issparse, "expecting sparse data"

    # ExpressionDataset emits X and label; we want identical results in both cases.
    for (x1, l1), (x2, l2) in zip(dense_dataset, sparse_dataset):
        assert l1 == l2, "Labels returned by ExpressionDataset are not the same"
        assert np.all(x1 == x2), "Data returned by ExpressionDataset is not the same"

    return
