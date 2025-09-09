import scipy.sparse as spsparse
import numpy as np
import polars as pl
import fast_formulaic


def test_from_formula():
    df = pl.DataFrame(
        {
            "cat1": ["a", "b", "c", "c", "c", "a"],
            "cat2": ["X", "X", "Y", "Y", "Z", "Z"],
            "y": [0, 2, 0, 3, 4, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )

    configs = [
        (
            "~ x : cat1",
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 6],
                    [0, 2, 0, 0, 0, 0],
                    [0, 0, 3, 4, 5, 0],
                ]
            ),
        ),
        (
            "~ 0 + cat1 : cat2",
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],  # a,X
                    [0, 1, 0, 0, 0, 0],  # b,X
                    [0, 0, 0, 0, 0, 0],  # c,X
                    [0, 0, 0, 0, 0, 0],  # a,Y
                    [0, 0, 0, 0, 0, 0],  # b,Y
                    [0, 0, 1, 1, 0, 0],  # c,Y
                    [0, 0, 0, 0, 0, 1],  # a,Z
                    [0, 0, 0, 0, 0, 0],  # b,Z
                    [0, 0, 0, 0, 1, 0],  # c,Z
                ]
            ),
        ),
    ]

    for formula, desired in configs:
        res = fast_formulaic.from_formula(formula, df)
        assert np.allclose(res.toarray(), desired.T)
