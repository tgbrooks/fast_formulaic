import numpy as np
import polars as pl
import fast_formulaic
import formulaic


def test_from_formula():
    df = pl.DataFrame(
        {
            "cat1": ["a", "b", "c", "c", "c", "a"],
            "cat2": ["X", "X", "Y", "Y", "Z", "Z"],
            "cat3": ["A", "A", "A", "A", "A", "A"],
            "y": [0, 2, 0, 3, 4, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )

    configs = [
        "~ x : cat1",
        "~ 0 + cat1 : cat2",
        "~ x",
        "~ x + y",
        "~ cat1",
        "~ 1 + x",
        "~ cat3",
        "~ x*y*cat1*cat2*cat3",
    ]

    for formula in configs:
        res = fast_formulaic.from_formula(formula, df)
        spec = formulaic.ModelSpec(
            formula,
            materializer=formulaic.materializers.NarwhalsMaterializer,
            output="sparse",
        )
        desired = formulaic.model_matrix(spec, df)
        assert np.allclose(res.toarray(), desired.toarray())
        assert res.model_spec.structure == desired.model_spec.structure
